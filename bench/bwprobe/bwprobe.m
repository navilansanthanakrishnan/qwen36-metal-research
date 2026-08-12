// bwprobe — minimal Metal streaming-read bandwidth probe.
//
// FROZEN. Do not tune this to make a number look better. If a later result
// looks strange, re-run this unchanged and compare against HARDWARE.md.
//
// Method: allocate a private-storage buffer several times larger than any
// cache (default 2 GiB, SLC on this chip class is tens of MB), read every
// byte of it exactly once per dispatch with fully coalesced float4 loads and
// 4-deep memory-level parallelism, and time many dispatches with the command
// buffer's own GPU timestamps. The accumulator is consumed by a comparison
// that is never true, so nothing is dead-code eliminated but nothing is
// written back either — this measures read bandwidth, not read+write.
//
// Reported: median and best GB/s over the measured dispatches. Use the
// median. "GB" here is 1e9 bytes, matching how vendors quote peak.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Fill so every page of the private buffer is real and resident.
kernel void fill(device float4 *dst [[buffer(0)]],
                 constant uint  &n   [[buffer(1)]],
                 uint gid [[thread_position_in_grid]],
                 uint gsz [[threads_per_grid]])
{
    for (uint i = gid; i < n; i += gsz) {
        dst[i] = float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

// Streaming read. Four independent grid-stride streams: each individual load
// is coalesced across the threadgroup, and the four are independent so the
// memory system sees four outstanding requests per thread.
kernel void read_bw(device const float4 *src [[buffer(0)]],
                    device float        *out [[buffer(1)]],
                    constant uint       &n   [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint gsz [[threads_per_grid]])
{
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const uint stride = gsz * 4u;
    for (uint i = gid; i + 3u * gsz < n; i += stride) {
        a0 += src[i];
        a1 += src[i + gsz];
        a2 += src[i + 2u * gsz];
        a3 += src[i + 3u * gsz];
    }
    const float4 a = (a0 + a1) + (a2 + a3);
    const float  s = a.x + a.y + a.z + a.w;
    if (s == -1.0e30f) {          // never true; defeats DCE, never stores
        out[gid] = s;
    }
}
)MSL";

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        // ---- frozen probe parameters ----------------------------------
        size_t   buf_bytes   = 2ull * 1024 * 1024 * 1024;  // 2 GiB
        uint32_t threads_tot = 1u << 20;                   // 1,048,576 threads
        uint32_t tg_size     = 256;
        int      warmup      = 3;
        int      iters       = 20;

        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--mib") && i + 1 < argc)
                buf_bytes = (size_t)atoll(argv[++i]) * 1024 * 1024;
            else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
                iters = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
                threads_tot = (uint32_t)atoll(argv[++i]);
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }

        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                               options:opts
                                                 error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.description.UTF8String); return 1; }

        id<MTLComputePipelineState> pFill =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fill"] error:&err];
        id<MTLComputePipelineState> pRead =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"read_bw"] error:&err];
        if (!pFill || !pRead) { fprintf(stderr, "pipeline: %s\n", err.description.UTF8String); return 1; }

        // n must be an exact multiple of 4*threads_tot so every element is read
        // exactly once and the loop has no tail.
        uint32_t chunk = threads_tot * 4u;
        uint32_t n     = (uint32_t)((buf_bytes / 16) / chunk) * chunk;
        size_t   bytes = (size_t)n * 16;

        id<MTLBuffer> src = [dev newBufferWithLength:bytes
                                             options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [dev newBufferWithLength:(size_t)threads_tot * 4
                                             options:MTLResourceStorageModePrivate];
        if (!src || !out) { fprintf(stderr, "alloc %zu failed\n", bytes); return 1; }

        id<MTLCommandQueue> q = [dev newCommandQueue];
        MTLSize grid = MTLSizeMake(threads_tot, 1, 1);
        MTLSize tg   = MTLSizeMake(tg_size, 1, 1);

        // ---- fill (also forces residency) ------------------------------
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pFill];
            [e setBuffer:src offset:0 atIndex:0];
            [e setBytes:&n length:4 atIndex:1];
            [e dispatchThreads:grid threadsPerThreadgroup:tg];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        double *gbs = (double *)calloc((size_t)iters, sizeof(double));

        for (int it = 0; it < warmup + iters; it++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pRead];
            [e setBuffer:src offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:grid threadsPerThreadgroup:tg];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (it >= warmup) {
                double dt = cb.GPUEndTime - cb.GPUStartTime;   // seconds
                gbs[it - warmup] = (double)bytes / dt / 1e9;
            }
        }

        qsort(gbs, (size_t)iters, sizeof(double), cmp_double);
        double med  = gbs[iters / 2];
        double best = gbs[iters - 1];
        double wrst = gbs[0];

        printf("device            %s\n", dev.name.UTF8String);
        printf("buffer_bytes      %zu (%.2f GiB)\n", bytes, bytes / 1073741824.0);
        printf("threads           %u  threadgroup %u\n", threads_tot, tg_size);
        printf("max_tg_threads    %lu\n", (unsigned long)pRead.maxTotalThreadsPerThreadgroup);
        printf("iters             %d (after %d warmup)\n", iters, warmup);
        printf("read_GBps_median  %.1f\n", med);
        printf("read_GBps_best    %.1f\n", best);
        printf("read_GBps_worst   %.1f\n", wrst);
        free(gbs);
    }
    return 0;
}
