// gpuinfo — query what the Metal driver will actually let us hold resident,
// plus a compute-bound GPU clock probe used as a thermal-throttle signal.
//
// FROZEN. Two modes:
//
//   gpuinfo              one line of key=value: device, working-set limits,
//                        current allocated size.
//   gpuinfo --clock      compute-bound FMA loop; prints GFLOPs. This tracks GPU
//                        clock, so it drops when the chip throttles. It is the
//                        thermal signal on this machine because powermetrics
//                        needs sudo and pmset -g therm only reports a warning
//                        level that is almost never set.
//
// recommendedMaxWorkingSetSize is the number that decides whether the model is
// fully resident. On Apple Silicon it tracks iogpu.wired_limit_mb when that is
// set, and defaults to a driver-chosen fraction of RAM when it is 0.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static const char *kClockSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Pure-FMA, register-resident, zero memory traffic. Throughput is proportional
// to (GPU clock x active cores), which is exactly what thermal throttling and
// power capping reduce.
kernel void clockprobe(device float *out [[buffer(0)]],
                       constant uint &iters [[buffer(1)]],
                       uint gid [[thread_position_in_grid]])
{
    float a = (float)gid * 1e-7f, b = 1.0000001f, c = 0.9999999f;
    float x0 = a, x1 = a + 1.0f, x2 = a + 2.0f, x3 = a + 3.0f;
    float y0 = a, y1 = a + 1.0f, y2 = a + 2.0f, y3 = a + 3.0f;
    for (uint i = 0; i < iters; ++i) {
        x0 = fma(x0, b, c); y0 = fma(y0, c, b);
        x1 = fma(x1, b, c); y1 = fma(y1, c, b);
        x2 = fma(x2, b, c); y2 = fma(y2, c, b);
        x3 = fma(x3, b, c); y3 = fma(y3, c, b);
    }
    float s = ((x0 + x1) + (x2 + x3)) + ((y0 + y1) + (y2 + y3));
    if (s == -1.0e30f) out[gid] = s;   // never true; defeats DCE
}
)MSL";

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }

        int want_clock = (argc > 1 && !strcmp(argv[1], "--clock"));

        if (!want_clock) {
            printf("device=\"%s\" ", dev.name.UTF8String);
            printf("recommendedMaxWorkingSetSize_MiB=%llu ",
                   (unsigned long long)(dev.recommendedMaxWorkingSetSize / 1048576ull));
            printf("maxBufferLength_MiB=%llu ",
                   (unsigned long long)(dev.maxBufferLength / 1048576ull));
            printf("currentAllocatedSize_MiB=%llu ",
                   (unsigned long long)(dev.currentAllocatedSize / 1048576ull));
            printf("hasUnifiedMemory=%d ", (int)dev.hasUnifiedMemory);
            printf("maxThreadgroupMemory_KiB=%lu\n",
                   (unsigned long)(dev.maxThreadgroupMemoryLength / 1024));
            return 0;
        }

        NSError *err = nil;
        id<MTLLibrary> lib =
            [dev newLibraryWithSource:[NSString stringWithUTF8String:kClockSrc]
                              options:[MTLCompileOptions new] error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.description.UTF8String); return 1; }
        id<MTLComputePipelineState> p =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"clockprobe"]
                                               error:&err];
        if (!p) { fprintf(stderr, "pipeline: %s\n", err.description.UTF8String); return 1; }

        const uint32_t threads = 1u << 17;      // 131072
        const uint32_t tg      = 256;
        const uint32_t iters   = 4000;
        // The GPU ramps its clock over the first few dispatches, so a short
        // run reads low on an idle machine and would be misread as throttling.
        // Discard the ramp, then take the best of the rest.
        const int      reps    = 40;
        const int      warmup  = 20;

        id<MTLBuffer> out = [dev newBufferWithLength:(size_t)threads * 4
                                             options:MTLResourceStorageModePrivate];
        id<MTLCommandQueue> q = [dev newCommandQueue];

        double best = 0.0;
        for (int r = 0; r < reps; r++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBuffer:out offset:0 atIndex:0];
            [e setBytes:&iters length:4 atIndex:1];
            [e dispatchThreads:MTLSizeMake(threads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (r < warmup) continue;
            double dt = cb.GPUEndTime - cb.GPUStartTime;
            // 8 fma per inner iteration, 2 flop per fma
            double flops = (double)threads * (double)iters * 8.0 * 2.0;
            double gf = flops / dt / 1e9;
            if (gf > best) best = gf;
        }
        printf("%.1f\n", best);
    }
    return 0;
}
