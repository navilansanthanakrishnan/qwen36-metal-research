// mmapeak.m — peak arithmetic throughput of the matrix paths on this GPU.
//
//   clang -fobjc-arc -O2 -framework Foundation -framework Metal probe/mmapeak.m -o probe/mmapeak
//   probe/mmapeak
//
// WHY THIS EXISTS.
//
// Two kernel families are on the table for speculative verify and they fail for
// opposite reasons:
//
//   - kernel_mul_mv_sgq4k_f32 (ours) holds the dequantized A fragment in
//     REGISTERS via simdgroup_float8x8::thread_elements() and never touches
//     threadgroup memory. Measured 4.10 TFLOP/s at n=8 on m=4096 k=14336.
//   - the tensor-API mul_mm reaches 16.41 TFLOP/s on the same Q4_K weights at
//     n=512, but must STAGE A through threadgroup memory, and LEDGER 034 showed
//     narrowing its N tile makes that staging dominate.
//
// Whether the first design has any headroom left depends on a number nobody in
// this project has measured: the peak rate of simdgroup_multiply_accumulate
// itself, with zero memory traffic. LEDGER 035 ASSUMED it was 17.6 TFLOP/s (the
// rate mul_mm attains at n=512) and every roofline since has inherited that.
// But on Apple GPUs before M5 the simdgroup matrix intrinsics are decomposed
// onto the ordinary FP32 ALUs, and M5's dedicated units are exposed through the
// tensor API, which is a DIFFERENT code path. If simdgroup MMA runs at scalar
// rate then the register-resident design has a hard ceiling near 6.3 TFLOP/s
// and no amount of load tuning gets past it; if it runs at 17 TFLOP/s then the
// kernel is load-bound and loads are the thing to fix.
//
// This probe answers exactly that and nothing else. All operands are register
// resident, the loop carries a true dependency through the accumulator so it
// cannot be hoisted, and the result is written out so it cannot be dead-coded.
// Iteration count is swept to confirm the time is linear in work (a constant
// offset would mean the loop was folded).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

static NSString * const kSrc = @R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Four independent accumulators so MMA issue latency is hidden and we measure
// throughput rather than dependent-op latency.
kernel void mma_f32(device float * out [[buffer(0)]],
                    constant uint & iters [[buffer(1)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    ushort tiisg [[thread_index_in_simdgroup]],
                    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    simdgroup_float8x8 a, b, c0, c1, c2, c3;
    { thread auto & e = a.thread_elements();  e[0] = 1.0f + tiisg; e[1] = 0.5f; }
    { thread auto & e = b.thread_elements();  e[0] = 0.25f; e[1] = 1.5f + sgitg; }
    { thread auto & e = c0.thread_elements(); e[0] = 0.0f; e[1] = 0.0f; }
    { thread auto & e = c1.thread_elements(); e[0] = 0.0f; e[1] = 0.0f; }
    { thread auto & e = c2.thread_elements(); e[0] = 0.0f; e[1] = 0.0f; }
    { thread auto & e = c3.thread_elements(); e[0] = 0.0f; e[1] = 0.0f; }

    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }

    thread auto & e0 = c0.thread_elements();
    thread auto & e1 = c1.thread_elements();
    thread auto & e2 = c2.thread_elements();
    thread auto & e3 = c3.thread_elements();
    out[tgid*32 + tiisg] = e0[0] + e1[0] + e2[0] + e3[0] + e0[1] + e1[1] + e2[1] + e3[1];
}

// Same shape, half A/B against a float accumulator. LEDGER 055 found this is
// accepted by MSL; whether it is FASTER is the open question, and 055 could only
// answer it inside a load-bound kernel where it would not show.
kernel void mma_f16(device float * out [[buffer(0)]],
                    constant uint & iters [[buffer(1)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    ushort tiisg [[thread_index_in_simdgroup]],
                    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    simdgroup_half8x8  a, b;
    simdgroup_float8x8 c0, c1, c2, c3;
    { thread auto & e = a.thread_elements();  e[0] = (half)(1.0f + tiisg); e[1] = (half)0.5f; }
    { thread auto & e = b.thread_elements();  e[0] = (half)0.25f; e[1] = (half)(1.5f + sgitg); }
    { thread auto & e = c0.thread_elements(); e[0] = 0.0f; e[1] = 0.0f; }
    { thread auto & e = c1.thread_elements(); e[0] = 0.0f; e[1] = 0.0f; }
    { thread auto & e = c2.thread_elements(); e[0] = 0.0f; e[1] = 0.0f; }
    { thread auto & e = c3.thread_elements(); e[0] = 0.0f; e[1] = 0.0f; }

    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }

    thread auto & e0 = c0.thread_elements();
    thread auto & e1 = c1.thread_elements();
    thread auto & e2 = c2.thread_elements();
    thread auto & e3 = c3.thread_elements();
    out[tgid*32 + tiisg] = e0[0] + e1[0] + e2[0] + e3[0] + e0[1] + e1[1] + e2[1] + e3[1];
}

// Scalar FP32 fma reference. Validates the harness against the 6258 GFLOP/s
// this machine is known to reach (HARDWARE.md), so a surprising MMA number can
// be trusted or blamed on the harness.
kernel void fma_f32(device float * out [[buffer(0)]],
                    constant uint & iters [[buffer(1)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    ushort tiisg [[thread_index_in_simdgroup]]) {
    float4 acc0 = float4(tiisg), acc1 = float4(tiisg+1u);
    float4 acc2 = float4(tiisg+2u), acc3 = float4(tiisg+3u);
    const float4 m = float4(1.0000001f), n = float4(0.9999999f);
    for (uint i = 0; i < iters; ++i) {
        acc0 = fma(acc0, m, n); acc1 = fma(acc1, m, n);
        acc2 = fma(acc2, m, n); acc3 = fma(acc3, m, n);
    }
    float4 s = acc0 + acc1 + acc2 + acc3;
    out[tgid*32 + tiisg] = s.x + s.y + s.z + s.w;
}
)MSL";

static double run(id<MTLDevice> dev, id<MTLCommandQueue> q,
                  id<MTLComputePipelineState> p,
                  id<MTLBuffer> out, uint iters, int ntg, int nsg) {
    // warm up, then time a second submission: first submit pays pipeline warm-up
    for (int rep = 0; rep < 2; ++rep) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:sizeof(iters) atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(ntg,1,1)
          threadsPerThreadgroup:MTLSizeMake(32*nsg,1,1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (rep == 1) return cb.GPUEndTime - cb.GPUStartTime;
    }
    return 0;
}

int main(void) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device: %s\n", dev.name.UTF8String);
    id<MTLCommandQueue> q = [dev newCommandQueue];

    NSError * err = nil;
    MTLCompileOptions * opt = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:opt error:&err];
    if (!lib) { printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }

    const char * names[3] = { "fma_f32", "mma_f32", "mma_f16" };
    // flops per thread-iteration:
    //   fma_f32: 4 float4 fma = 4*4*2 = 32 flops/thread
    //   mma_*  : 4 MMA of 8x8x8 per SIMDGROUP = 4*1024 flops / 32 threads = 128 flops/thread
    const double fpi[3] = { 32.0, 128.0, 128.0 };

    const int nsg = 4;            // simdgroups per threadgroup
    const int ntg = 16 * 32;      // 16 GPU cores, oversubscribed 32x

    id<MTLBuffer> out = [dev newBufferWithLength:sizeof(float)*ntg*32*nsg
                                         options:MTLResourceStorageModeShared];

    printf("\n%-9s %10s %12s %12s %10s\n", "kernel", "iters", "time_ms", "GFLOP", "TFLOP/s");
    for (int k = 0; k < 3; ++k) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:names[k]]];
        if (!fn) { printf("%-9s  MISSING\n", names[k]); continue; }
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!p) { printf("%-9s  pipeline failed: %s\n", names[k], err.localizedDescription.UTF8String); continue; }

        // sweep iteration counts: throughput must be flat and time linear, or the
        // loop was folded and the number is fiction
        for (uint it = 65536; it <= 262144; it *= 2) {
            double s = run(dev, q, p, out, it, ntg, nsg);
            // total threads = ntg * nsg * 32
            double threads = (double) ntg * nsg * 32.0;
            double gflop = threads * (double) it * fpi[k] / 1e9;
            printf("%-9s %10u %12.3f %12.1f %10.2f\n",
                   names[k], it, s*1e3, gflop, gflop/1e3/s);
        }
        printf("\n");
    }
    return 0;
}}
