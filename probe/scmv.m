// scmv.m — column-exact scalar multi-column Q4_K mat-vec, feasibility probe.
//
//   clang -fobjc-arc -O2 -framework Foundation -framework Metal probe/scmv.m -o probe/scmv
//   probe/scmv [M] [K]
//
// WHY THIS EXISTS (LEDGER 082/083).
//
// Verify at width W needs W columns of arithmetic, not eight. sgmv computes a
// fixed 8-wide simdgroup fragment whatever ne11 is and masks the store, so W=4
// costs the same ~109 ms as W=8. Against the measured ceilings -- 270.8 GB/s of
// bandwidth and 6.1 TFLOP/s of ALU (LEDGER 074, NOT the 17.6 that 035 assumed)
// -- a kernel doing exactly W columns costs
//
//     max(61 ms weight stream, 8.96*W ms arithmetic + 23 ms dequant) * 1.17
//
// which is FLAT at 71.4 ms out to W=4, because the arithmetic does not overtake
// the weight stream until W ~ 4.2. That is 39 ms, 28% of the cycle, at the width
// speculation actually runs.
//
// Neither existing kernel can do it. kernel_mul_mv_q4_K_f32_impl takes the
// column index as tgpig.y and re-streams all 16.52 GB per column (94.0 ms at
// ne11=2, 134.9 at ne11=3). mul_mv_ext reads them once and is still 1.73x off,
// and an 18-point sweep of its constants cannot fix it (LEDGER 083).
//
// So this is sgmv's structure -- register resident, weight stream read once,
// zero threadgroup traffic -- with the 8x8 MMA replaced by scalar fma into N
// accumulators. THE NUMBER TO BEAT is sgmv's effective rate; the number to reach
// is mul_mv's 231 GB/s at width 1, which would put T_ver(4) near 71 ms.
//
// Register pressure is the known trap: 067 and 070 both died of it in this
// kernel. Hence NR and N are swept rather than assumed.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256

typedef struct {
    __fp16  d;
    __fp16  dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;                       // 144 bytes, matches ggml exactly

static void get_scale_min_k4_host(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
    else       { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
                 *m = (q[j+4] >>  4) | ((q[j  ] >> 6) << 4); }
}

static NSString * const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define QK_K 256

typedef struct {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[128];
} block_q4_K;

static inline void scale_min_k4(int j, device const uchar * q,
                                thread uchar & d, thread uchar & m) {
    if (j < 4) { d = q[j] & 63; m = q[j+4] & 63; }
    else       { d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
                 m = (q[j+4] >>  4) | ((q[j  ] >> 6) << 4); }
}

#ifndef NCOL
#define NCOL 4
#endif
#ifndef NROW
#define NROW 4
#endif
#ifndef NSG
#define NSG 4
#endif

// C[M x NCOL] = A[M x K] (q4_K) * B[K x NCOL], B column-major (stride K).
//
// Lane t owns one uint of qs -- bytes [4t, 4t+4) -- which is 4 low nibbles and
// 4 high nibbles, i.e. 8 of the block's 256 values. All four bytes of a lane sit
// in the same 32-byte group, so g = t/8 and the in-group offset is 4*(t%8).
// That makes both of a lane's activation reads a single aligned float4, and the
// weight read a single aligned uint. 32 lanes x 8 values = 256. Nothing is read
// twice and nothing goes through threadgroup memory.
//
// The two sub-block scales a lane needs are fixed by g alone, so they are
// decoded once per row per block rather than per value -- the asymmetry LEDGER
// 067 found for Q6_K is avoided here because Q4_K's scales hoist.
kernel void scmv(
        device const block_q4_K * A    [[buffer(0)]],
        device const float      * B    [[buffer(1)]],
        device       float      * C    [[buffer(2)]],
        constant     int        & M    [[buffer(3)]],
        constant     int        & K    [[buffer(4)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    const int nb = K/QK_K;

    const ushort g   = tiisg/8;        // which 32-byte group of qs (0..3)
    const ushort off = 4*(tiisg%8);    // offset within the group

    const int first_row = (int)(tgpig*NSG + sgitg)*NROW;

    float acc[NROW][NCOL];
    for (short r = 0; r < NROW; ++r)
        for (short c = 0; c < NCOL; ++c) acc[r][c] = 0.0f;

    // activation base for this lane: block 0, group g, in-group offset
    device const float * yb = B + 64*g + off;

    for (int ib = 0; ib < nb; ++ib) {
        // one aligned float4 per column for each half of the group
        float4 yl[NCOL], yh[NCOL];
        float  syl[NCOL], syh[NCOL];

        for (short c = 0; c < NCOL; ++c) {
            device const float * y = yb + (size_t)c*K + (size_t)ib*QK_K;
            yl[c] = *((device const float4 *) y);
            yh[c] = *((device const float4 *)(y + 32));
            syl[c] = yl[c].x + yl[c].y + yl[c].z + yl[c].w;
            syh[c] = yh[c].x + yh[c].y + yh[c].z + yh[c].w;
        }

        for (short r = 0; r < NROW; ++r) {
            device const block_q4_K * xb = A + (size_t)(first_row + r)*nb + ib;

            const uint qv = *((device const uint *)(xb->qs + 32*g) + (tiisg%8));

            uchar scl, mml, sch, mmh;
            scale_min_k4((int)(2*g  ), xb->scales, scl, mml);
            scale_min_k4((int)(2*g+1), xb->scales, sch, mmh);

            // 255 undoes unpack_unorm4x8_to_float's divisor
            const float dd = (float) xb->d * 255.0f;
            const float dm = (float) xb->dmin;

            const float4 ql = unpack_unorm4x8_to_float(qv & 0x0F0F0F0Fu);
            const float4 qh = unpack_unorm4x8_to_float((qv >> 4) & 0x0F0F0F0Fu);

            const float dl = dd*(float) scl, ml = dm*(float) mml;
            const float dh = dd*(float) sch, mh = dm*(float) mmh;

            for (short c = 0; c < NCOL; ++c) {
                acc[r][c] += dl*dot(ql, yl[c]) - ml*syl[c]
                           + dh*dot(qh, yh[c]) - mh*syh[c];
            }
        }
    }

    for (short r = 0; r < NROW; ++r) {
        if (first_row + r >= M) break;
        for (short c = 0; c < NCOL; ++c) {
            const float s = simd_sum(acc[r][c]);
            if (tiisg == 0) C[(size_t)(first_row + r)*NCOL + c] = s;
        }
    }
}
)MSL";

int main(int argc, char **argv) { @autoreleasepool {
    const int M = (argc > 1) ? atoi(argv[1]) : 5120;
    const int K = (argc > 2) ? atoi(argv[2]) : 5120;
    const int nb = K/QK_K;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    printf("device : %s\nshape  : M=%d K=%d (%d q4_K blocks/row)\n", dev.name.UTF8String, M, K, nb);

    const size_t wbytes = (size_t)M*nb*sizeof(block_q4_K);
    printf("weights: %.1f MB\n\n", wbytes/1048576.0);

    const int NMAX = 8;
    id<MTLBuffer> bA = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bB = [dev newBufferWithLength:(size_t)K*NMAX*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bC = [dev newBufferWithLength:(size_t)M*NMAX*sizeof(float) options:MTLResourceStorageModeShared];

    block_q4_K *A = (block_q4_K *)bA.contents;
    float *Bx = (float *)bB.contents, *C = (float *)bC.contents;

    srand(1234);
    for (size_t i = 0; i < (size_t)M*nb; ++i) {
        A[i].d    = (__fp16)(6.187e-05 * (0.5 + (rand()/(double)RAND_MAX)));
        A[i].dmin = (__fp16)(3.0e-05  * (0.5 + (rand()/(double)RAND_MAX)));
        for (int j = 0; j < 12;  ++j) A[i].scales[j] = rand() & 0xFF;
        for (int j = 0; j < 128; ++j) A[i].qs[j]     = rand() & 0xFF;
    }
    for (int c = 0; c < NMAX; ++c)
        for (int j = 0; j < K; ++j) Bx[(size_t)c*K + j] = (rand()/(double)RAND_MAX)*2.0 - 1.0;

    const int NCHK = 16;
    double *ref = calloc((size_t)NCHK*NMAX, sizeof(double));
    for (int r = 0; r < NCHK; ++r) {
        const block_q4_K *row = A + (size_t)r*nb;
        for (int ib = 0; ib < nb; ++ib) {
            const block_q4_K *b = row + ib;
            const double d = (double)(float)b->d, dmin = (double)(float)b->dmin;
            for (int s = 0; s < 8; ++s) {
                uint8_t sc, mm; get_scale_min_k4_host(s, b->scales, &sc, &mm);
                const double ds = d*sc, ms = dmin*mm;
                const int g = s/2, hi = s & 1;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qq = b->qs[g*32 + l];
                    const double w = ds*(hi ? (qq >> 4) : (qq & 0xF)) - ms;
                    const size_t j = (size_t)ib*QK_K + (size_t)s*32 + l;
                    for (int c = 0; c < NMAX; ++c) ref[(size_t)r*NMAX + c] += w*Bx[(size_t)c*K + j];
                }
            }
        }
    }

    printf("%-22s %9s %9s %9s   %s\n", "variant", "ms", "GB/s", "GFLOP/s", "max rel err");
    printf("--------------------------------------------------------------------------\n");

    const int ncols[] = {1,2,3,4,6,8};
    const int nrows[] = {2,4,8};
    for (int ci = 0; ci < 6; ++ci) {
        for (int ri = 0; ri < 3; ++ri) {
            const int NC = ncols[ci], NR = nrows[ri], nsg = 4;
            char label[64]; snprintf(label, sizeof(label), "N=%d NROW=%d", NC, NR);

            MTLCompileOptions *opt = [MTLCompileOptions new];
            opt.preprocessorMacros = @{@"NCOL": [@(NC) stringValue],
                                       @"NROW": [@(NR) stringValue],
                                       @"NSG":  [@(nsg) stringValue]};
            NSError *err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:opt error:&err];
            if (!lib) { printf("%-22s COMPILE FAILED: %s\n", label, err.localizedDescription.UTF8String); continue; }
            id<MTLComputePipelineState> ps =
                [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"scmv"] error:&err];
            if (!ps) { printf("%-22s PIPELINE FAILED\n", label); continue; }

            const int rows_tg = nsg*NR;
            if (M % rows_tg) { printf("%-22s skipped (M %% %d)\n", label, rows_tg); continue; }

            memset(C, 0, (size_t)M*NMAX*sizeof(float));
            double best = 1e30;
            for (int rep = 0; rep < 12; ++rep) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                [e setBuffer:bA offset:0 atIndex:0];
                [e setBuffer:bB offset:0 atIndex:1];
                [e setBuffer:bC offset:0 atIndex:2];
                [e setBytes:&M length:4 atIndex:3];
                [e setBytes:&K length:4 atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake(M/rows_tg,1,1)
                  threadsPerThreadgroup:MTLSizeMake(32*nsg,1,1)];
                [e endEncoding];
                [cb commit]; [cb waitUntilCompleted];
                if (cb.error) { printf("%-22s RUN FAILED\n", label); best = -1; break; }
                const double ms = (cb.GPUEndTime - cb.GPUStartTime)*1e3;
                if (ms < best) best = ms;
            }
            if (best < 0) continue;

            double maxrel = 0;
            for (int r = 0; r < NCHK; ++r)
                for (int c = 0; c < NC; ++c) {
                    const double got = C[(size_t)r*NC + c], exp = ref[(size_t)r*NMAX + c];
                    maxrel = fmax(maxrel, fabs(got - exp)/fmax(fabs(exp), 1e-6));
                }

            printf("%-22s %9.3f %9.1f %9.1f   %.3e%s\n", label, best,
                   wbytes/(best/1e3)/1e9, 2.0*(double)M*K*NC/(best/1e3)/1e9,
                   maxrel, maxrel < 1e-3 ? "" : "   <-- WRONG");
        }
    }
    printf("\nceilings: 270.8 GB/s attained bandwidth; mul_mv reaches 231 GB/s at width 1.\n");
    free(ref);
    return 0;
}}
