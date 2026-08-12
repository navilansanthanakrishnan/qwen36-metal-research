// sgmv.m — register-resident Q4_K simdgroup mat-vec, standalone feasibility probe.
//
//   clang -fobjc-arc -O2 -framework Foundation -framework Metal bench/sgmv.m -o bench/sgmv
//   bench/sgmv
//
// WHY THIS EXISTS (LEDGER 035/036/037).
//
// 35 tok/s at MTP depth 4 requires T_ver(5) <= 95.7 ms. The bandwidth floor for
// one forward pass is 61 ms, so the arithmetic budget is 6.94 ms per verify
// column. Scalar FP32 peak on this part is 6.258 TFLOP/s = 9.36 ms/column, so
// *no* scalar kernel reaches the target even at 100% of peak. The matrix units
// run at 17.6 TFLOP/s = 3.33 ms/column and are the only thing that clears it.
//
// llama.cpp cannot use them here. mul_mv/mul_mv_ext are register-resident but
// scalar. mul_mm uses simdgroup matrix ops but stages dequantized A through
// threadgroup memory, which is why it costs the same as mat-vec at width 5
// (LEDGER 034) and why narrowing its N tile made it worse, not better: the wide
// tile amortises the staging, it is not waste.
//
// 036 found the way out: simdgroup_load() from `thread` address space is
// rejected, but simdgroup_matrix::thread_elements() COMPILES and is writable.
// So a kernel can dequantize a K-quant straight into the matrix registers and
// never touch threadgroup memory at all. 037 recovered the layout that makes
// that addressable:
//
//     row = 4*(lane/16) + (lane%8)/2
//     col = 4*((lane%16)/8) + 2*(lane%2)      e[0]=(row,col)  e[1]=(row,col+1)
//
// No llama.cpp backend does this. This probe is the smallest thing that can
// prove or kill it: real Q4_K bit layout, real shape, correctness against a CPU
// reference, and effective bandwidth at width 8.
//
// THE NUMBER TO BEAT. At n=5 llama.cpp moves the 16.5 GB weight stream in
// 137.6 ms = 120 GB/s, 44% of the 270.8 GB/s this machine actually attains.
// 95.7 ms would be 64%. If this kernel gets there, the target is reachable; if
// it stalls at 44% the matrix units are not the binding constraint and 035 is
// wrong.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------- host types

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

// ------------------------------------------------------------- Metal sources

// The MSL lives in bench/sgmv.metal and is compiled at runtime, so the kernel
// can be edited without rebuilding this harness.
static NSString *load_src(void) {
    NSString *self_dir = [@(__FILE__) stringByDeletingLastPathComponent];
    NSString *path = [self_dir stringByAppendingPathComponent:@"sgmv.metal"];
    NSError *e = nil;
    NSString *s = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&e];
    if (!s) { fprintf(stderr, "cannot read %s: %s\n", path.UTF8String, e.description.UTF8String); exit(1); }
    return s;
}

// ------------------------------------------------------------------ harness

typedef struct { const char *name; const char *fn; int nfrag, nsg, r8; } variant;

int main(int argc, char **argv) {
@autoreleasepool {
    const int M = (argc > 1) ? atoi(argv[1]) : 40960;
    const int K = (argc > 2) ? atoi(argv[2]) : 5120;
    const int N = 8;
    const int nb = K/QK_K;
    const int S  = K/32;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device : %s\n", dev.name.UTF8String);
    printf("shape  : M=%d K=%d N=%d  (%d q4_K blocks/row)\n", M, K, N, nb);

    const size_t wbytes = (size_t)M*nb*sizeof(block_q4_K);
    printf("weights: %.1f MB\n\n", wbytes/1048576.0);

    id<MTLBuffer> bA  = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bB  = [dev newBufferWithLength:(size_t)K*N*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bSY = [dev newBufferWithLength:(size_t)S*N*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bC  = [dev newBufferWithLength:(size_t)M*N*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bA8 = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];

    block_q4_K *A = (block_q4_K *)bA.contents;
    float *Bx = (float *)bB.contents, *SY = (float *)bSY.contents, *C = (float *)bC.contents;

    // Random but *representative* data: d near the real model's median 6.187e-05
    // (MODEL.md / LEDGER 003), nibbles and 6-bit scales uniform.
    srand(1234);
    for (size_t i = 0; i < (size_t)M*nb; ++i) {
        A[i].d    = (__fp16)(6.187e-05 * (0.5 + (rand()/(double)RAND_MAX)));
        A[i].dmin = (__fp16)(3.0e-05  * (0.5 + (rand()/(double)RAND_MAX)));
        for (int j = 0; j < 12;  ++j) A[i].scales[j] = rand() & 0xFF;
        for (int j = 0; j < 128; ++j) A[i].qs[j]     = rand() & 0xFF;
    }
    for (int c = 0; c < N; ++c)
        for (int j = 0; j < K; ++j) Bx[(size_t)c*K + j] = (rand()/(double)RAND_MAX)*2.0 - 1.0;
    for (int s = 0; s < S; ++s)
        for (int c = 0; c < N; ++c) {
            double t = 0; for (int l = 0; l < 32; ++l) t += Bx[(size_t)c*K + s*32 + l];
            SY[(size_t)s*N + c] = (float)t;
        }

    // ---- CPU reference for a sample of rows
    const int NCHK = 24;
    double *ref = calloc((size_t)NCHK*N, sizeof(double));
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
                    const uint8_t q = b->qs[g*32 + l];
                    const double w = ds*(hi ? (q >> 4) : (q & 0xF)) - ms;
                    const size_t j = (size_t)ib*QK_K + (size_t)s*32 + l;
                    for (int c = 0; c < N; ++c) ref[(size_t)r*N + c] += w*Bx[(size_t)c*K + j];
                }
            }
        }
    }

    // R8 repack: block (row, ib) -> ((row/8)*nb + ib)*8 + row%8
    block_q4_K *A8 = (block_q4_K *)bA8.contents;
    for (int row = 0; row < M; ++row)
        for (int ib = 0; ib < nb; ++ib)
            A8[((size_t)(row/8)*nb + ib)*8 + (row%8)] = A[(size_t)row*nb + ib];

    variant vs[] = {
        {"v13 nsg=2 nf=1", "mv_sg13", 1, 2, 0},
        {"v13 nsg=2 nf=2", "mv_sg13", 2, 2, 0},
        {"v13 nsg=2 nf=4", "mv_sg13", 4, 2, 0},
        {"v13 nsg=4 nf=1", "mv_sg13", 1, 4, 0},
        {"v13 nsg=4 nf=2", "mv_sg13", 2, 4, 0},
        {"v13 nsg=4 nf=4", "mv_sg13", 4, 4, 0},
        {"v13 nsg=8 nf=1", "mv_sg13", 1, 8, 0},
        {"v13 nsg=8 nf=2", "mv_sg13", 2, 8, 0},
        {"v13 nsg=8 nf=4", "mv_sg13", 4, 8, 0},
            };
    const int NV = sizeof(vs)/sizeof(vs[0]);
    const int REPS = 20;

    printf("%-24s  %9s  %9s  %9s   %s\n", "kernel", "ms", "GB/s", "GFLOP/s", "max rel err");
    printf("%.78s\n", "-----------------------------------------------------------------------------------");

    id<MTLCommandQueue> q = [dev newCommandQueue];

    for (int vi = 0; vi <= NV; ++vi) {
        const bool scalar = (vi == NV);
        const int nfrag = scalar ? 8 : vs[vi].nfrag;
        const int nsg   = scalar ? 4 : vs[vi].nsg;
        const char *name = scalar ? "mv_scalar (control)" : vs[vi].name;

        MTLCompileOptions *opt = [MTLCompileOptions new];
        opt.preprocessorMacros = @{@"NFRAG": [@(nfrag) stringValue], @"NSG": [@(nsg) stringValue]};
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:load_src() options:opt error:&err];
        if (!lib) { printf("%-24s  COMPILE FAILED\n%s\n", name, err.description.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:scalar ? @"mv_scalar" : @(vs[vi].fn)];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) { printf("%-24s  PIPELINE FAILED: %s\n", name, err.description.UTF8String); continue; }

        const bool ksplit = !scalar && strstr(vs[vi].fn, "mv_sg12");
        const int rows_tg = scalar ? nsg*4 : (ksplit ? 8*nfrag : nsg*8*nfrag);
        if (M % rows_tg) { printf("%-24s  skipped (M %% %d)\n", name, rows_tg); continue; }
        const int ntg = M/rows_tg;

        memset(C, 0, (size_t)M*N*sizeof(float));
        double best = 1e30;
        for (int rep = 0; rep < REPS; ++rep) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:(!scalar && vs[vi].r8) ? bA8 : bA offset:0 atIndex:0];
            [enc setBuffer:bB offset:0 atIndex:1];
            [enc setBuffer:bSY offset:0 atIndex:2];
            [enc setBuffer:bC offset:0 atIndex:3];
            [enc setBytes:&M length:4 atIndex:4];
            [enc setBytes:&K length:4 atIndex:5];
            [enc setBytes:&N length:4 atIndex:6];
            if (ksplit) [enc setThreadgroupMemoryLength:(NSUInteger)(nsg*nfrag*64*sizeof(float)) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(ntg,1,1)
                threadsPerThreadgroup:MTLSizeMake(nsg*32,1,1)];
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            if (cb.error) { printf("%-24s  RUN FAILED: %s\n", name, cb.error.description.UTF8String); best = -1; break; }
            const double ms = (cb.GPUEndTime - cb.GPUStartTime)*1e3;
            if (ms < best) best = ms;
        }
        if (best < 0) continue;

        double maxrel = 0;
        const bool nocheck = !scalar && strstr(vs[vi].fn, "bwonly");
        for (int r = 0; r < NCHK; ++r)
            for (int c = 0; c < N; ++c) {
                const double g = C[(size_t)r*N + c], e = ref[(size_t)r*N + c];
                const double den = fmax(fabs(e), 1e-6);
                maxrel = fmax(maxrel, fabs(g - e)/den);
            }

        const double gbs   = wbytes/(best/1e3)/1e9;
        const double gflop = 2.0*(double)M*K*N/(best/1e3)/1e9;
        printf("%-24s  %9.3f  %9.1f  %9.1f   %.3e%s\n",
               name, best, gbs, gflop, nocheck ? 0.0 : maxrel,
               (nocheck || maxrel < 1e-3) ? (nocheck ? "   (loads only)" : "") : "   <-- WRONG");
    }

    printf("\nreference points: llama.cpp moves the weight stream at 120 GB/s at n=5;\n");
    printf("attained peak on this machine is 270.8 GB/s; 35 tok/s needs ~64%% of it.\n");
    free(ref);
}
    return 0;
}
