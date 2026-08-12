#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K 256

typedef struct {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[128];
} block_q4_K;

static inline void get_scale_min_k4(int j, device const uchar * q,
                                    thread uchar & d, thread uchar & m) {
    if (j < 4) { d = q[j] & 63; m = q[j+4] & 63; }
    else       { d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
                 m = (q[j+4] >>  4) | ((q[j  ] >> 6) << 4); }
}

// NFRAG A-fragments per simdgroup (8*NFRAG rows), NSG simdgroups per threadgroup.
#ifndef NFRAG
#define NFRAG 8
#endif
#ifndef NSG
#define NSG 4
#endif

// ---------------------------------------------------------------------------
// The kernel 035 asked for: dequantize Q4_K directly into simdgroup matrix
// registers via thread_elements(), multiply on the matrix units, zero
// threadgroup traffic.
//
// C[M x 8] = A[M x K] (q4_K) * Bx[K x 8]
//
// Q4_K is w = d*sc[s]*q - dmin*m[s]. The first term is the main K-loop. The
// second is separable -- sum_j dmin*m[s(j)]*x[j] = sum_s (dmin*m[s]) * SY[s]
// where SY[s] is the sub-block sum of x -- so it becomes a second, 32x smaller
// matmul in the epilogue on the same accumulators, not a scalar fixup.
// ---------------------------------------------------------------------------
kernel void mv_sg(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],   // K x N, column-major (ggml src1)
        device const float      * SY [[buffer(2)]],   // (K/32) x N, row-major
        device       float      * C  [[buffer(3)]],   // M x N, row-major
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    // LEDGER 037: the empirically recovered thread_elements() layout.
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    // row this lane owns within each fragment, hoisted out of the K-loop
    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;
    }

    for (uint j0 = 0; j0 < (uint) K; j0 += 8) {
        // B fragment: B[k][c] = x[j0+k][c]. Lane holds (lrow, lcol/lcol+1).
        simdgroup_float8x8 b;
        {
            thread auto & e = b.thread_elements();
            e[0] = Bx[(ulong)(lcol  )*K + j0 + lrow];
            e[1] = Bx[(ulong)(lcol+1)*K + j0 + lrow];
        }

        // Nibble address is identical for every row, so compute it once.
        const uint blk  = j0 / QK_K;        // super-block
        const uint e0   = j0 % QK_K;        // element offset within it
        const uint g    = e0 / 64;
        const uint o    = e0 % 64;
        const bool hi   = o >= 32;          // constant across the 8-chunk
        const uint qoff = g*32 + (o % 32) + lcol;
        const int  is   = (int)(2*g + (hi ? 1 : 0));

        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * xb = base[f] + blk;
            uchar sc, mm;
            get_scale_min_k4(is, xb->scales, sc, mm);
            const float dall = (float) xb->d * (float) sc;

            const uchar q0 = xb->qs[qoff];
            const uchar q1 = xb->qs[qoff + 1];

            simdgroup_float8x8 a;
            {
                thread auto & e = a.thread_elements();
                e[0] = dall * (float)(hi ? (q0 >> 4) : (q0 & 0xF));
                e[1] = dall * (float)(hi ? (q1 >> 4) : (q1 & 0xF));
            }
            simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
        }
    }

    // Epilogue: the -dmin*m term, as a (M x S)*(S x N) matmul on the same
    // accumulators. S = K/32, so this is 1/32 of the main loop.
    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol;
        const uint sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// Scalar control, shaped like ggml's mul_mv: one simdgroup covers NR0 rows,
// each lane walks K with stride 32 and the tail is a simd_sum. Same data, same
// launch geometry class, no matrix units. This is what the register-resident
// kernel has to beat, and it keeps the comparison on one machine and one clock
// instead of against a remembered number.
// ---------------------------------------------------------------------------
#define NR0 4
kernel void mv_scalar(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint nb   = (uint) K / QK_K;
    const uint row0 = (tgpig*NSG + sgitg)*NR0;

    for (uint r = 0; r < NR0; ++r) {
        device const block_q4_K * xb = A + (ulong)(row0 + r)*nb;
        float sum[8];
        for (int c = 0; c < 8; ++c) sum[c] = 0.0f;

        // lane owns super-blocks lane, lane+32, ...
        for (uint ib = lane; ib < nb; ib += 32) {
            device const block_q4_K * b = xb + ib;
            const float d    = (float) b->d;
            const float dmin = (float) b->dmin;
            for (int s = 0; s < 8; ++s) {
                uchar sc, mm;
                get_scale_min_k4(s, b->scales, sc, mm);
                const float ds = d*(float)sc, ms = dmin*(float)mm;
                const uint g = s/2, hi = s & 1;
                for (int l = 0; l < 32; ++l) {
                    const uchar q = b->qs[g*32 + l];
                    const float w = ds*(float)(hi ? (q >> 4) : (q & 0xF)) - ms;
                    const ulong j = (ulong)ib*QK_K + (ulong)s*32 + l;
                    for (int c = 0; c < 8; ++c) sum[c] += w*Bx[(ulong)c*K + j];
                }
            }
        }
        for (int c = 0; c < 8; ++c) {
            const float t = simd_sum(sum[c]);
            if (lane == 0) C[(ulong)(row0 + r)*N + c] = t;
        }
    }
}

// ---------------------------------------------------------------------------
// v2. Same idea, loop restructured around the Q4_K sub-block.
//
// v1 recomputed get_scale_min_k4() and re-loaded ->d and ->scales for every
// 8-element chunk, i.e. 4x per 32-element sub-block per fragment: ~5 memory ops
// to dequantize 2 weights. Here the super-block pointer, d/dmin, the 6-bit
// scale and the nibble half are all hoisted, the 4 chunks of a sub-block are
// unrolled, and the low/high nibble select becomes a branch on a value that is
// uniform across the sub-block instead of a per-weight select.
// ---------------------------------------------------------------------------
kernel void mv_sg2(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q4_K * xb[NFRAG];
        float dv[NFRAG];
        for (int f = 0; f < NFRAG; ++f) { xb[f] = base[f] + ib; dv[f] = (float) xb[f]->d; }

        for (int s = 0; s < 8; ++s) {
            const uint g  = (uint) s / 2;
            const bool hi = (s & 1) != 0;

            float dall[NFRAG];
            device const uchar * qp[NFRAG];
            for (int f = 0; f < NFRAG; ++f) {
                uchar sc, mm;
                get_scale_min_k4(s, xb[f]->scales, sc, mm);
                dall[f] = dv[f] * (float) sc;
                qp[f]   = xb[f]->qs + g*32 + lcol;
            }

            const ulong jb = (ulong) ib*QK_K + (ulong) s*32 + lrow;
            for (uint t = 0; t < 4; ++t) {
                simdgroup_float8x8 b;
                {
                    thread auto & e = b.thread_elements();
                    e[0] = Bx[(ulong)(lcol  )*K + jb + t*8];
                    e[1] = Bx[(ulong)(lcol+1)*K + jb + t*8];
                }
                for (int f = 0; f < NFRAG; ++f) {
                    const uchar q0 = qp[f][t*8    ];
                    const uchar q1 = qp[f][t*8 + 1];
                    simdgroup_float8x8 a;
                    {
                        thread auto & e = a.thread_elements();
                        if (hi) { e[0] = dall[f]*(float)(q0 >> 4);  e[1] = dall[f]*(float)(q1 >> 4);  }
                        else    { e[0] = dall[f]*(float)(q0 & 0xF); e[1] = dall[f]*(float)(q1 & 0xF); }
                    }
                    simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
                }
            }
        }
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v3. Permute the reduction index so every lane's weights are contiguous.
//
// The sum over K is order-invariant, so the map from (fragment column k, chunk
// t) to element-within-sub-block is free to choose as long as A and B agree.
// v2 used the natural elem = t*8 + k, which makes a lane read bytes
// {lcol, lcol+1} at four 8-byte strides: four scattered 2-byte loads.
//
// Choose elem = k*4 + t instead. A lane owning columns (lcol, lcol+1) then
// needs elements lcol*4+t and lcol*4+4+t for t=0..3 -- exactly the 8 contiguous
// bytes at qs + lcol*4, which is 8-byte aligned because lcol is even. One
// uint2 load per sub-block per fragment covers all 8 of its weights, and the
// four lanes of a row tile the sub-block's 32 bytes perfectly.
//
// B gets the same treatment: column c needs elements lrow*4 + t for t=0..3,
// four contiguous floats, so one float4 per column instead of four scalars.
//
// Loads per 8 weights: v2 = 4 (A) + 4 (B), v3 = 1 (A) + 0.25 (B, amortised).
// ---------------------------------------------------------------------------
kernel void mv_sg3(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q4_K * xb[NFRAG];
        float dv[NFRAG];
        for (int f = 0; f < NFRAG; ++f) { xb[f] = base[f] + ib; dv[f] = (float) xb[f]->d; }

        for (int s = 0; s < 8; ++s) {
            const uint g  = (uint) s / 2;
            const bool hi = (s & 1) != 0;

            // B: four contiguous floats per column, one float4 load each.
            const ulong jb = (ulong) ib*QK_K + (ulong) s*32 + (ulong) lrow*4;
            const float4 b0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jb));
            const float4 b1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jb));

            // A: one aligned 8-byte load per fragment covers all 8 weights.
            float  dall[NFRAG];
            uint2  qv[NFRAG];
            for (int f = 0; f < NFRAG; ++f) {
                uchar sc, mm;
                get_scale_min_k4(s, xb[f]->scales, sc, mm);
                dall[f] = dv[f] * (float) sc;
                qv[f]   = *((device const uint2 *)(xb[f]->qs + g*32 + lcol*4));
            }

            for (uint t = 0; t < 4; ++t) {
                simdgroup_float8x8 b;
                {
                    thread auto & e = b.thread_elements();
                    e[0] = b0[t]; e[1] = b1[t];
                }
                const uint sh = hi ? (8*t + 4) : (8*t);
                for (int f = 0; f < NFRAG; ++f) {
                    simdgroup_float8x8 a;
                    {
                        thread auto & e = a.thread_elements();
                        e[0] = dall[f]*(float)((qv[f].x >> sh) & 0xF);
                        e[1] = dall[f]*(float)((qv[f].y >> sh) & 0xF);
                    }
                    simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
                }
            }
        }
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v4. Kill the per-weight dequant arithmetic with a hardware unpack.
//
// v3 still spends 4 scalar ops on every one of the 209M weights: shift, mask,
// int->float convert, multiply by the row scale. Two changes remove almost all
// of it.
//
// 1. unpack_unorm4x8_to_float() converts four packed bytes to a float4 in one
//    hardware instruction. Masking the loaded word with 0x0F0F0F0F leaves four
//    nibbles in four bytes, so one AND plus one unpack yields four dequantized
//    values -- and the v3 permutation is what makes those four the four the
//    lane actually wants. The instruction divides by 255, which folds into the
//    scale for free.
//
// 2. The row scale is constant across a sub-block, so it does not belong on
//    every weight. Accumulate the sub-block into a local fragment with
//    simdgroup_multiply (no zeroing needed for t=0), then fold dall*255 into
//    the running accumulator once: 4 ops per 8 weights instead of 8.
//
// Scalar ops per weight: v3 ~4, v4 ~1.5.
// ---------------------------------------------------------------------------
kernel void mv_sg4(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q4_K * xb[NFRAG];
        float dv[NFRAG];
        for (int f = 0; f < NFRAG; ++f) { xb[f] = base[f] + ib; dv[f] = (float) xb[f]->d; }

        for (int s = 0; s < 8; ++s) {
            const uint g  = (uint) s / 2;
            const bool hi = (s & 1) != 0;

            const ulong jb = (ulong) ib*QK_K + (ulong) s*32 + (ulong) lrow*4;
            const float4 b0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jb));
            const float4 b1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jb));

            simdgroup_float8x8 bf[4];
            for (uint t = 0; t < 4; ++t) {
                thread auto & e = bf[t].thread_elements();
                e[0] = b0[t]; e[1] = b1[t];
            }

            for (int f = 0; f < NFRAG; ++f) {
                uchar sc, mm;
                get_scale_min_k4(s, xb[f]->scales, sc, mm);
                const uint2 qv = *((device const uint2 *)(xb[f]->qs + g*32 + lcol*4));

                const uint w0 = hi ? ((qv.x >> 4) & 0x0F0F0F0Fu) : (qv.x & 0x0F0F0F0Fu);
                const uint w1 = hi ? ((qv.y >> 4) & 0x0F0F0F0Fu) : (qv.y & 0x0F0F0F0Fu);
                const float4 a0 = unpack_unorm4x8_to_float(w0);
                const float4 a1 = unpack_unorm4x8_to_float(w1);

                simdgroup_float8x8 dloc;
                {
                    simdgroup_float8x8 a;
                    thread auto & e = a.thread_elements();
                    e[0] = a0[0]; e[1] = a1[0];
                    simdgroup_multiply(dloc, a, bf[0]);
                }
                for (uint t = 1; t < 4; ++t) {
                    simdgroup_float8x8 a;
                    thread auto & e = a.thread_elements();
                    e[0] = a0[t]; e[1] = a1[t];
                    simdgroup_multiply_accumulate(dloc, a, bf[t], dloc);
                }

                const float ds = dv[f] * (float) sc * 255.0f;
                thread auto & ea = acc[f].thread_elements();
                thread auto & el = dloc.thread_elements();
                ea[0] += ds*el[0];
                ea[1] += ds*el[1];
            }
        }
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v5 = v3's loop exactly, with only the nibble extraction replaced.
//
// v4 bundled the unpack together with a local sub-block accumulator and a
// bf[4] fragment array, and lost 1.9x. Those are separable: this isolates the
// unpack so the measurement attributes the change to one cause. Per weight,
// (shift, mask, convert) becomes one AND and one unpack per four weights, and
// the row scale folds into the unorm divisor.
// ---------------------------------------------------------------------------
kernel void mv_sg5(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q4_K * xb[NFRAG];
        float dv[NFRAG];
        for (int f = 0; f < NFRAG; ++f) { xb[f] = base[f] + ib; dv[f] = (float) xb[f]->d; }

        for (int s = 0; s < 8; ++s) {
            const uint g  = (uint) s / 2;
            const bool hi = (s & 1) != 0;

            const ulong jb = (ulong) ib*QK_K + (ulong) s*32 + (ulong) lrow*4;
            const float4 b0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jb));
            const float4 b1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jb));

            float4 a0[NFRAG], a1[NFRAG];
            for (int f = 0; f < NFRAG; ++f) {
                uchar sc, mm;
                get_scale_min_k4(s, xb[f]->scales, sc, mm);
                const uint2 qv = *((device const uint2 *)(xb[f]->qs + g*32 + lcol*4));
                const uint w0 = hi ? ((qv.x >> 4) & 0x0F0F0F0Fu) : (qv.x & 0x0F0F0F0Fu);
                const uint w1 = hi ? ((qv.y >> 4) & 0x0F0F0F0Fu) : (qv.y & 0x0F0F0F0Fu);
                const float ds = dv[f] * (float) sc * 255.0f;
                a0[f] = unpack_unorm4x8_to_float(w0) * ds;
                a1[f] = unpack_unorm4x8_to_float(w1) * ds;
            }

            for (uint t = 0; t < 4; ++t) {
                simdgroup_float8x8 b;
                {
                    thread auto & e = b.thread_elements();
                    e[0] = b0[t]; e[1] = b1[t];
                }
                for (int f = 0; f < NFRAG; ++f) {
                    simdgroup_float8x8 a;
                    {
                        thread auto & e = a.thread_elements();
                        e[0] = a0[f][t]; e[1] = a1[f][t];
                    }
                    simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
                }
            }
        }
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v6 = v5 with the accumulator dependency chain split in two.
//
// Every simdgroup_multiply_accumulate reads the accumulator the previous one
// wrote, so at NFRAG=1 there is exactly one chain and the matrix unit's result
// latency is fully exposed. Two accumulators, even t into one and odd t into
// the other, halve the chain depth for two extra registers per lane and one
// add at the end.
// ---------------------------------------------------------------------------
kernel void mv_sg6(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG], acd[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements(); e[0] = 0.0f; e[1] = 0.0f;
        thread auto & o = acd[f].thread_elements(); o[0] = 0.0f; o[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q4_K * xb[NFRAG];
        float dv[NFRAG];
        for (int f = 0; f < NFRAG; ++f) { xb[f] = base[f] + ib; dv[f] = (float) xb[f]->d; }

        for (int s = 0; s < 8; ++s) {
            const uint g  = (uint) s / 2;
            const bool hi = (s & 1) != 0;

            const ulong jb = (ulong) ib*QK_K + (ulong) s*32 + (ulong) lrow*4;
            const float4 b0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jb));
            const float4 b1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jb));

            float4 a0[NFRAG], a1[NFRAG];
            for (int f = 0; f < NFRAG; ++f) {
                uchar sc, mm;
                get_scale_min_k4(s, xb[f]->scales, sc, mm);
                const uint2 qv = *((device const uint2 *)(xb[f]->qs + g*32 + lcol*4));
                const uint w0 = hi ? ((qv.x >> 4) & 0x0F0F0F0Fu) : (qv.x & 0x0F0F0F0Fu);
                const uint w1 = hi ? ((qv.y >> 4) & 0x0F0F0F0Fu) : (qv.y & 0x0F0F0F0Fu);
                const float ds = dv[f] * (float) sc * 255.0f;
                a0[f] = unpack_unorm4x8_to_float(w0) * ds;
                a1[f] = unpack_unorm4x8_to_float(w1) * ds;
            }

            for (uint t = 0; t < 4; ++t) {
                simdgroup_float8x8 b;
                {
                    thread auto & e = b.thread_elements();
                    e[0] = b0[t]; e[1] = b1[t];
                }
                for (int f = 0; f < NFRAG; ++f) {
                    simdgroup_float8x8 a;
                    {
                        thread auto & e = a.thread_elements();
                        e[0] = a0[f][t]; e[1] = a1[f][t];
                    }
                    if (t & 1) simdgroup_multiply_accumulate(acd[f], a, b, acd[f]);
                    else       simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
                }
            }
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        thread auto & o = acd[f].thread_elements();
        e[0] += o[0]; e[1] += o[1];
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v7 = v5 with the fragment loop moved outside the chunk loop.
//
// NFRAG>1 exists to amortise B: one pair of float4 B loads should serve NFRAG
// fragments, cutting B load traffic (838 MB at NFRAG=1, vs 112 MB for the
// weights themselves) by that factor. In v5 it did the opposite, because
// hoisting a0[NFRAG]/a1[NFRAG] out of the fragment loop put 8*NFRAG floats in
// registers and cost more occupancy than it saved in loads.
//
// Here b0/b1 are loaded once per sub-block and held across fragments, while
// the dequantized a0/a1 live only for the duration of one fragment's four
// chunks. Register cost stops growing with NFRAG except for the accumulators.
// ---------------------------------------------------------------------------
kernel void mv_sg7(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        for (int s = 0; s < 8; ++s) {
            const uint g  = (uint) s / 2;
            const bool hi = (s & 1) != 0;

            const ulong jb = (ulong) ib*QK_K + (ulong) s*32 + (ulong) lrow*4;
            const float4 b0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jb));
            const float4 b1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jb));

            for (int f = 0; f < NFRAG; ++f) {
                device const block_q4_K * xb = base[f] + ib;
                uchar sc, mm;
                get_scale_min_k4(s, xb->scales, sc, mm);
                const uint2 qv = *((device const uint2 *)(xb->qs + g*32 + lcol*4));
                const uint w0 = hi ? ((qv.x >> 4) & 0x0F0F0F0Fu) : (qv.x & 0x0F0F0F0Fu);
                const uint w1 = hi ? ((qv.y >> 4) & 0x0F0F0F0Fu) : (qv.y & 0x0F0F0F0Fu);
                const float ds = (float) xb->d * (float) sc * 255.0f;
                const float4 a0 = unpack_unorm4x8_to_float(w0) * ds;
                const float4 a1 = unpack_unorm4x8_to_float(w1) * ds;

                for (uint t = 0; t < 4; ++t) {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = b0[t]; e[1] = b1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = a0[t]; e[1] = a1[t]; }
                    simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
                }
            }
        }
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v8 = v7 with the row scale moved off the weights and onto the accumulator.
//
// dall is constant across a sub-block, so multiplying all 8 of a lane's
// weights by it (8 ops) is redundant: accumulate the sub-block raw into a
// local fragment and fold the scale in once (2 mul + 2 add). v4 tried this and
// lost, but v4 also carried a bf[4] fragment array; here it is the only change
// from v7.
// ---------------------------------------------------------------------------
kernel void mv_sg8(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        for (int s = 0; s < 8; ++s) {
            const uint g  = (uint) s / 2;
            const bool hi = (s & 1) != 0;

            const ulong jb = (ulong) ib*QK_K + (ulong) s*32 + (ulong) lrow*4;
            const float4 b0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jb));
            const float4 b1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jb));

            for (int f = 0; f < NFRAG; ++f) {
                device const block_q4_K * xb = base[f] + ib;
                uchar sc, mm;
                get_scale_min_k4(s, xb->scales, sc, mm);
                const uint2 qv = *((device const uint2 *)(xb->qs + g*32 + lcol*4));
                const uint w0 = hi ? ((qv.x >> 4) & 0x0F0F0F0Fu) : (qv.x & 0x0F0F0F0Fu);
                const uint w1 = hi ? ((qv.y >> 4) & 0x0F0F0F0Fu) : (qv.y & 0x0F0F0F0Fu);
                const float4 a0 = unpack_unorm4x8_to_float(w0);
                const float4 a1 = unpack_unorm4x8_to_float(w1);

                simdgroup_float8x8 dloc;
                {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = b0[0]; e[1] = b1[0]; }
                    { thread auto & e = a.thread_elements(); e[0] = a0[0]; e[1] = a1[0]; }
                    simdgroup_multiply(dloc, a, b);
                }
                for (uint t = 1; t < 4; ++t) {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = b0[t]; e[1] = b1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = a0[t]; e[1] = a1[t]; }
                    simdgroup_multiply_accumulate(dloc, a, b, dloc);
                }
                const float ds = (float) xb->d * (float) sc * 255.0f;
                thread auto & ea = acc[f].thread_elements();
                thread auto & el = dloc.thread_elements();
                ea[0] += ds*el[0];
                ea[1] += ds*el[1];
            }
        }
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v9 = v8 on an 8-row-interleaved weight layout.
//
// NFRAG>1 degrades monotonically in v7 even with register pressure removed,
// which points at the access pattern rather than occupancy. A simdgroup's 8
// rows are ne00/2 bytes apart (2880 here), so one load instruction issues 8
// scattered 32-byte requests into 8 different DRAM pages, and widening NFRAG
// only multiplies the scatter.
//
// R8 layout stores the 8 rows of a fragment interleaved: block (row, ib) lives
// at ((row/8)*nb + ib)*8 + row%8. The 8 rows a simdgroup needs for one
// super-block then occupy 1152 contiguous bytes instead of spanning 23 KB.
// Same bits, same values, same arithmetic -- purely an ordering of the buffer,
// which is a load-time repack in a real backend, not a quality change.
// ---------------------------------------------------------------------------
kernel void mv_sg9(
        device const block_q4_K * A  [[buffer(0)]],   // R8 layout
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    // start of this lane's row within each fragment's interleaved group
    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f)
        base[f] = A + (ulong)((row0 + f*8)/8)*nb*8 + lrow;

    for (uint ib = 0; ib < nb; ++ib) {
        for (int s = 0; s < 8; ++s) {
            const uint g  = (uint) s / 2;
            const bool hi = (s & 1) != 0;

            const ulong jb = (ulong) ib*QK_K + (ulong) s*32 + (ulong) lrow*4;
            const float4 b0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jb));
            const float4 b1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jb));

            for (int f = 0; f < NFRAG; ++f) {
                device const block_q4_K * xb = base[f] + (ulong) ib*8;
                uchar sc, mm;
                get_scale_min_k4(s, xb->scales, sc, mm);
                const uint2 qv = *((device const uint2 *)(xb->qs + g*32 + lcol*4));
                const uint w0 = hi ? ((qv.x >> 4) & 0x0F0F0F0Fu) : (qv.x & 0x0F0F0F0Fu);
                const uint w1 = hi ? ((qv.y >> 4) & 0x0F0F0F0Fu) : (qv.y & 0x0F0F0F0Fu);
                const float4 a0 = unpack_unorm4x8_to_float(w0);
                const float4 a1 = unpack_unorm4x8_to_float(w1);

                simdgroup_float8x8 dloc;
                {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = b0[0]; e[1] = b1[0]; }
                    { thread auto & e = a.thread_elements(); e[0] = a0[0]; e[1] = a1[0]; }
                    simdgroup_multiply(dloc, a, b);
                }
                for (uint t = 1; t < 4; ++t) {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = b0[t]; e[1] = b1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = a0[t]; e[1] = a1[t]; }
                    simdgroup_multiply_accumulate(dloc, a, b, dloc);
                }
                const float ds = (float) xb->d * (float) sc * 255.0f;
                thread auto & ea = acc[f].thread_elements();
                thread auto & el = dloc.thread_elements();
                ea[0] += ds*el[0];
                ea[1] += ds*el[1];
            }
        }
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + (ulong)(sA/8)*8;
            device const block_q4_K * x1 = base[f] + (ulong)(sB/8)*8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v10 = v9, taking both nibble halves from a single load.
//
// Sub-blocks 2g and 2g+1 share the same 32 bytes of qs -- one is the low
// nibbles, the other the high. v9 loads that word twice and pays a select each
// time. Handling the pair together halves the weight loads and the ->d loads
// and turns the hi select into straight-line code.
// ---------------------------------------------------------------------------
kernel void mv_sg10(
        device const block_q4_K * A  [[buffer(0)]],   // R8 layout
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f)
        base[f] = A + (ulong)((row0 + f*8)/8)*nb*8 + lrow;

    for (uint ib = 0; ib < nb; ++ib) {
        for (uint g = 0; g < 4; ++g) {
            const ulong jlo = (ulong) ib*QK_K + (ulong)(2*g  )*32 + (ulong) lrow*4;
            const ulong jhi = (ulong) ib*QK_K + (ulong)(2*g+1)*32 + (ulong) lrow*4;
            const float4 bl0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jlo));
            const float4 bl1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jlo));
            const float4 bh0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jhi));
            const float4 bh1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jhi));

            for (int f = 0; f < NFRAG; ++f) {
                device const block_q4_K * xb = base[f] + (ulong) ib*8;
                const uint2 qv = *((device const uint2 *)(xb->qs + g*32 + lcol*4));
                const float dd = (float) xb->d * 255.0f;

                uchar scl, mml, sch, mmh;
                get_scale_min_k4((int)(2*g  ), xb->scales, scl, mml);
                get_scale_min_k4((int)(2*g+1), xb->scales, sch, mmh);

                const float4 al0 = unpack_unorm4x8_to_float(qv.x & 0x0F0F0F0Fu);
                const float4 al1 = unpack_unorm4x8_to_float(qv.y & 0x0F0F0F0Fu);
                const float4 ah0 = unpack_unorm4x8_to_float((qv.x >> 4) & 0x0F0F0F0Fu);
                const float4 ah1 = unpack_unorm4x8_to_float((qv.y >> 4) & 0x0F0F0F0Fu);

                simdgroup_float8x8 dlo, dhi;
                {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = bl0[0]; e[1] = bl1[0]; }
                    { thread auto & e = a.thread_elements(); e[0] = al0[0]; e[1] = al1[0]; }
                    simdgroup_multiply(dlo, a, b);
                    { thread auto & e = b.thread_elements(); e[0] = bh0[0]; e[1] = bh1[0]; }
                    { thread auto & e = a.thread_elements(); e[0] = ah0[0]; e[1] = ah1[0]; }
                    simdgroup_multiply(dhi, a, b);
                }
                for (uint t = 1; t < 4; ++t) {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = bl0[t]; e[1] = bl1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = al0[t]; e[1] = al1[t]; }
                    simdgroup_multiply_accumulate(dlo, a, b, dlo);
                    { thread auto & e = b.thread_elements(); e[0] = bh0[t]; e[1] = bh1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = ah0[t]; e[1] = ah1[t]; }
                    simdgroup_multiply_accumulate(dhi, a, b, dhi);
                }
                const float dl = dd * (float) scl;
                const float dh = dd * (float) sch;
                thread auto & ea = acc[f].thread_elements();
                thread auto & el = dlo.thread_elements();
                thread auto & eh = dhi.thread_elements();
                ea[0] += dl*el[0] + dh*eh[0];
                ea[1] += dl*el[1] + dh*eh[1];
            }
        }
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + (ulong)(sA/8)*8;
            device const block_q4_K * x1 = base[f] + (ulong)(sB/8)*8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v11 = v10 on the stock ggml layout.
//
// v9 showed the 8-row interleave buys nothing (159.4 vs 161.8 GB/s), so the
// scattered-row hypothesis is refuted and v10's gain came from halving the
// loads and ALU, not from locality. That is the better outcome: the same win
// is available with no repack, no new buffer type, and no load-time pass.
// ---------------------------------------------------------------------------
kernel void mv_sg11(
        device const block_q4_K * A  [[buffer(0)]],   // stock ggml layout
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f)
        base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        for (uint g = 0; g < 4; ++g) {
            const ulong jlo = (ulong) ib*QK_K + (ulong)(2*g  )*32 + (ulong) lrow*4;
            const ulong jhi = (ulong) ib*QK_K + (ulong)(2*g+1)*32 + (ulong) lrow*4;
            const float4 bl0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jlo));
            const float4 bl1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jlo));
            const float4 bh0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jhi));
            const float4 bh1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jhi));

            for (int f = 0; f < NFRAG; ++f) {
                device const block_q4_K * xb = base[f] + ib;
                const uint2 qv = *((device const uint2 *)(xb->qs + g*32 + lcol*4));
                const float dd = (float) xb->d * 255.0f;

                uchar scl, mml, sch, mmh;
                get_scale_min_k4((int)(2*g  ), xb->scales, scl, mml);
                get_scale_min_k4((int)(2*g+1), xb->scales, sch, mmh);

                const float4 al0 = unpack_unorm4x8_to_float(qv.x & 0x0F0F0F0Fu);
                const float4 al1 = unpack_unorm4x8_to_float(qv.y & 0x0F0F0F0Fu);
                const float4 ah0 = unpack_unorm4x8_to_float((qv.x >> 4) & 0x0F0F0F0Fu);
                const float4 ah1 = unpack_unorm4x8_to_float((qv.y >> 4) & 0x0F0F0F0Fu);

                simdgroup_float8x8 dlo, dhi;
                {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = bl0[0]; e[1] = bl1[0]; }
                    { thread auto & e = a.thread_elements(); e[0] = al0[0]; e[1] = al1[0]; }
                    simdgroup_multiply(dlo, a, b);
                    { thread auto & e = b.thread_elements(); e[0] = bh0[0]; e[1] = bh1[0]; }
                    { thread auto & e = a.thread_elements(); e[0] = ah0[0]; e[1] = ah1[0]; }
                    simdgroup_multiply(dhi, a, b);
                }
                for (uint t = 1; t < 4; ++t) {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = bl0[t]; e[1] = bl1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = al0[t]; e[1] = al1[t]; }
                    simdgroup_multiply_accumulate(dlo, a, b, dlo);
                    { thread auto & e = b.thread_elements(); e[0] = bh0[t]; e[1] = bh1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = ah0[t]; e[1] = ah1[t]; }
                    simdgroup_multiply_accumulate(dhi, a, b, dhi);
                }
                const float dl = dd * (float) scl;
                const float dh = dd * (float) sch;
                thread auto & ea = acc[f].thread_elements();
                thread auto & el = dlo.thread_elements();
                thread auto & eh = dhi.thread_elements();
                ea[0] += dl*el[0] + dh*eh[0];
                ea[1] += dl*el[1] + dh*eh[1];
            }
        }
    }

    const uint S = (uint) K / 32;
    for (uint s0 = 0; s0 < S; s0 += 8) {
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// bwonly: every load this kernel family issues, and none of the arithmetic.
//
// Separates "compute bound" from "this access pattern cannot go faster". If
// this reaches the 270.8 GB/s the streaming probe attains, the remaining gap
// is ALU and worth chasing. If it plateaus near the real kernel, the pattern
// itself is the ceiling and the next move has to change the pattern.
// ---------------------------------------------------------------------------
kernel void mv_bwonly(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);
    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    float acc = 0.0f;
    for (uint ib = 0; ib < nb; ++ib) {
        for (uint g = 0; g < 4; ++g) {
            const ulong jlo = (ulong) ib*QK_K + (ulong)(2*g  )*32 + (ulong) lrow*4;
            const ulong jhi = (ulong) ib*QK_K + (ulong)(2*g+1)*32 + (ulong) lrow*4;
            const float4 bl0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jlo));
            const float4 bl1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jlo));
            const float4 bh0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jhi));
            const float4 bh1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jhi));
            acc += bl0.x + bl1.y + bh0.z + bh1.w;
            for (int f = 0; f < NFRAG; ++f) {
                device const block_q4_K * xb = base[f] + ib;
                const uint2 qv = *((device const uint2 *)(xb->qs + g*32 + lcol*4));
                acc += (float)(qv.x ^ qv.y) + (float) xb->d
                     + (float) xb->scales[2*g] + (float) xb->scales[2*g+1];
            }
        }
    }
    if (acc == 1234.5678f) C[(ulong)(row0 + lrow)*N + lcol] = acc;
}

// ---------------------------------------------------------------------------
// v12 = v11 with the reduction split across simdgroups.
//
// v11 parallelises over rows only, so it needs a tall matrix to fill the GPU:
// at M=40960 it reaches 170 GB/s, but at M=4096 there are just 128-256
// threadgroups for 16 cores and it falls to 100 GB/s -- worse than llama.cpp,
// whose mul_mv splits K across the lanes of a simdgroup and finishes with a
// simd_sum. Real models have both shapes (a wide LM head, a narrow down
// projection), so the kernel has to cover both.
//
// Here all NSG simdgroups of a threadgroup work the same 8*NFRAG rows and
// stride over super-blocks by NSG, then reduce their accumulators through
// threadgroup memory. Strided rather than blocked assignment so any nb works,
// including the nb=20 and nb=56 of this model's two shapes.
// ---------------------------------------------------------------------------
kernel void mv_sg12(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        threadgroup  float      * shm [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint row0 = tgpig*(8*NFRAG);
    const uint nb   = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = sgitg; ib < nb; ib += NSG) {
        for (uint g = 0; g < 4; ++g) {
            const ulong jlo = (ulong) ib*QK_K + (ulong)(2*g  )*32 + (ulong) lrow*4;
            const ulong jhi = (ulong) ib*QK_K + (ulong)(2*g+1)*32 + (ulong) lrow*4;
            const float4 bl0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jlo));
            const float4 bl1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jlo));
            const float4 bh0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jhi));
            const float4 bh1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jhi));

            for (int f = 0; f < NFRAG; ++f) {
                device const block_q4_K * xb = base[f] + ib;
                const uint2 qv = *((device const uint2 *)(xb->qs + g*32 + lcol*4));
                const float dd = (float) xb->d * 255.0f;

                uchar scl, mml, sch, mmh;
                get_scale_min_k4((int)(2*g  ), xb->scales, scl, mml);
                get_scale_min_k4((int)(2*g+1), xb->scales, sch, mmh);

                const float4 al0 = unpack_unorm4x8_to_float(qv.x & 0x0F0F0F0Fu);
                const float4 al1 = unpack_unorm4x8_to_float(qv.y & 0x0F0F0F0Fu);
                const float4 ah0 = unpack_unorm4x8_to_float((qv.x >> 4) & 0x0F0F0F0Fu);
                const float4 ah1 = unpack_unorm4x8_to_float((qv.y >> 4) & 0x0F0F0F0Fu);

                simdgroup_float8x8 dlo, dhi;
                {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = bl0[0]; e[1] = bl1[0]; }
                    { thread auto & e = a.thread_elements(); e[0] = al0[0]; e[1] = al1[0]; }
                    simdgroup_multiply(dlo, a, b);
                    { thread auto & e = b.thread_elements(); e[0] = bh0[0]; e[1] = bh1[0]; }
                    { thread auto & e = a.thread_elements(); e[0] = ah0[0]; e[1] = ah1[0]; }
                    simdgroup_multiply(dhi, a, b);
                }
                for (uint t = 1; t < 4; ++t) {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = bl0[t]; e[1] = bl1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = al0[t]; e[1] = al1[t]; }
                    simdgroup_multiply_accumulate(dlo, a, b, dlo);
                    { thread auto & e = b.thread_elements(); e[0] = bh0[t]; e[1] = bh1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = ah0[t]; e[1] = ah1[t]; }
                    simdgroup_multiply_accumulate(dhi, a, b, dhi);
                }
                const float dl = dd * (float) scl;
                const float dh = dd * (float) sch;
                thread auto & ea = acc[f].thread_elements();
                thread auto & el = dlo.thread_elements();
                thread auto & eh = dhi.thread_elements();
                ea[0] += dl*el[0] + dh*eh[0];
                ea[1] += dl*el[1] + dh*eh[1];
            }
        }
    }

    // min term, over this simdgroup's share of the super-blocks
    for (uint ib = sgitg; ib < nb; ib += NSG) {
        const uint s0 = ib*8;
        simdgroup_float8x8 b2;
        {
            thread auto & e = b2.thread_elements();
            e[0] = -SY[(ulong)(s0 + lrow)*N + lcol    ];
            e[1] = -SY[(ulong)(s0 + lrow)*N + lcol + 1];
        }
        const uint sA = s0 + lcol, sB = s0 + lcol + 1;
        for (int f = 0; f < NFRAG; ++f) {
            device const block_q4_K * x0 = base[f] + sA/8;
            device const block_q4_K * x1 = base[f] + sB/8;
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4((int)(sA % 8), x0->scales, sc0, m0);
            get_scale_min_k4((int)(sB % 8), x1->scales, sc1, m1);
            simdgroup_float8x8 a2;
            {
                thread auto & e = a2.thread_elements();
                e[0] = (float) x0->dmin * (float) m0;
                e[1] = (float) x1->dmin * (float) m1;
            }
            simdgroup_multiply_accumulate(acc[f], a2, b2, acc[f]);
        }
    }

    // cross-simdgroup reduction
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        shm[(sgitg*NFRAG + f)*64 + lane*2 + 0] = e[0];
        shm[(sgitg*NFRAG + f)*64 + lane*2 + 1] = e[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        for (int f = 0; f < NFRAG; ++f) {
            float v0 = 0.0f, v1 = 0.0f;
            for (uint u = 0; u < NSG; ++u) {
                v0 += shm[(u*NFRAG + f)*64 + lane*2 + 0];
                v1 += shm[(u*NFRAG + f)*64 + lane*2 + 1];
            }
            const ulong row = (ulong)(row0 + f*8 + lrow);
            C[row*N + lcol    ] = v0;
            C[row*N + lcol + 1] = v1;
        }
    }
}

// ---------------------------------------------------------------------------
// v13 = v11 with the min term folded into the weights instead of an epilogue.
//
// Q4_K is w = d*sc*q - dmin*m. Every version so far handled the second term as
// a separate (M x K/32)*(K/32 x N) matmul against SY, the sub-block sums of B.
// That is cheap in flops but expensive in plumbing: ggml would need an extra
// buffer and an extra kernel to produce SY on every decode step.
//
// Folding it per weight costs one fma where there was a multiply -- the scale
// can no longer be hoisted to the accumulator (v8's trick), so this trades
// some speed for dropping SY entirely. Measuring what that trade actually
// costs, because the simpler kernel is the one worth integrating unless the
// gap is large.
// ---------------------------------------------------------------------------
kernel void mv_sg13(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],   // unused
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        for (uint g = 0; g < 4; ++g) {
            const ulong jlo = (ulong) ib*QK_K + (ulong)(2*g  )*32 + (ulong) lrow*4;
            const ulong jhi = (ulong) ib*QK_K + (ulong)(2*g+1)*32 + (ulong) lrow*4;
            const float4 bl0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jlo));
            const float4 bl1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jlo));
            const float4 bh0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jhi));
            const float4 bh1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jhi));

            for (int f = 0; f < NFRAG; ++f) {
                device const block_q4_K * xb = base[f] + ib;
                const uint2 qv = *((device const uint2 *)(xb->qs + g*32 + lcol*4));
                const float dd = (float) xb->d * 255.0f;
                const float dm = (float) xb->dmin;

                uchar scl, mml, sch, mmh;
                get_scale_min_k4((int)(2*g  ), xb->scales, scl, mml);
                get_scale_min_k4((int)(2*g+1), xb->scales, sch, mmh);

                const float dl = dd * (float) scl, ml = dm * (float) mml;
                const float dh = dd * (float) sch, mh = dm * (float) mmh;

                const float4 al0 = fma(unpack_unorm4x8_to_float(qv.x & 0x0F0F0F0Fu), dl, -ml);
                const float4 al1 = fma(unpack_unorm4x8_to_float(qv.y & 0x0F0F0F0Fu), dl, -ml);
                const float4 ah0 = fma(unpack_unorm4x8_to_float((qv.x >> 4) & 0x0F0F0F0Fu), dh, -mh);
                const float4 ah1 = fma(unpack_unorm4x8_to_float((qv.y >> 4) & 0x0F0F0F0Fu), dh, -mh);

                for (uint t = 0; t < 4; ++t) {
                    simdgroup_float8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = bl0[t]; e[1] = bl1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = al0[t]; e[1] = al1[t]; }
                    simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
                    { thread auto & e = b.thread_elements(); e[0] = bh0[t]; e[1] = bh1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = ah0[t]; e[1] = ah1[t]; }
                    simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
                }
            }
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}

// ---------------------------------------------------------------------------
// v14 = v13 with half A/B fragments and a float accumulator.
//
// Apple GPUs run fp16 arithmetic at double rate, and the inner loop is
// dominated by thread_elements() register writes (32 per 16 weights) plus the
// matrix ops themselves. If MSL accepts mixed precision -- half inputs, float
// accumulator, as tensor cores generally do -- both halve.
//
// This variant converts the already-dequantized float values, so it measures
// the ceiling of the idea, not a shippable kernel: A holds values around 1e-2
// to 1e-5 and half's smallest normal is 6.1e-5, so a real version would carry
// A as the exact small integers q*sc (<= 945, exact in half) and fold the
// scale onto the float accumulator. Speed first; if it does not pay, the
// precision work is moot.
// ---------------------------------------------------------------------------
kernel void mv_sg14(
        device const block_q4_K * A  [[buffer(0)]],
        device const float      * Bx [[buffer(1)]],
        device const float      * SY [[buffer(2)]],   // unused
        device       float      * C  [[buffer(3)]],
        constant     int        & M  [[buffer(4)]],
        constant     int        & K  [[buffer(5)]],
        constant     int        & N  [[buffer(6)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]])
{
    const uint lrow = 4*(lane/16) + ((lane%8)/2);
    const uint lcol = 4*((lane%16)/8) + 2*(lane%2);

    const uint rows_sg = 8*NFRAG;
    const uint row0    = tgpig*(NSG*rows_sg) + sgitg*rows_sg;
    const uint nb      = (uint) K / QK_K;

    simdgroup_float8x8 acc[NFRAG];  // accumulate in float
    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        e[0] = 0.0f; e[1] = 0.0f;
    }

    device const block_q4_K * base[NFRAG];
    for (int f = 0; f < NFRAG; ++f) base[f] = A + (ulong)(row0 + f*8 + lrow)*nb;

    for (uint ib = 0; ib < nb; ++ib) {
        for (uint g = 0; g < 4; ++g) {
            const ulong jlo = (ulong) ib*QK_K + (ulong)(2*g  )*32 + (ulong) lrow*4;
            const ulong jhi = (ulong) ib*QK_K + (ulong)(2*g+1)*32 + (ulong) lrow*4;
            const float4 bl0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jlo));
            const float4 bl1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jlo));
            const float4 bh0 = *((device const float4 *)(Bx + (ulong)(lcol  )*K + jhi));
            const float4 bh1 = *((device const float4 *)(Bx + (ulong)(lcol+1)*K + jhi));

            for (int f = 0; f < NFRAG; ++f) {
                device const block_q4_K * xb = base[f] + ib;
                const uint2 qv = *((device const uint2 *)(xb->qs + g*32 + lcol*4));
                const float dd = (float) xb->d * 255.0f;
                const float dm = (float) xb->dmin;

                uchar scl, mml, sch, mmh;
                get_scale_min_k4((int)(2*g  ), xb->scales, scl, mml);
                get_scale_min_k4((int)(2*g+1), xb->scales, sch, mmh);

                const float dl = dd * (float) scl, ml = dm * (float) mml;
                const float dh = dd * (float) sch, mh = dm * (float) mmh;

                const float4 al0 = fma(unpack_unorm4x8_to_float(qv.x & 0x0F0F0F0Fu), dl, -ml);
                const float4 al1 = fma(unpack_unorm4x8_to_float(qv.y & 0x0F0F0F0Fu), dl, -ml);
                const float4 ah0 = fma(unpack_unorm4x8_to_float((qv.x >> 4) & 0x0F0F0F0Fu), dh, -mh);
                const float4 ah1 = fma(unpack_unorm4x8_to_float((qv.y >> 4) & 0x0F0F0F0Fu), dh, -mh);

                for (uint t = 0; t < 4; ++t) {
                    simdgroup_half8x8 a, b;
                    { thread auto & e = b.thread_elements(); e[0] = (half) bl0[t]; e[1] = (half) bl1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = (half) al0[t]; e[1] = (half) al1[t]; }
                    simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
                    { thread auto & e = b.thread_elements(); e[0] = (half) bh0[t]; e[1] = (half) bh1[t]; }
                    { thread auto & e = a.thread_elements(); e[0] = (half) ah0[t]; e[1] = (half) ah1[t]; }
                    simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
                }
            }
        }
    }

    for (int f = 0; f < NFRAG; ++f) {
        thread auto & e = acc[f].thread_elements();
        const ulong row = (ulong)(row0 + f*8 + lrow);
        C[row*N + lcol    ] = e[0];
        C[row*N + lcol + 1] = e[1];
    }
}
