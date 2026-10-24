#if defined(__AVX512__)||defined(__AVX2__)
inline void __transpose4x4_pd(
    __m256d v0, __m256d v1, __m256d v2, __m256d v3,
    __m256d& r0, __m256d& r1, __m256d& r2, __m256d& r3)
{
    __m256d t0 = _mm256_unpacklo_pd(v0, v1);
    __m256d t1 = _mm256_unpackhi_pd(v0, v1);
    __m256d t2 = _mm256_unpacklo_pd(v2, v3);
    __m256d t3 = _mm256_unpackhi_pd(v2, v3);
    r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}
#endif

#ifdef __AVX512__

#elif defined(__AVX2__)

#endif







#ifdef __AVX512__

template<>
inline void __pack_A_block<double, double>(
    const double* __restrict A,
    size_t ldA, size_t mc, size_t kc,
    double* __restrict Ap)
{
    constexpr int MR = 16;

    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const int mm = min(MR, (int)(mc - ip * MR));
        double* dst = Ap + ip * kc * MR;

        if (mm == MR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = A + p * ldA + ip * MR;
                _mm512_store_pd(dst, _mm512_loadu_pd(src));
                _mm512_store_pd(dst + 8, _mm512_loadu_pd(src + 8));
                dst += MR;
            }
        }
        else {
            const __mmask8 m0 = (mm >= 8) ? (__mmask8)0xFF
                : (__mmask8)((1u << mm) - 1u);
            const __mmask8 m1 = (mm > 8) ? (__mmask8)((1u << (mm - 8)) - 1u)
                : (__mmask8)0;
            for (size_t p = 0; p < kc; ++p) {
                const double* src = A + p * ldA + ip * MR;
                _mm512_store_pd(dst, _mm512_maskz_loadu_pd(m0, src));
                _mm512_store_pd(dst + 8, _mm512_maskz_loadu_pd(m1, src + 8));
                dst += MR;
            }
        }
    }
}
template<>
inline void __pack_AT_block<double, double>(
    const double* __restrict Vd,
    size_t ldV, size_t pc,
    size_t mc, size_t kc,
    double* __restrict Ap)
{
    constexpr size_t MR = 16;

    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const size_t mm = min(MR, mc - ip * MR);
        double* dst = Ap + ip * kc * MR;
        const double* base = Vd + (ip * MR) * ldV + pc;

        size_t q = 0;
        if (mm == MR) {
            for (; q + 4 <= kc; q += 4) {
                __m256d v0 = _mm256_loadu_pd(base + 0 * ldV + q);
                __m256d v1 = _mm256_loadu_pd(base + 1 * ldV + q);
                __m256d v2 = _mm256_loadu_pd(base + 2 * ldV + q);
                __m256d v3 = _mm256_loadu_pd(base + 3 * ldV + q);
                __m256d v4 = _mm256_loadu_pd(base + 4 * ldV + q);
                __m256d v5 = _mm256_loadu_pd(base + 5 * ldV + q);
                __m256d v6 = _mm256_loadu_pd(base + 6 * ldV + q);
                __m256d v7 = _mm256_loadu_pd(base + 7 * ldV + q);
                __m256d v8 = _mm256_loadu_pd(base + 8 * ldV + q);
                __m256d v9 = _mm256_loadu_pd(base + 9 * ldV + q);
                __m256d v10 = _mm256_loadu_pd(base + 10 * ldV + q);
                __m256d v11 = _mm256_loadu_pd(base + 11 * ldV + q);
                __m256d v12 = _mm256_loadu_pd(base + 12 * ldV + q);
                __m256d v13 = _mm256_loadu_pd(base + 13 * ldV + q);
                __m256d v14 = _mm256_loadu_pd(base + 14 * ldV + q);
                __m256d v15 = _mm256_loadu_pd(base + 15 * ldV + q);

                __m256d r0, r1, r2, r3, r4, r5, r6, r7;
                __m256d r8, r9, r10, r11, r12, r13, r14, r15;
                __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);
                __transpose4x4_pd(v4, v5, v6, v7, r4, r5, r6, r7);
                __transpose4x4_pd(v8, v9, v10, v11, r8, r9, r10, r11);
                __transpose4x4_pd(v12, v13, v14, v15, r12, r13, r14, r15);

                _mm256_storeu_pd(dst + (q + 0) * MR + 0, r0);
                _mm256_storeu_pd(dst + (q + 0) * MR + 4, r4);
                _mm256_storeu_pd(dst + (q + 0) * MR + 8, r8);
                _mm256_storeu_pd(dst + (q + 0) * MR + 12, r12);

                _mm256_storeu_pd(dst + (q + 1) * MR + 0, r1);
                _mm256_storeu_pd(dst + (q + 1) * MR + 4, r5);
                _mm256_storeu_pd(dst + (q + 1) * MR + 8, r9);
                _mm256_storeu_pd(dst + (q + 1) * MR + 12, r13);

                _mm256_storeu_pd(dst + (q + 2) * MR + 0, r2);
                _mm256_storeu_pd(dst + (q + 2) * MR + 4, r6);
                _mm256_storeu_pd(dst + (q + 2) * MR + 8, r10);
                _mm256_storeu_pd(dst + (q + 2) * MR + 12, r14);

                _mm256_storeu_pd(dst + (q + 3) * MR + 0, r3);
                _mm256_storeu_pd(dst + (q + 3) * MR + 4, r7);
                _mm256_storeu_pd(dst + (q + 3) * MR + 8, r11);
                _mm256_storeu_pd(dst + (q + 3) * MR + 12, r15);
            }
        }
        for (; q < kc; ++q) {
            const double* src = base + q;
            for (size_t i = 0; i < mm; ++i) dst[q * MR + i] = src[i * ldV];
            for (size_t i = mm; i < MR; ++i) dst[q * MR + i] = 0.0;
        }
    }
}

template<>
inline void __pack_B_block<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    double* __restrict Bp)
{
    constexpr int NR = 12;

    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const int nn = min(NR, (int)(nc - jp * NR));
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + (jp * NR) * ldB;

        if (nn == NR) {
            size_t p = 0;
            for (; p + 4 <= kc; p += 4) {
                __m256d v0 = _mm256_loadu_pd(Bcol + 0 * ldB + p);
                __m256d v1 = _mm256_loadu_pd(Bcol + 1 * ldB + p);
                __m256d v2 = _mm256_loadu_pd(Bcol + 2 * ldB + p);
                __m256d v3 = _mm256_loadu_pd(Bcol + 3 * ldB + p);
                __m256d v4 = _mm256_loadu_pd(Bcol + 4 * ldB + p);
                __m256d v5 = _mm256_loadu_pd(Bcol + 5 * ldB + p);
                __m256d v6 = _mm256_loadu_pd(Bcol + 6 * ldB + p);
                __m256d v7 = _mm256_loadu_pd(Bcol + 7 * ldB + p);
                __m256d v8 = _mm256_loadu_pd(Bcol + 8 * ldB + p);
                __m256d v9 = _mm256_loadu_pd(Bcol + 9 * ldB + p);
                __m256d v10 = _mm256_loadu_pd(Bcol + 10 * ldB + p);
                __m256d v11 = _mm256_loadu_pd(Bcol + 11 * ldB + p);

                __m256d r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;
                __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);
                __transpose4x4_pd(v4, v5, v6, v7, r4, r5, r6, r7);
                __transpose4x4_pd(v8, v9, v10, v11, r8, r9, r10, r11);

                _mm256_storeu_pd(dst + (p + 0) * NR + 0, r0);
                _mm256_storeu_pd(dst + (p + 0) * NR + 4, r4);
                _mm256_storeu_pd(dst + (p + 0) * NR + 8, r8);

                _mm256_storeu_pd(dst + (p + 1) * NR + 0, r1);
                _mm256_storeu_pd(dst + (p + 1) * NR + 4, r5);
                _mm256_storeu_pd(dst + (p + 1) * NR + 8, r9);

                _mm256_storeu_pd(dst + (p + 2) * NR + 0, r2);
                _mm256_storeu_pd(dst + (p + 2) * NR + 4, r6);
                _mm256_storeu_pd(dst + (p + 2) * NR + 8, r10);

                _mm256_storeu_pd(dst + (p + 3) * NR + 0, r3);
                _mm256_storeu_pd(dst + (p + 3) * NR + 4, r7);
                _mm256_storeu_pd(dst + (p + 3) * NR + 8, r11);
            }

            for (; p < kc; ++p)
                for (int j = 0; j < NR; ++j)
                    dst[p * NR + j] = Bcol[j * ldB + p];
        }
        else {
            for (int j = 0; j < nn; ++j) {
                const double* col = Bcol + j * ldB;
                for (size_t p = 0; p < kc; ++p)
                    dst[p * NR + j] = col[p];
            }
            for (int j = nn; j < NR; ++j)
                for (size_t p = 0; p < kc; ++p)
                    dst[p * NR + j] = 0.0;
        }
    }
}

template<>
inline void __pack_BT_block<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    double* __restrict Bp)
{
    constexpr int NR = 12;

    for (size_t jp = 0; jp* NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + jp * NR;

        if (nn == NR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                _mm512_storeu_pd(dst, _mm512_loadu_pd(src));
                _mm256_storeu_pd(dst + 8, _mm256_loadu_pd(src + 8));
                dst += NR;
            }
        }
        else {
            const __mmask8 m0 = (nn >= 8) ? (__mmask8)0xFF
                : (nn > 0) ? (__mmask8)((1u << nn) - 1u) : (__mmask8)0;
            const int rem = (nn > 8) ? (nn - 8) : 0;
            const __m256i m1 = _mm256_set_epi64x(
                rem > 3 ? -1LL : 0LL, rem > 2 ? -1LL : 0LL,
                rem > 1 ? -1LL : 0LL, rem > 0 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                _mm512_storeu_pd(dst, _mm512_maskz_loadu_pd(m0, src));
                _mm256_storeu_pd(dst + 8, _mm256_maskload_pd(src + 8, m1));
                dst += NR;
            }
        }
    }
}

template<>
inline void __pack_BT_block_d<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    const double* __restrict d,
    double* __restrict Bp)
{
    constexpr int NR = 12;

    for (size_t jp = 0; jp* NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + jp * NR;

        if (nn == NR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                __m512d d512 = _mm512_set1_pd(d[p]);
                __m256d d256 = _mm256_set1_pd(d[p]);
                _mm512_storeu_pd(dst, _mm512_mul_pd(_mm512_loadu_pd(src), d512));
                _mm256_storeu_pd(dst + 8, _mm256_mul_pd(_mm256_loadu_pd(src + 8), d256));
                dst += NR;
            }
        }
        else {
            const __mmask8 m0 = (nn >= 8) ? (__mmask8)0xFF
                : (nn > 0) ? (__mmask8)((1u << nn) - 1u) : (__mmask8)0;
            const int rem = (nn > 8) ? (nn - 8) : 0;
            const __m256i m1 = _mm256_set_epi64x(
                rem > 3 ? -1LL : 0LL, rem > 2 ? -1LL : 0LL,
                rem > 1 ? -1LL : 0LL, rem > 0 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                __m512d d512 = _mm512_set1_pd(d[p]);
                __m256d d256 = _mm256_set1_pd(d[p]);
                _mm512_storeu_pd(dst,
                    _mm512_mul_pd(_mm512_maskz_loadu_pd(m0, src), d512));
                _mm256_storeu_pd(dst + 8,
                    _mm256_mul_pd(_mm256_maskload_pd(src + 8, m1), d256));
                dst += NR;
            }
        }
    }
}

#elif defined(__AVX2__)

template<>
inline void __pack_A_block<double, double>(
    const double* __restrict A,
    size_t ldA, size_t mc, size_t kc,
    double* __restrict Ap)
{
    constexpr int MR = 8;

    for (size_t ip = 0; ip* MR < mc; ++ip) {
        const int mm = min(MR, (int)(mc - ip * MR));
        double* dst = Ap + ip * kc * MR;

        if (mm == MR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = A + p * ldA + ip * MR;
                _mm256_store_pd(dst, _mm256_loadu_pd(src));
                _mm256_store_pd(dst + 4, _mm256_loadu_pd(src + 4));
                dst += MR;
            }
        }
        else {
            const __m256i mask0 = _mm256_set_epi64x(
                mm > 3 ? -1LL : 0LL, mm > 2 ? -1LL : 0LL,
                mm > 1 ? -1LL : 0LL, mm > 0 ? -1LL : 0LL);
            const __m256i mask1 = _mm256_set_epi64x(
                mm > 7 ? -1LL : 0LL, mm > 6 ? -1LL : 0LL,
                mm > 5 ? -1LL : 0LL, mm > 4 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = A + p * ldA + ip * MR;
                _mm256_store_pd(dst, _mm256_maskload_pd(src, mask0));
                _mm256_store_pd(dst + 4, _mm256_maskload_pd(src + 4, mask1));
                dst += MR;
            }
        }
    }
}
template<>
inline void __pack_AT_block<double, double>(
    const double* __restrict Vd,
    size_t ldV, size_t pc,
    size_t mc, size_t kc,
    double* __restrict Ap)
{
    constexpr size_t MR = 8;

    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const size_t mm = min(MR, mc - ip * MR);
        double* dst = Ap + ip * kc * MR;
        const double* base = Vd + (ip * MR) * ldV + pc;

        size_t q = 0;
        if (mm == MR) {
            for (; q + 4 <= kc; q += 4) {
                __m256d v0 = _mm256_loadu_pd(base + 0 * ldV + q);
                __m256d v1 = _mm256_loadu_pd(base + 1 * ldV + q);
                __m256d v2 = _mm256_loadu_pd(base + 2 * ldV + q);
                __m256d v3 = _mm256_loadu_pd(base + 3 * ldV + q);
                __m256d v4 = _mm256_loadu_pd(base + 4 * ldV + q);
                __m256d v5 = _mm256_loadu_pd(base + 5 * ldV + q);
                __m256d v6 = _mm256_loadu_pd(base + 6 * ldV + q);
                __m256d v7 = _mm256_loadu_pd(base + 7 * ldV + q);

                __m256d r0, r1, r2, r3, r4, r5, r6, r7;
                __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);
                __transpose4x4_pd(v4, v5, v6, v7, r4, r5, r6, r7);

                _mm256_storeu_pd(dst + (q + 0) * MR + 0, r0);
                _mm256_storeu_pd(dst + (q + 0) * MR + 4, r4);

                _mm256_storeu_pd(dst + (q + 1) * MR + 0, r1);
                _mm256_storeu_pd(dst + (q + 1) * MR + 4, r5);

                _mm256_storeu_pd(dst + (q + 2) * MR + 0, r2);
                _mm256_storeu_pd(dst + (q + 2) * MR + 4, r6);

                _mm256_storeu_pd(dst + (q + 3) * MR + 0, r3);
                _mm256_storeu_pd(dst + (q + 3) * MR + 4, r7);
            }
        }
        for (; q < kc; ++q) {
            const double* src = base + q;
            for (size_t i = 0; i < mm; ++i) dst[q * MR + i] = src[i * ldV];
            for (size_t i = mm; i < MR; ++i) dst[q * MR + i] = 0.0;
        }
    }
}

template<>
inline void __pack_B_block<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    double* __restrict Bp)
{
    constexpr int NR = 6;

    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const int nn = min(NR, (int)(nc - jp * NR));
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + (jp * NR) * ldB;

        if (nn == NR) {
            size_t p = 0;
            for (; p + 4 <= kc; p += 4) {
                __m256d v0 = _mm256_loadu_pd(Bcol + 0 * ldB + p);
                __m256d v1 = _mm256_loadu_pd(Bcol + 1 * ldB + p);
                __m256d v2 = _mm256_loadu_pd(Bcol + 2 * ldB + p);
                __m256d v3 = _mm256_loadu_pd(Bcol + 3 * ldB + p);
                __m256d r0, r1, r2, r3;
                __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);

                __m256d v4 = _mm256_loadu_pd(Bcol + 4 * ldB + p);
                __m256d v5 = _mm256_loadu_pd(Bcol + 5 * ldB + p);
                __m256d t0 = _mm256_unpacklo_pd(v4, v5);
                __m256d t1 = _mm256_unpackhi_pd(v4, v5);

                _mm256_storeu_pd(dst + (p + 0) * NR + 0, r0);
                _mm_store_pd(dst + (p + 0) * NR + 4, _mm256_castpd256_pd128(t0));
                _mm256_storeu_pd(dst + (p + 1) * NR + 0, r1);
                _mm_store_pd(dst + (p + 1) * NR + 4, _mm256_castpd256_pd128(t1));
                _mm256_storeu_pd(dst + (p + 2) * NR + 0, r2);
                _mm_store_pd(dst + (p + 2) * NR + 4, _mm256_extractf128_pd(t0, 1));
                _mm256_storeu_pd(dst + (p + 3) * NR + 0, r3);
                _mm_store_pd(dst + (p + 3) * NR + 4, _mm256_extractf128_pd(t1, 1));
            }
            for (; p < kc; ++p)
                for (int j = 0; j < NR; ++j)
                    dst[p * NR + j] = Bcol[(size_t)j * ldB + p];
        }
        else {
            for (size_t j = 0; j < (size_t)nn; ++j) {
                const double* col = Bcol + j * ldB;
                for (size_t p = 0; p < kc; ++p)
                    dst[p * NR + j] = col[p];
            }
            for (size_t j = (size_t)nn; j < NR; ++j)
                for (size_t p = 0; p < kc; ++p)
                    dst[p * NR + j] = 0.0;
        }
    }
}

template<>
inline void __pack_BT_block<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    double* __restrict Bp)
{
    constexpr int NR = 6;

    for (size_t jp = 0; jp* NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + jp * NR;

        if (nn == NR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                _mm256_storeu_pd(dst, _mm256_loadu_pd(src));
                _mm_storeu_pd(dst + 4, _mm_loadu_pd(src + 4));
                dst += NR;
            }
        }
        else {
            const __m256i m0 = _mm256_set_epi64x(
                nn > 3 ? -1LL : 0LL, nn > 2 ? -1LL : 0LL,
                nn > 1 ? -1LL : 0LL, nn > 0 ? -1LL : 0LL);
            const int rem = (nn > 4) ? (nn - 4) : 0;
            const __m128i m1 = _mm_set_epi64x(
                rem > 1 ? -1LL : 0LL, rem > 0 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                _mm256_storeu_pd(dst, _mm256_maskload_pd(src, m0));
                _mm_storeu_pd(dst + 4, _mm_maskload_pd(src + 4, m1));
                dst += NR;
            }
        }
    }
}

template<>
inline void __pack_BT_block_d<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    const double* __restrict d,
    double* __restrict Bp)
{
    constexpr int NR = 6;

    for (size_t jp = 0; jp* NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + jp * NR;

        if (nn == NR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                __m256d d256 = _mm256_set1_pd(d[p]);
                __m128d d128 = _mm_set1_pd(d[p]);
                _mm256_storeu_pd(dst, _mm256_mul_pd(_mm256_loadu_pd(src), d256));
                _mm_storeu_pd(dst + 4, _mm_mul_pd(_mm_loadu_pd(src + 4), d128));
                dst += NR;
            }
        }
        else {
            const __m256i m0 = _mm256_set_epi64x(
                nn > 3 ? -1LL : 0LL, nn > 2 ? -1LL : 0LL,
                nn > 1 ? -1LL : 0LL, nn > 0 ? -1LL : 0LL);
            const int rem = (nn > 4) ? (nn - 4) : 0;
            const __m128i m1 = _mm_set_epi64x(
                rem > 1 ? -1LL : 0LL, rem > 0 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                __m256d d256 = _mm256_set1_pd(d[p]);
                __m128d d128 = _mm_set1_pd(d[p]);
                _mm256_storeu_pd(dst,
                    _mm256_mul_pd(_mm256_maskload_pd(src, m0), d256));
                _mm_storeu_pd(dst + 4,
                    _mm_mul_pd(_mm_maskload_pd(src + 4, m1), d128));
                dst += NR;
            }
        }
    }
}

#endif










#ifdef __AVX512__

template<>
inline void __pack_A_block_L<double, double>(
    const double* __restrict Vd,
    size_t ldV, size_t rows, size_t cols,
    double* __restrict Lp)
{
    constexpr size_t MR = 16;
    const size_t s = min(rows, ((cols + MR - 1) / MR) * MR);

    const __m512d ones = _mm512_set1_pd(1.0);

    for (size_t r = 0; r < s; r += MR) {
        const size_t m = min(MR, s - r);
        const size_t p_max = min(cols, r + m);
        double* dst = Lp + r * cols;

        if (m == MR) {
            for (size_t p = 0; p < p_max; ++p) {
                const double* Vc = Vd + p * ldV;
                const double* src = Vc + r;

                if (p < r) {
                    _mm512_storeu_pd(dst, _mm512_loadu_pd(src));
                    _mm512_storeu_pd(dst + 8, _mm512_loadu_pd(src + 8));
                }
                else {
                    const size_t j = p - r;

                    __mmask8 m_lo = (j >= 8) ? (__mmask8)0
                        : (__mmask8)(0xFEu << j);
                    __m512d v_lo = _mm512_maskz_loadu_pd(m_lo, src);
                    if (j < 8)
                        v_lo = _mm512_mask_mov_pd(v_lo, (__mmask8)(1u << j), ones);

                    __mmask8 m_hi = (j < 8) ? (__mmask8)0xFF
                        : (__mmask8)((j >= 15) ? 0 : (0xFEu << (j - 8)));
                    __m512d v_hi = _mm512_maskz_loadu_pd(m_hi, src + 8);
                    if (j >= 8 && j < 16)
                        v_hi = _mm512_mask_mov_pd(v_hi, (__mmask8)(1u << (j - 8)), ones);

                    _mm512_storeu_pd(dst, v_lo);
                    _mm512_storeu_pd(dst + 8, v_hi);
                }

                dst += MR;
            }
        }
        else {
            for (size_t p = 0; p < p_max; ++p) {
                const double* Vc = Vd + p * ldV;

                if (p < r) {
                    for (size_t i = 0; i < m; ++i)
                        dst[i] = Vc[r + i];
                }
                else {
                    for (size_t i = 0; i < m; ++i) {
                        const size_t row = r + i;
                        if (row > p)
                            dst[i] = Vc[row];
                        else if (row == p)
                            dst[i] = 1.0;
                        else
                            dst[i] = 0.0;
                    }
                }

                for (size_t i = m; i < MR; ++i)
                    dst[i] = 0.0;

                dst += MR;
            }
        }
    }
}


template<>
inline void __pack_A_block_LT<double, double>(
    const double* __restrict Vd,
    size_t ldV, size_t rows, size_t cols,
    double* __restrict LTp)
{
    constexpr size_t MR = 16;

    for (size_t r = 0; r < cols; r += MR) {
        const size_t m = min(MR, cols - r);
        double* dst0 = LTp + r * cols;

        if (m == MR) {
            if (r > 0)
                std::memset(dst0, 0, r * MR * sizeof(double));

            size_t p = r;
            for (; p + 4 <= cols; p += 4) {
                for (size_t i0 = 0; i0 < MR; i0 += 4) {
                    __m256d v0 = _mm256_loadu_pd(Vd + (r + i0 + 0) * ldV + p);
                    __m256d v1 = _mm256_loadu_pd(Vd + (r + i0 + 1) * ldV + p);
                    __m256d v2 = _mm256_loadu_pd(Vd + (r + i0 + 2) * ldV + p);
                    __m256d v3 = _mm256_loadu_pd(Vd + (r + i0 + 3) * ldV + p);

                    __m256d c0, c1, c2, c3;
                    __transpose4x4_pd(v0, v1, v2, v3, c0, c1, c2, c3);
                    __m256d cols_vec[4] = { c0, c1, c2, c3 };

                    for (int j = 0; j < 4; ++j) {
                        const size_t pj = p + j;
                        const long long kk = (long long)(pj - r) - (long long)i0;
                        __m256d v = cols_vec[j];

                        if (kk < 0) {
                            v = _mm256_setzero_pd();
                        }
                        else if (kk < 4) {
                            const __m256i keep = _mm256_set_epi64x(
                                (kk > 3) ? -1LL : 0LL,
                                (kk > 2) ? -1LL : 0LL,
                                (kk > 1) ? -1LL : 0LL,
                                (kk > 0) ? -1LL : 0LL);
                            const __m256i one = _mm256_set_epi64x(
                                (kk == 3) ? -1LL : 0LL,
                                (kk == 2) ? -1LL : 0LL,
                                (kk == 1) ? -1LL : 0LL,
                                (kk == 0) ? -1LL : 0LL);
                            v = _mm256_blendv_pd(
                                _mm256_and_pd(v, _mm256_castsi256_pd(keep)),
                                _mm256_set1_pd(1.0),
                                _mm256_castsi256_pd(one));
                        }

                        _mm256_storeu_pd(dst0 + pj * MR + i0, v);
                    }
                }
            }

            for (; p < cols; ++p) {
                double* dst = dst0 + p * MR;
                const size_t k = p - r;

                for (size_t i = 0; i < MR; ++i) {
                    if (i < k)
                        dst[i] = Vd[(r + i) * ldV + p];
                    else if (i == k)
                        dst[i] = 1.0;
                    else
                        dst[i] = 0.0;
                }
            }
        }
        else {
            for (size_t p = 0; p < cols; ++p) {
                double* dst = dst0 + p * MR;

                for (size_t i = 0; i < m; ++i) {
                    const size_t row = r + i;
                    if (row < p)
                        dst[i] = Vd[row * ldV + p];
                    else if (row == p)
                        dst[i] = 1.0;
                    else
                        dst[i] = 0.0;
                }

                for (size_t i = m; i < MR; ++i)
                    dst[i] = 0.0;
            }
        }
    }
}


template<>
inline void __pack_A_block_U<double, double>(
    const double* __restrict Tmat,
    size_t k, double* __restrict Up)
{
    constexpr size_t MR = 16;

    for (size_t r = 0; r < k; r += MR) {
        const size_t m = min(MR, k - r);
        double* dst = Up + r * (k + MR);

        if (m == MR) {
            for (size_t p = r; p < k; ++p) {
                const double* src = Tmat + p * k + r;
                const size_t k0 = p - r;

                if (k0 >= 15) {
                    _mm512_storeu_pd(dst, _mm512_loadu_pd(src));
                    _mm512_storeu_pd(dst + 8, _mm512_loadu_pd(src + 8));
                }
                else {
                    const __mmask8 m_lo = (k0 >= 7) ? (__mmask8)0xFF
                        : (__mmask8)((1u << (k0 + 1)) - 1u);
                    const __mmask8 m_hi = (k0 < 8) ? (__mmask8)0
                        : (__mmask8)((1u << (k0 - 7)) - 1u);
                    _mm512_storeu_pd(dst, _mm512_maskz_loadu_pd(m_lo, src));
                    _mm512_storeu_pd(dst + 8, _mm512_maskz_loadu_pd(m_hi, src + 8));
                }

                dst += MR;
            }
        }
        else {
            for (size_t p = r; p < k; ++p) {
                for (size_t i = 0; i < m; ++i) {
                    const size_t row = r + i;
                    dst[i] = (row > p) ? 0.0 : Tmat[p * k + row];
                }

                for (size_t i = m; i < MR; ++i)
                    dst[i] = 0.0;

                dst += MR;
            }
        }
    }
}


template<>
inline void __pack_A_block_UT<double, double>(
    const double* __restrict Tmat,
    size_t k, double* __restrict UTp)
{
    constexpr size_t MR = 16;

    for (size_t r = 0; r < k; r += MR) {
        const size_t m = min(MR, k - r);
        const size_t p_max = min(k, r + m);
        double* dst0 = UTp + r * k;

        if (m == MR) {
            size_t p = 0;
            for (; p + 4 <= p_max; p += 4) {
                for (size_t i0 = 0; i0 < MR; i0 += 4) {
                    __m256d v0 = _mm256_loadu_pd(Tmat + (r + i0 + 0) * k + p);
                    __m256d v1 = _mm256_loadu_pd(Tmat + (r + i0 + 1) * k + p);
                    __m256d v2 = _mm256_loadu_pd(Tmat + (r + i0 + 2) * k + p);
                    __m256d v3 = _mm256_loadu_pd(Tmat + (r + i0 + 3) * k + p);

                    __m256d c0, c1, c2, c3;
                    __transpose4x4_pd(v0, v1, v2, v3, c0, c1, c2, c3);
                    __m256d cols_vec[4] = { c0, c1, c2, c3 };

                    for (int j = 0; j < 4; ++j) {
                        const size_t pj = p + j;
                        const long long kk = (long long)(pj - r) - (long long)i0;
                        __m256d v = cols_vec[j];

                        if (kk >= 4) {
                            v = _mm256_setzero_pd();
                        }
                        else if (kk > 0) {
                            const __m256i keep = _mm256_set_epi64x(
                                (kk <= 3) ? -1LL : 0LL,
                                (kk <= 2) ? -1LL : 0LL,
                                (kk <= 1) ? -1LL : 0LL,
                                (kk <= 0) ? -1LL : 0LL);
                            v = _mm256_and_pd(v, _mm256_castsi256_pd(keep));
                        }

                        _mm256_storeu_pd(dst0 + pj * MR + i0, v);
                    }
                }
            }

            for (; p < p_max; ++p) {
                double* dst = dst0 + p * MR;
                const size_t k0 = (p <= r) ? 0 : p - r;

                for (size_t i = 0; i < MR; ++i) {
                    if (i < k0)
                        dst[i] = 0.0;
                    else
                        dst[i] = Tmat[(r + i) * k + p];
                }
            }
        }
        else {
            for (size_t p = 0; p < p_max; ++p) {
                double* dst = dst0 + p * MR;
                const size_t k0 = (p <= r) ? 0 : p - r;

                for (size_t i = 0; i < m; ++i) {
                    if (i < k0)
                        dst[i] = 0.0;
                    else
                        dst[i] = Tmat[(r + i) * k + p];
                }

                for (size_t i = m; i < MR; ++i)
                    dst[i] = 0.0;
            }
        }
    }
}


#elif defined(__AVX2__)

template<>
inline void __pack_A_block_L<double, double>(
    const double* __restrict Vd,
    size_t ldV, size_t rows, size_t cols,
    double* __restrict Lp)
{
    constexpr size_t MR = 8;
    const size_t s = min(rows, ((cols + MR - 1) / MR) * MR);

    const __m256d ones = _mm256_set1_pd(1.0);

    for (size_t r = 0; r < s; r += MR) {
        const size_t m = min(MR, s - r);
        const size_t p_max = min(cols, r + m);
        double* dst = Lp + r * cols;

        if (m == MR) {
            for (size_t p = 0; p < p_max; ++p) {
                const double* Vc = Vd + p * ldV;
                const double* src = Vc + r;

                if (p < r) {
                    _mm256_storeu_pd(dst, _mm256_loadu_pd(src));
                    _mm256_storeu_pd(dst + 4, _mm256_loadu_pd(src + 4));
                }
                else {
                    const size_t j = p - r;

                    const __m256i m_lo = _mm256_set_epi64x(
                        (3 > (long long)j) ? -1LL : 0LL,
                        (2 > (long long)j) ? -1LL : 0LL,
                        (1 > (long long)j) ? -1LL : 0LL,
                        (0 > (long long)j) ? -1LL : 0LL);
                    __m256d v_lo = _mm256_maskload_pd(src, m_lo);
                    if (j < 4) {
                        const __m256i eq_i = _mm256_set_epi64x(
                            (3 == (long long)j) ? -1LL : 0LL,
                            (2 == (long long)j) ? -1LL : 0LL,
                            (1 == (long long)j) ? -1LL : 0LL,
                            (0 == (long long)j) ? -1LL : 0LL);
                        v_lo = _mm256_blendv_pd(v_lo, ones, _mm256_castsi256_pd(eq_i));
                    }

                    const __m256i m_hi = _mm256_set_epi64x(
                        (7 > (long long)j) ? -1LL : 0LL,
                        (6 > (long long)j) ? -1LL : 0LL,
                        (5 > (long long)j) ? -1LL : 0LL,
                        (4 > (long long)j) ? -1LL : 0LL);
                    __m256d v_hi = _mm256_maskload_pd(src + 4, m_hi);
                    if (j >= 4 && j < 8) {
                        const __m256i eq_i = _mm256_set_epi64x(
                            (7 == (long long)j) ? -1LL : 0LL,
                            (6 == (long long)j) ? -1LL : 0LL,
                            (5 == (long long)j) ? -1LL : 0LL,
                            (4 == (long long)j) ? -1LL : 0LL);
                        v_hi = _mm256_blendv_pd(v_hi, ones, _mm256_castsi256_pd(eq_i));
                    }

                    _mm256_storeu_pd(dst, v_lo);
                    _mm256_storeu_pd(dst + 4, v_hi);
                }

                dst += MR;
            }
        }
        else {
            for (size_t p = 0; p < p_max; ++p) {
                const double* Vc = Vd + p * ldV;

                if (p < r) {
                    for (size_t i = 0; i < m; ++i)
                        dst[i] = Vc[r + i];
                }
                else {
                    for (size_t i = 0; i < m; ++i) {
                        const size_t row = r + i;
                        if (row > p)
                            dst[i] = Vc[row];
                        else if (row == p)
                            dst[i] = 1.0;
                        else
                            dst[i] = 0.0;
                    }
                }

                for (size_t i = m; i < MR; ++i)
                    dst[i] = 0.0;

                dst += MR;
            }
        }
    }
}


template<>
inline void __pack_A_block_LT<double, double>(
    const double* __restrict Vd,
    size_t ldV, size_t rows, size_t cols,
    double* __restrict LTp)
{
    constexpr size_t MR = 8;

    for (size_t r = 0; r < cols; r += MR) {
        const size_t m = min(MR, cols - r);
        double* dst0 = LTp + r * cols;

        if (m == MR) {
            if (r > 0)
                std::memset(dst0, 0, r * MR * sizeof(double));

            size_t p = r;
            for (; p + 4 <= cols; p += 4) {
                for (size_t i0 = 0; i0 < MR; i0 += 4) {
                    __m256d v0 = _mm256_loadu_pd(Vd + (r + i0 + 0) * ldV + p);
                    __m256d v1 = _mm256_loadu_pd(Vd + (r + i0 + 1) * ldV + p);
                    __m256d v2 = _mm256_loadu_pd(Vd + (r + i0 + 2) * ldV + p);
                    __m256d v3 = _mm256_loadu_pd(Vd + (r + i0 + 3) * ldV + p);

                    __m256d c0, c1, c2, c3;
                    __transpose4x4_pd(v0, v1, v2, v3, c0, c1, c2, c3);
                    __m256d cols_vec[4] = { c0, c1, c2, c3 };

                    for (int j = 0; j < 4; ++j) {
                        const size_t pj = p + j;
                        const long long kk = (long long)(pj - r) - (long long)i0;
                        __m256d v = cols_vec[j];

                        if (kk < 0) {
                            v = _mm256_setzero_pd();
                        }
                        else if (kk < 4) {
                            const __m256i keep = _mm256_set_epi64x(
                                (kk > 3) ? -1LL : 0LL,
                                (kk > 2) ? -1LL : 0LL,
                                (kk > 1) ? -1LL : 0LL,
                                (kk > 0) ? -1LL : 0LL);
                            const __m256i one = _mm256_set_epi64x(
                                (kk == 3) ? -1LL : 0LL,
                                (kk == 2) ? -1LL : 0LL,
                                (kk == 1) ? -1LL : 0LL,
                                (kk == 0) ? -1LL : 0LL);
                            v = _mm256_blendv_pd(
                                _mm256_and_pd(v, _mm256_castsi256_pd(keep)),
                                _mm256_set1_pd(1.0),
                                _mm256_castsi256_pd(one));
                        }

                        _mm256_storeu_pd(dst0 + pj * MR + i0, v);
                    }
                }
            }

            for (; p < cols; ++p) {
                double* dst = dst0 + p * MR;
                const size_t k = p - r;

                for (size_t i = 0; i < MR; ++i) {
                    if (i < k)
                        dst[i] = Vd[(r + i) * ldV + p];
                    else if (i == k)
                        dst[i] = 1.0;
                    else
                        dst[i] = 0.0;
                }
            }
        }
        else {
            for (size_t p = 0; p < cols; ++p) {
                double* dst = dst0 + p * MR;

                for (size_t i = 0; i < m; ++i) {
                    const size_t row = r + i;
                    if (row < p)
                        dst[i] = Vd[row * ldV + p];
                    else if (row == p)
                        dst[i] = 1.0;
                    else
                        dst[i] = 0.0;
                }

                for (size_t i = m; i < MR; ++i)
                    dst[i] = 0.0;
            }
        }
    }
}


template<>
inline void __pack_A_block_U<double, double>(
    const double* __restrict Tmat,
    size_t k, double* __restrict Up)
{
    constexpr size_t MR = 8;

    for (size_t r = 0; r < k; r += MR) {
        const size_t m = min(MR, k - r);
        double* dst = Up + r * (k + MR);

        if (m == MR) {
            for (size_t p = r; p < k; ++p) {
                const double* src = Tmat + p * k + r;
                const size_t k0 = p - r;

                if (k0 >= 7) {
                    _mm256_storeu_pd(dst, _mm256_loadu_pd(src));
                    _mm256_storeu_pd(dst + 4, _mm256_loadu_pd(src + 4));
                }
                else {
                    const __m256i m_lo = _mm256_set_epi64x(
                        (3 <= (long long)k0) ? -1LL : 0LL,
                        (2 <= (long long)k0) ? -1LL : 0LL,
                        (1 <= (long long)k0) ? -1LL : 0LL,
                        (0 <= (long long)k0) ? -1LL : 0LL);
                    const __m256i m_hi = _mm256_set_epi64x(
                        (7 <= (long long)k0) ? -1LL : 0LL,
                        (6 <= (long long)k0) ? -1LL : 0LL,
                        (5 <= (long long)k0) ? -1LL : 0LL,
                        (4 <= (long long)k0) ? -1LL : 0LL);
                    _mm256_storeu_pd(dst, _mm256_maskload_pd(src, m_lo));
                    _mm256_storeu_pd(dst + 4, _mm256_maskload_pd(src + 4, m_hi));
                }

                dst += MR;
            }
        }
        else {
            for (size_t p = r; p < k; ++p) {
                for (size_t i = 0; i < m; ++i) {
                    const size_t row = r + i;
                    dst[i] = (row > p) ? 0.0 : Tmat[p * k + row];
                }

                for (size_t i = m; i < MR; ++i)
                    dst[i] = 0.0;

                dst += MR;
            }
        }
    }
}


template<>
inline void __pack_A_block_UT<double, double>(
    const double* __restrict Tmat,
    size_t k, double* __restrict UTp)
{
    constexpr size_t MR = 8;

    for (size_t r = 0; r < k; r += MR) {
        const size_t m = min(MR, k - r);
        const size_t p_max = min(k, r + m);
        double* dst0 = UTp + r * k;

        if (m == MR) {
            size_t p = 0;
            for (; p + 4 <= p_max; p += 4) {
                for (size_t i0 = 0; i0 < MR; i0 += 4) {
                    __m256d v0 = _mm256_loadu_pd(Tmat + (r + i0 + 0) * k + p);
                    __m256d v1 = _mm256_loadu_pd(Tmat + (r + i0 + 1) * k + p);
                    __m256d v2 = _mm256_loadu_pd(Tmat + (r + i0 + 2) * k + p);
                    __m256d v3 = _mm256_loadu_pd(Tmat + (r + i0 + 3) * k + p);

                    __m256d c0, c1, c2, c3;
                    __transpose4x4_pd(v0, v1, v2, v3, c0, c1, c2, c3);
                    __m256d cols_vec[4] = { c0, c1, c2, c3 };

                    for (int j = 0; j < 4; ++j) {
                        const size_t pj = p + j;
                        const long long kk = (long long)(pj - r) - (long long)i0;
                        __m256d v = cols_vec[j];

                        if (kk >= 4) {
                            v = _mm256_setzero_pd();
                        }
                        else if (kk > 0) {
                            const __m256i keep = _mm256_set_epi64x(
                                (kk <= 3) ? -1LL : 0LL,
                                (kk <= 2) ? -1LL : 0LL,
                                (kk <= 1) ? -1LL : 0LL,
                                (kk <= 0) ? -1LL : 0LL);
                            v = _mm256_and_pd(v, _mm256_castsi256_pd(keep));
                        }

                        _mm256_storeu_pd(dst0 + pj * MR + i0, v);
                    }
                }
            }

            for (; p < p_max; ++p) {
                double* dst = dst0 + p * MR;
                const size_t k0 = (p <= r) ? 0 : p - r;

                for (size_t i = 0; i < MR; ++i) {
                    if (i < k0)
                        dst[i] = 0.0;
                    else
                        dst[i] = Tmat[(r + i) * k + p];
                }
            }
        }
        else {
            for (size_t p = 0; p < p_max; ++p) {
                double* dst = dst0 + p * MR;
                const size_t k0 = (p <= r) ? 0 : p - r;

                for (size_t i = 0; i < m; ++i) {
                    if (i < k0)
                        dst[i] = 0.0;
                    else
                        dst[i] = Tmat[(r + i) * k + p];
                }

                for (size_t i = m; i < MR; ++i)
                    dst[i] = 0.0;
            }
        }
    }
}

#endif
