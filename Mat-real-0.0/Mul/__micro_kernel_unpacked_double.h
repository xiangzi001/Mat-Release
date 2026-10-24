#if defined(__AVX512__)

static void __micro_kernel_mul_unpacked_full(
    const double* __restrict A, size_t ldA,
    const double* __restrict B, size_t ldB,
    double* __restrict C, size_t ldC,
    size_t k)
{
    __m512d c0_0 = _mm512_loadu_pd(C + 0 * ldC + 0);
    __m512d c0_1 = _mm512_loadu_pd(C + 0 * ldC + 8);
    __m512d c1_0 = _mm512_loadu_pd(C + 1 * ldC + 0);
    __m512d c1_1 = _mm512_loadu_pd(C + 1 * ldC + 8);
    __m512d c2_0 = _mm512_loadu_pd(C + 2 * ldC + 0);
    __m512d c2_1 = _mm512_loadu_pd(C + 2 * ldC + 8);
    __m512d c3_0 = _mm512_loadu_pd(C + 3 * ldC + 0);
    __m512d c3_1 = _mm512_loadu_pd(C + 3 * ldC + 8);
    __m512d c4_0 = _mm512_loadu_pd(C + 4 * ldC + 0);
    __m512d c4_1 = _mm512_loadu_pd(C + 4 * ldC + 8);
    __m512d c5_0 = _mm512_loadu_pd(C + 5 * ldC + 0);
    __m512d c5_1 = _mm512_loadu_pd(C + 5 * ldC + 8);
    __m512d c6_0 = _mm512_loadu_pd(C + 6 * ldC + 0);
    __m512d c6_1 = _mm512_loadu_pd(C + 6 * ldC + 8);
    __m512d c7_0 = _mm512_loadu_pd(C + 7 * ldC + 0);
    __m512d c7_1 = _mm512_loadu_pd(C + 7 * ldC + 8);
    __m512d c8_0 = _mm512_loadu_pd(C + 8 * ldC + 0);
    __m512d c8_1 = _mm512_loadu_pd(C + 8 * ldC + 8);
    __m512d c9_0 = _mm512_loadu_pd(C + 9 * ldC + 0);
    __m512d c9_1 = _mm512_loadu_pd(C + 9 * ldC + 8);
    __m512d c10_0 = _mm512_loadu_pd(C + 10 * ldC + 0);
    __m512d c10_1 = _mm512_loadu_pd(C + 10 * ldC + 8);
    __m512d c11_0 = _mm512_loadu_pd(C + 11 * ldC + 0);
    __m512d c11_1 = _mm512_loadu_pd(C + 11 * ldC + 8);

    for (size_t p = 0; p < k; ++p) {
        const double* ap = A + p * ldA;
        const __m512d a0 = _mm512_loadu_pd(ap + 0);
        const __m512d a1 = _mm512_loadu_pd(ap + 8);

        const double* bp = B + p;
        __m512d b;

        b = _mm512_set1_pd(bp[0 * ldB]);
        c0_0 = _mm512_fmadd_pd(a0, b, c0_0);
        c0_1 = _mm512_fmadd_pd(a1, b, c0_1);
        b = _mm512_set1_pd(bp[1 * ldB]);
        c1_0 = _mm512_fmadd_pd(a0, b, c1_0);
        c1_1 = _mm512_fmadd_pd(a1, b, c1_1);
        b = _mm512_set1_pd(bp[2 * ldB]);
        c2_0 = _mm512_fmadd_pd(a0, b, c2_0);
        c2_1 = _mm512_fmadd_pd(a1, b, c2_1);
        b = _mm512_set1_pd(bp[3 * ldB]);
        c3_0 = _mm512_fmadd_pd(a0, b, c3_0);
        c3_1 = _mm512_fmadd_pd(a1, b, c3_1);
        b = _mm512_set1_pd(bp[4 * ldB]);
        c4_0 = _mm512_fmadd_pd(a0, b, c4_0);
        c4_1 = _mm512_fmadd_pd(a1, b, c4_1);
        b = _mm512_set1_pd(bp[5 * ldB]);
        c5_0 = _mm512_fmadd_pd(a0, b, c5_0);
        c5_1 = _mm512_fmadd_pd(a1, b, c5_1);
        b = _mm512_set1_pd(bp[6 * ldB]);
        c6_0 = _mm512_fmadd_pd(a0, b, c6_0);
        c6_1 = _mm512_fmadd_pd(a1, b, c6_1);
        b = _mm512_set1_pd(bp[7 * ldB]);
        c7_0 = _mm512_fmadd_pd(a0, b, c7_0);
        c7_1 = _mm512_fmadd_pd(a1, b, c7_1);
        b = _mm512_set1_pd(bp[8 * ldB]);
        c8_0 = _mm512_fmadd_pd(a0, b, c8_0);
        c8_1 = _mm512_fmadd_pd(a1, b, c8_1);
        b = _mm512_set1_pd(bp[9 * ldB]);
        c9_0 = _mm512_fmadd_pd(a0, b, c9_0);
        c9_1 = _mm512_fmadd_pd(a1, b, c9_1);
        b = _mm512_set1_pd(bp[10 * ldB]);
        c10_0 = _mm512_fmadd_pd(a0, b, c10_0);
        c10_1 = _mm512_fmadd_pd(a1, b, c10_1);
        b = _mm512_set1_pd(bp[11 * ldB]);
        c11_0 = _mm512_fmadd_pd(a0, b, c11_0);
        c11_1 = _mm512_fmadd_pd(a1, b, c11_1);
    }

    _mm512_storeu_pd(C + 0 * ldC + 0, c0_0);
    _mm512_storeu_pd(C + 0 * ldC + 8, c0_1);
    _mm512_storeu_pd(C + 1 * ldC + 0, c1_0);
    _mm512_storeu_pd(C + 1 * ldC + 8, c1_1);
    _mm512_storeu_pd(C + 2 * ldC + 0, c2_0);
    _mm512_storeu_pd(C + 2 * ldC + 8, c2_1);
    _mm512_storeu_pd(C + 3 * ldC + 0, c3_0);
    _mm512_storeu_pd(C + 3 * ldC + 8, c3_1);
    _mm512_storeu_pd(C + 4 * ldC + 0, c4_0);
    _mm512_storeu_pd(C + 4 * ldC + 8, c4_1);
    _mm512_storeu_pd(C + 5 * ldC + 0, c5_0);
    _mm512_storeu_pd(C + 5 * ldC + 8, c5_1);
    _mm512_storeu_pd(C + 6 * ldC + 0, c6_0);
    _mm512_storeu_pd(C + 6 * ldC + 8, c6_1);
    _mm512_storeu_pd(C + 7 * ldC + 0, c7_0);
    _mm512_storeu_pd(C + 7 * ldC + 8, c7_1);
    _mm512_storeu_pd(C + 8 * ldC + 0, c8_0);
    _mm512_storeu_pd(C + 8 * ldC + 8, c8_1);
    _mm512_storeu_pd(C + 9 * ldC + 0, c9_0);
    _mm512_storeu_pd(C + 9 * ldC + 8, c9_1);
    _mm512_storeu_pd(C + 10 * ldC + 0, c10_0);
    _mm512_storeu_pd(C + 10 * ldC + 8, c10_1);
    _mm512_storeu_pd(C + 11 * ldC + 0, c11_0);
    _mm512_storeu_pd(C + 11 * ldC + 8, c11_1);
}

static void __micro_kernel_mul_unpacked_edge(
    const double* __restrict A, size_t ldA,
    const double* __restrict B, size_t ldB,
    double* __restrict C, size_t ldC,
    size_t k, int mr, int nr)
{
    const __mmask8 m0 = (mr >= 8) ? (__mmask8)0xFF
        : (__mmask8)((1u << mr) - 1u);
    const __mmask8 m1 = (mr >= 16) ? (__mmask8)0xFF
        : (mr > 8) ? (__mmask8)((1u << (mr - 8)) - 1u)
        : (__mmask8)0;

    __m512d c0_0 = _mm512_setzero_pd(), c0_1 = _mm512_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c0_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c0_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c1_0 = _mm512_setzero_pd(), c1_1 = _mm512_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c1_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c1_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c2_0 = _mm512_setzero_pd(), c2_1 = _mm512_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c2_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c2_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c3_0 = _mm512_setzero_pd(), c3_1 = _mm512_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c3_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c3_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c4_0 = _mm512_setzero_pd(), c4_1 = _mm512_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c4_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c4_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c5_0 = _mm512_setzero_pd(), c5_1 = _mm512_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c5_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c5_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c6_0 = _mm512_setzero_pd(), c6_1 = _mm512_setzero_pd();
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        c6_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c6_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c7_0 = _mm512_setzero_pd(), c7_1 = _mm512_setzero_pd();
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        c7_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c7_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c8_0 = _mm512_setzero_pd(), c8_1 = _mm512_setzero_pd();
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        c8_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c8_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c9_0 = _mm512_setzero_pd(), c9_1 = _mm512_setzero_pd();
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        c9_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c9_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c10_0 = _mm512_setzero_pd(), c10_1 = _mm512_setzero_pd();
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        c10_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c10_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c11_0 = _mm512_setzero_pd(), c11_1 = _mm512_setzero_pd();
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        c11_0 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c11_1 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }

    for (size_t p = 0; p < k; ++p) {
        const double* ap = A + p * ldA;
        const __m512d a0 = _mm512_maskz_loadu_pd(m0, ap + 0);
        const __m512d a1 = _mm512_maskz_loadu_pd(m1, ap + 8);

        const double* bp = B + p;
        double bv[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int j = 0; j < nr; ++j) {
            bv[j] = bp[(size_t)j * ldB];
        }

        __m512d b;
        b = _mm512_set1_pd(bv[0]);
        c0_0 = _mm512_fmadd_pd(a0, b, c0_0);
        c0_1 = _mm512_fmadd_pd(a1, b, c0_1);
        b = _mm512_set1_pd(bv[1]);
        c1_0 = _mm512_fmadd_pd(a0, b, c1_0);
        c1_1 = _mm512_fmadd_pd(a1, b, c1_1);
        b = _mm512_set1_pd(bv[2]);
        c2_0 = _mm512_fmadd_pd(a0, b, c2_0);
        c2_1 = _mm512_fmadd_pd(a1, b, c2_1);
        b = _mm512_set1_pd(bv[3]);
        c3_0 = _mm512_fmadd_pd(a0, b, c3_0);
        c3_1 = _mm512_fmadd_pd(a1, b, c3_1);
        b = _mm512_set1_pd(bv[4]);
        c4_0 = _mm512_fmadd_pd(a0, b, c4_0);
        c4_1 = _mm512_fmadd_pd(a1, b, c4_1);
        b = _mm512_set1_pd(bv[5]);
        c5_0 = _mm512_fmadd_pd(a0, b, c5_0);
        c5_1 = _mm512_fmadd_pd(a1, b, c5_1);
        b = _mm512_set1_pd(bv[6]);
        c6_0 = _mm512_fmadd_pd(a0, b, c6_0);
        c6_1 = _mm512_fmadd_pd(a1, b, c6_1);
        b = _mm512_set1_pd(bv[7]);
        c7_0 = _mm512_fmadd_pd(a0, b, c7_0);
        c7_1 = _mm512_fmadd_pd(a1, b, c7_1);
        b = _mm512_set1_pd(bv[8]);
        c8_0 = _mm512_fmadd_pd(a0, b, c8_0);
        c8_1 = _mm512_fmadd_pd(a1, b, c8_1);
        b = _mm512_set1_pd(bv[9]);
        c9_0 = _mm512_fmadd_pd(a0, b, c9_0);
        c9_1 = _mm512_fmadd_pd(a1, b, c9_1);
        b = _mm512_set1_pd(bv[10]);
        c10_0 = _mm512_fmadd_pd(a0, b, c10_0);
        c10_1 = _mm512_fmadd_pd(a1, b, c10_1);
        b = _mm512_set1_pd(bv[11]);
        c11_0 = _mm512_fmadd_pd(a0, b, c11_0);
        c11_1 = _mm512_fmadd_pd(a1, b, c11_1);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c0_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c0_1);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c1_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c1_1);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c2_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c2_1);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c3_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c3_1);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c4_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c4_1);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c5_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c5_1);
    }
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c6_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c6_1);
    }
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c7_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c7_1);
    }
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c8_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c8_1);
    }
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c9_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c9_1);
    }
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c10_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c10_1);
    }
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c11_0);
        _mm512_mask_storeu_pd(C + off + 8, m1, c11_1);
    }
}

#elif defined(__AVX2__)

static void __micro_kernel_mul_unpacked_full(
    const double* __restrict A, size_t ldA,
    const double* __restrict B, size_t ldB,
    double* __restrict C, size_t ldC,
    size_t k)
{
    __m256d c0_0 = _mm256_loadu_pd(C + 0 * ldC + 0);
    __m256d c0_1 = _mm256_loadu_pd(C + 0 * ldC + 4);
    __m256d c1_0 = _mm256_loadu_pd(C + 1 * ldC + 0);
    __m256d c1_1 = _mm256_loadu_pd(C + 1 * ldC + 4);
    __m256d c2_0 = _mm256_loadu_pd(C + 2 * ldC + 0);
    __m256d c2_1 = _mm256_loadu_pd(C + 2 * ldC + 4);
    __m256d c3_0 = _mm256_loadu_pd(C + 3 * ldC + 0);
    __m256d c3_1 = _mm256_loadu_pd(C + 3 * ldC + 4);
    __m256d c4_0 = _mm256_loadu_pd(C + 4 * ldC + 0);
    __m256d c4_1 = _mm256_loadu_pd(C + 4 * ldC + 4);
    __m256d c5_0 = _mm256_loadu_pd(C + 5 * ldC + 0);
    __m256d c5_1 = _mm256_loadu_pd(C + 5 * ldC + 4);

    for (size_t p = 0; p < k; ++p) {
        const double* ap = A + p * ldA;
        const __m256d a_0 = _mm256_loadu_pd(ap + 0);
        const __m256d a_1 = _mm256_loadu_pd(ap + 4);

        const double* bp = B + p;
        __m256d b;

        b = _mm256_set1_pd(bp[0 * ldB]);
        c0_0 = _mm256_fmadd_pd(a_0, b, c0_0);
        c0_1 = _mm256_fmadd_pd(a_1, b, c0_1);
        b = _mm256_set1_pd(bp[1 * ldB]);
        c1_0 = _mm256_fmadd_pd(a_0, b, c1_0);
        c1_1 = _mm256_fmadd_pd(a_1, b, c1_1);
        b = _mm256_set1_pd(bp[2 * ldB]);
        c2_0 = _mm256_fmadd_pd(a_0, b, c2_0);
        c2_1 = _mm256_fmadd_pd(a_1, b, c2_1);
        b = _mm256_set1_pd(bp[3 * ldB]);
        c3_0 = _mm256_fmadd_pd(a_0, b, c3_0);
        c3_1 = _mm256_fmadd_pd(a_1, b, c3_1);
        b = _mm256_set1_pd(bp[4 * ldB]);
        c4_0 = _mm256_fmadd_pd(a_0, b, c4_0);
        c4_1 = _mm256_fmadd_pd(a_1, b, c4_1);
        b = _mm256_set1_pd(bp[5 * ldB]);
        c5_0 = _mm256_fmadd_pd(a_0, b, c5_0);
        c5_1 = _mm256_fmadd_pd(a_1, b, c5_1);
    }

    _mm256_storeu_pd(C + 0 * ldC + 0, c0_0);
    _mm256_storeu_pd(C + 0 * ldC + 4, c0_1);
    _mm256_storeu_pd(C + 1 * ldC + 0, c1_0);
    _mm256_storeu_pd(C + 1 * ldC + 4, c1_1);
    _mm256_storeu_pd(C + 2 * ldC + 0, c2_0);
    _mm256_storeu_pd(C + 2 * ldC + 4, c2_1);
    _mm256_storeu_pd(C + 3 * ldC + 0, c3_0);
    _mm256_storeu_pd(C + 3 * ldC + 4, c3_1);
    _mm256_storeu_pd(C + 4 * ldC + 0, c4_0);
    _mm256_storeu_pd(C + 4 * ldC + 4, c4_1);
    _mm256_storeu_pd(C + 5 * ldC + 0, c5_0);
    _mm256_storeu_pd(C + 5 * ldC + 4, c5_1);
}

static void __micro_kernel_mul_unpacked_edge(
    const double* __restrict A, size_t ldA,
    const double* __restrict B, size_t ldB,
    double* __restrict C, size_t ldC,
    size_t k, int mr, int nr)
{
    const __m256i m0 = _mm256_set_epi64x(
        mr > 3 ? -1LL : 0LL, mr > 2 ? -1LL : 0LL,
        mr > 1 ? -1LL : 0LL, mr > 0 ? -1LL : 0LL);
    const __m256i m1 = _mm256_set_epi64x(
        mr > 7 ? -1LL : 0LL, mr > 6 ? -1LL : 0LL,
        mr > 5 ? -1LL : 0LL, mr > 4 ? -1LL : 0LL);

    __m256d c0_0 = _mm256_setzero_pd(), c0_1 = _mm256_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c0_0 = _mm256_maskload_pd(C + off + 0, m0);
        c0_1 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c1_0 = _mm256_setzero_pd(), c1_1 = _mm256_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c1_0 = _mm256_maskload_pd(C + off + 0, m0);
        c1_1 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c2_0 = _mm256_setzero_pd(), c2_1 = _mm256_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c2_0 = _mm256_maskload_pd(C + off + 0, m0);
        c2_1 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c3_0 = _mm256_setzero_pd(), c3_1 = _mm256_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c3_0 = _mm256_maskload_pd(C + off + 0, m0);
        c3_1 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c4_0 = _mm256_setzero_pd(), c4_1 = _mm256_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c4_0 = _mm256_maskload_pd(C + off + 0, m0);
        c4_1 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c5_0 = _mm256_setzero_pd(), c5_1 = _mm256_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c5_0 = _mm256_maskload_pd(C + off + 0, m0);
        c5_1 = _mm256_maskload_pd(C + off + 4, m1);
    }

    for (size_t p = 0; p < k; ++p) {
        const double* ap = A + p * ldA;
        const __m256d a0 = _mm256_maskload_pd(ap + 0, m0);
        const __m256d a1 = _mm256_maskload_pd(ap + 4, m1);

        const double* bp = B + p;
        double bv[6] = { 0, 0, 0, 0, 0, 0 };
        for (int j = 0; j < nr; ++j) {
            bv[j] = bp[(size_t)j * ldB];
        }

        __m256d b;
        b = _mm256_set1_pd(bv[0]);
        c0_0 = _mm256_fmadd_pd(a0, b, c0_0);
        c0_1 = _mm256_fmadd_pd(a1, b, c0_1);
        b = _mm256_set1_pd(bv[1]);
        c1_0 = _mm256_fmadd_pd(a0, b, c1_0);
        c1_1 = _mm256_fmadd_pd(a1, b, c1_1);
        b = _mm256_set1_pd(bv[2]);
        c2_0 = _mm256_fmadd_pd(a0, b, c2_0);
        c2_1 = _mm256_fmadd_pd(a1, b, c2_1);
        b = _mm256_set1_pd(bv[3]);
        c3_0 = _mm256_fmadd_pd(a0, b, c3_0);
        c3_1 = _mm256_fmadd_pd(a1, b, c3_1);
        b = _mm256_set1_pd(bv[4]);
        c4_0 = _mm256_fmadd_pd(a0, b, c4_0);
        c4_1 = _mm256_fmadd_pd(a1, b, c4_1);
        b = _mm256_set1_pd(bv[5]);
        c5_0 = _mm256_fmadd_pd(a0, b, c5_0);
        c5_1 = _mm256_fmadd_pd(a1, b, c5_1);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c0_0);
        _mm256_maskstore_pd(C + off + 4, m1, c0_1);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c1_0);
        _mm256_maskstore_pd(C + off + 4, m1, c1_1);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c2_0);
        _mm256_maskstore_pd(C + off + 4, m1, c2_1);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c3_0);
        _mm256_maskstore_pd(C + off + 4, m1, c3_1);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c4_0);
        _mm256_maskstore_pd(C + off + 4, m1, c4_1);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c5_0);
        _mm256_maskstore_pd(C + off + 4, m1, c5_1);
    }
}

#endif