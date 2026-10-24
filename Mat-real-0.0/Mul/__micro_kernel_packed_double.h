
#ifdef __AVX512__

template<>
constexpr int __micro_kernel_nr<double, double>() {
    return 12;
}

template<>
constexpr int __micro_kernel_mr<double, double>() {
    return 16;
}

static void __micro_kernel_mul_packed_full(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k)
{
    __m512d c00 = _mm512_loadu_pd(C + 0 * ldC + 0);
    __m512d c01 = _mm512_loadu_pd(C + 0 * ldC + 8);
    __m512d c10 = _mm512_loadu_pd(C + 1 * ldC + 0);
    __m512d c11 = _mm512_loadu_pd(C + 1 * ldC + 8);
    __m512d c20 = _mm512_loadu_pd(C + 2 * ldC + 0);
    __m512d c21 = _mm512_loadu_pd(C + 2 * ldC + 8);
    __m512d c30 = _mm512_loadu_pd(C + 3 * ldC + 0);
    __m512d c31 = _mm512_loadu_pd(C + 3 * ldC + 8);
    __m512d c40 = _mm512_loadu_pd(C + 4 * ldC + 0);
    __m512d c41 = _mm512_loadu_pd(C + 4 * ldC + 8);
    __m512d c50 = _mm512_loadu_pd(C + 5 * ldC + 0);
    __m512d c51 = _mm512_loadu_pd(C + 5 * ldC + 8);
    __m512d c60 = _mm512_loadu_pd(C + 6 * ldC + 0);
    __m512d c61 = _mm512_loadu_pd(C + 6 * ldC + 8);
    __m512d c70 = _mm512_loadu_pd(C + 7 * ldC + 0);
    __m512d c71 = _mm512_loadu_pd(C + 7 * ldC + 8);
    __m512d c80 = _mm512_loadu_pd(C + 8 * ldC + 0);
    __m512d c81 = _mm512_loadu_pd(C + 8 * ldC + 8);
    __m512d c90 = _mm512_loadu_pd(C + 9 * ldC + 0);
    __m512d c91 = _mm512_loadu_pd(C + 9 * ldC + 8);
    __m512d c100 = _mm512_loadu_pd(C + 10 * ldC + 0);
    __m512d c101 = _mm512_loadu_pd(C + 10 * ldC + 8);
    __m512d c110 = _mm512_loadu_pd(C + 11 * ldC + 0);
    __m512d c111 = _mm512_loadu_pd(C + 11 * ldC + 8);

    for (size_t p = 0; p < k; ++p) {
        const __m512d a0 = _mm512_load_pd(Ap + p * 16);
        const __m512d a1 = _mm512_load_pd(Ap + p * 16 + 8);

        __m512d b;
        b = _mm512_set1_pd(Bp[p * 12 + 0]);
        c00 = _mm512_fmadd_pd(a0, b, c00);
        c01 = _mm512_fmadd_pd(a1, b, c01);
        b = _mm512_set1_pd(Bp[p * 12 + 1]);
        c10 = _mm512_fmadd_pd(a0, b, c10);
        c11 = _mm512_fmadd_pd(a1, b, c11);
        b = _mm512_set1_pd(Bp[p * 12 + 2]);
        c20 = _mm512_fmadd_pd(a0, b, c20);
        c21 = _mm512_fmadd_pd(a1, b, c21);
        b = _mm512_set1_pd(Bp[p * 12 + 3]);
        c30 = _mm512_fmadd_pd(a0, b, c30);
        c31 = _mm512_fmadd_pd(a1, b, c31);
        b = _mm512_set1_pd(Bp[p * 12 + 4]);
        c40 = _mm512_fmadd_pd(a0, b, c40);
        c41 = _mm512_fmadd_pd(a1, b, c41);
        b = _mm512_set1_pd(Bp[p * 12 + 5]);
        c50 = _mm512_fmadd_pd(a0, b, c50);
        c51 = _mm512_fmadd_pd(a1, b, c51);
        b = _mm512_set1_pd(Bp[p * 12 + 6]);
        c60 = _mm512_fmadd_pd(a0, b, c60);
        c61 = _mm512_fmadd_pd(a1, b, c61);
        b = _mm512_set1_pd(Bp[p * 12 + 7]);
        c70 = _mm512_fmadd_pd(a0, b, c70);
        c71 = _mm512_fmadd_pd(a1, b, c71);
        b = _mm512_set1_pd(Bp[p * 12 + 8]);
        c80 = _mm512_fmadd_pd(a0, b, c80);
        c81 = _mm512_fmadd_pd(a1, b, c81);
        b = _mm512_set1_pd(Bp[p * 12 + 9]);
        c90 = _mm512_fmadd_pd(a0, b, c90);
        c91 = _mm512_fmadd_pd(a1, b, c91);
        b = _mm512_set1_pd(Bp[p * 12 + 10]);
        c100 = _mm512_fmadd_pd(a0, b, c100);
        c101 = _mm512_fmadd_pd(a1, b, c101);
        b = _mm512_set1_pd(Bp[p * 12 + 11]);
        c110 = _mm512_fmadd_pd(a0, b, c110);
        c111 = _mm512_fmadd_pd(a1, b, c111);
    }

    _mm512_storeu_pd(C + 0 * ldC + 0, c00);
    _mm512_storeu_pd(C + 0 * ldC + 8, c01);
    _mm512_storeu_pd(C + 1 * ldC + 0, c10);
    _mm512_storeu_pd(C + 1 * ldC + 8, c11);
    _mm512_storeu_pd(C + 2 * ldC + 0, c20);
    _mm512_storeu_pd(C + 2 * ldC + 8, c21);
    _mm512_storeu_pd(C + 3 * ldC + 0, c30);
    _mm512_storeu_pd(C + 3 * ldC + 8, c31);
    _mm512_storeu_pd(C + 4 * ldC + 0, c40);
    _mm512_storeu_pd(C + 4 * ldC + 8, c41);
    _mm512_storeu_pd(C + 5 * ldC + 0, c50);
    _mm512_storeu_pd(C + 5 * ldC + 8, c51);
    _mm512_storeu_pd(C + 6 * ldC + 0, c60);
    _mm512_storeu_pd(C + 6 * ldC + 8, c61);
    _mm512_storeu_pd(C + 7 * ldC + 0, c70);
    _mm512_storeu_pd(C + 7 * ldC + 8, c71);
    _mm512_storeu_pd(C + 8 * ldC + 0, c80);
    _mm512_storeu_pd(C + 8 * ldC + 8, c81);
    _mm512_storeu_pd(C + 9 * ldC + 0, c90);
    _mm512_storeu_pd(C + 9 * ldC + 8, c91);
    _mm512_storeu_pd(C + 10 * ldC + 0, c100);
    _mm512_storeu_pd(C + 10 * ldC + 8, c101);
    _mm512_storeu_pd(C + 11 * ldC + 0, c110);
    _mm512_storeu_pd(C + 11 * ldC + 8, c111);
}

static void __micro_kernel_mul_packed_edge(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k, int mr, int nr)
{
    const __mmask8 m0 = (mr >= 8) ? (__mmask8)0xFF
        : (__mmask8)((1u << mr) - 1u);
    const __mmask8 m1 = (mr >= 16) ? (__mmask8)0xFF
        : (mr > 8) ? (__mmask8)((1u << (mr - 8)) - 1u)
        : (__mmask8)0;

    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c00 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c01 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c10 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c11 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c20 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c21 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c30 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c31 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c40 = _mm512_setzero_pd(), c41 = _mm512_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c40 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c41 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c50 = _mm512_setzero_pd(), c51 = _mm512_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c50 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c51 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c60 = _mm512_setzero_pd(), c61 = _mm512_setzero_pd();
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        c60 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c61 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c70 = _mm512_setzero_pd(), c71 = _mm512_setzero_pd();
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        c70 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c71 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c80 = _mm512_setzero_pd(), c81 = _mm512_setzero_pd();
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        c80 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c81 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c90 = _mm512_setzero_pd(), c91 = _mm512_setzero_pd();
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        c90 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c91 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c100 = _mm512_setzero_pd(), c101 = _mm512_setzero_pd();
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        c100 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c101 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c110 = _mm512_setzero_pd(), c111 = _mm512_setzero_pd();
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        c110 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c111 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }

    for (size_t p = 0; p < k; ++p) {
        const __m512d a0 = _mm512_load_pd(Ap + p * 16);
        const __m512d a1 = _mm512_load_pd(Ap + p * 16 + 8);

        __m512d b;
        b = _mm512_set1_pd(Bp[p * 12 + 0]);
        c00 = _mm512_fmadd_pd(a0, b, c00);
        c01 = _mm512_fmadd_pd(a1, b, c01);
        b = _mm512_set1_pd(Bp[p * 12 + 1]);
        c10 = _mm512_fmadd_pd(a0, b, c10);
        c11 = _mm512_fmadd_pd(a1, b, c11);
        b = _mm512_set1_pd(Bp[p * 12 + 2]);
        c20 = _mm512_fmadd_pd(a0, b, c20);
        c21 = _mm512_fmadd_pd(a1, b, c21);
        b = _mm512_set1_pd(Bp[p * 12 + 3]);
        c30 = _mm512_fmadd_pd(a0, b, c30);
        c31 = _mm512_fmadd_pd(a1, b, c31);
        b = _mm512_set1_pd(Bp[p * 12 + 4]);
        c40 = _mm512_fmadd_pd(a0, b, c40);
        c41 = _mm512_fmadd_pd(a1, b, c41);
        b = _mm512_set1_pd(Bp[p * 12 + 5]);
        c50 = _mm512_fmadd_pd(a0, b, c50);
        c51 = _mm512_fmadd_pd(a1, b, c51);
        b = _mm512_set1_pd(Bp[p * 12 + 6]);
        c60 = _mm512_fmadd_pd(a0, b, c60);
        c61 = _mm512_fmadd_pd(a1, b, c61);
        b = _mm512_set1_pd(Bp[p * 12 + 7]);
        c70 = _mm512_fmadd_pd(a0, b, c70);
        c71 = _mm512_fmadd_pd(a1, b, c71);
        b = _mm512_set1_pd(Bp[p * 12 + 8]);
        c80 = _mm512_fmadd_pd(a0, b, c80);
        c81 = _mm512_fmadd_pd(a1, b, c81);
        b = _mm512_set1_pd(Bp[p * 12 + 9]);
        c90 = _mm512_fmadd_pd(a0, b, c90);
        c91 = _mm512_fmadd_pd(a1, b, c91);
        b = _mm512_set1_pd(Bp[p * 12 + 10]);
        c100 = _mm512_fmadd_pd(a0, b, c100);
        c101 = _mm512_fmadd_pd(a1, b, c101);
        b = _mm512_set1_pd(Bp[p * 12 + 11]);
        c110 = _mm512_fmadd_pd(a0, b, c110);
        c111 = _mm512_fmadd_pd(a1, b, c111);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c00);
        _mm512_mask_storeu_pd(C + off + 8, m1, c01);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c10);
        _mm512_mask_storeu_pd(C + off + 8, m1, c11);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c20);
        _mm512_mask_storeu_pd(C + off + 8, m1, c21);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c30);
        _mm512_mask_storeu_pd(C + off + 8, m1, c31);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c40);
        _mm512_mask_storeu_pd(C + off + 8, m1, c41);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c50);
        _mm512_mask_storeu_pd(C + off + 8, m1, c51);
    }
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c60);
        _mm512_mask_storeu_pd(C + off + 8, m1, c61);
    }
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c70);
        _mm512_mask_storeu_pd(C + off + 8, m1, c71);
    }
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c80);
        _mm512_mask_storeu_pd(C + off + 8, m1, c81);
    }
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c90);
        _mm512_mask_storeu_pd(C + off + 8, m1, c91);
    }
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c100);
        _mm512_mask_storeu_pd(C + off + 8, m1, c101);
    }
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c110);
        _mm512_mask_storeu_pd(C + off + 8, m1, c111);
    }
}


static void __micro_kernel_nmul_packed_full(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k)
{
    __m512d c00 = _mm512_loadu_pd(C + 0 * ldC + 0);
    __m512d c01 = _mm512_loadu_pd(C + 0 * ldC + 8);
    __m512d c10 = _mm512_loadu_pd(C + 1 * ldC + 0);
    __m512d c11 = _mm512_loadu_pd(C + 1 * ldC + 8);
    __m512d c20 = _mm512_loadu_pd(C + 2 * ldC + 0);
    __m512d c21 = _mm512_loadu_pd(C + 2 * ldC + 8);
    __m512d c30 = _mm512_loadu_pd(C + 3 * ldC + 0);
    __m512d c31 = _mm512_loadu_pd(C + 3 * ldC + 8);
    __m512d c40 = _mm512_loadu_pd(C + 4 * ldC + 0);
    __m512d c41 = _mm512_loadu_pd(C + 4 * ldC + 8);
    __m512d c50 = _mm512_loadu_pd(C + 5 * ldC + 0);
    __m512d c51 = _mm512_loadu_pd(C + 5 * ldC + 8);
    __m512d c60 = _mm512_loadu_pd(C + 6 * ldC + 0);
    __m512d c61 = _mm512_loadu_pd(C + 6 * ldC + 8);
    __m512d c70 = _mm512_loadu_pd(C + 7 * ldC + 0);
    __m512d c71 = _mm512_loadu_pd(C + 7 * ldC + 8);
    __m512d c80 = _mm512_loadu_pd(C + 8 * ldC + 0);
    __m512d c81 = _mm512_loadu_pd(C + 8 * ldC + 8);
    __m512d c90 = _mm512_loadu_pd(C + 9 * ldC + 0);
    __m512d c91 = _mm512_loadu_pd(C + 9 * ldC + 8);
    __m512d c100 = _mm512_loadu_pd(C + 10 * ldC + 0);
    __m512d c101 = _mm512_loadu_pd(C + 10 * ldC + 8);
    __m512d c110 = _mm512_loadu_pd(C + 11 * ldC + 0);
    __m512d c111 = _mm512_loadu_pd(C + 11 * ldC + 8);

    for (size_t p = 0; p < k; ++p) {
        const __m512d a0 = _mm512_load_pd(Ap + p * 16);
        const __m512d a1 = _mm512_load_pd(Ap + p * 16 + 8);

        __m512d b;
        b = _mm512_set1_pd(Bp[p * 12 + 0]);
        c00 = _mm512_fnmadd_pd(a0, b, c00);
        c01 = _mm512_fnmadd_pd(a1, b, c01);
        b = _mm512_set1_pd(Bp[p * 12 + 1]);
        c10 = _mm512_fnmadd_pd(a0, b, c10);
        c11 = _mm512_fnmadd_pd(a1, b, c11);
        b = _mm512_set1_pd(Bp[p * 12 + 2]);
        c20 = _mm512_fnmadd_pd(a0, b, c20);
        c21 = _mm512_fnmadd_pd(a1, b, c21);
        b = _mm512_set1_pd(Bp[p * 12 + 3]);
        c30 = _mm512_fnmadd_pd(a0, b, c30);
        c31 = _mm512_fnmadd_pd(a1, b, c31);
        b = _mm512_set1_pd(Bp[p * 12 + 4]);
        c40 = _mm512_fnmadd_pd(a0, b, c40);
        c41 = _mm512_fnmadd_pd(a1, b, c41);
        b = _mm512_set1_pd(Bp[p * 12 + 5]);
        c50 = _mm512_fnmadd_pd(a0, b, c50);
        c51 = _mm512_fnmadd_pd(a1, b, c51);
        b = _mm512_set1_pd(Bp[p * 12 + 6]);
        c60 = _mm512_fnmadd_pd(a0, b, c60);
        c61 = _mm512_fnmadd_pd(a1, b, c61);
        b = _mm512_set1_pd(Bp[p * 12 + 7]);
        c70 = _mm512_fnmadd_pd(a0, b, c70);
        c71 = _mm512_fnmadd_pd(a1, b, c71);
        b = _mm512_set1_pd(Bp[p * 12 + 8]);
        c80 = _mm512_fnmadd_pd(a0, b, c80);
        c81 = _mm512_fnmadd_pd(a1, b, c81);
        b = _mm512_set1_pd(Bp[p * 12 + 9]);
        c90 = _mm512_fnmadd_pd(a0, b, c90);
        c91 = _mm512_fnmadd_pd(a1, b, c91);
        b = _mm512_set1_pd(Bp[p * 12 + 10]);
        c100 = _mm512_fnmadd_pd(a0, b, c100);
        c101 = _mm512_fnmadd_pd(a1, b, c101);
        b = _mm512_set1_pd(Bp[p * 12 + 11]);
        c110 = _mm512_fnmadd_pd(a0, b, c110);
        c111 = _mm512_fnmadd_pd(a1, b, c111);
    }

    _mm512_storeu_pd(C + 0 * ldC + 0, c00);
    _mm512_storeu_pd(C + 0 * ldC + 8, c01);
    _mm512_storeu_pd(C + 1 * ldC + 0, c10);
    _mm512_storeu_pd(C + 1 * ldC + 8, c11);
    _mm512_storeu_pd(C + 2 * ldC + 0, c20);
    _mm512_storeu_pd(C + 2 * ldC + 8, c21);
    _mm512_storeu_pd(C + 3 * ldC + 0, c30);
    _mm512_storeu_pd(C + 3 * ldC + 8, c31);
    _mm512_storeu_pd(C + 4 * ldC + 0, c40);
    _mm512_storeu_pd(C + 4 * ldC + 8, c41);
    _mm512_storeu_pd(C + 5 * ldC + 0, c50);
    _mm512_storeu_pd(C + 5 * ldC + 8, c51);
    _mm512_storeu_pd(C + 6 * ldC + 0, c60);
    _mm512_storeu_pd(C + 6 * ldC + 8, c61);
    _mm512_storeu_pd(C + 7 * ldC + 0, c70);
    _mm512_storeu_pd(C + 7 * ldC + 8, c71);
    _mm512_storeu_pd(C + 8 * ldC + 0, c80);
    _mm512_storeu_pd(C + 8 * ldC + 8, c81);
    _mm512_storeu_pd(C + 9 * ldC + 0, c90);
    _mm512_storeu_pd(C + 9 * ldC + 8, c91);
    _mm512_storeu_pd(C + 10 * ldC + 0, c100);
    _mm512_storeu_pd(C + 10 * ldC + 8, c101);
    _mm512_storeu_pd(C + 11 * ldC + 0, c110);
    _mm512_storeu_pd(C + 11 * ldC + 8, c111);
}

static void __micro_kernel_nmul_packed_edge(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k, int mr, int nr)
{
    const __mmask8 m0 = (mr >= 8) ? (__mmask8)0xFF
        : (__mmask8)((1u << mr) - 1u);
    const __mmask8 m1 = (mr >= 16) ? (__mmask8)0xFF
        : (mr > 8) ? (__mmask8)((1u << (mr - 8)) - 1u)
        : (__mmask8)0;

    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c00 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c01 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c10 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c11 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c20 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c21 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c30 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c31 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c40 = _mm512_setzero_pd(), c41 = _mm512_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c40 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c41 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c50 = _mm512_setzero_pd(), c51 = _mm512_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c50 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c51 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c60 = _mm512_setzero_pd(), c61 = _mm512_setzero_pd();
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        c60 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c61 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c70 = _mm512_setzero_pd(), c71 = _mm512_setzero_pd();
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        c70 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c71 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c80 = _mm512_setzero_pd(), c81 = _mm512_setzero_pd();
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        c80 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c81 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c90 = _mm512_setzero_pd(), c91 = _mm512_setzero_pd();
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        c90 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c91 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c100 = _mm512_setzero_pd(), c101 = _mm512_setzero_pd();
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        c100 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c101 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c110 = _mm512_setzero_pd(), c111 = _mm512_setzero_pd();
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        c110 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c111 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }

    for (size_t p = 0; p < k; ++p) {
        const __m512d a0 = _mm512_load_pd(Ap + p * 16);
        const __m512d a1 = _mm512_load_pd(Ap + p * 16 + 8);

        __m512d b;
        b = _mm512_set1_pd(Bp[p * 12 + 0]);
        c00 = _mm512_fnmadd_pd(a0, b, c00);
        c01 = _mm512_fnmadd_pd(a1, b, c01);
        b = _mm512_set1_pd(Bp[p * 12 + 1]);
        c10 = _mm512_fnmadd_pd(a0, b, c10);
        c11 = _mm512_fnmadd_pd(a1, b, c11);
        b = _mm512_set1_pd(Bp[p * 12 + 2]);
        c20 = _mm512_fnmadd_pd(a0, b, c20);
        c21 = _mm512_fnmadd_pd(a1, b, c21);
        b = _mm512_set1_pd(Bp[p * 12 + 3]);
        c30 = _mm512_fnmadd_pd(a0, b, c30);
        c31 = _mm512_fnmadd_pd(a1, b, c31);
        b = _mm512_set1_pd(Bp[p * 12 + 4]);
        c40 = _mm512_fnmadd_pd(a0, b, c40);
        c41 = _mm512_fnmadd_pd(a1, b, c41);
        b = _mm512_set1_pd(Bp[p * 12 + 5]);
        c50 = _mm512_fnmadd_pd(a0, b, c50);
        c51 = _mm512_fnmadd_pd(a1, b, c51);
        b = _mm512_set1_pd(Bp[p * 12 + 6]);
        c60 = _mm512_fnmadd_pd(a0, b, c60);
        c61 = _mm512_fnmadd_pd(a1, b, c61);
        b = _mm512_set1_pd(Bp[p * 12 + 7]);
        c70 = _mm512_fnmadd_pd(a0, b, c70);
        c71 = _mm512_fnmadd_pd(a1, b, c71);
        b = _mm512_set1_pd(Bp[p * 12 + 8]);
        c80 = _mm512_fnmadd_pd(a0, b, c80);
        c81 = _mm512_fnmadd_pd(a1, b, c81);
        b = _mm512_set1_pd(Bp[p * 12 + 9]);
        c90 = _mm512_fnmadd_pd(a0, b, c90);
        c91 = _mm512_fnmadd_pd(a1, b, c91);
        b = _mm512_set1_pd(Bp[p * 12 + 10]);
        c100 = _mm512_fnmadd_pd(a0, b, c100);
        c101 = _mm512_fnmadd_pd(a1, b, c101);
        b = _mm512_set1_pd(Bp[p * 12 + 11]);
        c110 = _mm512_fnmadd_pd(a0, b, c110);
        c111 = _mm512_fnmadd_pd(a1, b, c111);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c00);
        _mm512_mask_storeu_pd(C + off + 8, m1, c01);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c10);
        _mm512_mask_storeu_pd(C + off + 8, m1, c11);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c20);
        _mm512_mask_storeu_pd(C + off + 8, m1, c21);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c30);
        _mm512_mask_storeu_pd(C + off + 8, m1, c31);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c40);
        _mm512_mask_storeu_pd(C + off + 8, m1, c41);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c50);
        _mm512_mask_storeu_pd(C + off + 8, m1, c51);
    }
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c60);
        _mm512_mask_storeu_pd(C + off + 8, m1, c61);
    }
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c70);
        _mm512_mask_storeu_pd(C + off + 8, m1, c71);
    }
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c80);
        _mm512_mask_storeu_pd(C + off + 8, m1, c81);
    }
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c90);
        _mm512_mask_storeu_pd(C + off + 8, m1, c91);
    }
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c100);
        _mm512_mask_storeu_pd(C + off + 8, m1, c101);
    }
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c110);
        _mm512_mask_storeu_pd(C + off + 8, m1, c111);
    }
}

#elif defined(__AVX2__)

template<>
constexpr int __micro_kernel_nr<double, double>() {
    return 6;
}

template<>
constexpr int __micro_kernel_mr<double, double>() {
    return 8;
}

static void __micro_kernel_mul_packed_full(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k)
{
    __m256d c00 = _mm256_loadu_pd(C + 0 * ldC + 0);
    __m256d c01 = _mm256_loadu_pd(C + 0 * ldC + 4);
    __m256d c10 = _mm256_loadu_pd(C + 1 * ldC + 0);
    __m256d c11 = _mm256_loadu_pd(C + 1 * ldC + 4);
    __m256d c20 = _mm256_loadu_pd(C + 2 * ldC + 0);
    __m256d c21 = _mm256_loadu_pd(C + 2 * ldC + 4);
    __m256d c30 = _mm256_loadu_pd(C + 3 * ldC + 0);
    __m256d c31 = _mm256_loadu_pd(C + 3 * ldC + 4);
    __m256d c40 = _mm256_loadu_pd(C + 4 * ldC + 0);
    __m256d c41 = _mm256_loadu_pd(C + 4 * ldC + 4);
    __m256d c50 = _mm256_loadu_pd(C + 5 * ldC + 0);
    __m256d c51 = _mm256_loadu_pd(C + 5 * ldC + 4);

    for (size_t p = 0; p < k; ++p) {
        const __m256d a0 = _mm256_load_pd(Ap + p * 8);
        const __m256d a1 = _mm256_load_pd(Ap + p * 8 + 4);

        __m256d b;
        b = _mm256_set1_pd(Bp[p * 6 + 0]);
        c00 = _mm256_fmadd_pd(a0, b, c00);
        c01 = _mm256_fmadd_pd(a1, b, c01);
        b = _mm256_set1_pd(Bp[p * 6 + 1]);
        c10 = _mm256_fmadd_pd(a0, b, c10);
        c11 = _mm256_fmadd_pd(a1, b, c11);
        b = _mm256_set1_pd(Bp[p * 6 + 2]);
        c20 = _mm256_fmadd_pd(a0, b, c20);
        c21 = _mm256_fmadd_pd(a1, b, c21);
        b = _mm256_set1_pd(Bp[p * 6 + 3]);
        c30 = _mm256_fmadd_pd(a0, b, c30);
        c31 = _mm256_fmadd_pd(a1, b, c31);
        b = _mm256_set1_pd(Bp[p * 6 + 4]);
        c40 = _mm256_fmadd_pd(a0, b, c40);
        c41 = _mm256_fmadd_pd(a1, b, c41);
        b = _mm256_set1_pd(Bp[p * 6 + 5]);
        c50 = _mm256_fmadd_pd(a0, b, c50);
        c51 = _mm256_fmadd_pd(a1, b, c51);
    }

    _mm256_storeu_pd(C + 0 * ldC + 0, c00);
    _mm256_storeu_pd(C + 0 * ldC + 4, c01);
    _mm256_storeu_pd(C + 1 * ldC + 0, c10);
    _mm256_storeu_pd(C + 1 * ldC + 4, c11);
    _mm256_storeu_pd(C + 2 * ldC + 0, c20);
    _mm256_storeu_pd(C + 2 * ldC + 4, c21);
    _mm256_storeu_pd(C + 3 * ldC + 0, c30);
    _mm256_storeu_pd(C + 3 * ldC + 4, c31);
    _mm256_storeu_pd(C + 4 * ldC + 0, c40);
    _mm256_storeu_pd(C + 4 * ldC + 4, c41);
    _mm256_storeu_pd(C + 5 * ldC + 0, c50);
    _mm256_storeu_pd(C + 5 * ldC + 4, c51);
}

static void __micro_kernel_mul_packed_edge(
    const double* __restrict Ap,
    const double* __restrict Bp,
    double* __restrict C, size_t ldC, size_t k,
    int mr, int nr)
{
    const __m256i m0 = _mm256_set_epi64x(
        mr > 3 ? -1LL : 0LL, mr > 2 ? -1LL : 0LL,
        mr > 1 ? -1LL : 0LL, mr > 0 ? -1LL : 0LL);
    const __m256i m1 = _mm256_set_epi64x(
        mr > 7 ? -1LL : 0LL, mr > 6 ? -1LL : 0LL,
        mr > 5 ? -1LL : 0LL, mr > 4 ? -1LL : 0LL);

    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c00 = _mm256_maskload_pd(C + off + 0, m0);
        c01 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c10 = _mm256_maskload_pd(C + off + 0, m0);
        c11 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c20 = _mm256_maskload_pd(C + off + 0, m0);
        c21 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c30 = _mm256_maskload_pd(C + off + 0, m0);
        c31 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c40 = _mm256_maskload_pd(C + off + 0, m0);
        c41 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c50 = _mm256_maskload_pd(C + off + 0, m0);
        c51 = _mm256_maskload_pd(C + off + 4, m1);
    }

    for (size_t p = 0; p < k; ++p) {
        const __m256d a0 = _mm256_load_pd(Ap + p * 8);
        const __m256d a1 = _mm256_load_pd(Ap + p * 8 + 4);

        __m256d b;
        b = _mm256_set1_pd(Bp[p * 6 + 0]);
        c00 = _mm256_fmadd_pd(a0, b, c00);
        c01 = _mm256_fmadd_pd(a1, b, c01);
        b = _mm256_set1_pd(Bp[p * 6 + 1]);
        c10 = _mm256_fmadd_pd(a0, b, c10);
        c11 = _mm256_fmadd_pd(a1, b, c11);
        b = _mm256_set1_pd(Bp[p * 6 + 2]);
        c20 = _mm256_fmadd_pd(a0, b, c20);
        c21 = _mm256_fmadd_pd(a1, b, c21);
        b = _mm256_set1_pd(Bp[p * 6 + 3]);
        c30 = _mm256_fmadd_pd(a0, b, c30);
        c31 = _mm256_fmadd_pd(a1, b, c31);
        b = _mm256_set1_pd(Bp[p * 6 + 4]);
        c40 = _mm256_fmadd_pd(a0, b, c40);
        c41 = _mm256_fmadd_pd(a1, b, c41);
        b = _mm256_set1_pd(Bp[p * 6 + 5]);
        c50 = _mm256_fmadd_pd(a0, b, c50);
        c51 = _mm256_fmadd_pd(a1, b, c51);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c00);
        _mm256_maskstore_pd(C + off + 4, m1, c01);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c10);
        _mm256_maskstore_pd(C + off + 4, m1, c11);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c20);
        _mm256_maskstore_pd(C + off + 4, m1, c21);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c30);
        _mm256_maskstore_pd(C + off + 4, m1, c31);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c40);
        _mm256_maskstore_pd(C + off + 4, m1, c41);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c50);
        _mm256_maskstore_pd(C + off + 4, m1, c51);
    }
}


static void __micro_kernel_nmul_packed_full(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k)
{
    __m256d c00 = _mm256_loadu_pd(C + 0 * ldC + 0);
    __m256d c01 = _mm256_loadu_pd(C + 0 * ldC + 4);
    __m256d c10 = _mm256_loadu_pd(C + 1 * ldC + 0);
    __m256d c11 = _mm256_loadu_pd(C + 1 * ldC + 4);
    __m256d c20 = _mm256_loadu_pd(C + 2 * ldC + 0);
    __m256d c21 = _mm256_loadu_pd(C + 2 * ldC + 4);
    __m256d c30 = _mm256_loadu_pd(C + 3 * ldC + 0);
    __m256d c31 = _mm256_loadu_pd(C + 3 * ldC + 4);
    __m256d c40 = _mm256_loadu_pd(C + 4 * ldC + 0);
    __m256d c41 = _mm256_loadu_pd(C + 4 * ldC + 4);
    __m256d c50 = _mm256_loadu_pd(C + 5 * ldC + 0);
    __m256d c51 = _mm256_loadu_pd(C + 5 * ldC + 4);

    for (size_t p = 0; p < k; ++p) {
        const __m256d a0 = _mm256_load_pd(Ap + p * 8);
        const __m256d a1 = _mm256_load_pd(Ap + p * 8 + 4);

        __m256d b;
        b = _mm256_set1_pd(Bp[p * 6 + 0]);
        c00 = _mm256_fnmadd_pd(a0, b, c00);
        c01 = _mm256_fnmadd_pd(a1, b, c01);
        b = _mm256_set1_pd(Bp[p * 6 + 1]);
        c10 = _mm256_fnmadd_pd(a0, b, c10);
        c11 = _mm256_fnmadd_pd(a1, b, c11);
        b = _mm256_set1_pd(Bp[p * 6 + 2]);
        c20 = _mm256_fnmadd_pd(a0, b, c20);
        c21 = _mm256_fnmadd_pd(a1, b, c21);
        b = _mm256_set1_pd(Bp[p * 6 + 3]);
        c30 = _mm256_fnmadd_pd(a0, b, c30);
        c31 = _mm256_fnmadd_pd(a1, b, c31);
        b = _mm256_set1_pd(Bp[p * 6 + 4]);
        c40 = _mm256_fnmadd_pd(a0, b, c40);
        c41 = _mm256_fnmadd_pd(a1, b, c41);
        b = _mm256_set1_pd(Bp[p * 6 + 5]);
        c50 = _mm256_fnmadd_pd(a0, b, c50);
        c51 = _mm256_fnmadd_pd(a1, b, c51);
    }

    _mm256_storeu_pd(C + 0 * ldC + 0, c00);
    _mm256_storeu_pd(C + 0 * ldC + 4, c01);
    _mm256_storeu_pd(C + 1 * ldC + 0, c10);
    _mm256_storeu_pd(C + 1 * ldC + 4, c11);
    _mm256_storeu_pd(C + 2 * ldC + 0, c20);
    _mm256_storeu_pd(C + 2 * ldC + 4, c21);
    _mm256_storeu_pd(C + 3 * ldC + 0, c30);
    _mm256_storeu_pd(C + 3 * ldC + 4, c31);
    _mm256_storeu_pd(C + 4 * ldC + 0, c40);
    _mm256_storeu_pd(C + 4 * ldC + 4, c41);
    _mm256_storeu_pd(C + 5 * ldC + 0, c50);
    _mm256_storeu_pd(C + 5 * ldC + 4, c51);
}

static void __micro_kernel_nmul_packed_edge(
    const double* __restrict Ap,
    const double* __restrict Bp,
    double* __restrict C, size_t ldC, size_t k,
    int mr, int nr)
{
    const __m256i m0 = _mm256_set_epi64x(
        mr > 3 ? -1LL : 0LL, mr > 2 ? -1LL : 0LL,
        mr > 1 ? -1LL : 0LL, mr > 0 ? -1LL : 0LL);
    const __m256i m1 = _mm256_set_epi64x(
        mr > 7 ? -1LL : 0LL, mr > 6 ? -1LL : 0LL,
        mr > 5 ? -1LL : 0LL, mr > 4 ? -1LL : 0LL);

    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c00 = _mm256_maskload_pd(C + off + 0, m0);
        c01 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c10 = _mm256_maskload_pd(C + off + 0, m0);
        c11 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c20 = _mm256_maskload_pd(C + off + 0, m0);
        c21 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c30 = _mm256_maskload_pd(C + off + 0, m0);
        c31 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c40 = _mm256_maskload_pd(C + off + 0, m0);
        c41 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c50 = _mm256_maskload_pd(C + off + 0, m0);
        c51 = _mm256_maskload_pd(C + off + 4, m1);
    }

    for (size_t p = 0; p < k; ++p) {
        const __m256d a0 = _mm256_load_pd(Ap + p * 8);
        const __m256d a1 = _mm256_load_pd(Ap + p * 8 + 4);

        __m256d b;
        b = _mm256_set1_pd(Bp[p * 6 + 0]);
        c00 = _mm256_fnmadd_pd(a0, b, c00);
        c01 = _mm256_fnmadd_pd(a1, b, c01);
        b = _mm256_set1_pd(Bp[p * 6 + 1]);
        c10 = _mm256_fnmadd_pd(a0, b, c10);
        c11 = _mm256_fnmadd_pd(a1, b, c11);
        b = _mm256_set1_pd(Bp[p * 6 + 2]);
        c20 = _mm256_fnmadd_pd(a0, b, c20);
        c21 = _mm256_fnmadd_pd(a1, b, c21);
        b = _mm256_set1_pd(Bp[p * 6 + 3]);
        c30 = _mm256_fnmadd_pd(a0, b, c30);
        c31 = _mm256_fnmadd_pd(a1, b, c31);
        b = _mm256_set1_pd(Bp[p * 6 + 4]);
        c40 = _mm256_fnmadd_pd(a0, b, c40);
        c41 = _mm256_fnmadd_pd(a1, b, c41);
        b = _mm256_set1_pd(Bp[p * 6 + 5]);
        c50 = _mm256_fnmadd_pd(a0, b, c50);
        c51 = _mm256_fnmadd_pd(a1, b, c51);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c00);
        _mm256_maskstore_pd(C + off + 4, m1, c01);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c10);
        _mm256_maskstore_pd(C + off + 4, m1, c11);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c20);
        _mm256_maskstore_pd(C + off + 4, m1, c21);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c30);
        _mm256_maskstore_pd(C + off + 4, m1, c31);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c40);
        _mm256_maskstore_pd(C + off + 4, m1, c41);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c50);
        _mm256_maskstore_pd(C + off + 4, m1, c51);
    }
}

#endif