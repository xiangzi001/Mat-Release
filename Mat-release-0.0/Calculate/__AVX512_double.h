
inline void __ApplyGivensOnTheRight(double* __restrict a, double* __restrict b, double c, double s, int len) {
    __m512d cc = _mm512_set1_pd(c);
    __m512d ss = _mm512_set1_pd(s);
    int n = (len / 8) * 8;
    int i;
    for (i = 0; i < n; i += 8) {
        __m512d ai = _mm512_loadu_pd(a + i);
        __m512d bi = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(a + i, _mm512_fmadd_pd(ai, cc, _mm512_mul_pd(bi, ss)));
        _mm512_storeu_pd(b + i, _mm512_fnmadd_pd(ai, ss, _mm512_mul_pd(bi, cc)));
    }
    int rem = len - n;
    if (rem != 0) {
        __mmask8 mask = (1 << rem) - 1;
        __m512d ai = _mm512_maskz_loadu_pd(mask, a + i);
        __m512d bi = _mm512_maskz_loadu_pd(mask, b + i);
        __m512d new_a = _mm512_fmadd_pd(ai, cc, _mm512_mul_pd(bi, ss));
        __m512d new_b = _mm512_fnmadd_pd(ai, ss, _mm512_mul_pd(bi, cc));
        _mm512_mask_storeu_pd(a + i, mask, new_a);
        _mm512_mask_storeu_pd(b + i, mask, new_b);
    }
}

inline void __Schur_Householder_Right(
    double* __restrict a, double* __restrict b, double* __restrict c,
    const double* __restrict w, int len) {
    __m512d w0 = _mm512_set1_pd(w[0]);
    __m512d w1 = _mm512_set1_pd(w[1]);
    __m512d w2 = _mm512_set1_pd(w[2]);

    int n = (len / 8) * 8;
    int i;
    for (i = 0; i < n; i += 8) {
        __m512d ai = _mm512_loadu_pd(a + i);
        __m512d bi = _mm512_loadu_pd(b + i);
        __m512d ci = _mm512_loadu_pd(c + i);
        __m512d sum =
            _mm512_fmadd_pd(ai, w0,
                _mm512_fmadd_pd(bi, w1,
                    _mm512_mul_pd(ci, w2)));
        _mm512_storeu_pd(a + i, _mm512_fnmadd_pd(sum, w0, ai));
        _mm512_storeu_pd(b + i, _mm512_fnmadd_pd(sum, w1, bi));
        _mm512_storeu_pd(c + i, _mm512_fnmadd_pd(sum, w2, ci));
    }
    int rem = len - n;
    if (rem != 0) {
        __mmask8 mask = (1 << rem) - 1;
        __m512d ai = _mm512_maskz_loadu_pd(mask, a + i);
        __m512d bi = _mm512_maskz_loadu_pd(mask, b + i);
        __m512d ci = _mm512_maskz_loadu_pd(mask, c + i);
        __m512d sum =
            _mm512_fmadd_pd(ai, w0,
                _mm512_fmadd_pd(bi, w1,
                    _mm512_mul_pd(ci, w2)));
        __m512d new_a = _mm512_fnmadd_pd(sum, w0, ai);
        __m512d new_b = _mm512_fnmadd_pd(sum, w1, bi);
        __m512d new_c = _mm512_fnmadd_pd(sum, w2, ci);
        _mm512_mask_storeu_pd(a + i, mask, new_a);
        _mm512_mask_storeu_pd(b + i, mask, new_b);
        _mm512_mask_storeu_pd(c + i, mask, new_c);
    }
}

inline void __times(double* __restrict a, double b, double* __restrict ans, int len) {
    __m512d bb = _mm512_set1_pd(b);
    int n = (len / 8) * 8;
    int i;
    for (i = 0; i < n; i += 8) {
        _mm512_storeu_pd(ans + i, _mm512_mul_pd(_mm512_loadu_pd(a + i), bb));
    }
    int rem = len - n;
    if (rem != 0) {
        __mmask8 mask = (1 << rem) - 1;
        __m512d ai = _mm512_maskz_loadu_pd(mask, a + i);
        _mm512_mask_storeu_pd(ans + i, mask, _mm512_mul_pd(ai, bb));
    }
}

inline void __add(double* __restrict a, double* __restrict b, double* __restrict ans, int len) {
    int n = (len / 8) * 8;
    int i;
    for (i = 0; i < n; i += 8) {
        _mm512_storeu_pd(ans + i, _mm512_add_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i)));
    }
    int rem = len - n;
    if (rem != 0) {
        __mmask8 mask = (1 << rem) - 1;
        __m512d ai = _mm512_maskz_loadu_pd(mask, a + i);
        __m512d bi = _mm512_maskz_loadu_pd(mask, b + i);
        _mm512_mask_storeu_pd(ans + i, mask, _mm512_add_pd(ai, bi));
    }
}

// ans = a + b * q
inline void __mixed(double* __restrict a, double* __restrict b, double q, double* __restrict ans, int len) {
    __m512d qq = _mm512_set1_pd(q);
    int n = (len / 8) * 8;
    int i;
    for (i = 0; i < n; i += 8) {
        _mm512_storeu_pd(ans + i, _mm512_fmadd_pd(_mm512_loadu_pd(b + i), qq, _mm512_loadu_pd(a + i)));
    }
    int rem = len - n;
    if (rem != 0) {
        __mmask8 mask = (1 << rem) - 1;
        __m512d ai = _mm512_maskz_loadu_pd(mask, a + i);
        __m512d bi = _mm512_maskz_loadu_pd(mask, b + i);
        _mm512_mask_storeu_pd(ans + i, mask, _mm512_fmadd_pd(bi, qq, ai));
    }
}

inline void __swap(double* __restrict a, double* __restrict b, int len) {
    int n = (len / 8) * 8;
    int i;
    for (i = 0; i < n; i += 8) {
        __m512d av = _mm512_loadu_pd(a + i);
        __m512d bv = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(a + i, bv);
        _mm512_storeu_pd(b + i, av);
    }
    int rem = len - n;
    if (rem != 0) {
        __mmask8 mask = (1 << rem) - 1;
        __m512d av = _mm512_maskz_loadu_pd(mask, a + i);
        __m512d bv = _mm512_maskz_loadu_pd(mask, b + i);
        _mm512_mask_storeu_pd(a + i, mask, bv);
        _mm512_mask_storeu_pd(b + i, mask, av);
    }
}

inline double __dot(double* __restrict a, double* __restrict b, int len) {
    int n = (len / 8) * 8;
    int i = 0;
    __m512d ans;
    if (n >= 32) {
        __m512d ans0 = _mm512_setzero_pd();
        __m512d ans1 = _mm512_setzero_pd();
        __m512d ans2 = _mm512_setzero_pd();
        __m512d ans3 = _mm512_setzero_pd();
        for (; i <= n - 32; i += 32) {
            ans0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i), ans0);
            ans1 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 8), _mm512_loadu_pd(b + i + 8), ans1);
            ans2 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 16), _mm512_loadu_pd(b + i + 16), ans2);
            ans3 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 24), _mm512_loadu_pd(b + i + 24), ans3);
        }
        ans = _mm512_add_pd(_mm512_add_pd(ans0, ans1), _mm512_add_pd(ans2, ans3));
    }
    else {
        ans = _mm512_setzero_pd();
    }
    for (; i < n; i += 8) {
        ans = _mm512_fmadd_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i), ans);
    }

    int rem = len - n;
    if (rem != 0) {
        __mmask8 mask = (1 << rem) - 1;
        __m512d av = _mm512_maskz_loadu_pd(mask, a + i);
        __m512d bv = _mm512_maskz_loadu_pd(mask, b + i);
        ans = _mm512_fmadd_pd(av, bv, ans);
    }
    return _mm512_reduce_add_pd(ans);
}

double __MaxAbs(double* a, int len) {
    int n = (len / 8) * 8;
    __m512d max0 = _mm512_setzero_pd();
    __m512d max1 = _mm512_setzero_pd();
    __m512d max2 = _mm512_setzero_pd();
    __m512d max3 = _mm512_setzero_pd();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        max0 = _mm512_max_pd(max0, _mm512_abs_pd(_mm512_loadu_pd(a + i)));
        max1 = _mm512_max_pd(max1, _mm512_abs_pd(_mm512_loadu_pd(a + i + 8)));
        max2 = _mm512_max_pd(max2, _mm512_abs_pd(_mm512_loadu_pd(a + i + 16)));
        max3 = _mm512_max_pd(max3, _mm512_abs_pd(_mm512_loadu_pd(a + i + 24)));
    }
    for (; i < n; i += 8) {
        max0 = _mm512_max_pd(max0, _mm512_abs_pd(_mm512_loadu_pd(a + i)));
    }
    __m512d max = _mm512_max_pd(_mm512_max_pd(max0, max1), _mm512_max_pd(max2, max3));

    int rem = len - n;
    if (rem != 0) {
        __mmask8 mask = (1 << rem) - 1;
        __m512d av = _mm512_maskz_loadu_pd(mask, a + i);
        max = _mm512_max_pd(max, _mm512_abs_pd(av));
    }
    return _mm512_reduce_max_pd(max);
}