
inline void __ApplyGivensOnTheRight(double* __restrict a, double* __restrict b, double c, double s, int len) {
    __m256d cc = _mm256_set1_pd(c);
    __m256d ss = _mm256_set1_pd(s);
    int n = (len / 4) * 4;
    int i;
    for (i = 0; i < n; i += 4) {
        __m256d ai = _mm256_loadu_pd(a + i);
        __m256d bi = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(a + i, _mm256_fmadd_pd(ai, cc, _mm256_mul_pd(bi, ss)));
        _mm256_storeu_pd(b + i, _mm256_fnmadd_pd(ai, ss, _mm256_mul_pd(bi, cc)));
    }
    int rem = len - n;
    if (rem != 0) {
        __m256i mask = _mm256_setr_epi64x(
            rem > 0 ? -1LL : 0LL,
            rem > 1 ? -1LL : 0LL,
            rem > 2 ? -1LL : 0LL,
            rem > 3 ? -1LL : 0LL
        );
        __m256d ai = _mm256_maskload_pd(a + i, mask);
        __m256d bi = _mm256_maskload_pd(b + i, mask);
        __m256d new_a = _mm256_fmadd_pd(ai, cc, _mm256_mul_pd(bi, ss));
        __m256d new_b = _mm256_fnmadd_pd(ai, ss, _mm256_mul_pd(bi, cc));
        _mm256_maskstore_pd(a + i, mask, new_a);
        _mm256_maskstore_pd(b + i, mask, new_b);
    }
}

inline void __Schur_Householder_Right(
    double* __restrict a, double* __restrict b, double* __restrict c,
    const double* __restrict w, int len) {
    __m256d w0 = _mm256_set1_pd(w[0]);
    __m256d w1 = _mm256_set1_pd(w[1]);
    __m256d w2 = _mm256_set1_pd(w[2]);

    int n = (len / 4) * 4;
    int i;
    for (i = 0; i < n; i += 4) {
        __m256d ai = _mm256_loadu_pd(a + i);
        __m256d bi = _mm256_loadu_pd(b + i);
        __m256d ci = _mm256_loadu_pd(c + i);
        __m256d sum =
            _mm256_fmadd_pd(ai, w0,
                _mm256_fmadd_pd(bi, w1,
                    _mm256_mul_pd(ci, w2)));
        _mm256_storeu_pd(a + i, _mm256_fnmadd_pd(sum, w0, ai));
        _mm256_storeu_pd(b + i, _mm256_fnmadd_pd(sum, w1, bi));
        _mm256_storeu_pd(c + i, _mm256_fnmadd_pd(sum, w2, ci));
    }
    int rem = len - n;
    if (rem != 0) {
        __m256i mask = _mm256_setr_epi64x(
            rem > 0 ? -1LL : 0LL,
            rem > 1 ? -1LL : 0LL,
            rem > 2 ? -1LL : 0LL,
            rem > 3 ? -1LL : 0LL
        );
        __m256d ai = _mm256_maskload_pd(a + i, mask);
        __m256d bi = _mm256_maskload_pd(b + i, mask);
        __m256d ci = _mm256_maskload_pd(c + i, mask);
        __m256d sum =
            _mm256_fmadd_pd(ai, w0,
                _mm256_fmadd_pd(bi, w1,
                    _mm256_mul_pd(ci, w2)));
        __m256d new_a = _mm256_fnmadd_pd(sum, w0, ai);
        __m256d new_b = _mm256_fnmadd_pd(sum, w1, bi);
        __m256d new_c = _mm256_fnmadd_pd(sum, w2, ci);
        _mm256_maskstore_pd(a + i, mask, new_a);
        _mm256_maskstore_pd(b + i, mask, new_b);
        _mm256_maskstore_pd(c + i, mask, new_c);
    }
}

inline void __times(double* __restrict a, double b, double* __restrict ans, int len) {
    __m256d bb = _mm256_set1_pd(b);
    int n = (len / 4) * 4;
    int i;
    for (i = 0; i < n; i += 4) {
        _mm256_storeu_pd(ans + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), bb));
    }
    int rem = len - n;
    if (rem != 0) {
        __m256i mask = _mm256_setr_epi64x(
            rem > 0 ? -1LL : 0LL,
            rem > 1 ? -1LL : 0LL,
            rem > 2 ? -1LL : 0LL,
            rem > 3 ? -1LL : 0LL
        );
        __m256d ai = _mm256_maskload_pd(a + i, mask);
        _mm256_maskstore_pd(ans + i, mask, _mm256_mul_pd(ai, bb));
    }
}

inline void __add(double* __restrict a, double* __restrict b, double* __restrict ans, int len) {
    int n = (len / 4) * 4;
    int i;
    for (i = 0; i < n; i += 4) {
        _mm256_storeu_pd(ans + i, _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
    }
    int rem = len - n;
    if (rem != 0) {
        __m256i mask = _mm256_setr_epi64x(
            rem > 0 ? -1LL : 0LL,
            rem > 1 ? -1LL : 0LL,
            rem > 2 ? -1LL : 0LL,
            rem > 3 ? -1LL : 0LL
        );
        __m256d ai = _mm256_maskload_pd(a + i, mask);
        __m256d bi = _mm256_maskload_pd(b + i, mask);
        _mm256_maskstore_pd(ans + i, mask, _mm256_add_pd(ai, bi));
    }
}

inline void __mixed(double* __restrict a, double* __restrict b, double q, double* __restrict ans, int len) {
    __m256d qq = _mm256_set1_pd(q);
    int n = (len / 4) * 4;
    int i;
    for (i = 0; i < n; i += 4) {
        _mm256_storeu_pd(ans + i, _mm256_fmadd_pd(_mm256_loadu_pd(b + i), qq, _mm256_loadu_pd(a + i)));
    }
    int rem = len - n;
    if (rem != 0) {
        __m256i mask = _mm256_setr_epi64x(
            rem > 0 ? -1LL : 0LL,
            rem > 1 ? -1LL : 0LL,
            rem > 2 ? -1LL : 0LL,
            rem > 3 ? -1LL : 0LL
        );
        __m256d ai = _mm256_maskload_pd(a + i, mask);
        __m256d bi = _mm256_maskload_pd(b + i, mask);
        _mm256_maskstore_pd(ans + i, mask, _mm256_fmadd_pd(bi, qq, ai));
    }
}

inline void __swap(double* __restrict a, double* __restrict b, int len) {
    int n = (len / 4) * 4;
    int i;
    for (i = 0; i < n; i += 4) {
        __m256d av = _mm256_loadu_pd(a + i);
        __m256d bv = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(a + i, bv);
        _mm256_storeu_pd(b + i, av);
    }
    int rem = len - n;
    if (rem != 0) {
        __m256i mask = _mm256_setr_epi64x(
            rem > 0 ? -1LL : 0LL,
            rem > 1 ? -1LL : 0LL,
            rem > 2 ? -1LL : 0LL,
            rem > 3 ? -1LL : 0LL
        );
        __m256d av = _mm256_maskload_pd(a + i, mask);
        __m256d bv = _mm256_maskload_pd(b + i, mask);
        _mm256_maskstore_pd(a + i, mask, bv);
        _mm256_maskstore_pd(b + i, mask, av);
    }
}

inline double __dot(double* __restrict a, double* __restrict b, int len) {
    int n = (len / 4) * 4;
    int i = 0;

    __m256d sum;
    if (n >= 16) {
        __m256d ans0 = _mm256_setzero_pd();
        __m256d ans1 = _mm256_setzero_pd();
        __m256d ans2 = _mm256_setzero_pd();
        __m256d ans3 = _mm256_setzero_pd();
        for (; i <= n - 16; i += 16) {
            ans0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), ans0);
            ans1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4), ans1);
            ans2 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 8), _mm256_loadu_pd(b + i + 8), ans2);
            ans3 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 12), _mm256_loadu_pd(b + i + 12), ans3);
        }
        sum = _mm256_add_pd(_mm256_add_pd(ans0, ans1), _mm256_add_pd(ans2, ans3));
    }
    else {
        sum = _mm256_setzero_pd();
    }

    for (; i < n; i += 4) {
        sum = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), sum);
    }

    int rem = len - n;
    if (rem != 0) {
        __m256i mask = _mm256_setr_epi64x(
            rem > 0 ? -1LL : 0LL,
            rem > 1 ? -1LL : 0LL,
            rem > 2 ? -1LL : 0LL,
            rem > 3 ? -1LL : 0LL
        );
        __m256d ai = _mm256_maskload_pd(a + i, mask);
        __m256d bi = _mm256_maskload_pd(b + i, mask);
        sum = _mm256_fmadd_pd(ai, bi, sum);   // 无效 lane 为 0，不影响 sum
    }

    __m128d low = _mm256_castpd256_pd128(sum);
    __m128d high = _mm256_extractf128_pd(sum, 1);
    __m128d pair_sum = _mm_add_pd(low, high);
    pair_sum = _mm_hadd_pd(pair_sum, pair_sum);
    return _mm_cvtsd_f64(pair_sum);
}

double __MaxAbs(double* a, int len) {
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    __m256d max0 = _mm256_setzero_pd();
    __m256d max1 = _mm256_setzero_pd();
    __m256d max2 = _mm256_setzero_pd();
    __m256d max3 = _mm256_setzero_pd();
    int n = (len / 4) * 4;
    int i = 0;
    for (; i <= n - 16; i += 16) {
        max0 = _mm256_max_pd(max0, _mm256_andnot_pd(sign_mask, _mm256_loadu_pd(a + i)));
        max1 = _mm256_max_pd(max1, _mm256_andnot_pd(sign_mask, _mm256_loadu_pd(a + i + 4)));
        max2 = _mm256_max_pd(max2, _mm256_andnot_pd(sign_mask, _mm256_loadu_pd(a + i + 8)));
        max3 = _mm256_max_pd(max3, _mm256_andnot_pd(sign_mask, _mm256_loadu_pd(a + i + 12)));
    }
    for (; i < n; i += 4) {
        max0 = _mm256_max_pd(max0, _mm256_andnot_pd(sign_mask, _mm256_loadu_pd(a + i)));
    }
    __m256d maxv = _mm256_max_pd(_mm256_max_pd(max0, max1), _mm256_max_pd(max2, max3));

    int rem = len - n;
    if (rem != 0) {
        __m256i mask = _mm256_setr_epi64x(
            rem > 0 ? -1LL : 0LL,
            rem > 1 ? -1LL : 0LL,
            rem > 2 ? -1LL : 0LL,
            rem > 3 ? -1LL : 0LL
        );
        __m256d av = _mm256_maskload_pd(a + i, mask);
        maxv = _mm256_max_pd(maxv, _mm256_andnot_pd(sign_mask, av)); // 无效 lane 为 0，不影响 maxv
    }

    __m128d low = _mm256_castpd256_pd128(maxv);
    __m128d high = _mm256_extractf128_pd(maxv, 1);
    __m128d max128 = _mm_max_pd(low, high);
    __m128d swapped = _mm_permute_pd(max128, 0x1);
    max128 = _mm_max_pd(max128, swapped);
    return _mm_cvtsd_f64(max128);
}