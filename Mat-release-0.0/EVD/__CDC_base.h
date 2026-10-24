
template<typename T>
inline T __EVD_H_CDC_Solve_EigenValue_Zero(int n, T* z, T* d) {
    T a = 0;
    for (int j = 0; j < n; ++j) {
        a += z[j] / d[j];
    }
    return a;
}
template <typename T>
inline double __EVD_H_CDC_Solve_EigenValue__(int n, T* z, T* d, T x, int k) {
    T a = 0;
    int j;
    for (j = 0; j < k; ++j) {
        a += z[j] / (d[j] - x);
    }
    for (j = k + 1; j < n; ++j) {
        a += z[j] / (d[j] - x);
    }
    return a;
}
template <typename T>
inline double __EVD_H_CDC_Solve_EigenValue__2(int n, T* z, T* d, T x, int k) {
    int j;
    T b = 0, temp;
    for (j = 0; j < k; ++j) {
        temp = d[j] - x;
        b += z[j] / (temp * temp);
    }
    for (j = k + 1; j < n; ++j) {
        temp = d[j] - x;
        b += z[j] / (temp * temp);
    }
    return b;
}
template<typename T>
inline void __EVD_H_CDC_u2(int n, T v, T* d, T* u, int k) {
    int i;
    for (i = 0; i < k; ++i) {
        u[i] *= (v - d[i]) / (d[k] - d[i]);
    }
    u[k] *= (v - d[k]);
    for (i = k + 1; i < n; ++i) {
        u[i] *= (v - d[i]) / (d[k] - d[i]);
    }
}
template<typename T>
inline void __EVD_H_CDC_v(int n, T* u, T* d, T* v, T x) {
    for (int j = 0; j < n; ++j) {
        v[j] = u[j] / (d[j] - x);
    }
}



template<typename T>
inline bool __EVD_H_CDC_Solve_EigenValue_if(int n, T* z, T* d, int i, T x) {
    return (1 + __EVD_H_CDC_Solve_EigenValue__(n, z, d, x, n)) > 0;
}

template<typename T>
inline T __EVD_H_CDC_Solve_EigenValue(int n, T* z, T* d, int i, int k, T x) {
    T a = 1 + __EVD_H_CDC_Solve_EigenValue__(n, z, d, x, k);
    T b = __EVD_H_CDC_Solve_EigenValue__2(n, z, d, x, k);

    T c, s, S;
    if (k == i) {
        int j = k + 1;
        c = a + (x - d[j]) * b;
        s = z[k];
        S = (x - d[j]) * (x - d[j]) * b;
    }
    else {
        c = a + (x - d[i]) * b;
        s = (x - d[i]) * (x - d[i]) * b;
        S = z[k];
    }

    a = (d[i] + d[i + 1]) * c + s + S;
    b = c * d[i] * d[i + 1] + s * d[i + 1] + S * d[i];

    if (a > 0)
        return (2 * b) / (a + sqrt(a * a - 4 * b * c));
    else
        return (a - sqrt(a * a - 4 * b * c)) / (2 * c);
}

template<typename T>
inline T __EVD_H_CDC_Solve_EigenValue2(int n, T* z, T* d, T x) {
    T a = 1 + __EVD_H_CDC_Solve_EigenValue__(n, z, d, x, n);
    T b = __EVD_H_CDC_Solve_EigenValue__2(n, z, d, x, n);

    if (a < 0) {
        return x - a / b;
    }

    return x - (a * x) / (a + b * x);
}

template<typename T>
inline void __EVD_H_CDC_u1(int n, T* u) {
    for (int i = 0; i < n; ++i) {
        if (u[i] < 0)
            u[i] = -1;
        else
            u[i] = 1;
    }
}

template<typename T>
inline void __EVD_H_CDC_u3(int n, T* u) {
    int i;

    for (i = 0; i < n; ++i) {
        if (u[i] < 0)
            u[i] = -sqrt(-u[i]);
        else
            u[i] = sqrt(u[i]);
    }
}

#if defined(__AVX512__)

// ¦² z[j]/d[j]
inline double __EVD_H_CDC_Solve_EigenValue_Zero(int n, double* z, double* d) {
    int i = 0;
    __m512d ans = _mm512_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        ans = _mm512_add_pd(ans, _mm512_div_pd(_mm512_loadu_pd(z + i), _mm512_loadu_pd(d + i)));
    }
    double s = _mm512_reduce_add_pd(ans);
    for (; i < n; ++i) s += z[i] / d[i];
    return s;
}

// ¦²_{j¡Ùk} z[j]/(d[j]-x)
inline double __EVD_H_CDC_Solve_EigenValue__(int n, double* z, double* d, double x, int k) {
    const __m512d xv = _mm512_set1_pd(x);
    __m512d ans = _mm512_setzero_pd();
    int i = 0;
    for (; i + 8 <= k; i += 8) {
        __m512d diff = _mm512_sub_pd(_mm512_loadu_pd(d + i), xv);
        ans = _mm512_add_pd(ans, _mm512_div_pd(_mm512_loadu_pd(z + i), diff));
    }
    double s = _mm512_reduce_add_pd(ans);
    for (; i < k; ++i) s += z[i] / (d[i] - x);

    __m512d ans2 = _mm512_setzero_pd();
    i = k + 1;
    for (; i + 8 <= n; i += 8) {
        __m512d diff = _mm512_sub_pd(_mm512_loadu_pd(d + i), xv);
        ans2 = _mm512_add_pd(ans2, _mm512_div_pd(_mm512_loadu_pd(z + i), diff));
    }
    s += _mm512_reduce_add_pd(ans2);
    for (; i < n; ++i) s += z[i] / (d[i] - x);
    return s;
}

// ¦²_{j¡Ùk} z[j]/(d[j]-x)^2
inline double __EVD_H_CDC_Solve_EigenValue__2(int n, double* z, double* d, double x, int k) {
    const __m512d xv = _mm512_set1_pd(x);
    __m512d ans = _mm512_setzero_pd();
    int i = 0;
    for (; i + 8 <= k; i += 8) {
        __m512d diff = _mm512_sub_pd(_mm512_loadu_pd(d + i), xv);
        ans = _mm512_add_pd(ans, _mm512_div_pd(_mm512_loadu_pd(z + i), _mm512_mul_pd(diff, diff)));
    }
    double s = _mm512_reduce_add_pd(ans);
    for (; i < k; ++i) { double t = d[i] - x; s += z[i] / (t * t); }

    __m512d ans2 = _mm512_setzero_pd();
    i = k + 1;
    for (; i + 8 <= n; i += 8) {
        __m512d diff = _mm512_sub_pd(_mm512_loadu_pd(d + i), xv);
        ans2 = _mm512_add_pd(ans2, _mm512_div_pd(_mm512_loadu_pd(z + i), _mm512_mul_pd(diff, diff)));
    }
    s += _mm512_reduce_add_pd(ans2);
    for (; i < n; ++i) { double t = d[i] - x; s += z[i] / (t * t); }
    return s;
}

// u[i] *= (v-d[i])/(d[k]-d[i])£»u[k] *= (v-d[k])
inline void __EVD_H_CDC_u2(int n, double v, double* d, double* u, int k) {
    const __m512d vv = _mm512_set1_pd(v);
    const __m512d dk = _mm512_set1_pd(d[k]);
    int i = 0;
    for (; i + 8 <= k; i += 8) {
        __m512d di = _mm512_loadu_pd(d + i);
        __m512d r = _mm512_div_pd(_mm512_sub_pd(vv, di), _mm512_sub_pd(dk, di));
        _mm512_storeu_pd(u + i, _mm512_mul_pd(_mm512_loadu_pd(u + i), r));
    }
    for (; i < k; ++i) u[i] *= (v - d[i]) / (d[k] - d[i]);
    u[k] *= (v - d[k]);
    i = k + 1;
    for (; i + 8 <= n; i += 8) {
        __m512d di = _mm512_loadu_pd(d + i);
        __m512d r = _mm512_div_pd(_mm512_sub_pd(vv, di), _mm512_sub_pd(dk, di));
        _mm512_storeu_pd(u + i, _mm512_mul_pd(_mm512_loadu_pd(u + i), r));
    }
    for (; i < n; ++i) u[i] *= (v - d[i]) / (d[k] - d[i]);
}

// v[j] = u[j]/(d[j]-x)
inline void __EVD_H_CDC_v(int n, double* u, double* d, double* v, double x) {
    const __m512d xv = _mm512_set1_pd(x);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d diff = _mm512_sub_pd(_mm512_loadu_pd(d + i), xv);
        _mm512_storeu_pd(v + i, _mm512_div_pd(_mm512_loadu_pd(u + i), diff));
    }
    for (; i < n; ++i) v[i] = u[i] / (d[i] - x);
}

// u[i] = (u[i] < 0) ? -1 : 1
inline void __EVD_H_CDC_u1(int n, double* u) {
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d neg_one = _mm512_set1_pd(-1.0);
    const __m512d zero = _mm512_setzero_pd();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __mmask8 neg = _mm512_cmp_pd_mask(_mm512_loadu_pd(u + i), zero, _CMP_LT_OQ);
        _mm512_storeu_pd(u + i, _mm512_mask_blend_pd(neg, one, neg_one));
    }
    for (; i < n; ++i) u[i] = (u[i] < 0.0) ? -1.0 : 1.0;
}

// u[i] = (u[i] < 0) ? -sqrt(-u[i]) : sqrt(u[i])
inline void __EVD_H_CDC_u3(int n, double* u) {
    const __m512d zero = _mm512_setzero_pd();
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d neg_one = _mm512_set1_pd(-1.0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d ui = _mm512_loadu_pd(u + i);
        __mmask8 neg = _mm512_cmp_pd_mask(ui, zero, _CMP_LT_OQ);
        __m512d sign = _mm512_mask_blend_pd(neg, one, neg_one);   // u<0 ? -1 : 1
        __m512d au = _mm512_mul_pd(ui, sign);                     // |u|
        _mm512_storeu_pd(u + i, _mm512_mul_pd(sign, _mm512_sqrt_pd(au)));
    }
    for (; i < n; ++i) u[i] = (u[i] < 0.0) ? -sqrt(-u[i]) : sqrt(u[i]);
}

#elif defined(__AVX2__)

inline double __EVD_H_CDC_Solve_EigenValue_Zero(int n, double* z, double* d) {
    int i = 0;
    __m256d ans = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        ans = _mm256_add_pd(ans, _mm256_div_pd(_mm256_loadu_pd(z + i), _mm256_loadu_pd(d + i)));
    }
    __m128d lo = _mm256_castpd256_pd128(ans);
    __m128d hi = _mm256_extractf128_pd(ans, 1);
    __m128d s2 = _mm_hadd_pd(_mm_add_pd(lo, hi), _mm_add_pd(lo, hi));
    double s = _mm_cvtsd_f64(s2);
    for (; i < n; ++i) s += z[i] / d[i];
    return s;
}

inline double __EVD_H_CDC_Solve_EigenValue__(int n, double* z, double* d, double x, int k) {
    const __m256d xv = _mm256_set1_pd(x);
    __m256d ans = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= k; i += 4) {
        __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(d + i), xv);
        ans = _mm256_add_pd(ans, _mm256_div_pd(_mm256_loadu_pd(z + i), diff));
    }
    __m128d lo = _mm256_castpd256_pd128(ans);
    __m128d hi = _mm256_extractf128_pd(ans, 1);
    __m128d s2 = _mm_hadd_pd(_mm_add_pd(lo, hi), _mm_add_pd(lo, hi));
    double s = _mm_cvtsd_f64(s2);
    for (; i < k; ++i) s += z[i] / (d[i] - x);

    __m256d ans2 = _mm256_setzero_pd();
    i = k + 1;
    for (; i + 4 <= n; i += 4) {
        __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(d + i), xv);
        ans2 = _mm256_add_pd(ans2, _mm256_div_pd(_mm256_loadu_pd(z + i), diff));
    }
    lo = _mm256_castpd256_pd128(ans2);
    hi = _mm256_extractf128_pd(ans2, 1);
    s2 = _mm_hadd_pd(_mm_add_pd(lo, hi), _mm_add_pd(lo, hi));
    s += _mm_cvtsd_f64(s2);
    for (; i < n; ++i) s += z[i] / (d[i] - x);
    return s;
}

inline double __EVD_H_CDC_Solve_EigenValue__2(int n, double* z, double* d, double x, int k) {
    const __m256d xv = _mm256_set1_pd(x);
    __m256d ans = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= k; i += 4) {
        __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(d + i), xv);
        ans = _mm256_add_pd(ans, _mm256_div_pd(_mm256_loadu_pd(z + i), _mm256_mul_pd(diff, diff)));
    }
    __m128d lo = _mm256_castpd256_pd128(ans);
    __m128d hi = _mm256_extractf128_pd(ans, 1);
    __m128d s2 = _mm_hadd_pd(_mm_add_pd(lo, hi), _mm_add_pd(lo, hi));
    double s = _mm_cvtsd_f64(s2);
    for (; i < k; ++i) { double t = d[i] - x; s += z[i] / (t * t); }

    __m256d ans2 = _mm256_setzero_pd();
    i = k + 1;
    for (; i + 4 <= n; i += 4) {
        __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(d + i), xv);
        ans2 = _mm256_add_pd(ans2, _mm256_div_pd(_mm256_loadu_pd(z + i), _mm256_mul_pd(diff, diff)));
    }
    lo = _mm256_castpd256_pd128(ans2);
    hi = _mm256_extractf128_pd(ans2, 1);
    s2 = _mm_hadd_pd(_mm_add_pd(lo, hi), _mm_add_pd(lo, hi));
    s += _mm_cvtsd_f64(s2);
    for (; i < n; ++i) { double t = d[i] - x; s += z[i] / (t * t); }
    return s;
}

inline void __EVD_H_CDC_u2(int n, double v, double* d, double* u, int k) {
    const __m256d vv = _mm256_set1_pd(v);
    const __m256d dk = _mm256_set1_pd(d[k]);
    int i = 0;
    for (; i + 4 <= k; i += 4) {
        __m256d di = _mm256_loadu_pd(d + i);
        __m256d r = _mm256_div_pd(_mm256_sub_pd(vv, di), _mm256_sub_pd(dk, di));
        _mm256_storeu_pd(u + i, _mm256_mul_pd(_mm256_loadu_pd(u + i), r));
    }
    for (; i < k; ++i) u[i] *= (v - d[i]) / (d[k] - d[i]);
    u[k] *= (v - d[k]);
    i = k + 1;
    for (; i + 4 <= n; i += 4) {
        __m256d di = _mm256_loadu_pd(d + i);
        __m256d r = _mm256_div_pd(_mm256_sub_pd(vv, di), _mm256_sub_pd(dk, di));
        _mm256_storeu_pd(u + i, _mm256_mul_pd(_mm256_loadu_pd(u + i), r));
    }
    for (; i < n; ++i) u[i] *= (v - d[i]) / (d[k] - d[i]);
}

inline void __EVD_H_CDC_v(int n, double* u, double* d, double* v, double x) {
    const __m256d xv = _mm256_set1_pd(x);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(d + i), xv);
        _mm256_storeu_pd(v + i, _mm256_div_pd(_mm256_loadu_pd(u + i), diff));
    }
    for (; i < n; ++i) v[i] = u[i] / (d[i] - x);
}

inline void __EVD_H_CDC_u1(int n, double* u) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg_one = _mm256_set1_pd(-1.0);
    const __m256d zero = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d neg = _mm256_cmp_pd(_mm256_loadu_pd(u + i), zero, _CMP_LT_OQ);
        _mm256_storeu_pd(u + i, _mm256_blendv_pd(one, neg_one, neg));
    }
    for (; i < n; ++i) u[i] = (u[i] < 0.0) ? -1.0 : 1.0;
}

inline void __EVD_H_CDC_u3(int n, double* u) {
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg_one = _mm256_set1_pd(-1.0);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d ui = _mm256_loadu_pd(u + i);
        __m256d neg = _mm256_cmp_pd(ui, zero, _CMP_LT_OQ);
        __m256d sign = _mm256_blendv_pd(one, neg_one, neg);
        __m256d au = _mm256_mul_pd(ui, sign);
        _mm256_storeu_pd(u + i, _mm256_mul_pd(sign, _mm256_sqrt_pd(au)));
    }
    for (; i < n; ++i) u[i] = (u[i] < 0.0) ? -sqrt(-u[i]) : sqrt(u[i]);
}

#endif
