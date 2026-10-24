template<typename t1, typename t2, typename t3>
void __mulAB_outer_vec(const MAT<t1>& A, const MAT<t2>& B, MAT<t3>& C)
{
    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    const int ldA = A.rows;
    const int ldB = B.rows;
    const int ldC = C.rows;

    __set0(C.data, (size_t)M * N);

    for (int k = 0; k < K; ++k) {
        t1* Ak = A.data + (size_t)k * ldA;

        for (int j = 0; j < N; ++j) {
            const t2 b = B.data[k + (size_t)j * ldB];
            t3* Cj = C.data + (size_t)j * ldC;

            __mixed(Cj, Ak, b, Cj, M);
        }
    }
}


template<typename t1, typename t2, typename t3>
inline void __mixed4(t3* __restrict ans,
    t1* __restrict a0, t2 q0,
    t1* __restrict a1, t2 q1,
    t1* __restrict a2, t2 q2,
    t1* __restrict a3, t2 q3,
    int len)
{
    for (int i = 0; i < len; ++i) {
        t3 c1 = a0[i] * q0;
        t3 c2 = a1[i] * q1;
        t3 c3 = a2[i] * q2;
        t3 c0 = ans[i] + a3[i] * q3;
        ans[i] = (c0 + c1) + (c2 + c3);
    }
}

#ifdef __AVX512__

inline void __mixed8(double* __restrict ans,
    double* __restrict a0, double q0,
    double* __restrict a1, double q1,
    double* __restrict a2, double q2,
    double* __restrict a3, double q3,
    double* __restrict a4, double q4,
    double* __restrict a5, double q5,
    double* __restrict a6, double q6,
    double* __restrict a7, double q7,
    int len)
{
    const __m512d vq0 = _mm512_set1_pd(q0);
    const __m512d vq1 = _mm512_set1_pd(q1);
    const __m512d vq2 = _mm512_set1_pd(q2);
    const __m512d vq3 = _mm512_set1_pd(q3);
    const __m512d vq4 = _mm512_set1_pd(q4);
    const __m512d vq5 = _mm512_set1_pd(q5);
    const __m512d vq6 = _mm512_set1_pd(q6);
    const __m512d vq7 = _mm512_set1_pd(q7);
    int n = (len / 8) * 8;
    int i = 0;
    for (; i < n; i += 8) {
        __m512d c1 = _mm512_mul_pd(_mm512_loadu_pd(a0 + i), vq0);
        __m512d c0 = _mm512_fmadd_pd(_mm512_loadu_pd(a1 + i), vq1, _mm512_loadu_pd(ans + i));
        c1 = _mm512_fmadd_pd(_mm512_loadu_pd(a2 + i), vq2, c1);
        c0 = _mm512_fmadd_pd(_mm512_loadu_pd(a3 + i), vq3, c0);
        c1 = _mm512_fmadd_pd(_mm512_loadu_pd(a4 + i), vq4, c1);
        c0 = _mm512_fmadd_pd(_mm512_loadu_pd(a5 + i), vq5, c0);
        c1 = _mm512_fmadd_pd(_mm512_loadu_pd(a6 + i), vq6, c1);
        c0 = _mm512_fmadd_pd(_mm512_loadu_pd(a7 + i), vq7, c0);
        _mm512_storeu_pd(ans + i, _mm512_add_pd(c0, c1));
    }
    int rem = len - n;
    if (rem != 0) {
        const __mmask8 mask = (__mmask8)((1u << rem) - 1u);
        __m512d c1 = _mm512_mul_pd(_mm512_maskz_loadu_pd(mask, a0 + i), vq0);
        __m512d c0 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(mask, a1 + i), vq1, _mm512_maskz_loadu_pd(mask, ans + i));
        c1 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(mask, a2 + i), vq2, c1);
        c0 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(mask, a3 + i), vq3, c0);
        c1 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(mask, a4 + i), vq4, c1);
        c0 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(mask, a5 + i), vq5, c0);
        c1 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(mask, a6 + i), vq6, c1);
        c0 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(mask, a7 + i), vq7, c0);
        _mm512_mask_storeu_pd(ans + i, mask, _mm512_add_pd(c0, c1));
    }
}

inline void __mulAB_vec(double* A, const double* b, double* c, int M, int K, int ldA) {
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __mixed8(c,
            A + (size_t)(k + 0) * ldA, b[k + 0],
            A + (size_t)(k + 1) * ldA, b[k + 1],
            A + (size_t)(k + 2) * ldA, b[k + 2],
            A + (size_t)(k + 3) * ldA, b[k + 3],
            A + (size_t)(k + 4) * ldA, b[k + 4],
            A + (size_t)(k + 5) * ldA, b[k + 5],
            A + (size_t)(k + 6) * ldA, b[k + 6],
            A + (size_t)(k + 7) * ldA, b[k + 7],
            M);
    }
    for (; k < K; ++k)
        __mixed(c, A + (size_t)k * ldA, b[k], c, M);
}

#elif defined(__AVX2__)

inline void __mixed6(double* __restrict ans,
    double* __restrict a0, double q0,
    double* __restrict a1, double q1,
    double* __restrict a2, double q2,
    double* __restrict a3, double q3,
    double* __restrict a4, double q4,
    double* __restrict a5, double q5,
    int len) {
    const __m256d vq0 = _mm256_set1_pd(q0);
    const __m256d vq1 = _mm256_set1_pd(q1);
    const __m256d vq2 = _mm256_set1_pd(q2);
    const __m256d vq3 = _mm256_set1_pd(q3);
    const __m256d vq4 = _mm256_set1_pd(q4);
    const __m256d vq5 = _mm256_set1_pd(q5);
    int n = (len / 4) * 4;
    int i = 0;
    for (; i < n; i += 4) {
        __m256d c1 = _mm256_mul_pd(_mm256_loadu_pd(a0 + i), vq0);
        __m256d c2 = _mm256_mul_pd(_mm256_loadu_pd(a1 + i), vq1);
        __m256d c3 = _mm256_mul_pd(_mm256_loadu_pd(a2 + i), vq2);
        __m256d c0 = _mm256_fmadd_pd(_mm256_loadu_pd(a3 + i), vq3, _mm256_loadu_pd(ans + i));
        c1 = _mm256_fmadd_pd(_mm256_loadu_pd(a4 + i), vq4, c1);
        c2 = _mm256_fmadd_pd(_mm256_loadu_pd(a5 + i), vq5, c2);
        _mm256_storeu_pd(ans + i, _mm256_add_pd(_mm256_add_pd(c0, c1), _mm256_add_pd(c2, c3)));
    }
    int rem = len - n;
    if (rem != 0) {
        const __m256i mask = _mm256_setr_epi64x(
            rem > 0 ? -1LL : 0LL,
            rem > 1 ? -1LL : 0LL,
            rem > 2 ? -1LL : 0LL,
            rem > 3 ? -1LL : 0LL);
        __m256d c1 = _mm256_mul_pd(_mm256_maskload_pd(a0 + i, mask), vq0);
        __m256d c2 = _mm256_mul_pd(_mm256_maskload_pd(a1 + i, mask), vq1);
        __m256d c3 = _mm256_mul_pd(_mm256_maskload_pd(a2 + i, mask), vq2);
        __m256d c0 = _mm256_fmadd_pd(_mm256_maskload_pd(a3 + i, mask), vq3, _mm256_maskload_pd(ans + i, mask));
        c1 = _mm256_fmadd_pd(_mm256_maskload_pd(a4 + i, mask), vq4, c1);
        c2 = _mm256_fmadd_pd(_mm256_maskload_pd(a5 + i, mask), vq5, c2);
        _mm256_maskstore_pd(ans + i, mask, _mm256_add_pd(_mm256_add_pd(c0, c1), _mm256_add_pd(c2, c3)));
    }
}

inline void __mulAB_vec(double* A, const double* b, double* c, int M, int K, int ldA) {
    int k = 0;
    for (; k + 6 <= K; k += 6) {
        __mixed6(c,
            A + (size_t)(k + 0) * ldA, b[k + 0],
            A + (size_t)(k + 1) * ldA, b[k + 1],
            A + (size_t)(k + 2) * ldA, b[k + 2],
            A + (size_t)(k + 3) * ldA, b[k + 3],
            A + (size_t)(k + 4) * ldA, b[k + 4],
            A + (size_t)(k + 5) * ldA, b[k + 5],
            M);
    }
    for (; k < K; ++k)
        __mixed(c, A + (size_t)k * ldA, b[k], c, M);
}

#endif

template<typename t1, typename t2, typename t3>
inline void __mulAB_vec(t1* A, const t2* b, t3* c, int M, int K, int ldA) {
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        __mixed4(c,
            A + (size_t)(k + 0) * ldA, b[k + 0],
            A + (size_t)(k + 1) * ldA, b[k + 1],
            A + (size_t)(k + 2) * ldA, b[k + 2],
            A + (size_t)(k + 3) * ldA, b[k + 3],
            M);
    }
    for (; k < K; ++k)
        __mixed(c, A + (size_t)k * ldA, b[k], c, M);
}

template<typename t1, typename t2, typename t3>
inline void __mulAB_vec(const MAT<t1>& A, const MAT<t2>& B, MAT<t3>& C) {
    __set0(C.data, (size_t)C.rows * C.cols);
    for (int i = 0; i < C.cols; ++i) {
        __mulAB_vec(A.data, B.data + C.rows * i, C.data + C.rows * i, A.rows, A.cols, A.rows);
    }
}






template<typename T>
void __mulAB_vec_AH_lower(T* __restrict A, size_t ldinc,
    const T* __restrict x,
    T* __restrict y, int n)
{
    for (int i = 0; i < n; ++i) y[i] = T(0);

    int j = 0;
    for (; j + 4 <= n; j += 4) {
        T yl[4];
        for (int t = 0; t < 4; ++t) yl[t] = y[j + t];
        for (int t = 0; t < 4; ++t) {
            const T* __restrict col = A + (size_t)(j + t) * ldinc;
            yl[t] += col[0] * x[j + t];
            for (int s = t + 1; s < 4; ++s) {
                const T a = col[s - t];
                yl[t] += a * x[j + s];
                yl[s] += a * x[j + t];
            }
        }
        for (int t = 0; t < 4; ++t) y[j + t] = yl[t];

        const int len = n - j - 4;
        if (len <= 0) continue;

        const T x0 = x[j + 0];
        const T x1 = x[j + 1];
        const T x2 = x[j + 2];
        const T x3 = x[j + 3];

        const T* __restrict c0 = A + (size_t)(j + 0) * ldinc + 4;
        const T* __restrict c1 = A + (size_t)(j + 1) * ldinc + 3;
        const T* __restrict c2 = A + (size_t)(j + 2) * ldinc + 2;
        const T* __restrict c3 = A + (size_t)(j + 3) * ldinc + 1;

        T yj0 = T(0), yj1 = T(0), yj2 = T(0), yj3 = T(0);

        for (int i = 0; i < len; ++i) {
            const T a0 = c0[i];
            const T a1 = c1[i];
            const T a2 = c2[i];
            const T a3 = c3[i];
            const T xv = x[j + 4 + i];

            y[j + 4 + i] += ((a0 * x0) + (a1 * x1)) + ((a2 * x2) + (a3 * x3));

            yj0 += a0 * xv;
            yj1 += a1 * xv;
            yj2 += a2 * xv;
            yj3 += a3 * xv;
        }

        y[j + 0] += yj0;
        y[j + 1] += yj1;
        y[j + 2] += yj2;
        y[j + 3] += yj3;
    }

    for (; j < n; ++j) {
        const T* __restrict col = A + (size_t)j * ldinc;
        const T xj = x[j];
        const int len = n - j;
        T yj = col[0] * xj;
        for (int i = 1; i < len; ++i) {
            const T a = col[i];
            y[j + i] += a * xj;
            yj += a * x[j + i];
        }
        y[j] += yj;
    }
}

#ifdef __AVX512__

inline void __mulAB_vec_AH_lower(double* __restrict A, size_t ldinc,
    const double* __restrict x,
    double* __restrict y, int n)
{
    for (int i = 0; i < n; ++i) y[i] = 0.0;

    const int r = n & 7;
    const int n_main = n - r;

    if (r > 0) {
        double yl[8];
        for (int t = 0; t < r; ++t) yl[t] = y[t];
        for (int t = 0; t < r; ++t) {
            const double* col = A + (size_t)t * ldinc;
            yl[t] += col[0] * x[t];
            for (int s = t + 1; s < r; ++s) {
                const double a = col[s - t];
                yl[t] += a * x[s];
                yl[s] += a * x[t];
            }
        }
        for (int t = 0; t < r; ++t) y[t] = yl[t];

        __m512d xv_col[8], acc[8];
        for (int t = 0; t < r; ++t) {
            xv_col[t] = _mm512_set1_pd(x[t]);
            acc[t] = _mm512_setzero_pd();
        }
        for (int i = 0; i < n_main; i += 8) {
            __m512d yv = _mm512_loadu_pd(y + r + i);
            const __m512d xv_rows = _mm512_loadu_pd(x + r + i);
            for (int t = 0; t < r; ++t) {
                const __m512d av = _mm512_loadu_pd(
                    A + (size_t)t * ldinc + (r - t) + i);
                yv = _mm512_fmadd_pd(av, xv_col[t], yv);
                acc[t] = _mm512_fmadd_pd(av, xv_rows, acc[t]);
            }
            _mm512_storeu_pd(y + r + i, yv);
        }
        for (int t = 0; t < r; ++t)
            y[t] += _mm512_reduce_add_pd(acc[t]);
    }

    for (int j = r; j + 8 <= n; j += 8) {
        double yl[8];
        for (int t = 0; t < 8; ++t) yl[t] = y[j + t];
        for (int t = 0; t < 8; ++t) {
            const double* col = A + (size_t)(j + t) * ldinc;
            yl[t] += col[0] * x[j + t];
            for (int s = t + 1; s < 8; ++s) {
                const double a = col[s - t];
                yl[t] += a * x[j + s];
                yl[s] += a * x[j + t];
            }
        }
        for (int t = 0; t < 8; ++t) y[j + t] = yl[t];

        const int len = n - j - 8;
        if (len <= 0) continue;

        const __m512d x0 = _mm512_set1_pd(x[j + 0]);
        const __m512d x1 = _mm512_set1_pd(x[j + 1]);
        const __m512d x2 = _mm512_set1_pd(x[j + 2]);
        const __m512d x3 = _mm512_set1_pd(x[j + 3]);
        const __m512d x4 = _mm512_set1_pd(x[j + 4]);
        const __m512d x5 = _mm512_set1_pd(x[j + 5]);
        const __m512d x6 = _mm512_set1_pd(x[j + 6]);
        const __m512d x7 = _mm512_set1_pd(x[j + 7]);

        const double* __restrict c0 = A + (size_t)(j + 0) * ldinc + 8;
        const double* __restrict c1 = A + (size_t)(j + 1) * ldinc + 7;
        const double* __restrict c2 = A + (size_t)(j + 2) * ldinc + 6;
        const double* __restrict c3 = A + (size_t)(j + 3) * ldinc + 5;
        const double* __restrict c4 = A + (size_t)(j + 4) * ldinc + 4;
        const double* __restrict c5 = A + (size_t)(j + 5) * ldinc + 3;
        const double* __restrict c6 = A + (size_t)(j + 6) * ldinc + 2;
        const double* __restrict c7 = A + (size_t)(j + 7) * ldinc + 1;

        __m512d acc0 = _mm512_setzero_pd(), acc1 = _mm512_setzero_pd();
        __m512d acc2 = _mm512_setzero_pd(), acc3 = _mm512_setzero_pd();
        __m512d acc4 = _mm512_setzero_pd(), acc5 = _mm512_setzero_pd();
        __m512d acc6 = _mm512_setzero_pd(), acc7 = _mm512_setzero_pd();

        for (int i = 0; i < len; i += 8) {
            const __m512d xv = _mm512_loadu_pd(x + j + 8 + i);

            const __m512d a0 = _mm512_loadu_pd(c0 + i);
            const __m512d a1 = _mm512_loadu_pd(c1 + i);
            const __m512d a2 = _mm512_loadu_pd(c2 + i);
            const __m512d a3 = _mm512_loadu_pd(c3 + i);
            const __m512d a4 = _mm512_loadu_pd(c4 + i);
            const __m512d a5 = _mm512_loadu_pd(c5 + i);
            const __m512d a6 = _mm512_loadu_pd(c6 + i);
            const __m512d a7 = _mm512_loadu_pd(c7 + i);

            __m512d yv0 = _mm512_loadu_pd(y + j + 8 + i);
            __m512d yv1 = _mm512_mul_pd(a4, x4);
            yv0 = _mm512_fmadd_pd(a0, x0, yv0);
            yv1 = _mm512_fmadd_pd(a5, x5, yv1);
            yv0 = _mm512_fmadd_pd(a1, x1, yv0);
            yv1 = _mm512_fmadd_pd(a6, x6, yv1);
            yv0 = _mm512_fmadd_pd(a2, x2, yv0);
            yv1 = _mm512_fmadd_pd(a7, x7, yv1);
            yv0 = _mm512_fmadd_pd(a3, x3, yv0);
            _mm512_storeu_pd(y + j + 8 + i, _mm512_add_pd(yv0, yv1));

            acc0 = _mm512_fmadd_pd(a0, xv, acc0);
            acc1 = _mm512_fmadd_pd(a1, xv, acc1);
            acc2 = _mm512_fmadd_pd(a2, xv, acc2);
            acc3 = _mm512_fmadd_pd(a3, xv, acc3);
            acc4 = _mm512_fmadd_pd(a4, xv, acc4);
            acc5 = _mm512_fmadd_pd(a5, xv, acc5);
            acc6 = _mm512_fmadd_pd(a6, xv, acc6);
            acc7 = _mm512_fmadd_pd(a7, xv, acc7);
        }

        y[j + 0] += _mm512_reduce_add_pd(acc0);
        y[j + 1] += _mm512_reduce_add_pd(acc1);
        y[j + 2] += _mm512_reduce_add_pd(acc2);
        y[j + 3] += _mm512_reduce_add_pd(acc3);
        y[j + 4] += _mm512_reduce_add_pd(acc4);
        y[j + 5] += _mm512_reduce_add_pd(acc5);
        y[j + 6] += _mm512_reduce_add_pd(acc6);
        y[j + 7] += _mm512_reduce_add_pd(acc7);
    }
}

#elif defined(__AVX2__)

inline double __ah_hsum256(__m256d v) {
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    const __m128d s = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}

inline void __mulAB_vec_AH_lower(double* __restrict A, size_t ldinc,
    const double* __restrict x,
    double* __restrict y, int n)
{
    for (int i = 0; i < n; ++i) y[i] = 0.0;

    const int r = n & 3;
    const int n_main = n - r;

    if (r > 0) {
        double yl[4];
        for (int t = 0; t < r; ++t) yl[t] = y[t];
        for (int t = 0; t < r; ++t) {
            const double* col = A + (size_t)t * ldinc;
            yl[t] += col[0] * x[t];
            for (int s = t + 1; s < r; ++s) {
                const double a = col[s - t];
                yl[t] += a * x[s];
                yl[s] += a * x[t];
            }
        }
        for (int t = 0; t < r; ++t) y[t] = yl[t];

        __m256d xv_col[4], acc[4];
        for (int t = 0; t < r; ++t) {
            xv_col[t] = _mm256_set1_pd(x[t]);
            acc[t] = _mm256_setzero_pd();
        }
        for (int i = 0; i < n_main; i += 4) {
            __m256d yv = _mm256_loadu_pd(y + r + i);
            const __m256d xv_rows = _mm256_loadu_pd(x + r + i);
            for (int t = 0; t < r; ++t) {
                const __m256d av = _mm256_loadu_pd(
                    A + (size_t)t * ldinc + (r - t) + i);
                yv = _mm256_fmadd_pd(av, xv_col[t], yv);
                acc[t] = _mm256_fmadd_pd(av, xv_rows, acc[t]);
            }
            _mm256_storeu_pd(y + r + i, yv);
        }
        for (int t = 0; t < r; ++t)
            y[t] += __ah_hsum256(acc[t]);
    }

    for (int j = r; j + 4 <= n; j += 4) {
        double yl[4];
        for (int t = 0; t < 4; ++t) yl[t] = y[j + t];
        for (int t = 0; t < 4; ++t) {
            const double* col = A + (size_t)(j + t) * ldinc;
            yl[t] += col[0] * x[j + t];
            for (int s = t + 1; s < 4; ++s) {
                const double a = col[s - t];
                yl[t] += a * x[j + s];
                yl[s] += a * x[j + t];
            }
        }
        for (int t = 0; t < 4; ++t) y[j + t] = yl[t];

        const int len = n - j - 4;
        if (len <= 0) continue;

        const __m256d x0 = _mm256_set1_pd(x[j + 0]);
        const __m256d x1 = _mm256_set1_pd(x[j + 1]);
        const __m256d x2 = _mm256_set1_pd(x[j + 2]);
        const __m256d x3 = _mm256_set1_pd(x[j + 3]);

        const double* __restrict c0 = A + (size_t)(j + 0) * ldinc + 4;
        const double* __restrict c1 = A + (size_t)(j + 1) * ldinc + 3;
        const double* __restrict c2 = A + (size_t)(j + 2) * ldinc + 2;
        const double* __restrict c3 = A + (size_t)(j + 3) * ldinc + 1;

        __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd();
        __m256d acc2 = _mm256_setzero_pd(), acc3 = _mm256_setzero_pd();

        for (int i = 0; i < len; i += 4) {
            const __m256d xv = _mm256_loadu_pd(x + j + 4 + i);

            const __m256d a0 = _mm256_loadu_pd(c0 + i);
            const __m256d a1 = _mm256_loadu_pd(c1 + i);
            const __m256d a2 = _mm256_loadu_pd(c2 + i);
            const __m256d a3 = _mm256_loadu_pd(c3 + i);

            __m256d yv0 = _mm256_loadu_pd(y + j + 4 + i);
            __m256d yv1 = _mm256_mul_pd(a2, x2);
            yv0 = _mm256_fmadd_pd(a0, x0, yv0);
            yv1 = _mm256_fmadd_pd(a3, x3, yv1);
            yv0 = _mm256_fmadd_pd(a1, x1, yv0);
            _mm256_storeu_pd(y + j + 4 + i, _mm256_add_pd(yv0, yv1));

            acc0 = _mm256_fmadd_pd(a0, xv, acc0);
            acc1 = _mm256_fmadd_pd(a1, xv, acc1);
            acc2 = _mm256_fmadd_pd(a2, xv, acc2);
            acc3 = _mm256_fmadd_pd(a3, xv, acc3);
        }

        y[j + 0] += __ah_hsum256(acc0);
        y[j + 1] += __ah_hsum256(acc1);
        y[j + 2] += __ah_hsum256(acc2);
        y[j + 3] += __ah_hsum256(acc3);
    }
}

#endif