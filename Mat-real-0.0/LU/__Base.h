
template<typename T>
void __Solve_direct_LX(__BlockMat<T>& L, __BlockMat<T>& X)
{
    constexpr int N = 4;
    constexpr int BS = __BlockSize_LU;

    const int K = L.cols;
    const int n = X.cols;
    const int ldX = X._rows;
    const int ldL = L._rows;

    alignas(64) T buf[BS][N];

    int c = 0;
    for (; c + N <= n; c += N) {
        for (int i = 0; i < K; ++i) {
            for (int r = 0; r < N; ++r)
                buf[i][r] = X.data[(size_t)(c + r) * ldX + i];
        }

        for (int i = 0; i < K; ++i) {
            const T* Li = L.data + i;
            for (int r = 0; r < N; ++r) {
                T acc = buf[i][r];
                for (int j = 0; j < i; ++j)
                    acc -= L.data[(size_t)j * ldL + i] * buf[j][r];
                buf[i][r] = acc;
            }
        }

        for (int i = 0; i < K; ++i) {
            for (int r = 0; r < N; ++r)
                X.data[(size_t)(c + r) * ldX + i] = buf[i][r];
        }
    }

    for (; c < n; ++c) {
        T* x = X[c];
        for (int i = 1; i < K; ++i) {
            const T a = x[i - 1];
            if (a == T(0)) continue;
            const T* Li = L.data + (size_t)(i - 1) * ldL + i;
            const int len = K - i;
            __mixed(x + i, const_cast<T*>(Li), -a, x + i, len);
        }
    }
}

template<typename T>
void __Solve_direct_UX(__BlockMat<T>& U, __BlockMat<T>& X) {
    constexpr int N = 4;
    constexpr int BS = __BlockSize_LU;
    const int K = U.cols;
    const int n = X.cols;
    const int ldX = X._rows;
    const int ldU = U._rows;
    alignas(64) T buf[BS][N];
    int c = 0;
    for (; c + N <= n; c += N) {
        for (int i = 0; i < K; ++i) {
            for (int r = 0; r < N; ++r)
                buf[i][r] = X.data[(size_t)(c + r) * ldX + i];
        }
        for (int i = K - 1; i >= 0; --i) {
            T diag = U.data[(size_t)i * ldU + i];
            for (int r = 0; r < N; ++r)
                buf[i][r] /= diag;
            const T* Ui = U.data + (size_t)i * ldU;
            for (int j = 0; j < i; ++j) {
                T uji = Ui[j];
                __mixed(buf[j], buf[i], -uji, buf[j], N);
            }
        }
        for (int i = 0; i < K; ++i) {
            for (int r = 0; r < N; ++r)
                X.data[(size_t)(c + r) * ldX + i] = buf[i][r];
        }
    }
    for (; c < n; ++c) {
        T* x = X[c];
        for (int i = K - 1; i >= 0; --i) {
            x[i] /= U.data[(size_t)i * ldU + i];
            T xi = x[i];
            const T* Ui = U.data + (size_t)i * ldU;
            __mixed(x, const_cast<T*>(Ui), -xi, x, i);
        }
    }
}


#if defined(__AVX512__)

template<>
void __Solve_direct_LX<double>(__BlockMat<double>& L, __BlockMat<double>& X)
{
    constexpr int N = 8;
    constexpr int BS = __BlockSize_LU;

    const int k = L.cols;
    const int n = X.cols;
    const int ldX = X._rows;
    const int ldL = L._rows;

    alignas(64) double buf[BS][N];

    int c = 0;
    for (; c + N <= n; c += N) {
        int i4 = 0;
        for (; i4 + 4 <= k; i4 += 4) {
            __m256d r0, r1, r2, r3;
            __m256d a0 = _mm256_loadu_pd(X.data + (size_t)(c + 0) * ldX + i4);
            __m256d a1 = _mm256_loadu_pd(X.data + (size_t)(c + 1) * ldX + i4);
            __m256d a2 = _mm256_loadu_pd(X.data + (size_t)(c + 2) * ldX + i4);
            __m256d a3 = _mm256_loadu_pd(X.data + (size_t)(c + 3) * ldX + i4);
            __transpose4x4_pd(a0, a1, a2, a3, r0, r1, r2, r3);
            _mm256_store_pd(&buf[i4 + 0][0], r0);
            _mm256_store_pd(&buf[i4 + 1][0], r1);
            _mm256_store_pd(&buf[i4 + 2][0], r2);
            _mm256_store_pd(&buf[i4 + 3][0], r3);

            a0 = _mm256_loadu_pd(X.data + (size_t)(c + 4) * ldX + i4);
            a1 = _mm256_loadu_pd(X.data + (size_t)(c + 5) * ldX + i4);
            a2 = _mm256_loadu_pd(X.data + (size_t)(c + 6) * ldX + i4);
            a3 = _mm256_loadu_pd(X.data + (size_t)(c + 7) * ldX + i4);
            __transpose4x4_pd(a0, a1, a2, a3, r0, r1, r2, r3);
            _mm256_store_pd(&buf[i4 + 0][4], r0);
            _mm256_store_pd(&buf[i4 + 1][4], r1);
            _mm256_store_pd(&buf[i4 + 2][4], r2);
            _mm256_store_pd(&buf[i4 + 3][4], r3);
        }
        for (int i = i4; i < k; ++i)
            for (int r = 0; r < N; ++r)
                buf[i][r] = X.data[(size_t)(c + r) * ldX + i];

        for (int i = 0; i < k; ++i) {
            const double* Li = L.data + i;
            __m512d s0 = _mm512_load_pd(&buf[i][0]);
            __m512d s1 = _mm512_setzero_pd();
            __m512d s2 = _mm512_setzero_pd();
            __m512d s3 = _mm512_setzero_pd();

            int j = 0;
            for (; j + 4 <= i; j += 4) {
                __m512d l0 = _mm512_set1_pd(Li[(size_t)(j + 0) * ldL]);
                __m512d l1 = _mm512_set1_pd(Li[(size_t)(j + 1) * ldL]);
                __m512d l2 = _mm512_set1_pd(Li[(size_t)(j + 2) * ldL]);
                __m512d l3 = _mm512_set1_pd(Li[(size_t)(j + 3) * ldL]);
                __m512d x0 = _mm512_load_pd(&buf[j + 0][0]);
                __m512d x1 = _mm512_load_pd(&buf[j + 1][0]);
                __m512d x2 = _mm512_load_pd(&buf[j + 2][0]);
                __m512d x3 = _mm512_load_pd(&buf[j + 3][0]);
                s0 = _mm512_fnmadd_pd(l0, x0, s0);
                s1 = _mm512_fnmadd_pd(l1, x1, s1);
                s2 = _mm512_fnmadd_pd(l2, x2, s2);
                s3 = _mm512_fnmadd_pd(l3, x3, s3);
            }
            for (; j < i; ++j) {
                __m512d lv = _mm512_set1_pd(Li[(size_t)j * ldL]);
                __m512d xv = _mm512_load_pd(&buf[j][0]);
                s0 = _mm512_fnmadd_pd(lv, xv, s0);
            }
            s0 = _mm512_add_pd(s0, _mm512_add_pd(s1, _mm512_add_pd(s2, s3)));
            _mm512_store_pd(&buf[i][0], s0);
        }

        i4 = 0;
        for (; i4 + 4 <= k; i4 += 4) {
            __m256d r0, r1, r2, r3, v0, v1, v2, v3;
            r0 = _mm256_load_pd(&buf[i4 + 0][0]);
            r1 = _mm256_load_pd(&buf[i4 + 1][0]);
            r2 = _mm256_load_pd(&buf[i4 + 2][0]);
            r3 = _mm256_load_pd(&buf[i4 + 3][0]);
            __transpose4x4_pd(r0, r1, r2, r3, v0, v1, v2, v3);
            _mm256_storeu_pd(X.data + (size_t)(c + 0) * ldX + i4, v0);
            _mm256_storeu_pd(X.data + (size_t)(c + 1) * ldX + i4, v1);
            _mm256_storeu_pd(X.data + (size_t)(c + 2) * ldX + i4, v2);
            _mm256_storeu_pd(X.data + (size_t)(c + 3) * ldX + i4, v3);

            r0 = _mm256_load_pd(&buf[i4 + 0][4]);
            r1 = _mm256_load_pd(&buf[i4 + 1][4]);
            r2 = _mm256_load_pd(&buf[i4 + 2][4]);
            r3 = _mm256_load_pd(&buf[i4 + 3][4]);
            __transpose4x4_pd(r0, r1, r2, r3, v0, v1, v2, v3);
            _mm256_storeu_pd(X.data + (size_t)(c + 4) * ldX + i4, v0);
            _mm256_storeu_pd(X.data + (size_t)(c + 5) * ldX + i4, v1);
            _mm256_storeu_pd(X.data + (size_t)(c + 6) * ldX + i4, v2);
            _mm256_storeu_pd(X.data + (size_t)(c + 7) * ldX + i4, v3);
        }
        for (int i = i4; i < k; ++i)
            for (int r = 0; r < N; ++r)
                X.data[(size_t)(c + r) * ldX + i] = buf[i][r];
    }

    for (; c < n; ++c) {
        double* x = X[c];
        for (int i = 1; i < k; ++i) {
            const double a = x[i - 1];
            if (a == 0.0) continue;
            double* xi = x + i;
            const double* Li = L.data + (size_t)(i - 1) * ldL + i;
            const int len = k - i;
            __mixed(xi, const_cast<double*>(Li), -a, xi, len);
        }
    }
}

template<>
void __Solve_direct_UX<double>(__BlockMat<double>& U, __BlockMat<double>& X) {
    constexpr int N = 8;
    constexpr int BS = __BlockSize_LU;
    const int k = U.cols;
    const int n = X.cols;
    const int ldX = X._rows;
    const int ldU = U._rows;
    alignas(64) double buf[BS][N];
    int c = 0;
    for (; c + N <= n; c += N) {
        int i4 = 0;
        for (; i4 + 4 <= k; i4 += 4) {
            __m256d r0, r1, r2, r3;
            __m256d a0 = _mm256_loadu_pd(X.data + (size_t)(c + 0) * ldX + i4);
            __m256d a1 = _mm256_loadu_pd(X.data + (size_t)(c + 1) * ldX + i4);
            __m256d a2 = _mm256_loadu_pd(X.data + (size_t)(c + 2) * ldX + i4);
            __m256d a3 = _mm256_loadu_pd(X.data + (size_t)(c + 3) * ldX + i4);
            __transpose4x4_pd(a0, a1, a2, a3, r0, r1, r2, r3);
            _mm256_store_pd(&buf[i4 + 0][0], r0);
            _mm256_store_pd(&buf[i4 + 1][0], r1);
            _mm256_store_pd(&buf[i4 + 2][0], r2);
            _mm256_store_pd(&buf[i4 + 3][0], r3);
            a0 = _mm256_loadu_pd(X.data + (size_t)(c + 4) * ldX + i4);
            a1 = _mm256_loadu_pd(X.data + (size_t)(c + 5) * ldX + i4);
            a2 = _mm256_loadu_pd(X.data + (size_t)(c + 6) * ldX + i4);
            a3 = _mm256_loadu_pd(X.data + (size_t)(c + 7) * ldX + i4);
            __transpose4x4_pd(a0, a1, a2, a3, r0, r1, r2, r3);
            _mm256_store_pd(&buf[i4 + 0][4], r0);
            _mm256_store_pd(&buf[i4 + 1][4], r1);
            _mm256_store_pd(&buf[i4 + 2][4], r2);
            _mm256_store_pd(&buf[i4 + 3][4], r3);
        }
        for (int i = i4; i < k; ++i)
            for (int r = 0; r < N; ++r)
                buf[i][r] = X.data[(size_t)(c + r) * ldX + i];
        for (int i = k - 1; i >= 0; --i) {
            __m512d diag = _mm512_set1_pd(U.data[(size_t)i * ldU + i]);
            __m512d vi = _mm512_load_pd(&buf[i][0]);
            vi = _mm512_div_pd(vi, diag);
            _mm512_store_pd(&buf[i][0], vi);
            const double* Ui = U.data + (size_t)i * ldU;
            for (int j = 0; j < i; ++j) {
                __m512d uji = _mm512_set1_pd(Ui[j]);
                __m512d vj = _mm512_load_pd(&buf[j][0]);
                vj = _mm512_fnmadd_pd(uji, vi, vj);
                _mm512_store_pd(&buf[j][0], vj);
            }
        }
        i4 = 0;
        for (; i4 + 4 <= k; i4 += 4) {
            __m256d r0, r1, r2, r3, v0, v1, v2, v3;
            r0 = _mm256_load_pd(&buf[i4 + 0][0]);
            r1 = _mm256_load_pd(&buf[i4 + 1][0]);
            r2 = _mm256_load_pd(&buf[i4 + 2][0]);
            r3 = _mm256_load_pd(&buf[i4 + 3][0]);
            __transpose4x4_pd(r0, r1, r2, r3, v0, v1, v2, v3);
            _mm256_storeu_pd(X.data + (size_t)(c + 0) * ldX + i4, v0);
            _mm256_storeu_pd(X.data + (size_t)(c + 1) * ldX + i4, v1);
            _mm256_storeu_pd(X.data + (size_t)(c + 2) * ldX + i4, v2);
            _mm256_storeu_pd(X.data + (size_t)(c + 3) * ldX + i4, v3);
            r0 = _mm256_load_pd(&buf[i4 + 0][4]);
            r1 = _mm256_load_pd(&buf[i4 + 1][4]);
            r2 = _mm256_load_pd(&buf[i4 + 2][4]);
            r3 = _mm256_load_pd(&buf[i4 + 3][4]);
            __transpose4x4_pd(r0, r1, r2, r3, v0, v1, v2, v3);
            _mm256_storeu_pd(X.data + (size_t)(c + 4) * ldX + i4, v0);
            _mm256_storeu_pd(X.data + (size_t)(c + 5) * ldX + i4, v1);
            _mm256_storeu_pd(X.data + (size_t)(c + 6) * ldX + i4, v2);
            _mm256_storeu_pd(X.data + (size_t)(c + 7) * ldX + i4, v3);
        }
        for (int i = i4; i < k; ++i)
            for (int r = 0; r < N; ++r)
                X.data[(size_t)(c + r) * ldX + i] = buf[i][r];
    }
    for (; c < n; ++c) {
        double* x = X[c];
        for (int i = k - 1; i >= 0; --i) {
            x[i] /= U.data[(size_t)i * ldU + i];
            double xi = x[i];
            const double* Ui = U.data + (size_t)i * ldU;
            __mixed(x, const_cast<double*>(Ui), -xi, x, i);
        }
    }
}


#elif defined(__AVX2__)

template<>
void __Solve_direct_LX<double>(__BlockMat<double>& L, __BlockMat<double>& X)
{
    constexpr int N = 4;
    constexpr int BS = __BlockSize_LU;

    const int k = L.cols;
    const int n = X.cols;
    const int ldX = X._rows;
    const int ldL = L._rows;

    alignas(64) double buf[BS][N];

    int c = 0;
    for (; c + N <= n; c += N) {
        int i4 = 0;
        for (; i4 + 4 <= k; i4 += 4) {
            __m256d v0 = _mm256_loadu_pd(X.data + (size_t)(c + 0) * ldX + i4);
            __m256d v1 = _mm256_loadu_pd(X.data + (size_t)(c + 1) * ldX + i4);
            __m256d v2 = _mm256_loadu_pd(X.data + (size_t)(c + 2) * ldX + i4);
            __m256d v3 = _mm256_loadu_pd(X.data + (size_t)(c + 3) * ldX + i4);
            __m256d r0, r1, r2, r3;
            __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);
            _mm256_store_pd(&buf[i4 + 0][0], r0);
            _mm256_store_pd(&buf[i4 + 1][0], r1);
            _mm256_store_pd(&buf[i4 + 2][0], r2);
            _mm256_store_pd(&buf[i4 + 3][0], r3);
        }
        for (int i = i4; i < k; ++i)
            for (int r = 0; r < N; ++r)
                buf[i][r] = X.data[(size_t)(c + r) * ldX + i];

        for (int i = 0; i < k; ++i) {
            const double* Li = L.data + i;
            __m256d a0 = _mm256_load_pd(&buf[i][0]);
            __m256d a1 = _mm256_setzero_pd();
            __m256d a2 = _mm256_setzero_pd();
            __m256d a3 = _mm256_setzero_pd();

            int j = 0;
            for (; j + 4 <= i; j += 4) {
                __m256d l0 = _mm256_set1_pd(Li[(size_t)(j + 0) * ldL]);
                __m256d l1 = _mm256_set1_pd(Li[(size_t)(j + 1) * ldL]);
                __m256d l2 = _mm256_set1_pd(Li[(size_t)(j + 2) * ldL]);
                __m256d l3 = _mm256_set1_pd(Li[(size_t)(j + 3) * ldL]);
                __m256d x0 = _mm256_load_pd(&buf[j + 0][0]);
                __m256d x1 = _mm256_load_pd(&buf[j + 1][0]);
                __m256d x2 = _mm256_load_pd(&buf[j + 2][0]);
                __m256d x3 = _mm256_load_pd(&buf[j + 3][0]);
                a0 = _mm256_fnmadd_pd(l0, x0, a0);
                a1 = _mm256_fnmadd_pd(l1, x1, a1);
                a2 = _mm256_fnmadd_pd(l2, x2, a2);
                a3 = _mm256_fnmadd_pd(l3, x3, a3);
            }
            for (; j < i; ++j) {
                __m256d lv = _mm256_set1_pd(Li[(size_t)j * ldL]);
                __m256d xv = _mm256_load_pd(&buf[j][0]);
                a0 = _mm256_fnmadd_pd(lv, xv, a0);
            }
            a0 = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
            _mm256_store_pd(&buf[i][0], a0);
        }

        i4 = 0;
        for (; i4 + 4 <= k; i4 += 4) {
            __m256d r0 = _mm256_load_pd(&buf[i4 + 0][0]);
            __m256d r1 = _mm256_load_pd(&buf[i4 + 1][0]);
            __m256d r2 = _mm256_load_pd(&buf[i4 + 2][0]);
            __m256d r3 = _mm256_load_pd(&buf[i4 + 3][0]);
            __m256d v0, v1, v2, v3;
            __transpose4x4_pd(r0, r1, r2, r3, v0, v1, v2, v3);
            _mm256_storeu_pd(X.data + (size_t)(c + 0) * ldX + i4, v0);
            _mm256_storeu_pd(X.data + (size_t)(c + 1) * ldX + i4, v1);
            _mm256_storeu_pd(X.data + (size_t)(c + 2) * ldX + i4, v2);
            _mm256_storeu_pd(X.data + (size_t)(c + 3) * ldX + i4, v3);
        }
        for (int i = i4; i < k; ++i)
            for (int r = 0; r < N; ++r)
                X.data[(size_t)(c + r) * ldX + i] = buf[i][r];
    }

    for (; c < n; ++c) {
        double* x = X[c];
        for (int i = 1; i < k; ++i) {
            const double a = x[i - 1];
            if (a == 0.0) continue;
            double* xi = x + i;
            const double* Li = L.data + (size_t)(i - 1) * ldL + i;
            const int len = k - i;
            __mixed(xi, const_cast<double*>(Li), -a, xi, len);
        }
    }
}

template<>
void __Solve_direct_UX<double>(__BlockMat<double>& U, __BlockMat<double>& X) {
    constexpr int N = 4;
    constexpr int BS = __BlockSize_LU;
    const int k = U.cols;
    const int n = X.cols;
    const int ldX = X._rows;
    const int ldU = U._rows;
    alignas(64) double buf[BS][N];
    int c = 0;
    for (; c + N <= n; c += N) {
        int i4 = 0;
        for (; i4 + 4 <= k; i4 += 4) {
            __m256d v0 = _mm256_loadu_pd(X.data + (size_t)(c + 0) * ldX + i4);
            __m256d v1 = _mm256_loadu_pd(X.data + (size_t)(c + 1) * ldX + i4);
            __m256d v2 = _mm256_loadu_pd(X.data + (size_t)(c + 2) * ldX + i4);
            __m256d v3 = _mm256_loadu_pd(X.data + (size_t)(c + 3) * ldX + i4);
            __m256d r0, r1, r2, r3;
            __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);
            _mm256_store_pd(&buf[i4 + 0][0], r0);
            _mm256_store_pd(&buf[i4 + 1][0], r1);
            _mm256_store_pd(&buf[i4 + 2][0], r2);
            _mm256_store_pd(&buf[i4 + 3][0], r3);
        }
        for (int i = i4; i < k; ++i)
            for (int r = 0; r < N; ++r)
                buf[i][r] = X.data[(size_t)(c + r) * ldX + i];
        for (int i = k - 1; i >= 0; --i) {
            __m256d diag = _mm256_set1_pd(U.data[(size_t)i * ldU + i]);
            __m256d vi = _mm256_load_pd(&buf[i][0]);
            vi = _mm256_div_pd(vi, diag);
            _mm256_store_pd(&buf[i][0], vi);
            const double* Ui = U.data + (size_t)i * ldU;
            for (int j = 0; j < i; ++j) {
                __m256d uji = _mm256_set1_pd(Ui[j]);
                __m256d vj = _mm256_load_pd(&buf[j][0]);
                vj = _mm256_fnmadd_pd(uji, vi, vj);
                _mm256_store_pd(&buf[j][0], vj);
            }
        }
        i4 = 0;
        for (; i4 + 4 <= k; i4 += 4) {
            __m256d r0 = _mm256_load_pd(&buf[i4 + 0][0]);
            __m256d r1 = _mm256_load_pd(&buf[i4 + 1][0]);
            __m256d r2 = _mm256_load_pd(&buf[i4 + 2][0]);
            __m256d r3 = _mm256_load_pd(&buf[i4 + 3][0]);
            __m256d v0, v1, v2, v3;
            __transpose4x4_pd(r0, r1, r2, r3, v0, v1, v2, v3);
            _mm256_storeu_pd(X.data + (size_t)(c + 0) * ldX + i4, v0);
            _mm256_storeu_pd(X.data + (size_t)(c + 1) * ldX + i4, v1);
            _mm256_storeu_pd(X.data + (size_t)(c + 2) * ldX + i4, v2);
            _mm256_storeu_pd(X.data + (size_t)(c + 3) * ldX + i4, v3);
        }
        for (int i = i4; i < k; ++i)
            for (int r = 0; r < N; ++r)
                X.data[(size_t)(c + r) * ldX + i] = buf[i][r];
    }
    for (; c < n; ++c) {
        double* x = X[c];
        for (int i = k - 1; i >= 0; --i) {
            x[i] /= U.data[(size_t)i * ldU + i];
            double xi = x[i];
            const double* Ui = U.data + (size_t)i * ldU;
            __mixed(x, const_cast<double*>(Ui), -xi, x, i);
        }
    }
}

#endif









template<typename T>
void __Solve_direct_XU(__BlockMat<T>& U, __BlockMat<T>& X) {
    constexpr int RB = 4;
    const int n = X.cols;
    const int len = X.rows;
    int r = 0;
    for (; r + RB <= n; r += RB) {
        for (int j = r; j < r + RB; ++j)
            for (int i = j + 1; i < r + RB; ++i)
                __mixed(X[i], X[j], -U(j, i), X[i], len);
        for (int i = r + RB; i < n; ++i) {
            T* __restrict Xi = X[i];
            const T* __restrict X0 = X[r + 0];
            const T* __restrict X1 = X[r + 1];
            const T* __restrict X2 = X[r + 2];
            const T* __restrict X3 = X[r + 3];
            const T q0 = -U(r + 0, i);
            const T q1 = -U(r + 1, i);
            const T q2 = -U(r + 2, i);
            const T q3 = -U(r + 3, i);
            for (int k = 0; k < len; ++k) {
                T a0 = q0 * X0[k];
                T a1 = q1 * X1[k];
                T a2 = q2 * X2[k];
                T a3 = q3 * X3[k];
                Xi[k] += (a0 + a1) + (a2 + a3);
            }
        }
    }
    for (; r < n; ++r)
        for (int i = r + 1; i < n; ++i)
            __mixed(X[i], X[r], -U(r, i), X[i], len);
}

template<typename T>
void __Solve_direct_XL(__BlockMat<T>& L, __BlockMat<T>& X) {
    constexpr int RB = 4;
    const int n = X.cols;
    const int len = X.rows;
    int r = n - 1;
    for (; r - RB + 1 >= 1; r -= RB) {
        for (int j = r; j > r - RB + 1; --j)
            for (int i = r - RB + 1; i < j; ++i)
                __mixed(X[i], X[j], -L(j, i), X[i], len);
        for (int i = 0; i < r - RB + 1; ++i) {
            T* __restrict Xi = X[i];
            const T* __restrict X0 = X[r - 0];
            const T* __restrict X1 = X[r - 1];
            const T* __restrict X2 = X[r - 2];
            const T* __restrict X3 = X[r - 3];
            const T q0 = -L(r - 0, i);
            const T q1 = -L(r - 1, i);
            const T q2 = -L(r - 2, i);
            const T q3 = -L(r - 3, i);
            for (int k = 0; k < len; ++k) {
                T a0 = q0 * X0[k];
                T a1 = q1 * X1[k];
                T a2 = q2 * X2[k];
                T a3 = q3 * X3[k];
                Xi[k] += (a0 + a1) + (a2 + a3);
            }
        }
    }
    for (; r > 0; --r)
        for (int i = 0; i < r; ++i)
            __mixed(X[i], X[r], -L(r, i), X[i], len);
}

// ==================== 板块2：AVX512 特化（RB = 8，树形归约） ====================

#if defined(__AVX512__)

template<>
void __Solve_direct_XU<double>(__BlockMat<double>& U, __BlockMat<double>& X) {
    constexpr int RB = 8;
    const int n = X.cols;
    const int len = X.rows;
    int r = 0;
    for (; r + RB <= n; r += RB) {
        for (int j = r; j < r + RB; ++j)
            for (int i = j + 1; i < r + RB; ++i)
                __mixed(X[i], X[j], -U(j, i), X[i], len);
        for (int i = r + RB; i < n; ++i) {
            double* __restrict Xi = X[i];
            const double* __restrict X0 = X[r + 0];
            const double* __restrict X1 = X[r + 1];
            const double* __restrict X2 = X[r + 2];
            const double* __restrict X3 = X[r + 3];
            const double* __restrict X4 = X[r + 4];
            const double* __restrict X5 = X[r + 5];
            const double* __restrict X6 = X[r + 6];
            const double* __restrict X7 = X[r + 7];
            const __m512d vq0 = _mm512_set1_pd(-U(r + 0, i));
            const __m512d vq1 = _mm512_set1_pd(-U(r + 1, i));
            const __m512d vq2 = _mm512_set1_pd(-U(r + 2, i));
            const __m512d vq3 = _mm512_set1_pd(-U(r + 3, i));
            const __m512d vq4 = _mm512_set1_pd(-U(r + 4, i));
            const __m512d vq5 = _mm512_set1_pd(-U(r + 5, i));
            const __m512d vq6 = _mm512_set1_pd(-U(r + 6, i));
            const __m512d vq7 = _mm512_set1_pd(-U(r + 7, i));
            int k = 0;
            for (; k + 8 <= len; k += 8) {
                __m512d p0 = _mm512_mul_pd(_mm512_loadu_pd(X0 + k), vq0);
                __m512d p1 = _mm512_mul_pd(_mm512_loadu_pd(X1 + k), vq1);
                __m512d p2 = _mm512_mul_pd(_mm512_loadu_pd(X2 + k), vq2);
                __m512d p3 = _mm512_mul_pd(_mm512_loadu_pd(X3 + k), vq3);
                __m512d p4 = _mm512_mul_pd(_mm512_loadu_pd(X4 + k), vq4);
                __m512d p5 = _mm512_mul_pd(_mm512_loadu_pd(X5 + k), vq5);
                __m512d p6 = _mm512_mul_pd(_mm512_loadu_pd(X6 + k), vq6);
                __m512d p7 = _mm512_mul_pd(_mm512_loadu_pd(X7 + k), vq7);
                __m512d e0 = _mm512_add_pd(p0, p1);
                __m512d e1 = _mm512_add_pd(p2, p3);
                __m512d e2 = _mm512_add_pd(p4, p5);
                __m512d e3 = _mm512_add_pd(p6, p7);
                __m512d d0 = _mm512_add_pd(e0, e1);
                __m512d d1 = _mm512_add_pd(e2, e3);
                __m512d s = _mm512_add_pd(d0, d1);
                _mm512_storeu_pd(Xi + k, _mm512_add_pd(_mm512_loadu_pd(Xi + k), s));
            }
            for (; k < len; ++k) {
                Xi[k] += (-U(r + 0, i)) * X0[k] + (-U(r + 1, i)) * X1[k]
                    + (-U(r + 2, i)) * X2[k] + (-U(r + 3, i)) * X3[k]
                    + (-U(r + 4, i)) * X4[k] + (-U(r + 5, i)) * X5[k]
                    + (-U(r + 6, i)) * X6[k] + (-U(r + 7, i)) * X7[k];
            }
        }
    }
    for (; r < n; ++r)
        for (int i = r + 1; i < n; ++i)
            __mixed(X[i], X[r], -U(r, i), X[i], len);
}

template<>
void __Solve_direct_XL<double>(__BlockMat<double>& L, __BlockMat<double>& X) {
    constexpr int RB = 8;
    const int n = X.cols;
    const int len = X.rows;
    int r = n - 1;
    for (; r - RB + 1 >= 1; r -= RB) {
        for (int j = r; j > r - RB + 1; --j)
            for (int i = r - RB + 1; i < j; ++i)
                __mixed(X[i], X[j], -L(j, i), X[i], len);
        for (int i = 0; i < r - RB + 1; ++i) {
            double* __restrict Xi = X[i];
            const double* __restrict X0 = X[r - 0];
            const double* __restrict X1 = X[r - 1];
            const double* __restrict X2 = X[r - 2];
            const double* __restrict X3 = X[r - 3];
            const double* __restrict X4 = X[r - 4];
            const double* __restrict X5 = X[r - 5];
            const double* __restrict X6 = X[r - 6];
            const double* __restrict X7 = X[r - 7];
            const __m512d vq0 = _mm512_set1_pd(-L(r - 0, i));
            const __m512d vq1 = _mm512_set1_pd(-L(r - 1, i));
            const __m512d vq2 = _mm512_set1_pd(-L(r - 2, i));
            const __m512d vq3 = _mm512_set1_pd(-L(r - 3, i));
            const __m512d vq4 = _mm512_set1_pd(-L(r - 4, i));
            const __m512d vq5 = _mm512_set1_pd(-L(r - 5, i));
            const __m512d vq6 = _mm512_set1_pd(-L(r - 6, i));
            const __m512d vq7 = _mm512_set1_pd(-L(r - 7, i));
            int k = 0;
            for (; k + 8 <= len; k += 8) {
                __m512d p0 = _mm512_mul_pd(_mm512_loadu_pd(X0 + k), vq0);
                __m512d p1 = _mm512_mul_pd(_mm512_loadu_pd(X1 + k), vq1);
                __m512d p2 = _mm512_mul_pd(_mm512_loadu_pd(X2 + k), vq2);
                __m512d p3 = _mm512_mul_pd(_mm512_loadu_pd(X3 + k), vq3);
                __m512d p4 = _mm512_mul_pd(_mm512_loadu_pd(X4 + k), vq4);
                __m512d p5 = _mm512_mul_pd(_mm512_loadu_pd(X5 + k), vq5);
                __m512d p6 = _mm512_mul_pd(_mm512_loadu_pd(X6 + k), vq6);
                __m512d p7 = _mm512_mul_pd(_mm512_loadu_pd(X7 + k), vq7);
                __m512d e0 = _mm512_add_pd(p0, p1);
                __m512d e1 = _mm512_add_pd(p2, p3);
                __m512d e2 = _mm512_add_pd(p4, p5);
                __m512d e3 = _mm512_add_pd(p6, p7);
                __m512d d0 = _mm512_add_pd(e0, e1);
                __m512d d1 = _mm512_add_pd(e2, e3);
                __m512d s = _mm512_add_pd(d0, d1);
                _mm512_storeu_pd(Xi + k, _mm512_add_pd(_mm512_loadu_pd(Xi + k), s));
            }
            for (; k < len; ++k) {
                Xi[k] += (-L(r - 0, i)) * X0[k] + (-L(r - 1, i)) * X1[k]
                    + (-L(r - 2, i)) * X2[k] + (-L(r - 3, i)) * X3[k]
                    + (-L(r - 4, i)) * X4[k] + (-L(r - 5, i)) * X5[k]
                    + (-L(r - 6, i)) * X6[k] + (-L(r - 7, i)) * X7[k];
            }
        }
    }
    for (; r > 0; --r)
        for (int i = 0; i < r; ++i)
            __mixed(X[i], X[r], -L(r, i), X[i], len);
}

// ==================== 板块3：AVX2 特化（RB = 8，树形归约） ====================

#elif defined(__AVX2__)

template<>
void __Solve_direct_XU<double>(__BlockMat<double>& U, __BlockMat<double>& X) {
    constexpr int RB = 8;
    const int n = X.cols;
    const int len = X.rows;
    int r = 0;
    for (; r + RB <= n; r += RB) {
        for (int j = r; j < r + RB; ++j)
            for (int i = j + 1; i < r + RB; ++i)
                __mixed(X[i], X[j], -U(j, i), X[i], len);
        for (int i = r + RB; i < n; ++i) {
            double* __restrict Xi = X[i];
            const double* __restrict X0 = X[r + 0];
            const double* __restrict X1 = X[r + 1];
            const double* __restrict X2 = X[r + 2];
            const double* __restrict X3 = X[r + 3];
            const double* __restrict X4 = X[r + 4];
            const double* __restrict X5 = X[r + 5];
            const double* __restrict X6 = X[r + 6];
            const double* __restrict X7 = X[r + 7];
            const __m256d vq0 = _mm256_set1_pd(-U(r + 0, i));
            const __m256d vq1 = _mm256_set1_pd(-U(r + 1, i));
            const __m256d vq2 = _mm256_set1_pd(-U(r + 2, i));
            const __m256d vq3 = _mm256_set1_pd(-U(r + 3, i));
            const __m256d vq4 = _mm256_set1_pd(-U(r + 4, i));
            const __m256d vq5 = _mm256_set1_pd(-U(r + 5, i));
            const __m256d vq6 = _mm256_set1_pd(-U(r + 6, i));
            const __m256d vq7 = _mm256_set1_pd(-U(r + 7, i));
            int k = 0;
            for (; k + 4 <= len; k += 4) {
                __m256d p0 = _mm256_mul_pd(_mm256_loadu_pd(X0 + k), vq0);
                __m256d p1 = _mm256_mul_pd(_mm256_loadu_pd(X1 + k), vq1);
                __m256d p2 = _mm256_mul_pd(_mm256_loadu_pd(X2 + k), vq2);
                __m256d p3 = _mm256_mul_pd(_mm256_loadu_pd(X3 + k), vq3);
                __m256d p4 = _mm256_mul_pd(_mm256_loadu_pd(X4 + k), vq4);
                __m256d p5 = _mm256_mul_pd(_mm256_loadu_pd(X5 + k), vq5);
                __m256d p6 = _mm256_mul_pd(_mm256_loadu_pd(X6 + k), vq6);
                __m256d p7 = _mm256_mul_pd(_mm256_loadu_pd(X7 + k), vq7);
                __m256d e0 = _mm256_add_pd(p0, p1);
                __m256d e1 = _mm256_add_pd(p2, p3);
                __m256d e2 = _mm256_add_pd(p4, p5);
                __m256d e3 = _mm256_add_pd(p6, p7);
                __m256d d0 = _mm256_add_pd(e0, e1);
                __m256d d1 = _mm256_add_pd(e2, e3);
                __m256d s = _mm256_add_pd(d0, d1);
                _mm256_storeu_pd(Xi + k, _mm256_add_pd(_mm256_loadu_pd(Xi + k), s));
            }
            for (; k < len; ++k) {
                Xi[k] += (-U(r + 0, i)) * X0[k] + (-U(r + 1, i)) * X1[k]
                    + (-U(r + 2, i)) * X2[k] + (-U(r + 3, i)) * X3[k]
                    + (-U(r + 4, i)) * X4[k] + (-U(r + 5, i)) * X5[k]
                    + (-U(r + 6, i)) * X6[k] + (-U(r + 7, i)) * X7[k];
            }
        }
    }
    for (; r < n; ++r)
        for (int i = r + 1; i < n; ++i)
            __mixed(X[i], X[r], -U(r, i), X[i], len);
}

template<>
void __Solve_direct_XL<double>(__BlockMat<double>& L, __BlockMat<double>& X) {
    constexpr int RB = 8;
    const int n = X.cols;
    const int len = X.rows;
    int r = n - 1;
    for (; r - RB + 1 >= 1; r -= RB) {
        for (int j = r; j > r - RB + 1; --j)
            for (int i = r - RB + 1; i < j; ++i)
                __mixed(X[i], X[j], -L(j, i), X[i], len);
        for (int i = 0; i < r - RB + 1; ++i) {
            double* __restrict Xi = X[i];
            const double* __restrict X0 = X[r - 0];
            const double* __restrict X1 = X[r - 1];
            const double* __restrict X2 = X[r - 2];
            const double* __restrict X3 = X[r - 3];
            const double* __restrict X4 = X[r - 4];
            const double* __restrict X5 = X[r - 5];
            const double* __restrict X6 = X[r - 6];
            const double* __restrict X7 = X[r - 7];
            const __m256d vq0 = _mm256_set1_pd(-L(r - 0, i));
            const __m256d vq1 = _mm256_set1_pd(-L(r - 1, i));
            const __m256d vq2 = _mm256_set1_pd(-L(r - 2, i));
            const __m256d vq3 = _mm256_set1_pd(-L(r - 3, i));
            const __m256d vq4 = _mm256_set1_pd(-L(r - 4, i));
            const __m256d vq5 = _mm256_set1_pd(-L(r - 5, i));
            const __m256d vq6 = _mm256_set1_pd(-L(r - 6, i));
            const __m256d vq7 = _mm256_set1_pd(-L(r - 7, i));
            int k = 0;
            for (; k + 4 <= len; k += 4) {
                __m256d p0 = _mm256_mul_pd(_mm256_loadu_pd(X0 + k), vq0);
                __m256d p1 = _mm256_mul_pd(_mm256_loadu_pd(X1 + k), vq1);
                __m256d p2 = _mm256_mul_pd(_mm256_loadu_pd(X2 + k), vq2);
                __m256d p3 = _mm256_mul_pd(_mm256_loadu_pd(X3 + k), vq3);
                __m256d p4 = _mm256_mul_pd(_mm256_loadu_pd(X4 + k), vq4);
                __m256d p5 = _mm256_mul_pd(_mm256_loadu_pd(X5 + k), vq5);
                __m256d p6 = _mm256_mul_pd(_mm256_loadu_pd(X6 + k), vq6);
                __m256d p7 = _mm256_mul_pd(_mm256_loadu_pd(X7 + k), vq7);
                __m256d e0 = _mm256_add_pd(p0, p1);
                __m256d e1 = _mm256_add_pd(p2, p3);
                __m256d e2 = _mm256_add_pd(p4, p5);
                __m256d e3 = _mm256_add_pd(p6, p7);
                __m256d d0 = _mm256_add_pd(e0, e1);
                __m256d d1 = _mm256_add_pd(e2, e3);
                __m256d s = _mm256_add_pd(d0, d1);
                _mm256_storeu_pd(Xi + k, _mm256_add_pd(_mm256_loadu_pd(Xi + k), s));
            }
            for (; k < len; ++k) {
                Xi[k] += (-L(r - 0, i)) * X0[k] + (-L(r - 1, i)) * X1[k]
                    + (-L(r - 2, i)) * X2[k] + (-L(r - 3, i)) * X3[k]
                    + (-L(r - 4, i)) * X4[k] + (-L(r - 5, i)) * X5[k]
                    + (-L(r - 6, i)) * X6[k] + (-L(r - 7, i)) * X7[k];
            }
        }
    }
    for (; r > 0; --r)
        for (int i = 0; i < r; ++i)
            __mixed(X[i], X[r], -L(r, i), X[i], len);
}

#endif