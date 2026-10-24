
template <typename t>
static void __LLT2(__BlockMat<t>& L) {
    t temp;
    int i, j;
    for (j = 0; j < L.cols; ++j) {
        temp = (t)1 / sqrt(L(j, j));
        __times(L[j] + j, temp, L[j] + j, L.rows - j);
        for (i = j + 1; i < L.rows; ++i) {
            __mixed(L[i] + i, L[j] + i, -L(i, j), L[i] + i, L.rows - i);
        }
    }
}

template <typename t>
static void __LDLT2(__BlockMat<t>& L, t* d) {
    t temp;
    int i, j;
    for (j = 0; j < L.cols; ++j) {
        d[j] = L(j, j);
        temp = (t)1 / d[j];
        __times(L[j] + j, temp, L[j] + j, L.rows - j);
        for (i = j + 1; i < L.rows; ++i) {
            __mixed(L[i] + i, L[j] + i, -L(i, j) * d[j], L[i] + i, L.rows - i);
        }
    }
}

template<typename T>
static void __Solve_direct_XLT(__BlockMat<T>& L, __BlockMat<T>& X) {
    T* x;
    int j, k;
    for (j = 0; j < X.cols; ++j) {
        x = X[j];
        __times(x, (T)1 / L(j, j), x, X.rows);
        for (k = j + 1; k < X.cols; ++k) {
            __mixed(X[k], x, -L(k, j), X[k], X.rows);
        }
    }
}

template<typename T>
static void __Solve_direct_XDLT(__BlockMat<T>& L, __BlockMat<T>& X, T* d) {
    T* x;
    int j, k;
    for (j = 0; j < X.cols; ++j) {
        x = X[j];
        __times(x, (T)1 / L(j, j), x, X.rows);
        for (k = j + 1; k < X.cols; ++k) {
            __mixed(X[k], x, -L(k, j), X[k], X.rows);
        }
    }
    for (j = 0; j < X.cols; ++j) {
        __times(X[j], (T)1 / d[j], X[j], X.rows);
    }
}

template<typename t>
static void __times_A_BT_LLT(
    const t* __restrict A_pack,
    const t* __restrict B_pack,
    __BlockMat<t>& C,
    int K)
{
    constexpr int MR = __micro_kernel_mr<t, t>();
    constexpr int NR = __micro_kernel_nr<t, t>();

    const int M = C.rows;
    const int N = C.cols;

    if (K == 0 || M == 0 || N == 0) return;

    const int mr_panels = (M + MR - 1) / MR;
    const int nr_panels = (N + NR - 1) / NR;

    for (int jp = 0; jp < nr_panels; ++jp) {
        const t* Bp = B_pack + (size_t)jp * K * NR;
        const int nn = (N - jp * NR < NR) ? (N - jp * NR) : NR;

        for (int ip = 0; ip < mr_panels; ++ip) {
            const t* Ap = A_pack + (size_t)ip * K * MR;
            const int mm = (M - ip * MR < MR) ? (M - ip * MR) : MR;

            t* Cp = C[jp * NR] + ip * MR;
            if (mm == MR && nn == NR)
                __micro_kernel_nmul_packed_full(Ap, Bp, Cp, C._rows, K);
            else
                __micro_kernel_nmul_packed_edge(Ap, Bp, Cp, C._rows, K, mm, nn);
        }
    }
}

template <typename t>
static void __LLT(MAT<t>& L, t* __restrict A_pack, t* __restrict B_pack) {
    int i, j, k;
    int m, n, l;
    constexpr int BS = __BlockSize_LLT;
    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int S = 2 * BS;
    __BlockMat<t> _L, _L1, _L2;

    if (L.cols <= S) {
        _L.set(L, 0, 0, L.cols, L.cols);
        __LLT2(_L);

        for (i = 0; i < L.cols; ++i) {
            __set0(L[i], i);
        }
        return;
    }

    for (j = 0; j < L.cols; j += BS) {
        n = (j + BS > L.cols) ? (L.cols - j) : BS;

        _L.set(L, j, j, n, n);
        __LLT2(_L);

        if (j + BS < L.cols) {
            for (i = j + BS; i < L.cols; i += BS) {
                m = (i + BS > L.cols) ? (L.cols - i) : BS;
                _L1.set(L, i, j, m, BS);
                __Solve_direct_XLT(_L, _L1);
            }

            _L.set(L, j + BS, j, L.rows - (j + BS), BS);
            __pack_A_block<t, t>(_L.data, _L._rows, _L.rows, BS, A_pack);

            for (i = j + BS; i < L.cols; i += BS) {
                m = (i + BS > L.cols) ? (L.cols - i) : BS;

                _L1.set(L, i, j, m, BS);
                __pack_BT_block<t, t>(_L1.data, _L1._rows, BS, m, B_pack);

                for (k = i; k < L.cols; k += MC) {
                    l = (k + MC > L.cols) ? (L.cols - k) : MC;

                    _L.set(L, k, i, l, m);

                    constexpr int MR = __micro_kernel_mr<t, t>();
                    const t* A_base = A_pack + (size_t)((k - (j + BS)) / MR) * BS * MR;
                    __times_A_BT_LLT<t>(A_base, B_pack, _L, BS);
                }
            }
        }
    }

    for (i = 0; i < L.cols; ++i) {
        __set0(L[i], i);
    }
}

template <typename t>
static void __LDLT(MAT<t>& L, t* d, t* __restrict A_pack, t* __restrict B_pack) {
    int i, j, k;
    int m, n, l;
    constexpr int BS = __BlockSize_LLT;
    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int S = 2 * BS;
    __BlockMat<t> _L, _L1, _L2;

    if (L.cols <= S) {
        _L.set(L, 0, 0, L.cols, L.cols);
        __LDLT2(_L, d);

        for (i = 0; i < L.cols; ++i) {
            __set0(L[i], i);
        }
        return;
    }

    for (j = 0; j < L.cols; j += BS) {
        n = (j + BS > L.cols) ? (L.cols - j) : BS;

        _L.set(L, j, j, n, n);
        __LDLT2(_L, d + j);

        if (j + BS < L.cols) {
            for (i = j + BS; i < L.cols; i += BS) {
                m = (i + BS > L.cols) ? (L.cols - i) : BS;
                _L1.set(L, i, j, m, n);
                __Solve_direct_XDLT(_L, _L1, d + j);
            }

            _L.set(L, j + BS, j, L.rows - (j + BS), BS);
            __pack_A_block<t, t>(_L.data, _L._rows, _L.rows, BS, A_pack);

            for (i = j + BS; i < L.cols; i += BS) {
                m = (i + BS > L.cols) ? (L.cols - i) : BS;

                _L1.set(L, i, j, m, n);
                __pack_BT_block_d<t, t>(_L1.data, _L1._rows, n, m, d + j, B_pack);

                for (k = i; k < L.cols; k += MC) {
                    l = (k + MC > L.cols) ? (L.cols - k) : MC;

                    _L.set(L, k, i, l, m);

                    constexpr int MR = __micro_kernel_mr<t, t>();
                    const t* A_base = A_pack + (size_t)((k - (j + BS)) / MR) * BS * MR;
                    __times_A_BT_LLT<t>(A_base, B_pack, _L, n);
                }
            }
        }
    }

    for (i = 0; i < L.cols; ++i) {
        __set0(L[i], i);
    }
}

template <typename t1, typename t2>
void LLT(MAT<t1>& A, MAT<t2>& L) {
    constexpr int BS = __BlockSize_LLT;
    constexpr int S = 2 * BS;
    constexpr int MR = __micro_kernel_mr<t2, t2>();
    constexpr int NR = __micro_kernel_nr<t2, t2>();
    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int NC = __BlockSize_Mul_sK_NC;

    for (int i = 0; i < A.cols; ++i) {
        __copy(L[i] + i, A[i] + i, A.rows - i);
    }

    if (L.cols <= S) {
        __BlockMat<t2> _L;
        _L.set(L, 0, 0, L.cols, L.cols);
        __LLT2(_L);
        for (int i = 0; i < L.cols; ++i) {
            __set0(L[i], i);
        }
        return;
    }

#define __align_up(x, a) (x + a - 1) & ~(a - 1)
    const size_t A_pack_size = __align_up(sizeof(t2) * ((L.cols + MR - 1) / MR) * BS * MR, 64);
    const size_t B_pack_size = __align_up(sizeof(t2) * ((BS + NR - 1) / NR) * BS * NR, 64);
#undef __align_up

    char* pack_buf = (char*)_mm_malloc(A_pack_size + B_pack_size, 64);
    t2* A_pack = (t2*)pack_buf;
    t2* B_pack = (t2*)(pack_buf + A_pack_size);

    __LLT(L, A_pack, B_pack);

    _mm_free(pack_buf);
}

template <typename t1, typename t2>
void LDLT(MAT<t1>& A, MAT<t2>& L, MAT<t2>& D) {
    constexpr int BS = __BlockSize_LLT;
    constexpr int S = 2 * BS;
    constexpr int MR = __micro_kernel_mr<t2, t2>();
    constexpr int NR = __micro_kernel_nr<t2, t2>();
    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int NC = __BlockSize_Mul_sK_NC;

    t2* d = D.data;
    __set0(d, D.rows * D.cols);
    for (int i = 0; i < A.cols; ++i) {
        __copy(L[i] + i, A[i] + i, A.rows - i);
    }

    if (L.cols <= S) {
        __BlockMat<t2> _L;
        _L.set(L, 0, 0, L.cols, L.cols);
        __LDLT2(_L, d);
        for (int i = 0; i < L.cols; ++i) {
            __set0(L[i], i);
        }
        for (int i = 1; i < D.cols; ++i) {
            D(i, i) = d[i];
        }
        __set0(d + 1, D.cols);
        return;
    }

#define __align_up(x, a) (x + a - 1) & ~(a - 1)
    const size_t A_pack_size = __align_up((size_t)((L.cols + MR - 1) / MR) * BS * MR, 64);
    const size_t B_pack_size = __align_up((size_t)((BS + NR - 1) / NR) * BS * NR, 64);
#undef __align_up

    t2* pack_buf = (t2*)_mm_malloc(sizeof(t2) * (A_pack_size + B_pack_size), 64);
    t2* A_pack = pack_buf;
    t2* B_pack = pack_buf + A_pack_size;

    __LDLT(L, d, A_pack, B_pack);

    _mm_free(pack_buf);

    for (int i = 1; i < D.cols; ++i) {
        D(i, i) = d[i];
    }
    __set0(d + 1, D.cols);

}