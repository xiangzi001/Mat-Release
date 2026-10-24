
template <typename T>
void __laswp_trailing(
    MAT<T>& U,
    int r, int kcount, int begin, int end,
    const int* __restrict piv)
{
    for (int col = begin; col < end; ++col) {
        T* __restrict colptr = U[col];
        for (int k = 0; k < kcount; ++k) {
            int i = r + k;
            int j = r + piv[k];
            if (i == j) continue;
            T tmp = colptr[i];
            colptr[i] = colptr[j];
            colptr[j] = tmp;
        }
    }
}

template <typename T>
void __PLU2(int* pos, __BlockMat<T>& L, __BlockMat<T>& U) {
    for (int r = 0; r < U.cols; ++r) {
        T* Temp = U[r];

        for (int i = 0; i < r; ++i) {
            __mixed(Temp + i + 1, L[i] + i + 1,
                -Temp[i], Temp + i + 1, U.rows - i - 1);
        }

        T maxv = __MaxAbs(U.data + (size_t)r * U._rows + r, U.rows - r);
        if (maxv == T(0)) break;

        int j = r;
        for (int i = r; i < U.rows; ++i) {
            if (abs(U(i, r)) == maxv) {
                j = i;
                break;
            }
        }

        if (j != r) {
            L.swap_rows(r, j, 0, r);
            U.swap_rows(r, j, r, U.cols);
            int tmp = pos[r]; pos[r] = pos[j]; pos[j] = tmp;
        }

        __times(Temp + r + 1, (T)1 / Temp[r],
            L[r] + r + 1, L.rows - r - 1);
    }
}

template <typename T>
void __PLU2_piv(int* pos, int* piv, __BlockMat<T>& L, __BlockMat<T>& U, int cL, int BS) {
    int r, i;

    for (r = 0; r < BS; ++r) piv[r] = r;

    T* Temp;
    for (r = 0; r < U.cols; ++r) {
        Temp = U[r];

        for (i = 0; i < r; ++i) {
            __mixed(Temp + i + 1, L[i] + i + 1, -Temp[i], Temp + i + 1, U.rows - i - 1);
        }

        T maxv = __MaxAbs(U.data + (size_t)r * U._rows + r, U.rows - r);
        if (maxv == T(0)) {
            piv[r] = r;
            break;
        }

        int j = r;
        for (i = r; i < U.rows; ++i) {
            if (abs(U(i, r)) == maxv) { j = i; break; }
        }
        piv[r] = j;

        if (j != r) {
            L.swap_rows(r, j, cL, r);
            U.swap_rows(r, j, r, U.cols);

            i = pos[j];
            pos[j] = pos[r];
            pos[r] = i;
        }

        __times(Temp + r + 1, (T)1 / Temp[r], L[r] + r + 1, L.rows - r - 1);
    }
}

template <typename T>
void __PLU(int* pos, MAT<T>& L, MAT<T>& U, T* __restrict A_pack, T* __restrict B_pack) {
    int r, s, i, j, m, n;
    __BlockMat<T> _L, _U, _A;

    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int BS = __BlockSize_LU;

    int* piv = new int[BS];

    s = (U.rows > U.cols) ? U.cols : U.rows;

    for (r = 0; r + BS <= s; r += BS) {
        _L.set(L, r, r, L.rows - r, BS);
        _U.set(U, r, r, U.rows - r, BS);
        __PLU2_piv(pos + r, piv, _L, _U, -r, BS);

        int panel_end = (r + BS < U.cols) ? (r + BS) : U.cols;
        __laswp_trailing(U, r, BS, panel_end, U.cols, piv);

        if (r + BS < s) {

            _L.set(L, r, r, L.rows - r, BS);
            _U.set(U, r, r + BS, BS, U.cols - (r + BS));
            __Solve_direct_LX(_L, _U);

            _L.set(L, r + BS, r, U.rows - (r + BS), BS);
            __pack_A_block<T, T>(_L.data, _L._rows, _L.rows, BS, A_pack);

            for (j = r + BS; j < U.cols; j += NC) {
                n = (j + NC > U.cols) ? (U.cols - j) : NC;

                _U.set(U, r, j, BS, n);
                __pack_B_block<T, T>(_U[0], _U._rows, BS, n, B_pack);

                for (i = r + BS; i < U.rows; i += MC) {
                    m = (i + MC > U.rows) ? (U.rows - i) : MC;

                    _A.set(U, i, j, m, n);
                    constexpr int MR = __micro_kernel_mr<T, T>();
                    const T* A_base = A_pack + (size_t)((i - (r + BS) + MR - 1) / MR) * BS * MR;
                    __mulAB_packed_n<T>(A_base, B_pack, _A.data, _A._rows,
                        _A.rows, _A.cols, BS);
                }
            }
        }
    }

    if (r < s) {
        int rem = s - r;
        _L.set(L, r, r, L.rows - r, rem);
        _U.set(U, r, r, U.rows - r, rem);
        __PLU2_piv(pos + r, piv, _L, _U, -r, rem);
    }

    delete[] piv;
}

template <typename t1, typename t2>
void PLU(MAT<t1>& A, int* pos, MAT<t2>& L, MAT<t2>& U) {
    constexpr int BS = __BlockSize_LU;
    int s = (U.rows > U.cols) ? U.cols : U.rows;

    if ((void*)A.data != (void*)U.data)
        __copy(U.data, A.data, U.rows * U.cols);
    for (int i = 0; i < U.rows; ++i) pos[i] = i;

    if (s <= BS * 1.5) {
        __BlockMat<t2> _L, _U;
        _L.set(L, 0, 0, L.rows, U.cols);
        _U.set(U, 0, 0, U.rows, U.cols);
        __PLU2(pos, _L, _U);
    }
    else {
        constexpr int NC = __BlockSize_Mul_sK_NC;
        constexpr int MR = __micro_kernel_mr<t2, t2>();
        constexpr int NR = __micro_kernel_nr<t2, t2>();

#define __align_up(x, a) (x + a - 1) & ~(a - 1)
        const size_t A_pack_size = __align_up(sizeof(t2) * ((U.rows + MR - 1) / MR) * BS * MR, 64);
        const size_t B_pack_size = __align_up(sizeof(t2) * ((NC + NR - 1) / NR) * BS * NR, 64);
#undef __align_up

        char* pack_buf = (char*)_mm_malloc(A_pack_size + B_pack_size, 64);
        t2* A_pack = (t2*)pack_buf;
        t2* B_pack = (t2*)(pack_buf + A_pack_size);

        __PLU(pos, L, U, A_pack, B_pack);

        _mm_free(pack_buf);
    }

    if (L.data != U.data) {
        for (int i = 0; i < s; ++i) __set0(L[i], i);
        for (int i = 0; i < s; ++i) L[i][i] = 1;
        for (int i = 0; i < s; ++i) __set0(U[i] + i + 1, U.rows - i - 1);
    }
}

template <typename t>
void PLU_getP(int* pos, MAT<t>& P) {
    __set0(P.data, P.rows * P.cols);
    for (int i = 0; i < P.rows; ++i) {
        P(pos[i], i) = 1;
    }
}