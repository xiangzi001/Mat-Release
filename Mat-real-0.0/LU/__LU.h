
template <typename T>
void __LU2(__BlockMat<T>& L, __BlockMat<T>& U) {
    int r, i;
    T* Temp;
    for (r = 0; r < U.cols; ++r) {
        Temp = U[r];
        for (i = 0; i < r; ++i) {
            __mixed(Temp + i + 1, L[i] + i + 1, -Temp[i], Temp + i + 1, U.rows - i - 1);
        }
        if (Temp[r] == 0) {
            break;
        }
        __times(Temp + r + 1, (T)1 / Temp[r], L[r] + r + 1, L.rows - r - 1);
    }
}

template <typename T>
void __LU(MAT<T>& L, MAT<T>& U, T* __restrict A_pack, T* __restrict B_pack) {
    int r, s, i, j, m, n;
    __BlockMat<T> _L, _U, _A;

    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int BS = __BlockSize_LU;

    s = (U.rows > U.cols) ? U.cols : U.rows;

    for (r = 0; r + BS <= s; r += BS) {
        _L.set(L, r, r, L.rows - r, BS);
        _U.set(U, r, r, U.rows - r, BS);
        __LU2(_L, _U);

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
        __LU2(_L, _U);
    }
}

template <typename t1, typename t2>
void LU(MAT<t1>& A, MAT<t2>& L, MAT<t2>& U) {
    constexpr int BS = __BlockSize_LU;
    const int s = (U.rows > U.cols) ? U.cols : U.rows;

    if ((void*)A.data != (void*)U.data)
        __copy(U.data, A.data, U.rows * U.cols);

    if (s <= BS * 1.5) {
        __BlockMat<t2> _L, _U;
        _L.set(L, 0, 0, L.rows, s);
        _U.set(U, 0, 0, U.rows, s);
        __LU2(_L, _U);
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

        __LU(L, U, A_pack, B_pack);

        _mm_free(pack_buf);
    }

    if (L.data != U.data) {
        for (int i = 0; i < s; ++i) __set0(L[i], i);
        for (int i = 0; i < s; ++i) L[i][i] = 1;
        for (int i = 0; i < s; ++i) __set0(U[i] + i + 1, U.rows - i - 1);
    }
}

template <typename t1, typename t2>
void LU_getLU(MAT<t1>& LU, MAT<t2>& L, MAT<t2>& U) {
    int i;
    int s = min(LU.rows, LU.cols);
    for (i = 0; i < s; ++i) {
        __set0(L[i], i);
        L(i, i) = 1;
        __copy(L[i] + i + 1, LU[i] + i + 1, L.rows - i - 1);
    }
    for (i = s; i < L.cols; ++i) {
        __set0(L[i], L.rows);
        L(i, i) = 1;
    }

    for (i = 0; i < s; ++i) {
        __copy(U[i], LU[i], i + 1);
        __set0(U[i] + i + 1, U.rows - i - 1);
    }
    for (i = s; i < U.cols; ++i) {
        __copy(U[i], LU[i], U.rows);
    }
}