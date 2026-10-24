//*

template <typename T>
void __Inv_X_LU(MAT<T>& A, MAT<T>& X, T* __restrict A_pack, T* __restrict B_pack) {
    int r, i, j, l, m, n;
    __BlockMat<T> _L, _U, _A;

    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int BS = __BlockSize_LU;

    for (r = 0; r < A.cols; r += BS) {
        l = (r + BS > A.cols) ? (A.cols - r) : BS;

        _U.set(A, r, r, l, l);

        for (i = 0; i <= r; i += BS) {
            m = (i + BS > X.rows) ? (X.rows - i) : BS;

            _A.set(X, i, r, m, l);
            __Solve_direct_XU(_U, _A);
        }

        int rows_to_pack = (r + l > X.rows) ? X.rows : (r + l);
        _L.set(X, 0, r, rows_to_pack, l);
        __pack_A_block<T, T>(_L.data, _L._rows, rows_to_pack, l, A_pack);

        for (j = r + BS; j < X.cols; j += NC) {
            n = (j + NC > X.cols) ? (X.cols - j) : NC;

            _U.set(A, r, j, l, n);
            __pack_B_block<T, T>(_U[0], _U._rows, l, n, B_pack);

            for (i = 0; i < rows_to_pack; i += MC) {
                m = (i + MC > rows_to_pack) ? (rows_to_pack - i) : MC;

                _A.set(X, i, j, m, n);
                constexpr int MR = __micro_kernel_mr<T, T>();
                const T* A_base = A_pack + (size_t)(i / MR) * (size_t)l * MR;
                __mulAB_packed_n<T>(A_base, B_pack, _A.data, _A._rows,
                    _A.rows, _A.cols, l);
            }
        }
    }

    for (r = ((X.rows - 1) / BS) * BS; r >= 0; r -= BS) {
        n = (r + BS < X.rows) ? BS : (X.rows - r);

        _L.set(A, r, r, n, n);

        for (i = 0; i < A.rows; i += BS) {
            m = (i + BS > X.rows) ? (X.rows - i) : BS;
            _A.set(X, i, r, m, n);
            __Solve_direct_XL(_L, _A);
        }

        _U.set(X, 0, r, X.rows, n);
        __pack_A_block<T, T>(_U.data, _U._rows, X.rows, n, A_pack);

        for (j = 0; j < r; j += NC) {
            int jc = (j + NC > r) ? (r - j) : NC;

            _L.set(A, r, j, n, jc);
            __pack_B_block<T, T>(_L[0], _L._rows, n, jc, B_pack);

            for (i = 0; i < X.rows; i += MC) {
                m = (i + MC > X.rows) ? (X.rows - i) : MC;

                _A.set(X, i, j, m, jc);
                constexpr int MR = __micro_kernel_mr<T, T>();
                const T* A_base = A_pack + (size_t)(i / MR) * (size_t)n * MR;
                __mulAB_packed_n<T>(A_base, B_pack, _A.data, _A._rows,
                    _A.rows, _A.cols, n);
            }
        }
    }
}

template<typename T>
void __Inv_swap(MAT<T>& A, int i, int* pos, int j, bool* flag) {
    int k = pos[j];
    if (k != i) {
        __Inv_swap(A, i, pos, k, flag);
    }
    __copy(A[k], A[j], A.rows);
    flag[j] = false;
}

template<typename T>
MAT<T> __Inv(MAT<T>& A) {
    MAT<T> B(A.rows, A.cols);
    int i;

    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int BS = __BlockSize_LU;
    constexpr int MR = __micro_kernel_mr<T, T>();
    constexpr int NR = __micro_kernel_nr<T, T>();

#define __align_up(x, a) (x + a - 1) & ~(a - 1)
    const size_t A_pack_size = __align_up(sizeof(T) * ((A.rows + MR - 1) / MR) * BS * MR, 64);
    const size_t B_pack_size = __align_up(sizeof(T) * ((NC + NR - 1) / NR) * BS * NR, 64);
#undef __align_up

    char* pack_buf = (char*)_mm_malloc(A_pack_size + B_pack_size, 64);
    T* A_pack = (T*)pack_buf;
    T* B_pack = (T*)(pack_buf + A_pack_size);

    int* pos = new int[B.rows];
    for (i = 0; i < A.rows; ++i) {
        pos[i] = i;
    }

    __PLU(pos, A, A, A_pack, B_pack);

    __set0(B.data, B.rows * B.cols);
    T temp;
    for (i = 0; i < A.cols; ++i) {
        temp = (T)1 / A(i, i);
        __times(A[i], temp, A[i], i);
        B(i, i) = temp;
    }

    __Inv_X_LU(A, B, A_pack, B_pack);

    bool* if_nswap = (bool*)A_pack;
    T* Temp = (T*)(if_nswap + B.cols);

    for (i = 0; i < B.cols; ++i) {
        if_nswap[i] = true;
    }
    int j;
    for (i = 0; i < B.cols; ++i) {
        if (if_nswap[i]) {
            j = pos[i];
            if (i != j) {
                __copy(Temp, B[i], B.rows);
                __Inv_swap(B, i, pos, i, if_nswap);
                __copy(B[j], Temp, B.rows);
            }
        }
    }
    delete[] pos;

    _mm_free(pack_buf);

    return B;
}


template <typename T>
void __Inv_X_LU_small(MAT<T>& A, MAT<T>& X) {
    __BlockMat<T> _B, _A;

    _B.set(A, 0, 0, A.rows, A.rows);
    _A.set(X, 0, 0, X.rows, X.rows);
    __Solve_direct_XU(_B, _A);

    _B.set(A, 0, 0, A.rows, A.rows);
    _A.set(X, 0, 0, X.rows, X.rows);
    __Solve_direct_XL(_B, _A);
}

template<typename T>
MAT<T> __Inv_small(MAT<T>& A) {
    MAT<T> B(A.rows, A.cols);
    int i;

    char* buf = new char[B.rows * sizeof(int) + B.rows * (sizeof(T) + sizeof(bool))];

    int* pos = (int*)buf;
    for (i = 0; i < A.rows; ++i) {
        pos[i] = i;
    }

    __BlockMat<T> _A;
    _A.set(A, 0, 0, A.rows, A.cols);
    __PLU2(pos, _A, _A);

    __set0(B.data, B.rows * B.cols);
    T temp;
    for (i = 0; i < A.cols; ++i) {
        temp = (T)1 / A(i, i);
        __times(A[i], temp, A[i], i);
        B(i, i) = temp;
    }

    __Inv_X_LU_small(A, B);

    bool* if_nswap = (bool*)(pos + B.rows);
    T* Temp = (T*)(if_nswap + B.cols);

    for (i = 0; i < B.cols; ++i) {
        if_nswap[i] = true;
    }
    int j;
    for (i = 0; i < B.cols; ++i) {
        if (if_nswap[i]) {
            j = pos[i];
            if (i != j) {
                __copy(Temp, B[i], B.rows);
                __Inv_swap(B, i, pos, i, if_nswap);
                __copy(B[j], Temp, B.rows);
            }
        }
    }
    delete[] buf;

    return B;
}

template<typename T>
MAT<T> Inv(MAT<T> A) {
    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int BS = __BlockSize_LU;
    constexpr int MR = __micro_kernel_mr<T, T>();
    constexpr int NR = __micro_kernel_nr<T, T>();

    if (A.rows <= BS * 1.5) {
        return __Inv_small(A);
    }

    return __Inv(A);
}
//*/
