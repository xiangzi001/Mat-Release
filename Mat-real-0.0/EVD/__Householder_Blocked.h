
/*
A22 <- A22 - V12*W12t - W12*V12t
*/
template<typename T>
void __EVD_H_Householder_Blocked2(
    __BlockMat<T> R, T* tau, T* y, __BlockMat<T> W, T* a, T* b)
{
    for (int r = 0; r < R.cols; ++r) {
        const int m = r + 1;
        const int n = R.cols - m;
        const int len = R.rows - m;

        T* v = R[r] + m;
        if (len == 1) {
            a[r] = v[-1];
            b[r] = -v[0];
            tau[r] = 2;
            v[0] = 1;

            ++r;
            a[r] = R(r, r);
            b[r] = T(0);
            tau[r] = T(0);
            return;
        }

        T cr = sqrt(__dot(v, v, len));
        if (cr == T(0)) {
            a[r] = v[-1];
            b[r] = T(0);
            tau[r] = T(0);
            continue;
        }
        if (v[0] < T(0)) cr = -cr;
        T w0 = v[0] + cr;
        T hr = T(1) / w0;

        a[r] = v[-1];
        b[r] = -cr;
        __times(v, hr, v, len);
        v[0] = T(1);
        hr = w0 / cr;
        tau[r] = hr;

        T* w = W[r] + m;
        __mulAB_vec_AH_lower(R[m] + m, (size_t)(R._rows + 1), v, y, len);

        if (len > n) {
            T* _y = y + n;
            const T* _v = v + n;
            const int temp_len = len - n;
            for (int j = 0; j < r; ++j) {
                T a = __dot(R[j] + R.cols, _v, temp_len);
                T b = __dot(W[j] + R.cols, _v, temp_len);
                __mixed(_y, W[j] + R.cols, -a, _y, temp_len);
                __mixed(_y, R[j] + R.cols, -b, _y, temp_len);
            }
        }

        __times(y, hr, y, len);

        __mixed(y, v, -__dot(y, v, len) * hr * T(0.5), w, len);
        for (int i = 0; i < n; ++i) {
            __mixed(R[m + i] + m + i, w + i, -v[i], R[m + i] + m + i, len - i);
            __mixed(R[m + i] + m + i, v + i, -w[i], R[m + i] + m + i, len - i);
        }
    }
}

template<typename t>
static void __EVD_H_Householder_Blocked_Mul(
    __BlockMat<t> V,
    __BlockMat<t> A22,
    __BlockMat<t> W12,
    int b, int k2,
    t* __restrict pack_A,
    t* __restrict pack_B)
{
    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int MR = __micro_kernel_mr<t, t>();

    t* V12 = V.data;
    t* A22d = A22.data;
    t* W12d = W12.data;

    __pack_A_block<t, t>(V12, V._rows, k2, b, pack_A);
    for (int jc = 0; jc < k2; jc += NC) {
        const int nc = min(NC, k2 - jc);
        __pack_BT_block<t, t>(W12d + jc, W12._rows, b, nc, pack_B);

        {
            const t* A_diag = pack_A + (size_t)(jc / MR) * b * MR;
            t* C_diag = A22d + jc + (size_t)jc * A22._rows;
            __mulAB_packed_n_lower<t>(A_diag, pack_B, C_diag, A22._rows, nc, nc, b);
        }

        for (int ic = jc + nc; ic < k2; ic += MC) {
            const int mc = min(MC, k2 - ic);
            const t* A_ic = pack_A + (size_t)(ic / MR) * b * MR;
            t* C = A22d + ic + (size_t)jc * A22._rows;
            __mulAB_packed_n<t>(A_ic, pack_B, C, A22._rows, mc, nc, b);
        }
    }

    __pack_A_block<t, t>(W12d, W12._rows, k2, b, pack_A);
    for (int jc = 0; jc < k2; jc += NC) {
        const int nc = min(NC, k2 - jc);
        __pack_BT_block<t, t>(V12 + jc, V._rows, b, nc, pack_B);

        {
            const t* A_diag = pack_A + (size_t)(jc / MR) * b * MR;
            t* C_diag = A22d + jc + (size_t)jc * A22._rows;
            __mulAB_packed_n_lower<t>(A_diag, pack_B, C_diag, A22._rows, nc, nc, b);
        }

        for (int ic = jc + nc; ic < k2; ic += MC) {
            const int mc = min(MC, k2 - ic);
            const t* A_ic = pack_A + (size_t)(ic / MR) * b * MR;
            t* C = A22d + ic + (size_t)jc * A22._rows;
            __mulAB_packed_n<t>(A_ic, pack_B, C, A22._rows, mc, nc, b);
        }
    }
}

template<typename t>
static void __EVD_H_Householder_Blocked(
    MAT<t>& R, t* v_buf, t* W, t* work1, t* work2, t* tau,
    const int m, t* a, t* b)
{
    constexpr int BS = __BlockSize_QR;

    for (int j = 0; j < m; j += BS) {
        const int bsize = (j + BS > m) ? (m - j) : BS;
        const int h = m - j;

        __BlockMat<t> _V(R, j, j, h, bsize);

        __BlockMat<t> _W;
        _W.rows = h;
        _W.cols = bsize;
        _W._rows = h;
        _W.data = W;

        __EVD_H_Householder_Blocked2<t>(_V, tau + j, v_buf, _W, a + j, b + j);

        const int k2 = h - bsize;
        if (k2 > 0) {
            _V.set(R, j + bsize, j, k2, bsize);
            _W.rows = k2;
            _W.cols = bsize;
            _W._rows = h;
            _W.data = W + bsize;

            __BlockMat<t> A22(R, j + bsize, j + bsize, k2, k2);
            __EVD_H_Householder_Blocked_Mul<t>(
                _V, A22, _W,
                bsize, k2,
                work1,
                work2);
        }
    }
}

template<typename t>
static void __EVD_H_Householder_BlockedQ(
    const MAT<t>& A, MAT<t>& Q,
    const t* __restrict tau,
    t* __restrict pack_A, t* __restrict pack_B,
    t* __restrict W, t* __restrict W2,
    t* __restrict T_buf,
    t* __restrict s, t* __restrict w,
    int m)
{
    constexpr int MR = __micro_kernel_mr<t, t>();
    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int KC = __BlockSize_Mul_KC;
    constexpr int BS = __BlockSize_QR;

    const int last = ((m - 1) / BS) * BS;

    for (int j = last; j >= 0; j -= BS) {
        const int bsize = (j + BS > m) ? (m - j) : BS;
        const int h = m - j;
        const int nc = (h == bsize) ? (bsize - 1) : bsize;
        if (nc <= 0) continue;

        __BlockMat<t> V(A, j + 1, j, h - 1, nc);

        __QR_Householder_Blocked_CalculateT(V, tau + j, T_buf, nc, s, w);

        const int N = (V.cols + MR - 1) / MR;
        const size_t tri_size = (size_t)N * V.cols * MR;
        const size_t full_block_size = (size_t)N * KC * MR;

        __QR_Householder_Blocked_pack_VT(V.data, V._rows, V.rows, V.cols, pack_A);

        for (int jc = 0; jc < m; jc += NC) {
            const int ncc = (jc + NC > m) ? (m - jc) : NC;
            t* Wj = W + (size_t)jc * nc;
            __set0(Wj, (size_t)nc * ncc);

            t* Temp = Q.data + (j + 1) + (size_t)jc * Q.rows;
            __pack_B_block<t, t>(Temp, Q.rows, nc, ncc, pack_B);
            __mulAB_packed_n_AU<t>(
                pack_A, pack_B, Wj, (size_t)nc, nc, ncc, nc, 0);

            t* TempA = pack_A + tri_size;
            for (int r = nc; r < V.rows; r += KC) {
                const int kc = min(KC, V.rows - r);
                __pack_B_block<t, t>(
                    Temp + r, Q.rows, kc, ncc, pack_B);
                __mulAB_packed_n<t>(TempA, pack_B, Wj, nc, nc, ncc, kc);
                TempA = TempA + full_block_size;
            }
        }

        __pack_A_block_U<t, t>(T_buf, nc, pack_A);
        for (int jc = 0; jc < m; jc += NC) {
            const int ncc = (jc + NC > m) ? (m - jc) : NC;
            t* Wj = W + (size_t)jc * nc;
            t* W2j = W2 + (size_t)jc * nc;
            __pack_B_block<t, t>(Wj, nc, nc, ncc, pack_B);
            __set0(W2j, (size_t)nc * ncc);
            __mulAB_packed_n_AU<t>(
                pack_A, pack_B, W2j, nc, nc, ncc, nc, 0);
        }

        __QR_Householder_Blocked_pack_V(V.data, V._rows, V.rows, V.cols, pack_A);
        const int tri_n = min(V.rows, ((V.cols + MR - 1) / MR) * MR);

        for (int jc = 0; jc < m; jc += NC) {
            const int ncc = (jc + NC > m) ? (m - jc) : NC;
            const t* W2j = W2 + (size_t)jc * nc;
            __pack_B_block<t, t>(W2j, nc, nc, ncc, pack_B);

            t* Temp = Q.data + (j + 1) + (size_t)jc * Q.rows;
            __mulAB_packed_n_AL<t>(
                pack_A, pack_B, Temp, Q.rows, tri_n, ncc, nc, 0);

            for (int i = tri_n; i < V.rows; i += MC) {
                const int mc = (i + MC > V.rows) ? (V.rows - i) : MC;
                const t* A2 = pack_A + (size_t)(i / MR) * nc * MR;
                __mulAB_packed_n<t>(
                    A2, pack_B, Temp + i, Q.rows, mc, ncc, nc);
            }
        }
    }
}