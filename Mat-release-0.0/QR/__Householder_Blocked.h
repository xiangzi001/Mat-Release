template<typename t>
struct __QR_Buffer {
    t* T_all;
    t* work1;
    t* work2;
    t* W;
    t* W2;
    t* buf;
};

template<typename t>
inline __QR_Buffer<t> QR_WorkAlloc(int m, int n) {
    constexpr int MR = __micro_kernel_mr<t, t>();
    constexpr int NR = __micro_kernel_nr<t, t>();
    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int KC2 = __BlockSize_Mul_sM_KC;
    constexpr int NC2 = __BlockSize_Mul_sM_NC;
    constexpr int BS = __BlockSize_QR;

    const int k = (m < n) ? m : n;
    const int nW = (m > n) ? m : n;

    const size_t t_pack_sz = ((size_t)k + BS - 1) / BS;

    const size_t v_pack_sz = ((size_t)(m + MR - 1) / MR) * BS * MR;

    const size_t np_b = ((size_t)BS + MR - 1) / MR;
    const size_t num_full = ((size_t)m + KC2 - 1) / KC2;
    const size_t vt_pack_sz =
        np_b * (size_t)BS * MR +
        np_b * num_full * (size_t)KC2 * MR;

    constexpr size_t r_pack_sz = ((size_t)(NC2 + NR - 1) / NR) * KC2 * NR;
    constexpr size_t w_pack_sz = ((size_t)(NC + NR - 1) / NR) * BS * NR;

    const size_t work1_sz = max(v_pack_sz, vt_pack_sz);
    const size_t work2_sz = max(r_pack_sz, w_pack_sz);

#define __align_up(x, a) (x + a - 1) & ~(a - 1)
    const size_t T_all_size = __align_up(sizeof(t) * t_pack_sz * BS * BS, 64);
    const size_t work1_size = __align_up(sizeof(t) * work1_sz, 64);
    const size_t work2_size = __align_up(sizeof(t) * work2_sz, 64);
    const size_t W_size = __align_up(sizeof(t) * BS * nW, 64);
    const size_t buf_size = __align_up(sizeof(t) * (m + 2 * BS + k), 64);
#undef __align_up

    __QR_Buffer<t> QR_buf;
    QR_buf.T_all = (t*)_mm_malloc(T_all_size + work1_size + work2_size + W_size * 2 + buf_size, 64);
    QR_buf.work1 = (t*)((char*)(QR_buf.T_all) + T_all_size);
    QR_buf.work2 = (t*)((char*)(QR_buf.work1) + work1_size);
    QR_buf.W = (t*)((char*)(QR_buf.work2) + work2_size);
    QR_buf.W2 = (t*)((char*)(QR_buf.W) + W_size);
    QR_buf.buf = (t*)((char*)(QR_buf.W2) + W_size);

    return QR_buf;
}

template<typename t>
inline void QR_WorkFree(__QR_Buffer<t>& QR_buf) {
    _mm_free(QR_buf.T_all);
}

template<typename t>
static void __QR_Householder_Blocked2(__BlockMat<t>& R, t* tau, t* v) {
    const int m = R.rows, n = R.cols;
    for (int r = 0; r < n; ++r) {
        t* w = R[r] + r;
        const int len = m - r;
        t cr = sqrt(__dot(w, w, len));
        if (cr == t(0)) {
            tau[r] = t(0);
            continue;
        }
        if (w[0] < t(0)) cr = -cr;
        t w0 = w[0] + cr;
        t hr = t(1) / w0;
        w[0] = -cr;
        __times(w + 1, hr, w + 1, len - 1);
        v[0] = t(1);
        __copy(v + 1, w + 1, len - 1);
        tau[r] = w0 / cr;
        for (int c = r + 1; c < n; ++c) {
            t* col = R[c] + r;
            t dot = col[0] + __dot(v + 1, col + 1, len - 1);
            __mixed(col, v, -dot * tau[r], col, len);
        }
    }
}

template<typename t>
static void __QR_Householder_Blocked_CalculateT(__BlockMat<t>& panel, const t* tau, t* Tmat, int k, t* s, t* w) {
    const int h = panel.rows;
    const int ldT = k;
    Tmat[0] = tau[0];
    for (int i = 1; i < k; ++i) {
        for (int l = 0; l < i; ++l) {
            const t* Vl = panel[l];
            const t* Vi = panel[i];
            s[l] = Vl[i] + __dot(Vl + i + 1, Vi + i + 1, h - i - 1);
        }
        __set0(w, i);
        for (int c = 0; c < i; ++c)
            __mixed(w, Tmat + (size_t)c * ldT, s[c], w, c + 1);
        __times(w, -tau[i], Tmat + (size_t)i * ldT, i);
        Tmat[i + (size_t)i * ldT] = tau[i];
    }
}

template<typename t>
static void __QR_Householder_Blocked_pack_V(
    const t* __restrict Vd, int ldV,
    int rows, int cols, t* __restrict V_pack)
{
    constexpr int MR = __micro_kernel_mr<t, t>();
    const int s = min(rows, ((cols + MR - 1) / MR) * MR);

    __pack_A_block_L<t, t>(Vd, ldV, rows, cols, V_pack);

    if (s < rows) {
        __pack_A_block<t, t>(Vd + s, ldV, rows - s, cols,
            V_pack + (size_t)(s / MR) * cols * MR);
    }
}

template<typename t>
static void __QR_Householder_Blocked_pack_VT(
    const t* __restrict Vd, int ldV,
    int rows, int cols, t* __restrict VT_pack)
{
    constexpr int KC = __BlockSize_Mul_KC;
    constexpr int MR = __micro_kernel_mr<t, t>();

    __pack_A_block_LT<t, t>(Vd, ldV, rows, cols, VT_pack);

    const int N = (cols + MR - 1) / MR;
    const size_t tri_size = (size_t)N * cols * MR;
    const size_t full_block_size = (size_t)N * KC * MR;

    t* dst = VT_pack + tri_size;
    for (int r = cols; r < rows; r += KC) {
        const int kc = min(KC, rows - r);
        __pack_AT_block<t, t>(Vd, ldV, r, cols, kc, dst);
        dst = dst + full_block_size;
    }
}

template<typename t>
static void __QR_Householder_Blocked_Mul(
    __BlockMat<t> V, __BlockMat<t> R, const t* Tmat, int k,
    t* __restrict pack_A, t* __restrict pack_B, t* __restrict W, t* __restrict W2)
{
    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int KC2 = __BlockSize_Mul_sM_KC;
    constexpr int NC2 = __BlockSize_Mul_sM_NC;
    constexpr int MR = __micro_kernel_mr<t, t>();

    const int N = (k + MR - 1) / MR;
    const size_t tri_size = (size_t)N * k * MR;
    const size_t full_block_size = (size_t)N * KC2 * MR;

    __QR_Householder_Blocked_pack_VT(V.data, V._rows, V.rows, k, pack_A);

    for (int j = 0; j < R.cols; j += NC2) {
        const int nc = (j + NC2 > R.cols) ? (R.cols - j) : NC2;
        t* Wj = W + (size_t)j * k;
        __set0(Wj, (size_t)k * nc);

        t* Temp = R.data + (size_t)j * R._rows;
        __pack_B_block<t, t>(
            Temp, R._rows, k, nc, pack_B);
        __mulAB_packed_n_AU<t>(
            pack_A, pack_B, Wj, k, k, nc, k, 0);

        t* TempA = pack_A + tri_size;
        for (int r = k; r < V.rows; r += KC2) {
            const int kc = min(KC2, V.rows - r);
            __pack_B_block<t, t>(
                Temp + r, R._rows, kc, nc, pack_B);
            __mulAB_packed_n<t>(TempA, pack_B, Wj, k, k, nc, kc);
            TempA = TempA + full_block_size;
        }
    }

    __pack_A_block_UT<t, t>(Tmat, k, pack_A);
    for (int j = 0; j < R.cols; j += NC) {
        const int nc = (j + NC > R.cols) ? (R.cols - j) : NC;
        t* Wj = W + (size_t)j * k;
        t* W2j = W2 + (size_t)j * k;

        __pack_B_block<t, t>(Wj, k, k, nc, pack_B);
        __set0(W2j, (size_t)k * nc);
        __mulAB_packed_n_AL<t>(
            pack_A, pack_B, W2j, k, k, nc, k, 0);
    }

    __QR_Householder_Blocked_pack_V(V.data, V._rows, V.rows, k, pack_A);

    const int tri_n = min(V.rows, ((k + MR - 1) / MR) * MR);

    for (int j = 0; j < R.cols; j += NC) {
        const int nc = (j + NC > R.cols) ? (R.cols - j) : NC;
        const t* W2j = W2 + (size_t)j * k;
        __pack_B_block<t, t>(W2j, k, k, nc, pack_B);

        t* Temp = R.data + (size_t)j * R._rows;
        __mulAB_packed_n_AL<t>(
            pack_A, pack_B, Temp, R._rows, tri_n, nc, k, 0);

        for (int i = tri_n; i < V.rows; i += MC) {
            const int mc = (i + MC > V.rows) ? (V.rows - i) : MC;
            const t* A = pack_A + (size_t)(i / MR) * k * MR;
            __mulAB_packed_n<t>(
                A, pack_B, Temp + i, R._rows, mc, nc, k);
        }
    }
}

template<typename t>
static void __QR_Householder_Blocked(MAT<t>& R, __QR_Buffer<t> QR_buf, int m, int n) {
    const int k = (m < n) ? m : n;

    constexpr int BS = __BlockSize_QR;

    t* v = QR_buf.buf;
    t* s = v + m;
    t* w = s + BS;
    t* tau = w + BS;

    for (int j = 0; j < k; j += BS) {
        const int b = (j + BS > k) ? (k - j) : BS;
        const int h = m - j;

        __BlockMat<t> _V(R, j, j, h, b);
        __QR_Householder_Blocked2(_V, tau + j, v);
        __QR_Householder_Blocked_CalculateT(_V, tau + j, QR_buf.T_all + (size_t)j * BS, b, s, w);

        if (j + b < n) {
            __BlockMat<t> _R(R, j, j + b, h, n - j - b);
            __QR_Householder_Blocked_Mul<t>(
                _V, _R, QR_buf.T_all + (size_t)j * BS, b,
                QR_buf.work1, QR_buf.work2, QR_buf.W, QR_buf.W2);
        }
    }
}

template<typename t>
static void __QR_Householder_Blocked_CalculateQ(const MAT<t>& A, MAT<t>& Q, 
    const t* __restrict T_all,
    t* __restrict pack_A, t* __restrict pack_B,
    t* __restrict W, t* __restrict W2,
    int A_rows, int A_cols) 
{
    const int k = (A_rows < A_cols) ? A_rows : A_cols;
    __set0(Q.data, A_rows * A_rows);
    for (int i = 0; i < Q.rows; ++i)
        Q(i, i) = t(1);

    constexpr int MR = __micro_kernel_mr<t, t>();
    constexpr int NR = __micro_kernel_nr<t, t>();
    constexpr int MC = __BlockSize_Mul_sK_MC;
    constexpr int NC = __BlockSize_Mul_sK_NC;
    constexpr int KC = __BlockSize_Mul_KC;
    constexpr int BS = __BlockSize_QR;

    const int last = ((k - 1) / BS) * BS;
    for (int j = last; j >= 0; j -= BS) {

        __BlockMat<t> V(A, j, j, A_rows - j, (j + BS > k) ? (k - j) : BS);
        const t* Tj = T_all + (size_t)j * BS;

        const int N = (V.cols + MR - 1) / MR;
        const size_t tri_size = (size_t)N * V.cols * MR;
        const size_t full_block_size = (size_t)N * KC * MR;

        __QR_Householder_Blocked_pack_VT(V.data, V._rows, V.rows, V.cols, pack_A);

        for (int jc = 0; jc < A_rows; jc += NC) {
            const int nc = (jc + NC > A_rows) ? (A_rows - jc) : NC;
            t* Wj = W + (size_t)jc * V.cols;
            __set0(Wj, (size_t)V.cols * nc);

            t* Temp = Q.data + j + (size_t)jc * Q.rows;
            __pack_B_block<t, t>(
                Temp, Q.rows, V.cols, nc, pack_B);
            __mulAB_packed_n_AU<t>(
                pack_A, pack_B, Wj, (size_t)V.cols, V.cols, nc, V.cols, 0);

            t* TempA = pack_A + tri_size;
            for (int r = V.cols; r < V.rows; r += KC) {
                const int kc = min(KC, V.rows - r);
                __pack_B_block<t, t>(
                    Temp + r, Q.rows, kc, nc, pack_B);
                __mulAB_packed_n<t>(TempA, pack_B, Wj, V.cols, V.cols, nc, kc);
                TempA = TempA + full_block_size;
            }
        }

        __pack_A_block_U<t, t>(Tj, V.cols, pack_A);
        for (int jc = 0; jc < A_rows; jc += NC) {
            const int nc = (jc + NC > A_rows) ? (A_rows - jc) : NC;
            t* Wj = W + (size_t)jc * V.cols;
            t* W2j = W2 + (size_t)jc * V.cols;
            __pack_B_block<t, t>(Wj, V.cols, V.cols, nc, pack_B);
            __set0(W2j, (size_t)V.cols * nc);
            __mulAB_packed_n_AU<t>(
                pack_A, pack_B, W2j, V.cols, V.cols, nc, V.cols, 0);
        }

        __QR_Householder_Blocked_pack_V(V.data, V._rows, V.rows, V.cols, pack_A);

        const int tri_n = min(V.rows, ((V.cols + MR - 1) / MR) * MR);

        for (int jc = 0; jc < A_rows; jc += NC) {
            const int nc = (jc + NC > A_rows) ? (A_rows - jc) : NC;
            const t* W2j = W2 + (size_t)jc * V.cols;
            __pack_B_block<t, t>(W2j, V.cols, V.cols, nc, pack_B);

            t* Temp = Q.data + j + (size_t)jc * Q.rows;
            __mulAB_packed_n_AL<t>(
                pack_A, pack_B,
                Temp, Q.rows,
                tri_n, nc, V.cols, 0);

            for (int i = tri_n; i < V.rows; i += MC) {
                const int mc = (i + MC > V.rows) ? (V.rows - i) : MC;
                const t* A2 = pack_A + (size_t)(i / MR) * V.cols * MR;
                __mulAB_packed_n<t>(
                    A2, pack_B, Temp + i, Q.rows, mc, nc, V.cols);
            }
        }
    }
}

template<typename t>
void __QR_Householder_Blocked(MAT<t>& Q, MAT<t>& R, const __QR_Buffer<t> QR_buf) {
    const int m = R.rows;
    const int n = R.cols;

    __QR_Householder_Blocked(R, QR_buf, m, n);
    Q.init(m, m);
    __QR_Householder_Blocked_CalculateQ(R, Q, 
        QR_buf.T_all, QR_buf.work1, QR_buf.work2, QR_buf.W, QR_buf.W2,
        m, n);

    for (int j = 0; j < n; ++j) {
        int n0 = m - j - 1;
        if (n0 > 0)
            __set0(R.data + (size_t)j * m + (j + 1), n0);
        else
            break;
    }
}