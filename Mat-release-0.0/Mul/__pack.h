
template<typename t1, typename t2>
inline void __pack_A_block(
    const t1* __restrict A,
    size_t ldA, size_t mc, size_t kc,
    t1* __restrict Ap)
{
    constexpr size_t MR = 4;

    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const size_t mm = min(MR, mc - ip * MR);
        t1* dst = Ap + ip * kc * MR;

        for (size_t p = 0; p < kc; ++p) {
            const t1* src = A + p * ldA + ip * MR;
            for (size_t i = 0; i < mm; ++i)
                dst[i] = src[i];
            for (size_t i = mm; i < MR; ++i)
                dst[i] = t1(0);
            dst += MR;
        }
    }
}
template<typename t1, typename t2>
inline void __pack_AT_block(
    const t1* __restrict Vd,
    size_t ldV, size_t pc,
    size_t mc, size_t kc,
    t1* __restrict Ap)
{
    constexpr size_t MR = 4;
    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const size_t mm = min(MR, mc - ip * MR);
        t1* dst = Ap + ip * kc * MR;
        for (size_t q = 0; q < kc; ++q) {
            const t1* src = Vd + (ip * MR) * ldV + pc + q;

            for (size_t i = 0; i < mm; ++i)
                dst[i] = src[i * ldV];
            for (size_t i = mm; i < MR; ++i)
                dst[i] = 0;
            dst += MR;
        }
    }
}

template<typename t1, typename t2>
inline void __pack_B_block(
    const t2* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    t2* __restrict Bp)
{
    constexpr size_t NR = 4;

    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const size_t nn = min(NR, nc - jp * NR);
        t2* dst = Bp + jp * kc * NR;

        for (size_t p = 0; p < kc; ++p) {
            for (size_t j = 0; j < nn; ++j)
                dst[p * NR + j] = B[(jp * NR + j) * ldB + p];
            for (size_t j = nn; j < NR; ++j)
                dst[p * NR + j] = 0;
        }
    }
}

template<typename t1, typename t2>
inline void __pack_BT_block(
    const t2* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    t2* __restrict Bp)
{
    constexpr int NR = 4;
    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        t2* dst = Bp + jp * kc * NR;
        for (size_t p = 0; p < kc; ++p) {
            const t2* src = B + p * ldB + jp * NR;

            for (int j = 0; j < nn; ++j)
                dst[j] = src[j];
            for (int j = nn; j < NR; ++j)
                dst[j] = 0;
            dst += NR;
        }
    }
}

template<typename t1, typename t2>
inline void __pack_BT_block_d(
    const t2* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    const t2* __restrict d,
    t2* __restrict Bp)
{
    constexpr int NR = 4;
    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        t2* dst = Bp + jp * kc * NR;
        for (size_t p = 0; p < kc; ++p) {
            const t2 dp = d[p];
            const t2* src = B + p * ldB + jp * NR;

            for (int j = 0; j < nn; ++j)
                dst[j] = src[j] * dp;
            for (int j = nn; j < NR; ++j)
                dst[j] = 0;
            dst += NR;
        }
    }
}

template<typename t1, typename t2>
inline void __pack_A_block_L(
    const t1* __restrict Vd,
    size_t ldV, size_t rows, size_t cols,
    t1* __restrict Lp)
{
    constexpr size_t MR = __micro_kernel_mr<t1, t1>();
    const size_t s = min(rows, ((cols + MR - 1) / MR) * MR);

    for (size_t r = 0; r < s; r += MR) {
        const size_t m = min(MR, s - r);
        const size_t p_max = min(cols, r + m);
        t1* dst = Lp + r * cols;

        for (size_t p = 0; p < p_max; ++p) {
            const t1* Vc = Vd + p * ldV;
            if (p < r) {
                for (size_t i = 0; i < m; ++i)
                    dst[i] = Vc[r + i];
            }
            else {
                for (size_t i = 0; i < m; ++i) {
                    size_t row = r + i;
                    if (row > p) {
                        dst[i] = Vc[row];
                    }
                    else if (row == p) {
                        dst[i] = 1;
                    }
                    else {
                        dst[i] = 0;
                    }
                }
            }
            for (size_t i = m; i < MR; ++i)
                dst[i] = 0;
            dst += MR;
        }
    }

    if (s < rows) {
        __pack_A_block<t1, t1>(Vd + s, ldV, rows - s, cols,
            Lp + (s / MR) * cols * MR);
    }
}

template<typename t1, typename t2>
inline void __pack_A_block_LT(
    const t1* __restrict Vd,
    size_t ldV, size_t rows, size_t cols,
    t1* __restrict LTp)
{
    constexpr size_t KC = __BlockSize_Mul_KC;
    constexpr size_t MR = __micro_kernel_mr<t1, t1>();

    for (size_t r = 0; r < cols; r += MR) {
        const size_t m = min(MR, cols - r);
        t1* dst = LTp + r * cols;
        for (size_t p = 0; p < cols; ++p) {
            for (size_t i = 0; i < m; ++i) {
                const size_t row = r + i;
                if (row < p) {
                    dst[i] = Vd[row * ldV + p];
                }
                else if (row == p) {
                    dst[i] = 1;
                }
                else {
                    dst[i] = 0;
                }
            }
            for (size_t i = m; i < MR; ++i)
                dst[i] = 0;
            dst += MR;
        }
    }

    const size_t N = (cols + MR - 1) / MR;
    const size_t tri_size = N * cols * MR;
    const size_t full_block_size = N * KC * MR;

    t1* dst = LTp + tri_size;
    for (size_t r = cols; r < rows; r += KC) {
        const size_t kc = min(KC, rows - r);
        __pack_AT_block<t1, t1>(Vd, ldV, r, cols, kc, dst);
        dst += full_block_size;
    }
}

template<typename t1, typename t2>
inline void __pack_A_block_U(
    const t1* __restrict Tmat,
    size_t k, t1* __restrict Up)
{
    constexpr size_t MR = __micro_kernel_mr<t1, t1>();

    for (size_t r = 0; r < k; r += MR) {
        const size_t m = min(MR, k - r);
        t1* dst = Up + r * (k + MR);
        for (size_t p = r; p < k; ++p) {
            if (r + m <= p + 1) {
                for (size_t i = 0; i < m; ++i)
                    dst[i] = Tmat[p * k + (r + i)];
            }
            else {
                for (size_t i = 0; i < m; ++i) {
                    size_t row = r + i;
                    dst[i] = (row > p) ? 0 : Tmat[p * k + row];
                }
            }
            for (size_t i = m; i < MR; ++i)
                dst[i] = 0;
            dst += MR;
        }
    }
}

template<typename t1, typename t2>
inline void __pack_A_block_UT(
    const t1* __restrict Tmat,
    size_t k, t1* __restrict UTp)
{
    constexpr size_t MR = __micro_kernel_mr<t1, t1>();

    for (size_t r = 0; r < k; r += MR) {
        const size_t m = min(MR, k - r);
        const size_t p_max = min(k, r + m);
        t1* dst = UTp + r * k;
        for (size_t p = 0; p < p_max; ++p) {
            if (r > p) {
                for (size_t i = 0; i < m; ++i)
                    dst[i] = Tmat[(r + i) * k + p];
            }
            else {
                for (size_t i = 0; i < m; ++i) {
                    size_t row = r + i;
                    dst[i] = (row < p) ? 0 : Tmat[row * k + p];
                }
            }
            for (size_t i = m; i < MR; ++i)
                dst[i] = 0;
            dst += MR;
        }
    }
}