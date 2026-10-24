
template<typename t1, typename t2, typename t3>
void __mulAB_scalar(const MAT<t1>& A, const MAT<t2>& B, MAT<t3>& C)
{
    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    const int ldA = A.rows;
    const int ldB = B.rows;
    const int ldC = C.rows;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            t3 sum = t3(0);
            for (int k = 0; k < K; ++k) {
                sum += A.data[i + (size_t)k * ldA]
                    * B.data[k + (size_t)j * ldB];
            }
            C.data[i + (size_t)j * ldC] = sum;
        }
    }
}

template<typename t1, typename t2, typename t3>
void __mulAB_unpacked(
    const MAT<t1>& A, const MAT<t2>& B, MAT<t3>& C)
{
    constexpr int MR = __micro_kernel_mr<t1, t2>();
    constexpr int NR = __micro_kernel_nr<t1, t2>();

    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    const int ldA = A.rows;
    const int ldB = B.rows;
    const int ldC = C.rows;

    __set0(C.data, (size_t)M * N);

    for (int jp = 0; jp < N; jp += NR) {
        const int nn = (N - jp < NR) ? (N - jp) : NR;

        for (int ip = 0; ip < M; ip += MR) {
            const int mm = (M - ip < MR) ? (M - ip) : MR;

            const t1* Ap = A.data + ip;
            const t2* Bp = B.data + (size_t)jp * ldB;
            t3* Cp = C.data + (size_t)jp * ldC + ip;

            if (mm == MR && nn == NR) {
                __micro_kernel_mul_unpacked_full(
                    Ap, ldA, Bp, ldB, Cp, ldC, K);
            }
            else {
                __micro_kernel_mul_unpacked_edge(
                    Ap, ldA, Bp, ldB, Cp, ldC, K, mm, nn);
            }
        }
    }
}







template<typename t1, typename t2, typename t3>
void __mulAB_packed(
    const t1* __restrict A_pack, const t2* __restrict B_pack, t3* __restrict C,
    int ldC, int mc, int nc, int kc)
{
    constexpr int MR = __micro_kernel_mr<t1, t2>();
    constexpr int NR = __micro_kernel_nr<t1, t2>();

    const int mr_panels = (mc + MR - 1) / MR;
    const int nr_panels = (nc + NR - 1) / NR;

    for (int jp = 0; jp < nr_panels; ++jp) {
        const t2* Bp = B_pack + (size_t)jp * kc * NR;
        const int nn = (nc - jp * NR < NR) ? (nc - jp * NR) : NR;

        for (int ip = 0; ip < mr_panels; ++ip) {
            const t1* Ap = A_pack + (size_t)ip * kc * MR;
            const int mm = (mc - ip * MR < MR) ? (mc - ip * MR) : MR;

            t3* Cp = C + (size_t)(jp * NR) * ldC + ip * MR;

            if (mm == MR && nn == NR) {
                __micro_kernel_mul_packed_full(Ap, Bp, Cp, ldC, kc);
            }
            else {
                __micro_kernel_mul_packed_edge(Ap, Bp, Cp, ldC, kc, mm, nn);
            }
        }
    }
}

template<typename t>
static void __mulAB_packed_n(
    const t* __restrict A_pack,
    const t* __restrict B_pack,
    t* __restrict C, size_t ldC,
    int m, int n, int k)
{
    constexpr int MR = __micro_kernel_mr<t, t>();
    constexpr int NR = __micro_kernel_nr<t, t>();
    const int mp = (m + MR - 1) / MR;
    const int np = (n + NR - 1) / NR;
    for (int jp = 0; jp < np; ++jp) {
        const t* Bp = B_pack + (size_t)jp * k * NR;
        const int nn = (n - jp * NR < NR) ? (n - jp * NR) : NR;
        for (int ip = 0; ip < mp; ++ip) {
            const t* Ap = A_pack + (size_t)ip * k * MR;
            const int mm = (m - ip * MR < MR) ? (m - ip * MR) : MR;
            t* Cp = C + (size_t)(jp * NR) * ldC + ip * MR;
            if (mm == MR && nn == NR)
                __micro_kernel_nmul_packed_full(Ap, Bp, Cp, ldC, (size_t)k);
            else
                __micro_kernel_nmul_packed_edge(Ap, Bp, Cp, ldC, (size_t)k, mm, nn);
        }
    }
}

template<typename t>
static void __mulAB_packed_n_AL(
    const t* __restrict A_pack,
    const t* __restrict B_pack,
    t* __restrict C, size_t ldC,
    int m, int n, int k, int k_base)
{
    constexpr int MR = __micro_kernel_mr<t, t>();
    constexpr int NR = __micro_kernel_nr<t, t>();
    const int mp = (m + MR - 1) / MR;
    const int np = (n + NR - 1) / NR;
    for (int jp = 0; jp < np; ++jp) {
        const t* Bp_base = B_pack + (size_t)jp * k * NR;
        const int nn = (n - jp * NR < NR) ? (n - jp * NR) : NR;
        for (int ip = 0; ip < mp; ++ip) {
            const t* Ap = A_pack + (size_t)ip * k * MR;
            const int mm = (m - ip * MR < MR) ? (m - ip * MR) : MR;
            const int k_eff = min(k, k_base + ip * MR + mm);
            if (k_eff <= 0) continue;
            t* Cp = C + (size_t)(jp * NR) * ldC + ip * MR;
            if (mm == MR && nn == NR)
                __micro_kernel_nmul_packed_full(Ap, Bp_base, Cp, ldC, (size_t)k_eff);
            else
                __micro_kernel_nmul_packed_edge(Ap, Bp_base, Cp, ldC, (size_t)k_eff, mm, nn);
        }
    }
}

template<typename t>
static void __mulAB_packed_n_AU(
    const t* __restrict A_pack,
    const t* __restrict B_pack,
    t* __restrict C, size_t ldC,
    int m, int n, int k, int k_base)
{
    constexpr int MR = __micro_kernel_mr<t, t>();
    constexpr int NR = __micro_kernel_nr<t, t>();
    const int mp = (m + MR - 1) / MR;
    const int np = (n + NR - 1) / NR;
    for (int jp = 0; jp < np; ++jp) {
        const t* Bp_base = B_pack + (size_t)jp * k * NR;
        const int nn = (n - jp * NR < NR) ? (n - jp * NR) : NR;
        for (int ip = 0; ip < mp; ++ip) {
            const t* Ap_base = A_pack + (size_t)ip * k * MR;
            const int mm = (m - ip * MR < MR) ? (m - ip * MR) : MR;
            const int k_start = max(0, ip * MR - k_base);
            const int k_eff = k - k_start;
            if (k_eff <= 0) continue;
            const t* Ap = Ap_base + (size_t)k_start * MR;
            const t* Bp = Bp_base + (size_t)k_start * NR;
            t* Cp = C + (size_t)(jp * NR) * ldC + ip * MR;
            if (mm == MR && nn == NR)
                __micro_kernel_nmul_packed_full(Ap, Bp, Cp, ldC, (size_t)k_eff);
            else
                __micro_kernel_nmul_packed_edge(Ap, Bp, Cp, ldC, (size_t)k_eff, mm, nn);
        }
    }
}

template<typename t>
static void __mulAB_packed_n_lower(
    const t* __restrict A_pack, const t* __restrict B_pack,
    t* __restrict C, size_t ldC,
    int m, int n, int k)
{
    constexpr int MR = __micro_kernel_mr<t, t>();
    constexpr int NR = __micro_kernel_nr<t, t>();
    const int mp = (m + MR - 1) / MR;
    const int np = (n + NR - 1) / NR;

    for (int jp = 0; jp < np; ++jp) {
        const t* Bp = B_pack + (size_t)jp * k * NR;
        const int nn = (n - jp * NR < NR) ? (n - jp * NR) : NR;

        for (int ip = 0; ip < mp; ++ip) {
            const int mm = (m - ip * MR < MR) ? (m - ip * MR) : MR;

            if (ip * MR + mm <= jp * NR) continue;

            const t* Ap = A_pack + (size_t)ip * k * MR;
            t* Cp = C + (size_t)(jp * NR) * ldC + ip * MR;

            if (mm == MR && nn == NR)
                __micro_kernel_nmul_packed_full(Ap, Bp, Cp, ldC, (size_t)k);
            else
                __micro_kernel_nmul_packed_edge(Ap, Bp, Cp, ldC, (size_t)k, mm, nn);
        }
    }
}
