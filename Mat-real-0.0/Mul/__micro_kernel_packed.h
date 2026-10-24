template<typename t1, typename t2>
constexpr int __micro_kernel_mr() {
    return 4;
}

template<typename t1, typename t2>
constexpr int __micro_kernel_nr() {
    return 4;
}

template<typename t1, typename t2, typename t3>
static void __micro_kernel_mul_packed_full(
    const t1* __restrict Ap,
    const t2* __restrict Bp,
    t3* __restrict C,
    size_t ldC, size_t k)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    t3 c00 = C[0 * ldC + 0];
    t3 c10 = C[0 * ldC + 1];
    t3 c20 = C[0 * ldC + 2];
    t3 c30 = C[0 * ldC + 3];
    t3 c01 = C[1 * ldC + 0];
    t3 c11 = C[1 * ldC + 1];
    t3 c21 = C[1 * ldC + 2];
    t3 c31 = C[1 * ldC + 3];
    t3 c02 = C[2 * ldC + 0];
    t3 c12 = C[2 * ldC + 1];
    t3 c22 = C[2 * ldC + 2];
    t3 c32 = C[2 * ldC + 3];
    t3 c03 = C[3 * ldC + 0];
    t3 c13 = C[3 * ldC + 1];
    t3 c23 = C[3 * ldC + 2];
    t3 c33 = C[3 * ldC + 3];

    for (size_t p = 0; p < k; ++p) {
        const t1 a0 = Ap[p * MR + 0];
        const t1 a1 = Ap[p * MR + 1];
        const t1 a2 = Ap[p * MR + 2];
        const t1 a3 = Ap[p * MR + 3];
        const t2 b0 = Bp[p * NR + 0];
        const t2 b1 = Bp[p * NR + 1];
        const t2 b2 = Bp[p * NR + 2];
        const t2 b3 = Bp[p * NR + 3];

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
    }

    C[0 * ldC + 0] = c00;
    C[0 * ldC + 1] = c10;
    C[0 * ldC + 2] = c20;
    C[0 * ldC + 3] = c30;
    C[1 * ldC + 0] = c01;
    C[1 * ldC + 1] = c11;
    C[1 * ldC + 2] = c21;
    C[1 * ldC + 3] = c31;
    C[2 * ldC + 0] = c02;
    C[2 * ldC + 1] = c12;
    C[2 * ldC + 2] = c22;
    C[2 * ldC + 3] = c32;
    C[3 * ldC + 0] = c03;
    C[3 * ldC + 1] = c13;
    C[3 * ldC + 2] = c23;
    C[3 * ldC + 3] = c33;
}

template<typename t1, typename t2, typename t3>
static void __micro_kernel_mul_packed_edge(
    const t1* __restrict Ap,
    const t2* __restrict Bp,
    t3* __restrict C,
    size_t ldC, size_t k,
    int mr, int nr)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    t3 c00, c10, c20, c30;
    t3 c01, c11, c21, c31;
    t3 c02, c12, c22, c32;
    t3 c03, c13, c23, c33;

    if (nr > 0) {
        c00 = C[0 * ldC + 0];
        c10 = C[0 * ldC + 1];
        c20 = C[0 * ldC + 2];
        c30 = C[0 * ldC + 3];
    }
    if (nr > 1) {
        c01 = C[1 * ldC + 0];
        c11 = C[1 * ldC + 1];
        c21 = C[1 * ldC + 2];
        c31 = C[1 * ldC + 3];
    }
    if (nr > 2) {
        c02 = C[2 * ldC + 0];
        c12 = C[2 * ldC + 1];
        c22 = C[2 * ldC + 2];
        c32 = C[2 * ldC + 3];
    }
    if (nr > 3) {
        c03 = C[3 * ldC + 0];
        c13 = C[3 * ldC + 1];
        c23 = C[3 * ldC + 2];
        c33 = C[3 * ldC + 3];
    }

    for (size_t p = 0; p < k; ++p) {
        const t1 a0 = Ap[p * MR + 0];
        const t1 a1 = Ap[p * MR + 1];
        const t1 a2 = Ap[p * MR + 2];
        const t1 a3 = Ap[p * MR + 3];
        const t2 b0 = Bp[p * NR + 0];
        const t2 b1 = Bp[p * NR + 1];
        const t2 b2 = Bp[p * NR + 2];
        const t2 b3 = Bp[p * NR + 3];

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
    }

    if (nr > 0) {
        const t3* Cj = C + 0 * ldC;
        if (mr > 0) Cj[0] = c00;
        if (mr > 1) Cj[1] = c10;
        if (mr > 2) Cj[2] = c20;
        if (mr > 3) Cj[3] = c30;
    }
    if (nr > 1) {
        const t3* Cj = C + 1 * ldC;
        if (mr > 0) Cj[0] = c01;
        if (mr > 1) Cj[1] = c11;
        if (mr > 2) Cj[2] = c21;
        if (mr > 3) Cj[3] = c31;
    }
    if (nr > 2) {
        const t3* Cj = C + 2 * ldC;
        if (mr > 0) Cj[0] = c02;
        if (mr > 1) Cj[1] = c12;
        if (mr > 2) Cj[2] = c22;
        if (mr > 3) Cj[3] = c32;
    }
    if (nr > 3) {
        const t3* Cj = C + 3 * ldC;
        if (mr > 0) Cj[0] = c03;
        if (mr > 1) Cj[1] = c13;
        if (mr > 2) Cj[2] = c23;
        if (mr > 3) Cj[3] = c33;
    }
}


template<typename t1, typename t2, typename t3>
static void __micro_kernel_nmul_packed_full(
    const t1* __restrict Ap,
    const t2* __restrict Bp,
    t3* __restrict C,
    size_t ldC, size_t k)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    t3 c00 = C[0 * ldC + 0];
    t3 c10 = C[0 * ldC + 1];
    t3 c20 = C[0 * ldC + 2];
    t3 c30 = C[0 * ldC + 3];
    t3 c01 = C[1 * ldC + 0];
    t3 c11 = C[1 * ldC + 1];
    t3 c21 = C[1 * ldC + 2];
    t3 c31 = C[1 * ldC + 3];
    t3 c02 = C[2 * ldC + 0];
    t3 c12 = C[2 * ldC + 1];
    t3 c22 = C[2 * ldC + 2];
    t3 c32 = C[2 * ldC + 3];
    t3 c03 = C[3 * ldC + 0];
    t3 c13 = C[3 * ldC + 1];
    t3 c23 = C[3 * ldC + 2];
    t3 c33 = C[3 * ldC + 3];

    for (size_t p = 0; p < k; ++p) {
        const t1 a0 = Ap[p * MR + 0];
        const t1 a1 = Ap[p * MR + 1];
        const t1 a2 = Ap[p * MR + 2];
        const t1 a3 = Ap[p * MR + 3];
        const t2 b0 = Bp[p * NR + 0];
        const t2 b1 = Bp[p * NR + 1];
        const t2 b2 = Bp[p * NR + 2];
        const t2 b3 = Bp[p * NR + 3];

        c00 -= a0 * b0; c01 -= a0 * b1; c02 -= a0 * b2; c03 -= a0 * b3;
        c10 -= a1 * b0; c11 -= a1 * b1; c12 -= a1 * b2; c13 -= a1 * b3;
        c20 -= a2 * b0; c21 -= a2 * b1; c22 -= a2 * b2; c23 -= a2 * b3;
        c30 -= a3 * b0; c31 -= a3 * b1; c32 -= a3 * b2; c33 -= a3 * b3;
    }

    C[0 * ldC + 0] = c00;
    C[0 * ldC + 1] = c10;
    C[0 * ldC + 2] = c20;
    C[0 * ldC + 3] = c30;
    C[1 * ldC + 0] = c01;
    C[1 * ldC + 1] = c11;
    C[1 * ldC + 2] = c21;
    C[1 * ldC + 3] = c31;
    C[2 * ldC + 0] = c02;
    C[2 * ldC + 1] = c12;
    C[2 * ldC + 2] = c22;
    C[2 * ldC + 3] = c32;
    C[3 * ldC + 0] = c03;
    C[3 * ldC + 1] = c13;
    C[3 * ldC + 2] = c23;
    C[3 * ldC + 3] = c33;
}

template<typename t1, typename t2, typename t3>
static void __micro_kernel_nmul_packed_edge(
    const t1* __restrict Ap,
    const t2* __restrict Bp,
    t3* __restrict C,
    size_t ldC, size_t k,
    int mr, int nr)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    t3 c00, c10, c20, c30;
    t3 c01, c11, c21, c31;
    t3 c02, c12, c22, c32;
    t3 c03, c13, c23, c33;

    if (nr > 0) {
        c00 = C[0 * ldC + 0];
        c10 = C[0 * ldC + 1];
        c20 = C[0 * ldC + 2];
        c30 = C[0 * ldC + 3];
    }
    if (nr > 1) {
        c01 = C[1 * ldC + 0];
        c11 = C[1 * ldC + 1];
        c21 = C[1 * ldC + 2];
        c31 = C[1 * ldC + 3];
    }
    if (nr > 2) {
        c02 = C[2 * ldC + 0];
        c12 = C[2 * ldC + 1];
        c22 = C[2 * ldC + 2];
        c32 = C[2 * ldC + 3];
    }
    if (nr > 3) {
        c03 = C[3 * ldC + 0];
        c13 = C[3 * ldC + 1];
        c23 = C[3 * ldC + 2];
        c33 = C[3 * ldC + 3];
    }

    for (size_t p = 0; p < k; ++p) {
        const t1 a0 = Ap[p * MR + 0];
        const t1 a1 = Ap[p * MR + 1];
        const t1 a2 = Ap[p * MR + 2];
        const t1 a3 = Ap[p * MR + 3];

        const t2 b0 = Bp[p * NR + 0];
        const t2 b1 = Bp[p * NR + 1];
        const t2 b2 = Bp[p * NR + 2];
        const t2 b3 = Bp[p * NR + 3];

        c00 -= a0 * b0; c01 -= a0 * b1; c02 -= a0 * b2; c03 -= a0 * b3;
        c10 -= a1 * b0; c11 -= a1 * b1; c12 -= a1 * b2; c13 -= a1 * b3;
        c20 -= a2 * b0; c21 -= a2 * b1; c22 -= a2 * b2; c23 -= a2 * b3;
        c30 -= a3 * b0; c31 -= a3 * b1; c32 -= a3 * b2; c33 -= a3 * b3;
    }

    if (nr > 0) {
        const t3* Cj = C + 0 * ldC;
        if (mr > 0) Cj[0] = c00;
        if (mr > 1) Cj[1] = c10;
        if (mr > 2) Cj[2] = c20;
        if (mr > 3) Cj[3] = c30;
    }
    if (nr > 1) {
        const t3* Cj = C + 1 * ldC;
        if (mr > 0) Cj[0] = c01;
        if (mr > 1) Cj[1] = c11;
        if (mr > 2) Cj[2] = c21;
        if (mr > 3) Cj[3] = c31;
    }
    if (nr > 2) {
        const t3* Cj = C + 2 * ldC;
        if (mr > 0) Cj[0] = c02;
        if (mr > 1) Cj[1] = c12;
        if (mr > 2) Cj[2] = c22;
        if (mr > 3) Cj[3] = c32;
    }
    if (nr > 3) {
        const t3* Cj = C + 3 * ldC;
        if (mr > 0) Cj[0] = c03;
        if (mr > 1) Cj[1] = c13;
        if (mr > 2) Cj[2] = c23;
        if (mr > 3) Cj[3] = c33;
    }
}