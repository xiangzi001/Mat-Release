template<typename t1, typename t2, typename t3>
static void __micro_kernel_mul_unpacked_full(
    const t1* __restrict A, size_t ldA,
    const t2* __restrict B, size_t ldB,
    t3* __restrict C, size_t ldC,
    size_t k)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    // 先从 C 读出初值
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
        const t1 a0 = A[p * ldA + 0];
        const t1 a1 = A[p * ldA + 1];
        const t1 a2 = A[p * ldA + 2];
        const t1 a3 = A[p * ldA + 3];
        const t2 b0 = B[p + (size_t)0 * ldB];
        const t2 b1 = B[p + (size_t)1 * ldB];
        const t2 b2 = B[p + (size_t)2 * ldB];
        const t2 b3 = B[p + (size_t)3 * ldB];

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
static void __micro_kernel_mul_unpacked_edge(
    const t1* __restrict A, size_t ldA,
    const t2* __restrict B, size_t ldB,
    t3* __restrict C, size_t ldC,
    size_t k, int mr, int nr)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    const t3 z = t3(0);

    // 第 0 列
    t3 c00 = (nr > 0 && mr > 0) ? C[0 * ldC + 0] : z;
    t3 c10 = (nr > 0 && mr > 1) ? C[0 * ldC + 1] : z;
    t3 c20 = (nr > 0 && mr > 2) ? C[0 * ldC + 2] : z;
    t3 c30 = (nr > 0 && mr > 3) ? C[0 * ldC + 3] : z;
    // 第 1 列
    t3 c01 = (nr > 1 && mr > 0) ? C[1 * ldC + 0] : z;
    t3 c11 = (nr > 1 && mr > 1) ? C[1 * ldC + 1] : z;
    t3 c21 = (nr > 1 && mr > 2) ? C[1 * ldC + 2] : z;
    t3 c31 = (nr > 1 && mr > 3) ? C[1 * ldC + 3] : z;
    // 第 2 列
    t3 c02 = (nr > 2 && mr > 0) ? C[2 * ldC + 0] : z;
    t3 c12 = (nr > 2 && mr > 1) ? C[2 * ldC + 1] : z;
    t3 c22 = (nr > 2 && mr > 2) ? C[2 * ldC + 2] : z;
    t3 c32 = (nr > 2 && mr > 3) ? C[2 * ldC + 3] : z;
    // 第 3 列
    t3 c03 = (nr > 3 && mr > 0) ? C[3 * ldC + 0] : z;
    t3 c13 = (nr > 3 && mr > 1) ? C[3 * ldC + 1] : z;
    t3 c23 = (nr > 3 && mr > 2) ? C[3 * ldC + 2] : z;
    t3 c33 = (nr > 3 && mr > 3) ? C[3 * ldC + 3] : z;

    for (size_t p = 0; p < k; ++p) {
        // unpacked 无补 0，越界处条件读
        const t1 a0 = A[p * ldA + 0];
        const t1 a1 = (mr > 1) ? A[p * ldA + 1] : t1(0);
        const t1 a2 = (mr > 2) ? A[p * ldA + 2] : t1(0);
        const t1 a3 = (mr > 3) ? A[p * ldA + 3] : t1(0);
        const t2 b0 = B[p + (size_t)0 * ldB];
        const t2 b1 = (nr > 1) ? B[p + (size_t)1 * ldB] : t2(0);
        const t2 b2 = (nr > 2) ? B[p + (size_t)2 * ldB] : t2(0);
        const t2 b3 = (nr > 3) ? B[p + (size_t)3 * ldB] : t2(0);

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
    }

    // 按列写回
    if (nr > 0) {
        t3* __restrict Cj = C + 0 * ldC;
        if (mr > 0) Cj[0] = c00;
        if (mr > 1) Cj[1] = c10;
        if (mr > 2) Cj[2] = c20;
        if (mr > 3) Cj[3] = c30;
    }
    if (nr > 1) {
        t3* __restrict Cj = C + 1 * ldC;
        if (mr > 0) Cj[0] = c01;
        if (mr > 1) Cj[1] = c11;
        if (mr > 2) Cj[2] = c21;
        if (mr > 3) Cj[3] = c31;
    }
    if (nr > 2) {
        t3* __restrict Cj = C + 2 * ldC;
        if (mr > 0) Cj[0] = c02;
        if (mr > 1) Cj[1] = c12;
        if (mr > 2) Cj[2] = c22;
        if (mr > 3) Cj[3] = c32;
    }
    if (nr > 3) {
        t3* __restrict Cj = C + 3 * ldC;
        if (mr > 0) Cj[0] = c03;
        if (mr > 1) Cj[1] = c13;
        if (mr > 2) Cj[2] = c23;
        if (mr > 3) Cj[3] = c33;
    }
}