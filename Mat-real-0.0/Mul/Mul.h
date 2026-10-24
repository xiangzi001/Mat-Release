
#include"__micro_kernel_packed.h"
#include"__micro_kernel_unpacked.h"
#include"__micro_kernel_packed_double.h"
#include"__micro_kernel_unpacked_double.h"

#include"__pack.h"
#include"__pack_double.h"

#include"__MulAB.h"
#include"__MulAB_vec.h"

template<typename t1, typename t2>
auto operator * (const MAT<t1>& A, const MAT<t2>& B) {
    using t3 = decltype(std::declval<t1>()* std::declval<t2>());
    MAT<t3> C(A.rows, B.cols);

    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    if ((M <= 4 && N <= 4 && K <= 4) || (long long)M * (long long)N * (long long)K <= 64) {
        __mulAB_scalar(A, B, C);
        return C;
    }

    if (K <= 4) {
        __mulAB_outer_vec(A, B, C);
        return C;
    }

    if ((M <= 32 && K <= 32) || N <= 16) {
        __mulAB_vec(A, B, C);
        return C;
    }

    if (M <= 128 && N <= 128 && K <= 128) {
        __mulAB_unpacked(A, B, C);
        return C;
    }

    constexpr int MR = __micro_kernel_mr<t1, t2>();
    constexpr int NR = __micro_kernel_nr<t1, t2>();

    constexpr int MC = __BlockSize_Mul_MC;
    constexpr int KC = __BlockSize_Mul_KC;
    constexpr int NC = __BlockSize_Mul_NC;

    __set0(C.data, (size_t)M * N);

#define __align_up(x, a) (x + a - 1) & ~(a - 1)
    const size_t A_pack_size = __align_up(sizeof(t1) * ((MC + MR - 1) / MR) * KC * MR, 64);
    const size_t B_pack_size = __align_up(sizeof(t2) * ((NC + NR - 1) / NR) * KC * NR, 64);
#undef __align_up

    char* pack_buf = (char*)_mm_malloc(A_pack_size + B_pack_size, 64);
    t1* A_pack = (t1*)(pack_buf);
    t2* B_pack = (t2*)(pack_buf + A_pack_size);

    for (int pc = 0; pc < K; pc += KC) {
        const int kc = (K - pc < KC) ? (K - pc) : KC;

        for (int jc = 0; jc < N; jc += NC) {
            const int nc = (N - jc < NC) ? (N - jc) : NC;

            __pack_B_block<t1, t2>(B.data + (size_t)pc + (size_t)jc * B.rows,
                B.rows, kc, nc, B_pack);

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = (M - ic < MC) ? (M - ic) : MC;

                __pack_A_block<t1, t2>(A.data + (size_t)pc * A.rows + ic,
                    A.rows, mc, kc, A_pack);

                __mulAB_packed<t1, t2, t3>(
                    A_pack, B_pack,
                    C.data + (size_t)jc * C.rows + ic,
                    C.rows, mc, nc, kc);
            }
        }
    }

    _mm_free(pack_buf);

    return C;
}

