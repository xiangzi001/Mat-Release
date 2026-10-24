
#ifndef XiangZiMatImport
#define XiangZiMatImport

//
// 
// void random(MAT<T>& A, int r, int c, int Big, int Small)
// void random(MAT<complex<T>>& A, int r, int c, int Big, int Small)
// MAT<t> _T(MAT<t>& A)
// MAT<complex<t>> _H(MAT<complex<t>>& A)
// 
// double Norm_one(MAT<T>& A)
// double Norm_inf(MAT<T>& A)
// T Norm_F(MAT<T>& A)
// 
// T Det(MAT<T>& n)
// void Solve_direct(MAT<T>& A, MAT<T>& B)
// MAT<T> Inv(MAT<T>& A)
// 
// void LU(MAT<t1>& A, MAT<t2>& L, MAT<t2>& U)
// void LU_getLU(MAT<t1>& LU, MAT<t2>& L, MAT<t2>& U)
// void PLU_getP(int* pos, MAT<t>& P)
// void LLT(MAT<T>& A, MAT<T>& L)
// void LDLT(MAT<T>& A, MAT<T>& L, MAT<T>& D)
//  
// void QR(MAT<t1>& A, MAT<t2>& Q, MAT<t2>& R, QR_type type)
// void QR(MAT<t1>& A, MAT<complex<t2>>& Q, MAT<complex<t2>>& R, QR_type type)
// 
// void Schur_R(MAT<t1>& A, MAT<t2>& T, MAT<t2>& U, t2 eps)
// void Schur_I(MAT<t1>& A, MAT<complex<t2>>& T, MAT<complex<t2>>& U, t2 eps)
// 
// void EVD_H(MAT<t1>& A, MAT<t2>& E, MAT<t2>& Q, t2 eps, EVD_type type)
// void EVD(MAT<t1>& A, MAT<complex<t2>>& E, MAT<complex<t2>>& X, t2 eps)
// void EVD(MAT<complex<t1>>& A, MAT<complex<t2>>& E, MAT<complex<t2>>& X, t2 eps)
// 
// void EV(MAT<t1>& A, complex<t2>* E, t2 eps)
// void EV(MAT<complex<t1>>& A, complex<t2>* E, t2 eps)
// void Roots(Poly<t1>& func, complex<t2>* ans, t2 eps)
// 
// void SVD(MAT<t1>& A, MAT<t2>& U, MAT<t2>& S, MAT<t2>& V, t2 eps, SVD_type type)
// 

enum QR_type {
	QR_Householder,
	QR_Givens,
	QR_GramSchmidt
};
enum EVD_type {
	EVD_QR,
	EVD_Jacobi,
	EVD_CDC
};
enum SVD_type {
	SVD_QR,
	SVD_Jacobi
};

#define __AVX2__

#ifdef  __AVX512__

#ifndef __BlockSize_Mul_MC
#define __BlockSize_Mul_MC 288
#endif // !__BlockSize_Mul_MC

#ifndef __BlockSize_Mul_KC
#define __BlockSize_Mul_KC 192
#endif // !__BlockSize_Mul_KC

#ifndef __BlockSize_Mul_NC
#define __BlockSize_Mul_NC 288 * 2
#endif // !__BlockSize_Mul_NC


#ifndef __BlockSize_Mul_sK_MC
#define __BlockSize_Mul_sK_MC 288 * 4
#endif // !__BlockSize_Mul_sK_MC

#ifndef __BlockSize_Mul_sK_NC
#define __BlockSize_Mul_sK_NC 192
#endif // !__BlockSize_Mul_sK_NC


#ifndef __BlockSize_Mul_sM_KC
#define __BlockSize_Mul_sM_KC 192
#endif // !__BlockSize_Mul_sM_KC

#ifndef __BlockSize_Mul_sM_NC
#define __BlockSize_Mul_sM_NC 288 * 4
#endif // !__BlockSize_Mul_sM_NC

#ifndef __BlockSize_LU
#define __BlockSize_LU 60
#endif // !__BlockSize_LU

#ifndef __BlockSize_LLT
#define __BlockSize_LLT 48
#endif // !__BlockSize_LLT

#ifndef __BlockSize_QR
#define __BlockSize_QR 48
#endif

#elif defined(__AVX2__)

#ifndef __BlockSize_Mul_MC
#define __BlockSize_Mul_MC 288
#endif // !__BlockSize_Mul_sK_MC

#ifndef __BlockSize_Mul_KC
#define __BlockSize_Mul_KC 192 * 2
#endif // !__BlockSize_Mul_KC

#ifndef __BlockSize_Mul_NC
#define __BlockSize_Mul_NC 288 * 2
#endif // !__BlockSize_Mul_sK_NC


#ifndef __BlockSize_Mul_sK_MC
#define __BlockSize_Mul_sK_MC 288 * 4
#endif // !__BlockSize_Mul_sK_MC

#ifndef __BlockSize_Mul_sK_NC
#define __BlockSize_Mul_sK_NC 144
#endif // !__BlockSize_Mul_sK_NC


#ifndef __BlockSize_Mul_sM_KC
#define __BlockSize_Mul_sM_KC 192 * 2
#endif // !__BlockSize_Mul_sK_MC

#ifndef __BlockSize_Mul_sM_NC
#define __BlockSize_Mul_sM_NC 288 * 2
#endif // !__BlockSize_Mul_sK_NC

#ifndef __BlockSize_LU
#define __BlockSize_LU 64
#endif // !__BlockSize_LU

#ifndef __BlockSize_LLT
#define __BlockSize_LLT 48
#endif // !__BlockSize_LLT

#ifndef __BlockSize_QR
#define __BlockSize_QR 64
#endif

#endif

#include "Calculate/Calculate.h"
#include "Base/Base.h"

template<typename T>
std::ostream& operator << (std::ostream& out, const MAT<T>& A) {
	for (int x = 0; x < A.rows; ++x) {
		for (int y = 0; y < A.cols; ++y) {
			out << A.data[y * A.rows + x];
			out << ",";
		}
		out << std::endl;
	}

	return out;
}
#include "Mul/Mul.h"

#include "LU/LU.h"

#include "QR/QR.h"

#include "Schur/Schur.h"
#include "EVD/EVD.h"

#include "EVD/EVD_H.h"

#include "EV/EV.h"

#include "SVD/SVD.h"


template<typename t1, typename t2>
void LU_init(MAT<t1>& A, MAT<t2>& L, MAT<t2>& U) {
	L.init(A.rows, A.rows);
	U.init(A.rows, A.cols);
}
template<typename t1, typename t2>
void QR_init(MAT<t1>& A, MAT<t2>& Q, MAT<t2>& R) {
	Q.init(A.rows, A.rows);
	R.init(A.rows, A.cols);
}
template<typename t1, typename t2>
void EVD_init(MAT<t1>& A, MAT<t2>& E, MAT<t2>& X) {
	E.init(A.rows, A.rows);
	X.init(A.rows, A.rows);
}
template<typename t1, typename t2>
void SVD_init(MAT<t1>& A, MAT<t2>& U, MAT<t2>& S, MAT<t2>& V) {
	U.init(A.rows, A.rows);
	S.init(A.rows, A.cols);
	V.init(A.cols, A.cols);
}
#endif