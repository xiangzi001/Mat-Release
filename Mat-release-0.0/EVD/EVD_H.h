
#include"__Householder_Blocked.h"

#include"__QR.h"

#include"__CDC_base.h"
#include"__CDC.h"

#include"__Jacobi.h"

template<typename t1, typename t2>
inline void EVD_H(MAT<t1>& A, MAT<t2>& E, MAT<t2>& Q, t2 eps, EVD_type type) {
	if((void*)A.data != (void*)E.data)
		__copy(E.data, A.data, E.rows * E.cols);
	switch (type)
	{
	case EVD_QR: {
		__EVD_H_QR(E, Q, eps);
		break;
	}
	case EVD_Jacobi: {
		__EVD_H_Jacobi(E, Q, eps);
		break;
	}
	case EVD_CDC: {
		if (A.rows <= 72) {
			__EVD_H_QR(E, Q, eps);
		}
		else {
			__EVD_H_CDC(E, Q, eps);
		}
		break;
	}
	default:
		break;
	}
}

template<EVD_type type, typename t1, typename t2>
inline void EVD_H(MAT<t1>& A, MAT<t2>& E, MAT<t2>& Q, t2 eps) {
	if ((void*)A.data != (void*)E.data)
		__copy(E.data, A.data, E.rows * E.cols);
	if constexpr (type == EVD_QR) {
		__EVD_H_QR(E, Q, eps);
	}
	else if constexpr (type == EVD_Jacobi) {
		__EVD_H_Jacobi(E, Q, eps);
	}
}

