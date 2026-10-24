
#include"__QR_I.h"
#include"__QR_R.h"

#include"__QR_H.h"

template<typename t>
void __EV(MAT<t>& A, complex<t>* v, t eps) {
	int i;
	t a;
	complex<t> b;

	__EV_Householder_R(A, (t*)v, (t*)v + A.rows);

	__EV_QR_R(A, 0, A.rows, eps);

	i = 0;
	while (i < A.cols - 1) {
		if (abs(A(i + 1, i)) < eps) {
			v[i] = A(i, i);
			++i;
		}
		else {
			a = A(i, i) + A(i + 1, i + 1);
			b = sqrt((complex<t>)(a * a - (A(i, i) * A(i + 1, i + 1) - A(i, i + 1) * A(i + 1, i)) * 4));
			v[i] = ((complex<t>)a + b) * 0.5;
			v[i + 1] = ((complex<t>)a - b) * 0.5;
			i += 2;
		}
	}
	if (i < A.cols) {
		v[i] = A(i, i);
	}
}
template<typename t>
void __EV(MAT<complex<t>>& A, complex<t>* v, t eps) {
	int i;
	complex<t>* w = new complex<t>[A.cols];
	__EV_Householder_I(A, w, v);
	delete[] w;
	__EV_Givens_I(A, 0, A.rows, eps);

	for (i = 0; i < A.rows; ++i) {
		v[i] = A(i, i);
	}
}

template<typename t1, typename t2>
void EV(MAT<t1>& A, complex<t2>* v, t2 eps) {
	MAT<t2>B = A;
	__EV(B, v, eps);
}

template<typename t1, typename t2>
void EV(MAT<complex<t1>>& A, complex<t2>* v, t2 eps) {
	MAT<complex<t2>>B = A;
	__EV(B, v, eps);
}

template<typename t1, typename t2>
void EV_H(MAT<t1>& A, t2* v, t2 eps) {
	MAT<t2>B = A;

	__EV_H(B, v, eps);
}

#ifdef XiangZiFuncImport

template<typename t1, typename t2>
void Roots(Poly<complex<t1>>& func, complex<t2>* ans, t2 eps) {
	int n (func.times);
	complex<t1>* data (func.data);
	complex<t2> temp(-complex<t2>(1) / data[n]);
	MAT<complex<t2>> A(n, n);

	--n;
	int i;
	for (i = 0; i < A.rows; ++i)
		A(0, i) = data[n - i] * temp;

	n = A.rows * A.rows;
	for (i = A.rows; i < n; ++i)
		A.data[i] = 0;
	for (i = 1; i < A.rows; ++i) 
		A(i, i - 1) = 1;
	
	__EV_Givens_I(A, 0, A.rows, eps);

	for (i = 0; i < A.rows; ++i) {
		ans[i] = A(i, i);
	}
}
template<typename t1, typename t2>
void Roots(Poly<t1>& func, complex<t2>* ans, t2 eps) {
	int n(func.times);
	t1* data(func.data);
	t2 temp(-1 / (t2)(data[n]));
	complex<t2> a, b;
	MAT<t2> A(n, n);

	int i;
	__set0(A.data, n * n);
	--n;
	for (i = 0; i < A.rows; ++i)
		A(0, i) = (t2)(data[n - i]) * temp;
	for (i = 1; i < A.rows; ++i)
		A(i, i - 1) = 1;

	__EV_QR_R(A, 0, A.rows, eps);

	n = 0;
	while (n < A.cols - 1) {
		if (abs(A(n + 1, n)) < eps) {
			ans[n] = A(n, n);
			++n;
		}
		else {
			temp = A(n, n) + A(n + 1, n + 1);
			b = complex<t2>(0, sqrt(-(temp * temp - (A(n, n) * A(n + 1, n + 1) - A(n, n + 1) * A(n + 1, n)) * 4)));
			ans[n] = ((complex<t2>)temp + b) * 0.5;
			ans[n + 1] = ((complex<t2>)temp - b) * 0.5;
			n += 2;
		}
	}
	if (n < A.cols) {
		ans[n] = A(n, n);
	}
}

#endif // XiangZiFuncImport
