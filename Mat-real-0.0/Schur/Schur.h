
#include"__Schur_R.h"
#include"__Schur_I.h"

template<typename t1, typename t2>
void Schur_R(MAT<t1>& A, MAT<t2>& T, MAT<t2>& U, t2 eps) {

	if (A.data != T.data)
		__copy(T.data, A.data, T.rows * T.cols);

	t2* v = new t2[A.rows];
	__Schur_Householder_R(U, T, v);
	
	int i, j;
	for (i = 1; i < T.rows; ++i) {
		v[i] = T(i, i - 1);
	}
	for (i = 0; i < T.rows; ++i) {
		for (j = 0; j < i; ++j) {
			T(i, j) = T(j, i);
		}
	}
	for (i = 2; i < T.rows; ++i) {
		__set0(T[i], i - 1);
	}
	for (i = 1; i < T.rows; ++i) {
		T(i - 1, i) = v[i];
	}

	__Schur_QR_R(U, T, 0, T.rows, eps);

	for (i = 1; i < T.rows; ++i) {
		v[i] = T(i - 1, i);
	}
	for (i = 0; i < T.rows; ++i) {
		for (j = i + 1; j < T.cols; ++j) {
			T(i, j) = T(j, i);
		}
	}
	for (i = 0; i < T.rows - 2; ++i) {
		__set0(T[i] + i + 2, T.rows - i - 2);
	}
	for (i = 1; i < T.rows; ++i) {
		T(i, i - 1) = v[i];
	}

	delete[] v;
}

template<typename t1, typename t2>
void Schur_I(MAT<t1>& A, MAT<complex<t2>>& T, MAT<complex<t2>>& U, t2 eps) {

	if (A.data != T.data)
		__copy(T.data, A.data, T.rows * T.cols);

	complex<t2>* v = new complex<t2>[A.rows];

	__Schur_Householder_I(U, T, v);

	delete[] v;

	__Schur_Givens_I(U, T, 0, T.rows, eps);
}
