
template<typename T>
void __LQT_HouseHolder(MAT<T>& Q, MAT<T>& R, T* w, T* v) {
	int n = 0;
	if (R.rows > R.cols) {
		n = R.cols;
	}
	else {
		n = R.rows;
	};
	int r;

	T cr, hr;

	for (r = 0; r < n; ++r) {
		for (int i = r; i < R.cols; ++i) {
			w[i] = R(r, i);
		}
		cr = sqrt(__dot(w + r, w + r, R.cols - r));

		if (w[r] < 0.0)cr = -cr;
		w[r] += cr;
		hr = (T)1 / sqrt(cr * w[r]);

		__times(w + r, hr, w + r, R.cols - r);
		__ApplyHouseHolderOnTheRight(Q, w, v, r);
		__ApplyHouseHolderOnTheRight(R, w, v, r);
	}
}

template <typename T>
inline void __SVD_Jacobi_Rot(MAT<T>& U, MAT<T>& S, MAT<T>& V, int i, int j, T eps) {
	T c, s;
	double Angle;
	T* Temp;
	T temp;

	c = S(i, i) * S(i, j) + S(j, j) * S(j, i);
	s = S(i, j) * S(i, j) + S(j, j) * S(j, j) - S(i, i) * S(i, i) - S(j, i) * S(j, i);

	Angle = -0.5 * atan2(2 * c, s);
	c = cos(Angle);
	s = sin(Angle);

	__ApplyGivensOnTheRight(V[i], V[j], c, s, V.rows);
	__ApplyGivensOnTheRight(S[i], S[j], c, s, S.cols);

	c = S(i, i) * S(j, i) + S(j, j) * S(i, j);
	s = S(j, i) * S(j, i) + S(j, j) * S(j, j) - S(i, i) * S(i, i) - S(i, j) * S(i, j);

	Angle = -0.5 * atan2(2 * c, s);
	c = cos(Angle);
	s = sin(Angle);

	__ApplyGivensOnTheRight(U[i], U[j], c, s, V.rows);

	for (int k = 0; k < S.cols; ++k) {
		Temp = S[k];
		temp = Temp[i];
		Temp[i] = Temp[j] * s + temp * c;
		Temp[j] = Temp[j] * c - temp * s;
	}
}

template <typename T>
void __SVD_Jacobi_2(MAT<T>& U, MAT<T>& S, MAT<T>& V, T eps) {
	int m, n;
	bool flag = true;
	while (flag) {
		flag = false;
		for (m = 1; m < S.cols; ++m) {
			for (n = 0; n < m; ++n) {
				if (fabs(S(n, m)) > eps || fabs(S(m, n)) > eps) {
					flag = true;
					__SVD_Jacobi_Rot(U, S, V, n, m, eps);
				}
			}
		}
	}
}

template <typename T>
void __SVD_Jacobi(MAT<T>& U, MAT<T>& S, MAT<T>& V, T eps) {
	int i;
	__set0(V.data, V.rows * V.cols);
	for (i = 0; i < V.rows; ++i) {
		V[i][i] = 1;
	}
	__set0(U.data, U.rows * U.cols);
	for (i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}

	if (S.rows >= S.cols) {

		__QR_HouseHolder(U, S);

		__SVD_Jacobi_2(U, S, V, eps);
	}
	else {

		T* w = new T[S.cols];
		T* v = new T[S.cols];
		__LQT_HouseHolder(V, S, w, v);
		delete[] w;
		delete[] v;

		i = S.cols;
		S.cols = S.rows;
		__SVD_Jacobi_2(U, S, V, eps);
		S.cols = i;
	}
}
