
template<typename T>
void __SVD_HouseHolder_Up(MAT<T>& A, MAT<T>& U, MAT<T>& V, T* v) {
	int m, n, i;

	T cr, hr;
	T* w;
	T* a = A.data + (A.rows - 1) * A.cols, * b = a - A.cols + 1;

	MAT<T> Temp, Temp2;
	Temp.rows = A.rows;
	Temp.cols = A.cols;
	Temp.data = A.data;
	Temp2.data = A.data;

	for (m = 0; m < A.cols - 1; ++m) {

		w = U.data + 2 * U.rows * m - m * m;
		__copy(w, Temp.data, Temp.rows);
		cr = sqrt(__dot(w, w, Temp.rows));
		if (*w < 0.0)cr = -cr;
		*w += cr;
		__times(w, (T)1 / sqrt(cr * *w), w, Temp.rows);
		for (i = 1; i < Temp.cols; ++i) {
			__mixed(Temp[i], w, -__dot(w, Temp[i], Temp.rows), Temp[i], Temp.rows);
		}
		hr = -cr;

		Temp2.rows = Temp.rows - 1;
		Temp2.cols = Temp.cols - 1;
		n = m + 1;

		w = V.data + 2 * V.rows * m - m * m;
		for (i = 1; i < Temp.cols; ++i) {
			w[i] = Temp(0, i);
		}
		w = w + 1;
		cr = sqrt(__dot(w, w, Temp2.cols));
		if (*w < 0.0)cr = -cr;
		*w += cr;
		__times(w, (T)1 / sqrt(cr * *w), w, Temp2.cols);
		__set0(v, Temp2.rows);
		for (i = 0; i < Temp2.cols; ++i) {
			__mixed(v, Temp[i + 1] + 1, w[i], v, Temp2.rows);
		}
		for (i = 0; i < Temp2.cols; ++i) {
			__mixed(Temp[i + 1] + 1, v, -w[i], Temp2[i], Temp2.rows);
		}

		a[m] = hr;
		b[m] = -cr;

		Temp.rows = Temp2.rows;
		Temp.cols = Temp2.cols;
	}

	w = U.data + 2 * U.rows * m - m * m;
	__copy(w, Temp.data, Temp.rows);
	cr = sqrt(__dot(w, w, Temp.rows));
	if (*w < 0.0)cr = -cr;
	*w += cr;
	__times(w, (T)1 / sqrt(cr * *w), w, Temp.rows);
	for (i = 0; i < Temp.cols; ++i) {
		__mixed(Temp[i], w, -__dot(w, Temp[i], Temp.rows), Temp[i], Temp.rows);
	}
	a[m] = -cr;

	__HouseHolder_Q1(U, A.cols);
	__HouseHolder_Q3(V);

	Temp.data = 0;
	Temp2.data = 0;
}

template<typename T>
void __SVD_HouseHolder_Down(MAT<T>& A, MAT<T>& U, MAT<T>& V, T* v) {
	int m, n, i;

	T cr, hr;
	T* w;
	T* a = A.data + (A.cols - 1) * A.rows, * b = a - A.rows + 1;

	MAT<T> Temp, Temp2;
	Temp.rows = A.rows;
	Temp.cols = A.cols;
	Temp.data = A.data;
	Temp2.data = A.data;

	for (m = 0; m < A.rows - 1; ++m) {

		w = V.data + 2 * V.rows * m - m * m;
		for (i = 0; i < Temp.cols; ++i) {
			w[i] = Temp(0, i);
		}
		cr = sqrt(__dot(w, w, Temp.cols));
		if (*w < 0.0)cr = -cr;
		*w += cr;
		__times(w, (T)1 / sqrt(cr * *w), w, Temp.cols);
		__set0(v, Temp.rows);
		for (i = 0; i < Temp.cols; ++i) {
			__mixed(v, Temp[i], w[i], v, Temp.rows);
		}
		for (i = 0; i < Temp.cols; ++i) {
			__mixed(Temp[i], v, -w[i], Temp[i], Temp.rows);
		}
		hr = -cr;

		Temp2.rows = Temp.rows - 1;
		Temp2.cols = Temp.cols - 1;
		n = m + 1;

		w = U.data + 2 * U.rows * m - m * m + 1;
		__copy(w, Temp.data + 1, Temp2.rows);
		cr = sqrt(__dot(w, w, Temp2.rows));
		if (*w < 0.0)cr = -cr;
		*w += cr;
		__times(w, (T)1 / sqrt(cr * *w), w, Temp2.rows);
		for (i = 0; i < Temp2.cols; ++i) {
			__mixed(Temp[i + 1] + 1, w, -__dot(w, Temp[i + 1] + 1, Temp2.rows), Temp2[i], Temp2.rows);
		}

		a[m] = hr;
		b[m] = -cr;

		Temp.rows = Temp2.rows;
		Temp.cols = Temp2.cols;
	}

	w = V.data + 2 * V.rows * m - m * m;
	for (i = 0; i < Temp.cols; ++i) {
		w[i] = Temp(0, i);
	}
	cr = sqrt(__dot(w, w, Temp.cols));
	if (*w < 0.0)cr = -cr;
	*w += cr;
	__times(w, (T)1 / sqrt(cr * *w), w, Temp.cols);
	__set0(v, Temp.rows);
	for (i = 0; i < Temp.cols; ++i) {
		__mixed(v, Temp[i], w[i], v, Temp.rows);
	}
	for (i = 0; i < Temp.cols; ++i) {
		__mixed(Temp[i], v, -w[i], Temp[i], Temp.rows);
	}
	a[m] = -cr;

	__HouseHolder_Q3(U);
	__HouseHolder_Q1(V, A.rows);

	Temp.data = 0;
	Temp2.data = 0;
}


template<typename T>
void __SVD_Givens_UP(T* a, T* b, MAT<T>& U, MAT<T>& V, int m, int n, T eps) {
	int i;
	T temp, t, s, c, B;

	while (n > m) {
		if (fabs(b[m]) < eps) {
			++m;
			continue;
		}
		if (fabs(b[n - 1]) < eps) {
			--n;
			continue;
		}
		for (i = m + 1; i < n - 1; ++i) {
			if (fabs(b[i]) < eps) {
				__SVD_Givens_UP(a, b, U, V, m, i, eps);
				m = i + 1;
			}
		}

		temp = a[m] * a[m] - a[n] * a[n] - b[n - 1] * b[n - 1];
		if (fabs(b[m] * a[m]) > fabs(temp)) {
			t = temp / (b[m] * a[m]);
			s = 1 / sqrt(t * t + 1);
			c = t * s;
		}
		else {
			t = b[m] * a[m] / temp;
			c = 1 / sqrt(t * t + 1);
			s = t * c;
		}

		temp = c * a[m] + s * b[m];
		b[m] = c * b[m] - s * a[m];
		a[m] = temp;
		B = s * a[m + 1];
		a[m + 1] = c * a[m + 1];

		__ApplyGivensOnTheRight(V[m], V[m + 1], c, s, V.rows);

		if (fabs(B) > fabs(a[m])) {
			t = a[m] / B;
			s = 1 / sqrt(t * t + 1);
			c = t * s;
		}
		else {
			t = B / a[m];
			c = 1 / sqrt(t * t + 1);
			s = t * c;
		}

		a[m] = c * a[m] + s * B;
		temp = c * b[m] + s * a[m + 1];
		a[m + 1] = c * a[m + 1] - s * b[m];
		b[m] = temp;
		B = s * b[m + 1];
		b[m + 1] = c * b[m + 1];

		__ApplyGivensOnTheRight(U[m], U[m + 1], c, s, U.rows);

		for (i = m + 1; i < n; ++i) {

			if (fabs(B) > fabs(b[i - 1])) {
				t = b[i - 1] / B;
				s = 1 / sqrt(t * t + 1);
				c = t * s;
			}
			else {
				t = B / b[i - 1];
				c = 1 / sqrt(t * t + 1);
				s = t * c;
			}

			b[i - 1] = c * b[i - 1] + s * B;
			temp = c * a[i] + s * b[i];
			b[i] = c * b[i] - s * a[i];
			a[i] = temp;
			B = s * a[i + 1];
			a[i + 1] = c * a[i + 1];

			__ApplyGivensOnTheRight(V[i], V[i + 1], c, s, V.rows);

			if (fabs(B) > fabs(a[i])) {
				t = a[i] / B;
				s = 1 / sqrt(t * t + 1);
				c = t * s;
			}
			else {
				t = B / a[i];
				c = 1 / sqrt(t * t + 1);
				s = t * c;
			}

			a[i] = c * a[i] + s * B;
			temp = c * b[i] + s * a[i + 1];
			a[i + 1] = c * a[i + 1] - s * b[i];
			b[i] = temp;
			B = s * b[i + 1];
			b[i + 1] = c * b[i + 1];

			__ApplyGivensOnTheRight(U[i], U[i + 1], c, s, U.rows);

		}
	}
}

template<typename T>
void __SVD_QR(MAT<T>& U, MAT<T>& S, MAT<T>& V, T eps) {
	int i = 0;
	int r;

	T* v;
	if (S.rows > S.cols) {

		v = new T[S.rows];
		__SVD_HouseHolder_Up(S, U, V, v);
		delete[] v;

		T* a = S.data + (S.rows - 1) * S.cols;
		T* b = a - S.cols;

		r = S.cols - 1;
		__copy(b, b + 1, r);
		b[r] = 0;
		__SVD_Givens_UP(a, b, U, V, 0, r, eps);

		__set0(S.data, (S.rows - 1) * S.cols);
		T temp = a[r];
		for (i = 0; i < r; ++i) {
			S(i, i) = a[i];
		}
		__set0(a, S.cols);
		S(r, r) = temp;

	}
	else {
		v = new T[S.cols];
		__SVD_HouseHolder_Down(S, U, V, v);
		delete[] v;

		T* a = S.data + (S.cols - 1) * S.rows;
		T* b = a - S.rows;

		r = S.rows - 1;
		__copy(b, b + 1, r);
		__SVD_Givens_UP(a, b, V, U, 0, r, eps);

		__set0(S.data, (S.cols - 1)* S.rows);
		T temp = a[r];
		for (i = 0; i < r; ++i) {
			S(i, i) = a[i];
		}
		__set0(a, S.rows);
		S(r, r) = temp;
	}
}
