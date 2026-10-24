
template<typename T>
void __EVD_H_Householder(MAT<T>& Q, MAT<T>& R, T* v) {
	int m, n;

	T temp;
	T cr, hr;
	T* w, * a = R[R.rows - 1], * b = R[R.rows - 2] + 1;

	for (m = 0; m < R.rows - 2; ++m) {
		w = Q.data + 2 * Q.rows * m - m * m + 1;

		temp = *R.data;

		n = R.rows - m - 1;
		__copy(w, R.data + 1, n);
		cr = sqrt(__dot(w, w, n));

		if (cr == 0) {
			continue;
		}

		if (*w < 0.0)cr = -cr;
		*w += cr;
		hr = (T)1 / sqrt(cr * *w);

		__times(w, hr, w, n);

		__set0(v, n);
		for (int i = 0; i < n; ++i) {
			__mixed(v + i, R.data + (n + 2) * (i + 1), w[i], v + i, n - i);
		}
		for (int i = 0; i < n; ++i) {
			v[i] += __dot(R.data + (n + 2) * (i + 1) + 1, w + i + 1, n - i - 1);
		}

		hr = __dot(v, w, n);
		for (int i = 0; i < n; ++i) {
			__mixed(R.data + (n + 2) * (i + 1), v + i, -w[i], R.data + (n + 1) * i, n - i);
			__mixed(R.data + (n + 1) * i, w + i, -v[i] + w[i] * hr, R.data + (n + 1) * i, n - i);
		}
		a[m] = temp;
		b[m] = -cr;
	}
	a[R.rows - 2] = R.data[0];
	a[R.rows - 1] = R.data[3];
	b[R.rows - 2] = R.data[1];
	__Householder_Q2(Q);
}


template<typename T>
void __EVD_H_Givens(T* a, T* b, MAT<T>& Q, int m, int n, T eps) {

	T t, temp, temp_2, c2, s2, cs;
	T B(0);
	int i;

	while (n > m) {
		if (abs(b[m]) < eps) {
			++m;
			continue;
		}
		if (abs(b[n - 1]) < eps) {
			--n;
			continue;
		}
		for (i = m + 1; i < n - 1; ++i) {
			if (abs(b[i]) < eps) {
				__EVD_H_Givens(a, b, Q, m, i, eps);
				m = i + 1;
			}
		}

		c2 = a[n - 1] + a[n];
		s2 = sqrt(c2 * c2 - (a[n - 1] * a[n] - b[n - 1] * b[n - 1]) * 4);
		if (c2 - 2 * a[n] > 0) t = (c2 - s2) * 0.5;
		else t = (c2 + s2) * 0.5;

		if (abs(b[m]) > abs(a[m] - t)) {
			t = (a[m] - t) / b[m];
			s2 = 1 / (t * t + 1);
			c2 = 1 - s2;
			cs = t * s2;
			temp = a[m];
			temp_2 = a[m + 1];

			a[m] = c2 * temp + s2 * temp_2 + 2 * cs * b[m];
			a[m + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[m];
			b[m] = -cs * temp + cs * temp_2 + (c2 - s2) * b[m];
			s2 = sqrt(s2);
			c2 = t * s2;
		}
		else {
			t = b[m] / (a[m] - t);
			c2 = 1 / (t * t + 1);
			s2 = 1 - c2;
			cs = t * c2;
			temp = a[m];
			temp_2 = a[m + 1];

			a[m] = c2 * temp + s2 * temp_2 + 2 * cs * b[m];
			a[m + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[m];
			b[m] = -cs * temp + cs * temp_2 + (c2 - s2) * b[m];
			c2 = sqrt(c2);
			s2 = t * c2;
		}
		B = s2 * b[m + 1];
		b[m + 1] = c2 * b[m + 1];

		__ApplyGivensOnTheRight(Q[m], Q[m + 1], c2, s2, Q.rows);

		for (i = m + 1; i < n; ++i) {
			if (abs(B) > abs(b[i - 1])) {
				t = b[i - 1] / B;
				s2 = 1 / (t * t + 1);
				c2 = 1 - s2;
				cs = t * s2;
				temp = a[i];
				temp_2 = a[i + 1];

				a[i] = c2 * temp + s2 * temp_2 + 2 * cs * b[i];
				a[i + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[i];
				b[i] = -cs * temp + cs * temp_2 + (c2 - s2) * b[i];
				s2 = sqrt(s2);
				c2 = t * s2;
			}
			else {
				t = B / b[i - 1];
				c2 = 1 / (t * t + 1);
				s2 = 1 - c2;
				cs = t * c2;
				temp = a[i];
				temp_2 = a[i + 1];

				a[i] = c2 * temp + s2 * temp_2 + 2 * cs * b[i];
				a[i + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[i];
				b[i] = -cs * temp + cs * temp_2 + (c2 - s2) * b[i];
				c2 = sqrt(c2);
				s2 = t * c2;
			}
			b[i - 1] = c2 * b[i - 1] + s2 * B;
			B = s2 * b[i + 1];
			b[i + 1] = c2 * b[i + 1];

			__ApplyGivensOnTheRight(Q[i], Q[i + 1], c2, s2, Q.rows);
		}
	}
}

template<typename T>
void __EVD_H_QR(MAT<T>& A, MAT<T>& Q, T eps) {
	int i;

	T* v = new T[A.cols];
	__EVD_H_Householder(Q, A, v);
	delete[] v;

	T* a(A[A.cols - 1]);
	T* b(A[A.cols - 2]);

	__copy(b, b + 1, A.cols - 1);
	b[A.cols - 1] = 0;

	__EVD_H_Givens(a, b, Q, 0, A.rows - 1, eps);

	__set0(A.data, A.rows * (A.cols - 1));

	for (i = 0; i < A.cols - 1; ++i) {
		A[i][i] = a[i];
	}
	__set0(a, A.rows - 1);
}
