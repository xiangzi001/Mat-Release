
template<typename T>
void __Schur_Householder_R(MAT<T>& Q, MAT<T>& R, T* v) {
	int m, n;
	T cr, hr;
	T* w;

	for (m = 0; m < R.rows - 2; ++m) {
		w = Q.data + 2 * Q.rows * m - m * m + 1;
		n = m + 1;
		__copy(w, R[m] + n, R.rows - n);
		cr = sqrt(__dot(w, w, R.rows - n));

		if (cr == 0) {
			continue;
		}

		if (*w < 0.0)cr = -cr;
		*w += cr;

		hr = (T)1 / sqrt(cr * *w);

		__times(w, hr, w, R.rows - n);

		__ApplyHouseholderOnTheRight(R, w - n, v, n);
		__ApplyHouseholderOnTheLeft2(R, w, m, n);
	}

	__Householder_Q2(Q);
}

template<typename T>
inline void __Schur_Householder_Right(T* __restrict a, T* __restrict b, T* __restrict c, T* w, int n) {
	T sum;
	T x = w[0], y = w[1], z = w[2];
	for (int i = 0; i < n; ++i) {
		sum = a[i] * x + b[i] * y + c[i] * z;
		a[i] -= sum * x;
		b[i] -= sum * y;
		c[i] -= sum * z;
	}
}
template<typename T>
inline void __Schur_Householder_Left(T* Adata, T* w, int Arows, int n) {
	T sum;
	T x = w[0], y = w[1], z = w[2];
	T* Temp = Adata;
	for (int i = 0; i < n; ++i) {
		sum = Temp[0] * x + Temp[1] * y + Temp[2] * z;
		Temp[0] -= sum * x;
		Temp[1] -= sum * y;
		Temp[2] -= sum * z;

		Temp = Temp + Arows;
	}
}

template<typename T>
void __Schur_QR_R(MAT<T>& Q, MAT<T>& R, int m, int n, T eps) {
	int i, k, r;
	complex<T> shift;
	T temp, x, y;
	T w[3];
	T* Temp;

	while (m < n) {

		if (abs(R(m, m + 1)) < eps) {
			++m;
			continue;
		}
		for (r = n - 1; r > m; --r) {
			if (abs(R(r - 1, r)) < eps) {
				if (n - r <= 2) {
					n = r;
					break;
				}
				else {
					__Schur_QR_R(Q, R, r, n, eps);
					n = r;
					break;
				}
			}
		}

		x = R(n - 2, n - 2) + R(n - 1, n - 1);
		y = x * x - (R(n - 2, n - 2) * R(n - 1, n - 1) - R(n - 2, n - 1) * R(n - 1, n - 2)) * 4;
		if (n - m <= 2) {
			if (n - m < 2) {
				break;
			}
			else if (y < 0) {
				break;
			}
		}

		if (y <= 0) {
			w[0] = R(m, m) * R(m, m) + R(m, m + 1) * R(m + 1, m) - x * R(m, m) + (x * x - y) / 4;
			if (abs(w[0]) <= eps) {
				x = x + R(m, m + 1);
				w[0] = R(m, m) * R(m, m) + R(m, m + 1) * R(m + 1, m) - x * R(m, m) + (x * x - y) / 4;
			}
			w[1] = R(m, m + 1) * (R(m, m) + R(m + 1, m + 1) - x);
			w[2] = R(m, m + 1) * R(m + 1, m + 2);

			x = w[2] / w[1];
			y = (T)1 / sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
			if (w[0] < 0) y = -y;
			temp = w[1] * y;

			w[1] = sqrt((1 + w[0] * y) / (1 + x * x));
			w[2] = w[1] * x;
			w[0] = -temp / w[1];

			if (m + 4 < n) k = m + 4;
			else k = n; 
			__Schur_Householder_Left(R.data + m, w, R.rows, k);
			__Schur_Householder_Right(R[m] + m, R[m + 1] + m, R[m + 2] + m, w, R.cols - m);
			__Schur_Householder_Right(Q[m], Q[m + 1], Q[m + 2], w, Q.rows);

			for (r = m + 1; r < n - 2; ++r) {
				w[0] = R(r - 1, r); w[1] = R(r - 1, r + 1); w[2] = R(r - 1, r + 2);
				temp = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
				if (w[0] < 0.0)temp = -temp;
				w[0] += temp;
				temp = (T)1 / sqrt(temp * w[0]);
				w[0] *= temp; w[1] *= temp; w[2] *= temp;

				if (r + 4 < n) k = r + 4;
				else k = n;
				__Schur_Householder_Left(R.data + r, w, R.rows, k);
				k = r - 1;
				__Schur_Householder_Right(R[r] + k, R[r + 1] + k, R[r + 2] + k, w, R.rows - k);
				__Schur_Householder_Right(Q[r], Q[r + 1], Q[r + 2], w, Q.rows);
			}

			if (abs(R(r - 1, r + 1)) > abs(R(r - 1, r))) {
				temp = R(r - 1, r) / R(r - 1, r + 1);
				y = 1 / sqrt(temp * temp + 1);
				x = temp * y;
			}
			else {
				temp = R(r - 1, r + 1) / R(r - 1, r);
				x = 1 / sqrt(temp * temp + 1);
				y = temp * x;
			}

			__ApplyGivensOnTheLeft(R.data + r, x, y, R.rows, n);
			k = r - 1;
			__ApplyGivensOnTheRight(R[r] + k, R[r + 1] + k, x, y, R.cols - k);
			__ApplyGivensOnTheRight(Q[r], Q[r + 1], x, y, Q.rows);
		}
		else {
			if (x - 2 * R(n - 1, n - 1) > 0) temp = (x - sqrt(y)) * 0.5;
			else temp = (x + sqrt(y)) * 0.5;

			if (abs(R(m, m + 1)) > abs(R(m, m) - temp)) {
				temp = (R(m, m) - temp) / R(m, m + 1);
				y = 1 / sqrt(temp * temp + 1);
				x = temp * y;
			}
			else {
				temp = R(m, m + 1) / (R(m, m) - temp);
				x = 1 / sqrt(temp * temp + 1);
				y = temp * x;
			}

			if (m + 3 < n) k = m + 3;
			else k = n;
			__ApplyGivensOnTheLeft(R.data + m, x, y, R.rows, k);
			__ApplyGivensOnTheRight(R[m] + m, R[m + 1] + m, x, y, R.cols - m);
			__ApplyGivensOnTheRight(Q[m], Q[m + 1], x, y, Q.rows);

			for (r = m + 1; r < n - 1; ++r) {
				if (abs(R(r - 1, r + 1)) > abs(R(r - 1, r))) {
					temp = R(r - 1, r) / R(r - 1, r + 1);
					y = 1 / sqrt(temp * temp + 1);
					x = temp * y;
				}
				else {
					temp = R(r - 1, r + 1) / R(r - 1, r);
					x = 1 / sqrt(temp * temp + 1);
					y = temp * x;
				}

				if (r + 3 < n) k = r + 3;
				else k = n;
				__ApplyGivensOnTheLeft(R.data + r, x, y, R.rows, k);
				k = r - 1;
				__ApplyGivensOnTheRight(R[r] + k, R[r + 1] + k, x, y, R.rows - k);
				__ApplyGivensOnTheRight(Q[r], Q[r + 1], x, y, Q.rows);
			}
		}
	}
}
