
template<typename T>
void __EV_Householder_R(MAT<T>& R, T* w, T* v) {
	int m, n;
	T cr, hr;

	for (m = 0; m < R.rows - 2; ++m) {
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
}

template<typename T>
void __EV_QR_R(MAT<T>& R, int m, int n, T eps) {
	int i, k, r;
	complex<T> shift;
	T temp, x, y;
	T w[3];
	T* Temp;

	while (m < n) {

		if (abs(R(m + 1, m)) < eps) {
			++m;
			continue;
		}
		for (r = n - 1; r > m; --r) {
			if (abs(R(r, r - 1)) < eps) {
				if (n - r <= 2) {
					n = r;
					break;
				}
				else {
					__EV_QR_R(R, r, n, eps);
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
			w[1] = R(m + 1, m) * (R(m, m) + R(m + 1, m + 1) - x);
			w[2] = R(m + 1, m) * R(m + 2, m + 1);
			if (abs(w[0]) <= eps) {
				x = x + R(m + 1, m);
				w[0] = R(m, m) * R(m, m) + R(m, m + 1) * R(m + 1, m) - x * R(m, m) + (x * x - y) / 4;
				w[1] = R(m + 1, m) * (R(m, m) + R(m + 1, m + 1) - x);
				w[2] = R(m + 1, m) * R(m + 2, m + 1);
			}

			x = w[2] / w[1];
			y = (T)1 / sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
			if (w[0] < 0) y = -y;
			temp = w[1] * y;

			w[1] = sqrt((1 + w[0] * y) / (1 + x * x));
			w[2] = w[1] * x;
			w[0] = -temp / w[1];

			if (m + 4 < n) k = m + 4;
			else k = n;
			__Schur_Householder_Right(R[m], R[m + 1], R[m + 2], w, k);
			for (i = m; i < n; ++i) {
				Temp = R[i] + m;

				temp = Temp[0] * w[0] + Temp[1] * w[1] + Temp[2] * w[2];
				Temp[0] -= w[0] * temp;
				Temp[1] -= w[1] * temp;
				Temp[2] -= w[2] * temp;
			}

			for (r = m + 1; r < n - 2; ++r) {
				Temp = R[r - 1] + r;
				w[0] = Temp[0]; w[1] = Temp[1]; w[2] = Temp[2];
				temp = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
				if (w[0] < 0.0)temp = -temp;
				w[0] += temp;
				temp = (T)1 / sqrt(temp * w[0]);
				w[0] *= temp; w[1] *= temp; w[2] *= temp;

				if (r + 4 < n) k = r + 4;
				else k = n;
				__Schur_Householder_Right(R[r], R[r + 1], R[r + 2], w, k);
				for (i = r - 1; i < n; ++i) {
					Temp = R[i] + r;

					temp = Temp[0] * w[0] + Temp[1] * w[1] + Temp[2] * w[2];
					Temp[0] -= w[0] * temp;
					Temp[1] -= w[1] * temp;
					Temp[2] -= w[2] * temp;
				}
			}

			temp = R(r + 1, r - 1) / R(r, r - 1);
			x = 1 / sqrt(temp * temp + 1);
			y = temp * x;

			__ApplyGivensOnTheRight(R[r], R[r + 1], x, y, n);
			for (i = r - 1; i < n; ++i) {
				Temp = R[i] + r;
				temp = Temp[0] * x + Temp[1] * y;
				Temp[1] = Temp[1] * x - Temp[0] * y;
				Temp[0] = temp;
			}
		}
		else {

			if (x - 2 * R(n - 1, n - 1) > 0) temp = (x - sqrt(y)) * 0.5;
			else temp = (x + sqrt(y)) * 0.5;

			if (abs(R(m + 1, m)) > abs(R(m, m) - temp)) {
				temp = (R(m, m) - temp) / R(m + 1, m);
				y = 1 / sqrt(temp * temp + 1);
				x = temp * y;
			}
			else {
				temp = R(m + 1, m) / (R(m, m) - temp);
				x = 1 / sqrt(temp * temp + 1);
				y = temp * x;
			}

			if (m + 3 < n) k = m + 3;
			else k = n;
			__ApplyGivensOnTheRight(R[m], R[m + 1], x, y, k);
			for (i = m; i < n; ++i) {
				Temp = R[i] + m;
				temp = Temp[0] * x + Temp[1] * y;
				Temp[1] = Temp[1] * x - Temp[0] * y;
				Temp[0] = temp;
			}

			for (r = m + 1; r < n - 1; ++r) {
				if (abs(R(r + 1, r - 1)) > abs(R(r, r - 1))) {
					temp = R(r, r - 1) / R(r + 1, r - 1);
					y = 1 / sqrt(temp * temp + 1);
					x = temp * y;
				}
				else {
					temp = R(r + 1, r - 1) / R(r, r - 1);
					x = 1 / sqrt(temp * temp + 1);
					y = temp * x;
				}

				if (r + 3 < n) k = r + 3;
				else k = n;
				__ApplyGivensOnTheRight(R[r], R[r + 1], x, y, k);
				for (i = r - 1; i < n; ++i) {
					Temp = R[i] + r;
					temp = Temp[0] * x + Temp[1] * y;
					Temp[1] = Temp[1] * x - Temp[0] * y;
					Temp[0] = temp;
				}
			}
		}
	}
}
