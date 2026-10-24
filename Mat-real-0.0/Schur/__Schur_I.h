
template<typename t>
void __Schur_Householder_I(MAT<complex<t>>& U, MAT<complex<t>>& R, complex<t>* v) {

	complex<t> sum, _sign;
	t cr, hr;
	complex<t>* w;

	int m, n;
	for (m = 0; m < R.rows - 2; ++m) {
		w = U.data + 2 * U.rows * m - m * m + 1;
		n = m + 1;
		__copy(w, R[m] + n, R.rows - n);
		cr = sqrt(__dot((t*)w, (t*)w, (R.rows - n) * 2));

		if (cr == 0) {
			continue;
		}

		_sign = sign(*w) * cr;
		hr = 1 / sqrt(cr * cr - times_conj(_sign, *w).real);
		*w -= _sign;

		__times((t*)w, hr, (t*)w, (R.rows - n) * 2);

		__ApplyHouseholderOnTheRight(R, w - n, v, n);
		__ApplyHouseholderOnTheLeft2(R, w, m, n);
	}
	__Householder_Q2(U);
}

template<typename t>
void __Schur_Givens_I(MAT<complex<t>>& U, MAT<complex<t>>& R, int m, int n, t eps) {

	t c;
	complex<t>* Temp;
	complex<t> t, s, _s, temp;
	int r, i, k;
	while (n - m > 1) {
		if (abs2(R(m + 1,m)) < eps) {
			m++;
			continue;
		}
		for (r = n - 1; r > m; --r) {
			if (abs2(R(r, r - 1)) < eps) {
				if (n - r < 2) {
					n = r;
					break;
				}
				else {
					__Schur_Givens_I(U, R, r, n, eps);
					n = r;
					break;
				}
			}
		}

		t = R(n - 2, n - 2) + R(n - 1, n - 1);
		s = R(n - 2, n - 2) * R(n - 1, n - 1) - R(n - 2, n - 1) * R(n - 1, n - 2);
		s = sqrt(t * t - s * 4);
		temp = t - R(n - 1, n - 1) * 2;
		if ((temp + s).len2() > (temp - s).len2()) {
			temp = (t - s) * 0.5;
		}
		else {
			temp = (t + s) * 0.5;
		}

		t = R(m + 1, m) / (R(m, m) - temp);
		c = 1 / sqrt(t.len2() + 1);
		s = t * c;
		_s = s.conj();

		for (i = m; i < R.rows; ++i) {
			Temp = R[i];

			temp = Temp[m] * c + Temp[m + 1] * _s;
			Temp[m + 1] = Temp[m + 1] * c - Temp[m] * s;
			Temp[m] = temp;
		}
		if (m + 3 < n) k = m + 3;
		else k = n;
		__ApplyGivensOnTheRight(R[m], R[m + 1], c, s, k);
		__ApplyGivensOnTheRight(U[m], U[m + 1], c, s, U.rows);

		for (r = m + 1; r < n - 1; ++r) {
			t = R(r + 1, r - 1) / R(r, r - 1);
			c = 1 / sqrt(t.len2() + 1);
			s = t * c;
			_s = s.conj();

			for (i = r - 1; i < R.rows; ++i) {
				Temp = R[i];

				temp = Temp[r] * c + Temp[r + 1] * _s;
				Temp[r + 1] = Temp[r + 1] * c - Temp[r] * s;
				Temp[r] = temp;
			}
			if (r + 3 < n) k = r + 3;
			else k = n;
			__ApplyGivensOnTheRight(R[r], R[r + 1], c, s, k);
			__ApplyGivensOnTheRight(U[r], U[r + 1], c, s, U.rows);
		}
	}
}
