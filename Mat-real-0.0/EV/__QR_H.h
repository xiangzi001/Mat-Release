
template<typename T>
void __EV_H_Householder(MAT<T>& R, T* v, T* w) {
	int m, n;

	T temp;
	T cr, hr;
	T* a = R[R.rows - 1], * b = R[R.rows - 2] + 1;

	for (m = 0; m < R.rows - 2; ++m) {
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
}

template<typename T>
void __EV_H_Givens(T* a, T* b, int m, int n, T eps) {

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
				__EV_H_Givens(a, b, m, i, eps);
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
		}
	}
}

template<typename t>
void __EV_H(MAT<t>& A, t* v, t eps) {

	const int m = A.rows;

	if (m <= __BlockSize_QR * 2) {
		t* w = new t[A.cols];
		__EV_H_Householder(A, v, w);
		delete[] w;

		t* a(A[A.cols - 1]);
		t* b(A[A.cols - 2]);

		__copy(b, b + 1, A.cols - 1);
		b[A.cols - 1] = 0;

		__EV_H_Givens(a, b, 0, A.rows - 1, eps);

		__copy(v, a, A.rows);
	}
	else {
		constexpr int MR = __micro_kernel_mr<t, t>();
		constexpr int NR = __micro_kernel_nr<t, t>();
		constexpr int NC = __BlockSize_Mul_sK_NC;
		constexpr int KC2 = __BlockSize_Mul_sM_KC;
		constexpr int NC2 = __BlockSize_Mul_sM_NC;
		constexpr int BS = __BlockSize_QR;

		const size_t v_pack_sz = ((size_t)(m + MR - 1) / MR) * BS * MR;

		const size_t np_b = ((size_t)BS + MR - 1) / MR;
		const size_t num_full = ((size_t)m + KC2 - 1) / KC2;
		const size_t vt_pack_sz =
			np_b * (size_t)BS * MR +
			np_b * num_full * (size_t)KC2 * MR;

		constexpr size_t r_pack_sz = ((size_t)(NC2 + NR - 1) / NR) * KC2 * NR;
		constexpr size_t w_pack_sz = ((size_t)(NC + NR - 1) / NR) * BS * NR;

		const size_t work1_sz = max(v_pack_sz, vt_pack_sz);
		const size_t work2_sz = max(r_pack_sz, w_pack_sz);

#define __align_up(x, a) (x + a - 1) & ~(a - 1)
		const size_t work1_size = __align_up(sizeof(t) * work1_sz, 64);
		const size_t work2_size = __align_up(sizeof(t) * work2_sz, 64);
		const size_t W_size = __align_up(sizeof(t) * BS * m, 64);
		const size_t buf_size = __align_up(sizeof(t) * m, 64);

		const size_t ab_size = __align_up(sizeof(t) * m, 64);
#undef __align_up

		t* work1 = (t*)_mm_malloc(work1_size + work2_size + W_size + buf_size + ab_size * 3, 64);
		t* work2 = (t*)((char*)(work1)+work1_size);
		t* W = (t*)((char*)(work2)+work2_size);
		t* buf = (t*)((char*)(W)+W_size);
		t* a = (t*)((char*)(buf)+buf_size);
		t* b = (t*)((char*)(a)+ab_size);
		t* tau = (t*)((char*)(b)+ab_size);

		__EVD_H_Householder_Blocked<t>(A, buf, W, work1, work2, tau, m, a, b);

		__EV_H_Givens(a, b, 0, m - 1, eps);
		__copy(v, a, m);

		_mm_free(work1);
	}
}

