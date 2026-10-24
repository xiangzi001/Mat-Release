
template <class T>
void quickSort(T* arr, int left, int right, int* pos) {
	if (left >= right) {
		return;
	}

	int i = left;
	int j = right;
	T Temp = arr[left];
	int temp = pos[left];
	while (i < j) {
		while (i < j && arr[j] >= Temp) {
			j--;
		}
		arr[i] = arr[j];
		pos[i] = pos[j];

		while (i < j && arr[i] <= Temp) {
			i++;
		}
		arr[j] = arr[i];
		pos[j] = pos[i];
	}
	arr[i] = Temp;
	pos[i] = temp;

	quickSort(arr, left, i - 1, pos);
	quickSort(arr, i + 1, right, pos);
}

template<typename T>
inline void __add(T* a, T p, T* b, int len) {
	for (int i = 0; i < len; ++i) {
		b[i] = a[i] + p;
	}
}

template<typename T>
void times_A_B2(const MAT<T>& A, const MAT<T>& B, T* Cdata, int ldc, T* A_pack, T* B_pack) {
	const int M = A.rows;
	const int N = B.cols;
	const int K = A.cols;

	if (M <= 128 && N <= 128 && K <= 128) {
		constexpr int MR = __micro_kernel_mr<T, T>();
		constexpr int NR = __micro_kernel_nr<T, T>();
		const int ldA = A.rows;
		const int ldB = B.rows;

		for (int jp = 0; jp < N; jp += NR) {
			const int nn = (N - jp < NR) ? (N - jp) : NR;
			for (int ip = 0; ip < M; ip += MR) {
				const int mm = (M - ip < MR) ? (M - ip) : MR;
				const T* Ap = A.data + ip;
				const T* Bp = B.data + (size_t)jp * ldB;
				T* Cp = Cdata + (size_t)jp * ldc + ip;
				if (mm == MR && nn == NR)
					__micro_kernel_mul_unpacked_full(
						Ap, ldA, Bp, ldB, Cp, ldc, (size_t)K);
				else
					__micro_kernel_mul_unpacked_edge(
						Ap, ldA, Bp, ldB, Cp, ldc, (size_t)K, mm, nn);
			}
		}
		return;
	}
	else {
		constexpr int MC = __BlockSize_Mul_MC;
		constexpr int KC = __BlockSize_Mul_KC;
		constexpr int NC = __BlockSize_Mul_NC;

		for (int pc = 0; pc < K; pc += KC) {
			const int kc = (K - pc < KC) ? (K - pc) : KC;

			for (int jc = 0; jc < N; jc += NC) {
				const int nc = (N - jc < NC) ? (N - jc) : NC;

				__pack_B_block<T, T>(B.data + (size_t)pc + (size_t)jc * B.rows,
					B.rows, kc, nc, B_pack);

				for (int ic = 0; ic < M; ic += MC) {
					const int mc = (M - ic < MC) ? (M - ic) : MC;

					__pack_A_block<T, T>(A.data + (size_t)pc * A.rows + ic,
						A.rows, mc, kc, A_pack);

					__mulAB_packed<T, T, T>(
						A_pack, B_pack,
						Cdata + (size_t)jc * ldc + ic,
						ldc, mc, nc, kc);
				}
			}
		}
		return;
	}
}

template<typename T>
void __EVD_H_CDC_Solve(T* w, T* v, MAT<T>& Q, T* Temp, T* Temp2, T* A_pack, T* B_pack, int* pos, T eps) {
	if (Q.rows <= 64) {
		__set0(Q.data, Q.rows * Q.cols);
		for (int i = 0; i < Q.rows; ++i) {
			Q[i][i] = 1;
		}

		__EVD_H_Givens(w, v, Q, 0, Q.rows - 1, eps);
	}
	else {
		int s = Q.rows / 2;
		int t = Q.rows - s;

		MAT<T> Q1, Q2;
		Q1.rows = s;
		Q1.cols = s;
		Q1.data = Temp;
		Q2.rows = t;
		Q2.cols = t;
		Q2.data = Temp + s * s;

		T p = v[s - 1];
		v[s - 1] = 0;
		w[s - 1] -= p;
		w[s] -= p;

		__EVD_H_CDC_Solve(w, v, Q1, Temp + t * t + s * s, Temp2, A_pack, B_pack, pos, eps);
		__EVD_H_CDC_Solve(w + s, v + s, Q2, Temp + t * t + s * s, Temp2, A_pack, B_pack, pos, eps);

		T* u = Temp + t * t + s * s;
		T* _u = u + Q.rows;
		T* _d = _u + Q.rows;
		T* d = _d + Q.rows;

		int i, j, n = Q.rows, k;
		T x, temp;

		__times(w, 1 / p, v, Q.rows);
		__copy(_d, v, Q.rows);
		for (i = 0; i < s; ++i) {
			u[i] = Q1(s - 1, i);
		}
		for (i = 0; i < t; ++i) {
			u[i + s] = Q2(0, i);
		}

		for (i = Q.rows - 1; i >= 0; --i) {
			if (fabs(u[i]) < eps) {
				--n;

				pos[i] = n;
				pos[n] = i;

				_d[i] = v[n];
				_d[n] = v[i];
			}
			else {
				pos[i] = i;
			}
		}
		quickSort(_d, 0, n - 1, pos);
		__copy(d, _d, Q.rows);

		for (i = 0; i < Q.rows; ++i) {
			_u[i] = u[pos[i]];
		}
		for (i = 0; i < Q.rows; ++i) {
			u[i] = _u[i] * _u[i];
		}

		for (i = 1; i < n; ++i) {
			T temp = max(fabs(d[i]), fabs(d[i - 1]));
			if (fabs(d[i] - d[i - 1]) < eps * temp) {
				//MessageBoxA(NULL, "wrong", "demo", MB_OK);
				//printf("%s", "BUG");
				d[i] += temp * eps;
			}
		}

		// EigenValues

		__EVD_H_CDC_u1(n, _u);
		for (i = 0; i < n - 1; ++i) {
			temp = (_d[i] + _d[i + 1]) / 2;

			if (__EVD_H_CDC_Solve_EigenValue_if(n, u, _d, i, temp)) {
				k = i;
			}
			else {
				k = i + 1;
			}
			temp -= _d[k];
			__add(d, -d[k], _d, n);
			x = __EVD_H_CDC_Solve_EigenValue(n, u, _d, i, k, temp);

			while (fabs(temp - x) > eps) {
				temp = x;
				x = __EVD_H_CDC_Solve_EigenValue(n, u, _d, i, k, x);
			}
			__EVD_H_CDC_u2(n, x, _d, _u, i);
			w[i] = x;
		}

		__add(d, -d[i], _d, n);
		temp = u[i] / __EVD_H_CDC_Solve_EigenValue_Zero(i, u, _d);
		x = __EVD_H_CDC_Solve_EigenValue2(n, u, _d, temp);
		while (fabs(temp - x) > eps) {
			temp = x;
			x = __EVD_H_CDC_Solve_EigenValue2(n, u, _d, x);
		}
		__EVD_H_CDC_u2(n, x, _d, _u, i);
		w[i] = x;

		for (i = n; i < Q.rows; ++i) {
			_u[i] = 0;
			w[i] = 0;
		}

		__EVD_H_CDC_u3(n, _u);

		for (i = 0; i < Q.rows; ++i) {
			u[pos[i]] = _u[i];
		}

		// EigenVectors
		__set0(Q.data, Q.rows * Q.cols);
		MAT<T> W1, W2;
		W1.rows = s;
		W1.cols = n;
		W1.data = Temp2;
		W2.rows = t;
		W2.cols = n;
		W2.data = Temp2 + s * n;

		T shift;
		__copy(d, v, Q.rows);
		for (i = 0; i < n; ++i) {
			if (w[i] > 0) {
				shift = d[pos[i]];
			}
			else {
				shift = d[pos[i + 1]];
			}
			__add(d, -shift, _d, Q.rows);

			x = w[i];
			w[i] = (shift + x) * p;

			__EVD_H_CDC_v(Q.rows, u, _d, v, x);
			temp = (T)1 / sqrt(__dot(v, v, Q.rows));
			__times(v, temp, W1[i], s);
			__times(v + s, temp, W2[i], t);
		}
		__set0(Q.data, Q.rows * Q.cols);
		times_A_B2(Q1, W1, Q.data, Q.rows, A_pack, B_pack);
		times_A_B2(Q2, W2, Q.data + s, Q.rows, A_pack, B_pack);

		for (i = n; i < Q.rows; ++i) {
			j = pos[i]; 
			w[i] = d[j] * p;
			if (j < s)
				__copy(Q[i], Q1[j], s);
			else
				__copy(Q[i] + s, Q2[j - s], t);
		}

		Q1.data = 0;
		Q2.data = 0;
		W1.data = 0;
		W2.data = 0;
	}
}


template<typename T>
void __EVD_H_Householder2(MAT<T>& Q, MAT<T>& R, T* v) {
	int m, n;

	T temp;
	T cr, hr;
	T* w, * a = R[R.rows - 1], * b = R[R.rows - 2] + 1;

	for (m = 0; m < R.rows - 2; ++m) {
		w = Q.data + m * (2 * Q.rows - m + 1) / 2;

		temp = *R.data;

		n = R.rows - m - 1;
		__copy(w, R.data + 1, n);
		cr = sqrt(__dot(w, w, n));

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
void __EVD_H_Householder_Q(MAT<T>& Q, MAT<T>& R) {
	int m, n, i;

	T* w;

	for (m = R.rows - 3; m >= 0; --m) {
		w = Q.data + m * (2 * Q.rows - m + 1) / 2;
		n = m + 1;

		for (i = 0; i < R.cols; ++i) {
			__mixed(R[i] + n, w, -__dot(w, R[i] + n, R.rows - n), R[i] + n, R.rows - n);
		}
	}
}

template<typename T>
void __EVD_H_CDC(MAT<T>& A, MAT<T>& Q, T eps) {
	int i;

	const int m = A.rows;

	constexpr int MR = __micro_kernel_mr<T, T>();
	constexpr int NR = __micro_kernel_nr<T, T>();

	constexpr int MC = __BlockSize_Mul_MC;
	constexpr int NC = __BlockSize_Mul_NC;
	constexpr int KC = __BlockSize_Mul_KC;

	constexpr int NC1 = __BlockSize_Mul_sK_NC;
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
	constexpr size_t w_pack_sz = ((size_t)(NC1 + NR - 1) / NR) * BS * NR;

	const size_t work1_sz = max(v_pack_sz, vt_pack_sz);
	const size_t work2_sz = max(r_pack_sz, w_pack_sz);

#define __align_up(x, a) (x + a - 1) & ~(a - 1)
	const size_t work1_size = __align_up(sizeof(T) * work1_sz, 64);
	const size_t work2_size = __align_up(sizeof(T) * work2_sz, 64);
	const size_t W_size = __align_up(sizeof(T) * BS * m, 64);
	const size_t buf_size = __align_up(sizeof(T) * (m + 2 * BS), 64);
	const size_t W2_size = __align_up(sizeof(T) * BS * m, 64);
	const size_t T_size = __align_up(sizeof(T) * BS * BS, 64);

	const size_t A_pack_size = __align_up(sizeof(T) * ((MC + MR - 1) / MR) * KC * MR, 64);
	const size_t B_pack_size = __align_up(sizeof(T) * ((NC + NR - 1) / NR) * KC * NR, 64);
	const size_t pos_size = sizeof(int) * m;
	const size_t Temp_size = sizeof(T) * m * m;

	const size_t total_size = max(
		work1_size + work2_size + W_size + buf_size + W2_size + T_size,
		A_pack_size + B_pack_size + pos_size + Temp_size * 2
	);

	const size_t ab_size = sizeof(T) * (m);
#undef __align_up

	T* work1 = (T*)_mm_malloc(total_size + 3 * ab_size, 64);
	T* work2 = (T*)((char*)(work1)+work1_size);
	T* W = (T*)((char*)(work2)+work2_size);
	T* buf = (T*)((char*)(W)+W_size);
	T* s = buf + m;
	T* w = s + BS;
	T* W2 = (T*)((char*)(buf)+buf_size);
	T* T_buf = (T*)((char*)(W2)+W2_size);

	T* A_pack = work1;
	T* B_pack = (T*)((char*)(A_pack)+A_pack_size);
	int* pos = (int*)((char*)(B_pack)+B_pack_size);
	T* Temp = (T*)((char*)(pos)+pos_size);
	T* Temp2 = (T*)((char*)(Temp)+Temp_size);

	T* a = (T*)((char*)(work1)+total_size);
	T* b = (T*)((char*)(a)+ab_size);
	T* tau = (T*)((char*)(b)+ab_size);

	//_EVD_H_Householder(TempQ, A, a);

	__EVD_H_Householder_Blocked<T>(A, buf, W, work1, work2, tau, m, a, b);

	__EVD_H_CDC_Solve(a, b, Q, Temp, Temp2, A_pack, B_pack, pos, eps);

	__EVD_H_Householder_BlockedQ<T>(
		A, Q, tau, work1, work2, W, W2, T_buf, s, w, m);

	__set0(A.data, m * m);
	for (i = 0; i < m; ++i) {
		A[i][i] = a[i];
	}

	_mm_free(work1);
}
