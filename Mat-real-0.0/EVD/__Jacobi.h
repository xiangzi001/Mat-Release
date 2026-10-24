
template <typename T>
void __EVD_H_Jacobi(MAT<T>& A, MAT<T>& Q, T eps)
{
	int m, n, i;
	__set0(Q.data, Q.rows * Q.cols);
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T temp;
	T* Temp;
	bool flag = true;

	while (flag){
		flag = false;
		for (m = 1; m < A.rows; ++m) {
			for (n = 0; n < m; ++n) {
				flag = flag || abs(A(n, m)) > eps;
				/*
				double angle = 0.5 * atan2(-2 * A[m][n], A[n][n] - A[m][m]);
				T s = sin(angle);
				T c = cos(angle);
				//*/
				//*
				T c, s;
				T x = A[n][n] - A[m][m];
				T y = A[m][n];
				T r = 1.0 / sqrt(x * x + y * y * 4);
				x = x * r;
				y = -y * r;
				if (x > 0.0) {
					c = sqrt(0.5 + x * 0.5);
					s = y / c;
				}
				else {
					s = sqrt(0.5 - x * 0.5);
					c = y / s;
				}
				//*/

				for (i = 0; i < A.rows; ++i) {
					Temp = A[i];
					temp = Temp[m];
					Temp[m] = Temp[n] * s + temp * c;
					Temp[n] = Temp[n] * c - temp * s;
				}
				__ApplyGivensOnTheRight(A[m], A[n], c, s, Q.rows);
				__ApplyGivensOnTheRight(Q[m], Q[n], c, s, Q.rows);
			}
		}
	}
}
