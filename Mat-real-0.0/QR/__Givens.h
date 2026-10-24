template<typename T>
inline void __ApplyGivensOnTheRight(T* __restrict a, T* __restrict b, T c, T s, int len) {
    T ai, bi;
    for (int i = 0; i < len; ++i) {
        ai = a[i];
        bi = b[i];
        a[i] = ai * c + bi * s;
        b[i] = -ai * s + bi * c;
    }
}
template<typename T>
inline void __ApplyGivensOnTheLeft(T* Adata, T c, T s, int Arows, int len) {
    T ai, bi;
    T* Temp;
    for (int i = 0; i < len; ++i) {
        Temp = Adata + Arows * i;
        ai = Temp[0];
        bi = Temp[1];
        Temp[0] = ai * c + bi * s;
        Temp[1] = -ai * s + bi * c;
    }
}

template<typename t>
inline void __ApplyGivensOnTheRight(complex<t>* __restrict a, complex<t>* __restrict b, t c, complex<t>& s, int len) {
    complex<t> _s(s.conj()), ai;
    for (int i = 0; i < len; ++i) {
        ai = a[i];
        a[i] = ai * c + b[i] * s;
        b[i] = -ai * _s + b[i] * c;
    }
}
template<typename t>
inline void __ApplyGivensOnTheLeft(complex<t>* Adata, t c, complex<t>& s, int Arows, int len) {
    complex<t> _s(s.conj()), ai;
    complex<t>* Temp;
    for (int i = 0; i < len; ++i) {
        Temp = Adata + Arows * i;
        ai = Temp[0];
        Temp[0] = ai * c + Temp[1] * _s;
        Temp[1] = -ai * s + Temp[1] * c;
    }
}

template<typename T>
void __QR_Givens(MAT<T>& Q, MAT<T>& R) {
    T temp, c, s;

    int i, j;
    for (j = 0; j < R.cols; ++j) {
        for (i = R.rows - 1; i > j; i--) {
            //double theta = atan2(R[i][j], R[i - 1][j]);
            //double c = cos(theta);
            //double s = sin(theta);
            temp = R(i, j) / R(i - 1, j);
            c = 1 / sqrt(temp * temp + 1);
            s = temp * c;

            __ApplyGivensOnTheLeft(R.data + j * R.rows + i - 1, c, s, R.rows, R.cols - j);
            __ApplyGivensOnTheRight(Q[i - 1], Q[i], c, s, Q.rows);
        }
    }
}


template<typename t>
void __QR_Givens(MAT<complex<t>>& Q, MAT<complex<t>>& R) {
    t c;
    complex<t> temp, s;

    int i, j;
    for (j = 0; j < R.cols; ++j) {
        for (i = R.rows - 1; i > j; --i) {
            temp = R(i, j) / R(i - 1, j);
            c = 1 / sqrt(temp.len2() + 1);
            s = temp * c;

            __ApplyGivensOnTheLeft(R.data + j * R.rows + i - 1, c, s, R.rows, R.cols - j);
            __ApplyGivensOnTheRight(Q[i - 1], Q[i], c, s, Q.rows);
        }
    }
}
