
template<typename T>
inline void __ApplyHouseholderOnTheRight(MAT<T>& A, T* w, T* v, int r) {
    __set0(v, A.rows);
    for (int i = r; i < A.cols; ++i) {
        __mixed(v, A[i], w[i], v, A.rows);
    }
    for (int i = r; i < A.cols; ++i) {
        __mixed(A[i], v, -w[i], A[i], A.rows);
    }
}
template<typename T>
inline void __ApplyHouseholderOnTheLeft(MAT<T>& A, T* w, int r) {
    for (int i = 0; i < A.cols; ++i) {
        __mixed(A[i] + r, w, -__dot(w, A[i] + r, A.rows - r), A[i] + r, A.rows - r);
    }
}
template<typename t>
inline void __ApplyHouseholderOnTheRight(MAT<complex<t>>& A, complex<t>* w, complex<t>* v, int r) {
    __set0(v, A.rows);
    for (int i = r; i < A.cols; ++i) {
        __mixed(v, A[i], w[i], v, A.rows);
    }
    for (int i = r; i < A.cols; ++i) {
        __mixed(A[i], v, -w[i].conj(), A[i], A.rows);
    }
}

template<typename T>
inline void __ApplyHouseholderOnTheRight2(MAT<T>& A, T* w, T* v, int m, int n) {
    __set0(v, A.rows - m);
    for (int i = n; i < A.cols; ++i) {
        __mixed(v, A[i] + m, w[i], v, A.rows - m);
    }
    for (int i = n; i < A.cols; ++i) {
        __mixed(A[i] + m, v, -w[i], A[i] + m, A.rows - m);
    }
}
template<typename T>
inline void __ApplyHouseholderOnTheLeft2(MAT<T>& A, T* w, int m, int n) {
    for (int i = m; i < A.cols; ++i) {
        __mixed(A[i] + n, w, -__dot(w, A[i] + n, A.rows - n), A[i] + n, A.rows - n);
    }
}

///*
template<typename T>
void __Householder_Q1(MAT<T>& Q, int n) {
    T temp;
    T* w;
    MAT<T> Temp1, Temp2;

    Temp2.rows = Q.rows - n;
    Temp2.data = Q.data + Q.rows * Q.rows - Temp2.rows * Temp2.rows;
    __set0(Temp2.data, Temp2.rows * Temp2.rows);

    for (int r = 0; r < Temp2.rows; ++r) {
        Temp2[r][r] = 1;
    }
    for (int r = n - 1; r >= 0; --r) {
        Temp1.rows = Q.rows - r;
        Temp1.data = Q.data + Q.rows * Q.rows - Temp1.rows * Temp1.rows;
        __set0(Temp1.data + Temp1.rows, Temp1.rows - 1);

        w = Temp1[0];
        for (int i = 1; i < Temp1.rows; ++i) {
            temp = -__dot(w + 1, Temp2[i - 1], Temp2.rows);
            Temp1[i][0] = w[0] * temp;
            __mixed(Temp2[i - 1], w + 1, temp, Temp1[i] + 1, Temp2.rows);
        }
        __times(w, -w[0], w, Temp1.rows);
        *w += 1;

        Temp2.rows = Temp1.rows;
        Temp2.data = Temp1.data;
    }
    Temp1.data = 0;
    Temp2.data = 0;
}

template<typename t>
void __Householder_Q1(MAT<complex<t>>& Q, int n) {
    complex<t> temp;
    complex<t>* w;
    MAT<complex<t>> Temp1, Temp2;

    Temp2.rows = Q.rows - n;
    Temp2.data = Q.data + Q.rows * Q.rows - Temp2.rows * Temp2.rows;
    __set0(Temp2.data, Temp2.rows * Temp2.rows);

    for (int r = 0; r < Temp2.rows; ++r) {
        Temp2[r][r] = 1;
    }
    for (int r = n - 1; r >= 0; --r) {
        Temp1.rows = Q.rows - r;
        Temp1.data = Q.data + Q.rows * Q.rows - Temp1.rows * Temp1.rows;
        __set0(Temp1.data + Temp1.rows, Temp1.rows - 1);

        w = Temp1[0];
        for (int i = 1; i < Temp1.rows; ++i) {
            temp = -__dot(w + 1, Temp2[i - 1], Temp2.rows);
            Temp1[i][0] = w[0] * temp;
            __mixed(Temp2[i - 1], w + 1, temp, Temp1[i] + 1, Temp2.rows);
        }
        __times(w, -w[0].conj(), w, Temp1.rows);
        *w += 1;

        Temp2.rows = Temp1.rows;
        Temp2.data = Temp1.data;
    }
    Temp1.data = 0;
    Temp2.data = 0;
}

template<typename T>
void __Householder_Q2(MAT<T>& Q) {
    T* w;
    MAT<T> Temp1, Temp2;

    Temp2.rows = 2;
    Temp2.data = Q.data + Q.rows * Q.rows - 4;
    Temp2.data[0] = 1;
    Temp2.data[1] = 0;
    Temp2.data[2] = 0;
    Temp2.data[3] = 1;

    for (int r = Q.rows - 3; r >= 0; --r) {
        Temp1.rows = Q.rows - r;
        Temp1.data = Q.data + Q.rows * Q.rows - Temp1.rows * Temp1.rows;
        __set0(Temp1.data + Temp1.rows, Temp1.rows - 1);

        w = Temp1[0];
        for (int i = 1; i < Temp1.rows; ++i) {
            *(Temp1[i]) = 0;
            __mixed(Temp2[i - 1], w + 1, -__dot(w + 1, Temp2[i - 1], Temp2.rows), Temp1[i] + 1, Temp2.rows);
        }
        __set0(w, Temp1.rows);
        *w = 1;

        Temp2.rows = Temp1.rows;
        Temp2.data = Temp1.data;
    }
    Temp1.data = 0;
    Temp2.data = 0;
}

template<typename T>
void __Householder_Q3(MAT<T>& Q) {
    T* w;
    MAT<T> Temp1, Temp2;

    Temp2.rows = 1;
    Temp2.data = Q.data + Q.rows * Q.rows - 1;
    Temp2.data[0] = 1;

    for (int r = Q.rows - 2; r >= 0; --r) {
        Temp1.rows = Q.rows - r;
        Temp1.data = Q.data + Q.rows * Q.rows - Temp1.rows * Temp1.rows;
        __set0(Temp1.data + Temp1.rows, Temp1.rows - 1);

        w = Temp1[0];
        for (int i = 1; i < Temp1.rows; ++i) {
            *(Temp1[i]) = 0;
            __mixed(Temp2[i - 1], w + 1, -__dot(w + 1, Temp2[i - 1], Temp2.rows), Temp1[i] + 1, Temp2.rows);
        }
        __set0(w, Temp1.rows);
        *w = 1;

        Temp2.rows = Temp1.rows;
        Temp2.data = Temp1.data;
    }
    Temp1.data = 0;
    Temp2.data = 0;
}

//*/

template<typename T>
void __QR_Householder(MAT<T>& Q, MAT<T>& R) {
    int n = 0;
    if (R.rows > R.cols) {
        n = R.cols;
    }
    else {
        n = R.rows;
    };
    int r;

    T cr, hr;
    T* w;
    for (r = 0; r < n; ++r) {
        w = Q.data + 2 * Q.rows * r - r * r;
        __copy(w, R[r] + r, R.rows - r);
        cr = sqrt(__dot(w, w, R.rows - r));

        if (cr == 0) {
            continue;
        }

        if (w[0] < 0.0)cr = -cr;
        w[0] += cr;
        hr = (T)1 / sqrt(cr * w[0]);

        __times(w, hr, w, R.rows - r);
        __ApplyHouseholderOnTheLeft2(R, w, r, r);
    }
    __Householder_Q1(Q, n);
}

template<typename t>
void __QR_Householder(MAT<complex<t>>& Q, MAT<complex<t>>& R) {
    int n = 0;
    if (R.rows > R.cols) {
        n = R.cols;
    }
    else {
        n = R.rows;
    };
    int r;

    complex<t> sum, _sign;
    t cr, hr;
    complex<t>* w;
    for (r = 0; r < n; ++r) {
        w = Q.data + 2 * Q.rows * r - r * r;
        __copy(w, R[r] + r, R.rows - r);
        cr = sqrt(__dot((t*)w, (t*)w, (R.rows - r) * 2));

        if (cr == 0) {
            continue;
        }

        _sign = sign(w[0]) * cr;
        hr = 1 / sqrt(cr * cr - times_conj(_sign, w[0]).real);
        w[0] -= _sign;

        __times((t*)w, hr, (t*)w, (R.rows - r) * 2);
        __ApplyHouseholderOnTheLeft2(R, w, r, r);
    }
    __Householder_Q1(Q, n);
}
