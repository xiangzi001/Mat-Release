
template<typename T>
void __QR_GramSchmidt(MAT<T>& Q, MAT<T>& R) {
    int i, j;
    T len;

    for (j = 0; j < Q.cols; ++j) {
        for (i = 0; i < j; ++i) {
            len = __dot(Q[i], Q[j], Q.rows);
            __mixed(Q[j], Q[i], -len, Q[j], Q.rows);
            R(i, j) = len;
        }

        len = sqrt(__dot(Q[j], Q[j], Q.rows));
        R(j, j) = len;
        __times(Q[j], (T)1 / len, Q[j], Q.rows);
    }
}


template<typename t>
void __QR_GramSchmidt(MAT<complex<t>>& Q, MAT<complex<t>>& R) {
    int i, j, k;
    complex<t> temp;
    t len;
    complex<t> * a, * b;

    for (j = 0; j < Q.cols; ++j) {
        a = Q[j];
        for (i = 0; i < j; ++i) {
            b = Q[i];
            temp = 0;
            for (k = 0; k < Q.rows; ++k) {
                temp += b[k].conj() * a[k];
            }
            for (k = 0; k < Q.rows; ++k) {
                a[k] -= b[k] * temp;
            }
            R(i, j) = temp;
        }

        len = sqrt(__dot((t*)a, (t*)a, Q.rows * 2));
        R(j, j) = len;
        __times((t*)a, (t)1 / len, (t*)a, Q.rows * 2);
    }
}