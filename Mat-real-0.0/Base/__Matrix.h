
template <typename T>
class MAT {
public:
	int rows;
	int cols;
	T* data;

	MAT() :
		data(0), rows(0), cols(0)
	{
	}

	MAT(int r, int c) :
		data(new T[r * c]), rows(r), cols(c)
	{
	}

	template<class t>
	MAT(int r, int c, t* p) :
		rows(r), cols(c)
	{
		int s = r * c;
		data = new T[s];
		for (int i = 0; i < s; ++i) {
			data[i] = p[i];
		}
	}
	MAT(MAT& a) :
		rows(a.rows),
		cols(a.cols)
	{
		int s = rows * cols;
		data = new T[s];
		__copy(data, a.data, s);
	}
	MAT(MAT&& a) :
		rows(a.rows),
		cols(a.cols),
		data(a.data)
	{
		a.data = 0;
	}
	template <class t>
	MAT(const MAT<t>& a) :
		rows(a.rows),
		cols(a.cols)
	{
		int s = rows * cols;
		data = new T[s];
		__copy(data, a.data, s);
	}

	~MAT() {
		if (data != 0) {
			delete[] data;
			data = 0;
		}
	}

	void init(int r, int c) {
		if (rows != r or cols != c) {
			rows = r;
			cols = c;
			delete[] data;
			data = new T[r * c];
		}
	}
	template<typename t>
	void init(int r, int c, t* p) {
		int s = r * c;
		if (rows != r or cols != c) {
			rows = r;
			cols = c;
			delete[] data;
			data = new T[s];
		}
		for (int i = 0; i < s; ++i) {
			data[i] = p[i];
		}
	}
	void iden(int a) {
		int s = a * a;
		if (rows != a or cols != a) {
			rows = a;
			cols = a;
			delete[] data;
			data = new T[s];
		}
		for (int i = 0; i < s; ++i)
			data[i] = 0;
		for (int i = 0; i < a; ++i)
			data[i * cols + i] = 1;
	}
	void fill(T a) {
		int s = rows * cols;
		for (int i = 0; i < s; ++i)
			data[i] = a;
	}
	void diagonal(T* a) {
		int s = rows * cols;
		for (int i = 0; i < s; ++i)
			data[i] = 0;
		for (int i = 0; i < rows; ++i)
			(data + i * rows)[i] = a[i];
	}
	template<typename t>
	void diagonal(t* a, int size) {
		init(size, size);
		int s = rows * cols;
		for (int i = 0; i < s; ++i)
			data[i] = 0;
		for (int i = 0; i < rows; ++i)
			(data + i * rows)[i] = (T)(a[i]);
	}

	auto transpose();
	auto dagger();

	void swap_rows(int r1, int r2) {
		T temp;
		T* Temp1 = (data + r1), * Temp2 = (data + r2);
		int k;
		for (int i = 0; i < cols; ++i) {
			k = i * rows;
			temp = Temp1[k];
			Temp1[k] = Temp2[k];
			Temp2[k] = temp;
		}
	}
	void swap_cols(int c1, int c2) {
		T temp;
		T* Temp1 = (data + rows * c1), * Temp2 = (data + rows * c2);
		for (int i = 0; i < rows; ++i) {
			temp = Temp1[i];
			Temp1[i] = Temp2[i];
			Temp2[i] = temp;
		}
	}

	T* operator [] (int c) const {
		return (data + rows * c);
	}
	T& operator () (int r, int c) const {
		return data[rows * c + r];
	}

	//copy B to A
	void operator = (MAT<T> B) {
		rows = B.rows;
		cols = B.cols;
		T* temp = B.data;
		B.data = data;
		data = temp;
	}

	//exchange A and B
	void operator & (MAT& B) {
		int temp = rows;
		rows = B.rows;
		B.rows = temp;

		temp = cols;
		cols = B.cols;
		B.cols = temp;

		T* _temp = data;
		data = B.data;
		B.data = _temp;
	}

	template <typename U>
	friend class VEC;
};

template<typename t>
MAT<t> _T(const MAT<t>& A) {
	MAT<t> n(A.cols, A.rows);
	int i, j;
	for (i = 0; i < n.rows; ++i)
		for (j = 0; j < n.cols; ++j) {
			n.data[j * n.rows + i] = A.data[i * A.rows + j];
		}

	return n;
}
template<typename T>
auto MAT<T>::transpose() {
	return _T(*this);
}

template<typename t>
MAT<t> _H(const MAT<t>& A) {
	MAT<t> n(A.cols, A.rows);
	int i, j;
	for (i = 0; i < n.rows; ++i)
		for (j = 0; j < n.cols; ++j) {
			n.data[j * n.rows + i] = A.data[i * A.rows + j];
		}

	return n;
}
template<typename t>
MAT<complex<t>> _H(const MAT<complex<t>>& A) {
	MAT<complex<t>> n(A.cols, A.rows);
	int i, j;
	for (i = 0; i < n.rows; ++i)
		for (j = 0; j < n.cols; ++j) {
			n.data[j * n.rows + i] = A.data[i * A.rows + j].conj();
		}

	return n;
}
template<typename T>
auto MAT<T>::dagger() {
	return _H(*this);
}


template<class t>
MAT<t> operator * (t a, const MAT<t>& A) {
	MAT<t> n(A.rows, A.cols);

	__times(A.data, a, n.data, A.rows * A.cols);

	return n;
}
template<class t>
MAT<t> operator * (const MAT<t>& A, t a) {
	MAT<t> n(A.rows, A.cols);

	__times(A.data, a, n.data, A.rows * A.cols);

	return n;
}
template<class t>
MAT<t> operator / (const MAT<t>& A, t a) {
	MAT<t> n(A.rows, A.cols);

	__times(A.data, t(1) / a, n.data, A.rows * A.cols);

	return n;
}

template<typename t1, typename t2>
auto operator + (const MAT<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>() + std::declval<t2>())> n(A.rows, A.cols);
	__add(A.data, B.data, n.data, A.rows * A.cols);
	return n;
}
template<typename t1, typename t2>
auto operator - (const MAT<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>() - std::declval<t2>())> n(A.rows, A.cols);
	__sub(A.data, B.data, n.data, A.rows * A.cols);
	return n;
}

/*
template<typename T>
class __TemplateTranspose {
public:
	const MAT<T>& __Mat;
	__TemplateTranspose(const MAT<T>& a) : __Mat(a)
	{
	}
	operator MAT<T>() {
		MAT<T> n(__Mat.cols, __Mat.rows);
		int i, j;
		for (i = 0; i < n.rows; ++i)
			for (j = 0; j < n.cols; ++j) {
				n.data[j * n.rows + i] = __Mat.data[i * n.cols + j];
			}

		return n;
	}
};
template<typename t>
__TemplateTranspose<t> _T(const MAT<t>& A) {
	return __TemplateTranspose<t>(A);
}
template<typename T>
auto MAT<T>::transpose() {
	return __TemplateTranspose<T>(*this);
}

template<typename T>
class __TemplateDagger {
public:
	const MAT<T>& __Mat;
	__TemplateDagger(const MAT<T>& a) : __Mat(a)
	{
	}
	operator MAT<T>() {
		MAT<T> n(__Mat.cols, __Mat.rows);
		int i, j;
		for (i = 0; i < n.rows; ++i)
			for (j = 0; j < n.cols; ++j) {
				n.data[j * n.rows + i] = __Mat.data[i * n.cols + j].conj();
			}

		return n;
	}
};
template<typename t>
__TemplateDagger<complex<t>> _H(const MAT<complex<t>>& A) {
	return __TemplateDagger<complex<t>>(A);
}
template<typename T>
auto MAT<T>::dagger() {
	return __TemplateDagger<T>(*this);
}
*/
