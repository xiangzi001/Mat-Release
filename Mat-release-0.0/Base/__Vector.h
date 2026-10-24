

template <typename T>
class VEC : public MAT<T>
{
public:
	VEC() {
		this->rows = 0;
		this->cols = 1;
		this->data = nullptr;
	}

	VEC(int len) : MAT<T>(len, 1) {
	}

	template<typename t>
	VEC(int len, t* p) : MAT<T>(len, 1, p) {
	}
	template<typename t>
	VEC(int len, t value, ...) : MAT<T>(len, 1) {
		va_list num;
		va_start(num, value);
		this->data[0] = value;
		for (int i = 1; i < len; ++i) {
			this->data[i] = va_arg(num, t);
		}
		va_end(num);
	}

	VEC(VEC& a) : MAT<T>(a.rows, 1, a.data) {
	}
	VEC(VEC&& a) {
		this->rows = a.rows;
		this->cols = 1;
		this->data = a.data;
		a.data = 0;
	}
	template <class t>
	VEC(const VEC<t>& a) {
		this->rows = a.rows;
		this->cols = 1;
		this->data = new T[this->rows];
		__copy(this->data, a.data, this->rows);
	}
	VEC(MAT<T>& A) {
		this->rows = A.rows * A.cols;
		this->cols = 1;
		this->data = new T[this->rows];
		__copy(this->data, A.data, A.rows * A.cols);
	}
	VEC(MAT<T>&& A) {
		this->rows = A.rows * A.cols;
		this->cols = 1;
		this->data = A.data;
		A.data = 0;
	}
	template <class t>
	VEC(const MAT<t>& A) {
		this->rows = A.rows * A.cols;
		this->cols = 1;
		this->data = new T[this->rows];
		__copy(this->data, A.data, this->rows);
	}

	void init(int r) {
		if (this->rows != r) {
			this->rows = r;
			this->cols = 1;
			delete[] this->data;
			this->data = new T[r];
		}
	}
	template<typename t>
	void init(int r, t* p) {
		if (this->rows != r) {
			this->rows = r;
			this->cols = 1;
			delete[] this->data;
			this->data = new T[r];
		}
		for (int i = 0; i < r; ++i) {
			this->data[i] = p[i];
		}
	}
	template<typename t>
	void init(int r, t value, ...) {
		if (this->rows != r) {
			this->rows = r;
			this->cols = 1;
			delete[] this->data;
			this->data = new T[r];
		}
		va_list num;
		va_start(num, value);
		this->data[0] = value;
		for (int i = 1; i < r; ++i) {
			this->data[i] = va_arg(num, t);
		}
		va_end(num);
	}

	template<typename t>
	void diagonal(const MAT<t>& A) {
		for (int i = 0; i < this->rows; ++i)
			this->data[i] = A(i, i);
	}

	void operator = (VEC<T> x) {
		this->rows = x.rows;
		this->cols = 1;
		T* temp = x.data;
		x.data = this->data;
		this->data = temp;
	}

	T& operator [](int n) const {
		return (this->data)[n];
	}
	T& operator ()(int n) const {
		return (this->data)[n];
	}
};

template<typename t1, typename t2>
auto operator + (const VEC<t1>& A, const VEC<t2>& B) {
	VEC<decltype(std::declval<t1>() + std::declval<t2>())> n(A.rows);
	__add(A.data, B.data, n.data, A.rows);
	return n;
}
template<typename t1, typename t2>
auto operator - (const VEC<t1>& A, const VEC<t2>& B) {
	VEC<decltype(std::declval<t1>() - std::declval<t2>())> n(A.rows);
	__sub(A.data, B.data, n.data, A.rows);
	return n;
}