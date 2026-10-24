
template <typename T>
class __BlockMat {
public:
	int rows;
	int cols;
	int _rows;
	T* data;
	__BlockMat() :
		rows(0),
		cols(0),
		_rows(0),
		data(nullptr)
	{
	}
	__BlockMat(const MAT<T>& A, int m, int n, int r, int c) :
		rows(r),
		cols(c),
		_rows(A.rows),
		data(A.data + A.rows * n + m)
	{
	}
	void set(const MAT<T>& A, int m, int n, int r, int c) {
		rows = r;
		cols = c;
		_rows = A.rows;
		data = A.data + A.rows * n + m;
	}
	T* operator [] (int c) {
		return (data + _rows * c);
	}
	T& operator () (int r, int c) {
		return data[_rows * c + r];
	}

	void swap_rows(int r1, int r2, int c1, int c2) {
		T temp;
		T* Temp1 = (data + r1), * Temp2 = (data + r2);
		int k = c1 * _rows;
		for (int i = c1; i < c2; ++i) {
			temp = Temp1[k];
			Temp1[k] = Temp2[k];
			Temp2[k] = temp;
			k += _rows;
		}
	}
};
