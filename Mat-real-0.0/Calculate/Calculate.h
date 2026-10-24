
#ifndef XiangZiMatImport_SIMD
#define XiangZiMatImport_SIMD

#include<immintrin.h>

#ifdef __AVX512__
#include "__AVX512_double.h"

#elif defined(__AVX2__)
#include "__AVX2_double.h"
#endif

#endif // !XiangZiMatImport_SIMD

template<typename t1, typename t2, typename t3>
inline void __times(t1* __restrict a, t2 b, t3* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] * b;
	}
}

template<typename t1, typename t2, typename t3>
inline void __add(t1* __restrict a, t2* __restrict b, t3* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b[i];
	}
}
template<typename t1, typename t2, typename t3>
inline void __add(complex<t1>* a, complex<t2>* b, complex<t3>* ans, int len) {
	__add((t1*)a, (t2*)b, (t3*)ans, len * 2);
}

template<typename t1, typename t2, typename t3>
inline void __sub(t1* __restrict a, t2* __restrict b, t3* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] - b[i];
	}
}
template<typename t1, typename t2, typename t3>
inline void __sub(complex<t1>* a, complex<t2>* b, complex<t3>* ans, int len) {
	__sub((t1*)a, (t2*)b, (t3*)ans, len * 2);
}

//ans=a+b*p
template<typename t1, typename t2, typename t3, typename t4>
inline void __mixed(t1* __restrict a, t2* __restrict b, t3 q, t4* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b[i] * q;
	}
}

template<typename T>
inline void __swap(T* __restrict a, T* __restrict b, int len) {
	for (int i = 0; i < len; ++i) {
		T temp = a[i];
		a[i] = b[i];
		b[i] = temp;
	}
}
template<typename t1, typename t2, typename T = decltype(std::declval<t1>()* std::declval<t2>())>
inline T __dot(t1* __restrict a, t2* __restrict b, int len) {
	T temp0(0), temp1(0), temp2(0), temp3(0);
	int n = 4 * ((int)(len / 4));
	for (int i = 0; i < n; i += 4) {
		temp0 += a[i] * b[i];
		temp1 += a[i + 1] * b[i + 1];
		temp2 += a[i + 2] * b[i + 2];
		temp3 += a[i + 3] * b[i + 3];
	}
	T ans = temp0 + temp1 + temp2 + temp3;
	for (int i = n; i < len; ++i) {
		ans += a[i] * b[i];
	}
	return ans;
}

template<typename t1, typename t2,
	typename t3 = decltype(std::declval<t1>()* std::declval<t2>())>
inline complex<t3> __dot(complex<t1>* __restrict a, complex<t2>* __restrict b, int len) {
	complex<t3> temp0(0), temp1(0), temp2(0), temp3(0);
	int i = 0;
	for (; i + 3 < len; i += 4) {
		temp0 += a[i].conj() * b[i];
		temp1 += a[i + 1].conj() * b[i + 1];
		temp2 += a[i + 2].conj() * b[i + 2];
		temp3 += a[i + 3].conj() * b[i + 3];
	}
	complex<t3> ans = temp0 + temp1 + temp2 + temp3;
	for (; i < len; ++i) {
		ans += a[i].conj() * b[i];
	}
	return ans;
}

template<typename t, typename t2 = decltype(abs(std::declval<t>()))>
t2 __MaxAbs(t* a, int len) {
	int i = 0;
	t2 ans(0);
	if (len > 3) {
		t2 max0(abs(a[0])), max1(abs(a[1])), max2(abs(a[2])), max3(abs(a[3]));
		for (i = 4; i + 3 < len; i += 4) {
			t2 v0 = abs(a[i]);
			t2 v1 = abs(a[i + 1]);
			t2 v2 = abs(a[i + 2]);
			t2 v3 = abs(a[i + 3]);
			if (v0 > max0) max0 = v0;
			if (v1 > max1) max1 = v1;
			if (v2 > max2) max2 = v2;
			if (v3 > max3) max3 = v3;
		}
		ans = max0;
		if (max0 < max1) max0 = max1;
		if (max2 < max3) max2 = max3;
		if (max0 < max2) ans = max2;
		else ans = max0;
	}
	for (; i < len; ++i) {
		t2 temp = abs(a[i]);
		if (temp > ans) ans = temp;
	}
	return ans;
}
