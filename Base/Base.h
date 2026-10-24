
#ifndef va_start
#define va_start(ap, x) (void)(__vcrt_assert_va_start_is_not_reference<decltype(x)>(), (void)(__va_start(&ap, x)))
#endif // !va_start

#ifndef va_arg
#define va_arg(ap, t)                                               \
        ((sizeof(t) > sizeof(__int64) || (sizeof(t) & (sizeof(t) - 1)) != 0) \
            ? **(t**)((ap += sizeof(__int64)) - sizeof(__int64))             \
            :  *(t* )((ap += sizeof(__int64)) - sizeof(__int64)))
#endif // !va_arg

#ifndef va_end
#define va_end(ap)        ((void)(ap = (va_list)0))
#endif // !va_end

template <typename T>
T gcd(T a, T b) {
	while (b != 0) {
		T t = b;
		b = a % b;
		a = t;
	}
	return a;
}

typedef int int32;
typedef unsigned int uint32;
typedef long long int64;
typedef unsigned long long uint64;
typedef float float32;
typedef double float64;

#include "__Complex.h"

template<typename T, typename t>
inline void __copy(T* dst, const t* src, size_t len) {
	for (size_t i = 0; i < len; ++i) {
		dst[i] = (T)(src[i]);
	}
}
template<typename T>
inline void __copy(T* dst, const T* src, size_t len) {
	memcpy(dst, src, sizeof(T) * len);
}

template<typename T>
inline void __set0(T* dst, size_t len) {
	for (size_t i = 0; i < len; ++i) {
		dst[i] = T(0);
	}
}
inline void __set0(int* dst, size_t len) {
	memset(dst, 0, sizeof(int) * len);
}
inline void __set0(long long* dst, size_t len) {
	memset(dst, 0, sizeof(long long) * len);
}
inline void __set0(float* dst, size_t len) {
	memset(dst, 0, sizeof(float) * len);
}
inline void __set0(double* dst, size_t len) {
	memset(dst, 0, sizeof(double) * len);
}
template<typename t>
inline void __set0(complex<t>* dst, size_t len) {
	__set0((t*)(dst), len * 2);
}

using std::max;
using std::min;
