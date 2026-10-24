
template <typename T, typename R = decltype(abs(std::declval<T>()))>
R Norm_one(const MAT<T>& A) {
	R ret(0);
	R temp;
	int i;
	T* a;
	for (int j = 0; j < A.cols; ++j) {
		temp = 0;
		a = A.data + A.rows * j;
		for (i = 0; i < A.rows; ++i) {
			temp += abs(a[i]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

template <typename T, typename R = decltype(abs(std::declval<T>()))>
R Norm_inf(const MAT<T>& A) {
	R ret(0);
	R temp;
	int j;
	for (int i = 0; i < A.rows; ++i) {
		temp = 0;
		for (j = 0; j < A.cols; ++j) {
			temp += abs(A.data[j * A.rows + i]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

template <typename T>
auto Norm_F(const MAT<T>& A) {
	return sqrt(__dot(A.data, A.data, A.rows * A.cols));
}

template <typename T>
auto MaxAbs(const MAT<T>& A) {
	return __MaxAbs(A.data, A.rows * A.cols);
}


#ifdef XiangZiMatImport_SIMD

#ifdef __AVX512__

double Norm_one(const MAT<double>& A) {
    int i, j;
    int n = A.rows / 8;
    int m = A.rows - n * 8;
    double max = 0.0, t;
    __m512d sum;
    __m512d* temp;
    __mmask8 mask = (1 << m) - 1;

    for (i = 0; i < A.cols; ++i) {
        sum = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_abs_pd(*(__m512d*)(A.data + i * A.rows)));
        temp = (__m512d*)(A.data + i * A.rows + m);
        for (j = 0; j < n; ++j) {
            sum = _mm512_add_pd(sum, _mm512_abs_pd(temp[j]));
        }
        t = _mm512_reduce_add_pd(sum);
        if (t > max) max = t;
    }
    return max;
}

float Norm_one(const MAT<float>& A) {
    int i, j;
    int n = A.rows / 16;
    int m = A.rows - n * 16;
    float max = 0.0f, t;
    __m512 sum;
    __m512* temp;
    __mmask16 mask = (1 << m) - 1;

    for (i = 0; i < A.cols; ++i) {
        sum = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_abs_ps(*(__m512*)(A.data + i * A.rows)));
        temp = (__m512*)(A.data + i * A.rows + m);
        for (j = 0; j < n; ++j) {
            sum = _mm512_add_ps(sum, _mm512_abs_ps(temp[j]));
        }
        t = _mm512_reduce_add_ps(sum);
        if (t > max) max = t;
    }
    return max;
}

int32_t Norm_one(const MAT<int32_t>& A) {
    int i, j;
    int n = A.rows / 16;
    int m = A.rows - n * 16;
    int32_t max = 0, t;
    __m512i sum;
    __m512i* temp;
    __mmask16 mask = (1 << m) - 1;

    for (i = 0; i < A.cols; ++i) {
        sum = _mm512_mask_blend_epi32(mask, _mm512_setzero_si512(), _mm512_abs_epi32(*(__m512i*)(A.data + i * A.rows)));
        temp = (__m512i*)(A.data + i * A.rows + m);
        for (j = 0; j < n; ++j) {
            sum = _mm512_add_epi32(sum, _mm512_abs_epi32(temp[j]));
        }
        t = _mm512_reduce_add_epi32(sum);
        if (t > max) max = t;
    }
    return max;
}

int64_t Norm_one(const MAT<int64_t>& A) {
    int i, j;
    int n = A.rows / 8;
    int m = A.rows - n * 8;
    int64_t max = 0, t;
    __m512i sum;
    __m512i* temp;
    __mmask8 mask = (1 << m) - 1;

    for (i = 0; i < A.cols; ++i) {
        sum = _mm512_mask_blend_epi64(mask, _mm512_setzero_si512(), _mm512_abs_epi64(*(__m512i*)(A.data + i * A.rows)));
        temp = (__m512i*)(A.data + i * A.rows + m);
        for (j = 0; j < n; ++j) {
            sum = _mm512_add_epi64(sum, _mm512_abs_epi64(temp[j]));
        }
        t = _mm512_reduce_add_epi64(sum);
        if (t > max) max = t;
    }
    return max;
}

#elif defined(__AVX2__)

double Norm_one(const MAT<double>& A) {
    int i, j;
    int n = A.rows / 4;
    int m = A.rows - n * 4;
    double max = 0.0, t;
    __m256d sum;
    __m256d* temp;
    __m256d maskd = _mm256_castsi256_pd(_mm256_setr_epi64x(
        m > 0 ? -1LL : 0LL,
        m > 1 ? -1LL : 0LL,
        m > 2 ? -1LL : 0LL,
        m > 3 ? -1LL : 0LL
    ));

    for (i = 0; i < A.cols; ++i) {
        __m256d first = *(__m256d*)(A.data + i * A.rows);
        first = _mm256_andnot_pd(_mm256_set1_pd(-0.0), first);
        sum = _mm256_blendv_pd(_mm256_setzero_pd(), first, maskd);
        temp = (__m256d*)(A.data + i * A.rows + m);
        for (j = 0; j < n; ++j) {
            __m256d val = _mm256_andnot_pd(_mm256_set1_pd(-0.0), temp[j]);
            sum = _mm256_add_pd(sum, val);
        }
        __m128d low = _mm256_castpd256_pd128(sum);
        __m128d high = _mm256_extractf128_pd(sum, 1);
        __m128d sum128 = _mm_add_pd(low, high);
        sum128 = _mm_hadd_pd(sum128, sum128);
        t = _mm_cvtsd_f64(sum128);
        if (t > max) max = t;
    }
    return max;
}

float Norm_one(const MAT<float>& A) {
    int i, j;
    int n = A.rows / 8;
    int m = A.rows - n * 8;
    float max = 0.0f, t;
    __m256 sum;
    __m256* temp;
    __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(
        m > 0 ? -1 : 0,
        m > 1 ? -1 : 0,
        m > 2 ? -1 : 0,
        m > 3 ? -1 : 0,
        m > 4 ? -1 : 0,
        m > 5 ? -1 : 0,
        m > 6 ? -1 : 0,
        m > 7 ? -1 : 0
    ));

    for (i = 0; i < A.cols; ++i) {
        __m256 first = *(__m256*)(A.data + i * A.rows);
        first = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), first);
        sum = _mm256_blendv_ps(_mm256_setzero_ps(), first, mask);
        temp = (__m256*)(A.data + i * A.rows + m);
        for (j = 0; j < n; ++j) {
            __m256 val = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), temp[j]);
            sum = _mm256_add_ps(sum, val);
        }
        __m128 low = _mm256_castps256_ps128(sum);
        __m128 high = _mm256_extractf128_ps(sum, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        t = _mm_cvtss_f32(sum128);
        if (t > max) max = t;
    }
    return max;
}

int32_t Norm_one(const MAT<int32_t>& A) {
    int i, j;
    int n = A.rows / 8;
    int m = A.rows - n * 8;
    int32_t max = 0, t;
    __m256i sum;
    __m256i* temp;
    __m256i mask = _mm256_setr_epi32(
        m > 0 ? -1 : 0,
        m > 1 ? -1 : 0,
        m > 2 ? -1 : 0,
        m > 3 ? -1 : 0,
        m > 4 ? -1 : 0,
        m > 5 ? -1 : 0,
        m > 6 ? -1 : 0,
        m > 7 ? -1 : 0
    );

    for (i = 0; i < A.cols; ++i) {
        __m256i first = *(__m256i*)(A.data + i * A.rows);
        first = _mm256_abs_epi32(first);
        sum = _mm256_blendv_epi8(_mm256_setzero_si256(), first, mask);
        temp = (__m256i*)(A.data + i * A.rows + m);
        for (j = 0; j < n; ++j) {
            __m256i val = _mm256_abs_epi32(temp[j]);
            sum = _mm256_add_epi32(sum, val);
        }
        __m128i low = _mm256_castsi256_si128(sum);
        __m128i high = _mm256_extracti128_si256(sum, 1);
        __m128i sum128 = _mm_add_epi32(low, high);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        t = _mm_cvtsi128_si32(sum128);
        if (t > max) max = t;
    }
    return max;
}

#endif

#ifdef __AVX512__

double Norm_inf(const MAT<double>& A) {
    int i;
    int n = (int)(A.rows / 8) * 8;
    __m512d max = _mm512_setzero_pd();
    __m512d sum;

    for (int j = 0; j < n; j += 8) {
        sum = _mm512_setzero_pd();
        for (i = 0; i < A.cols; ++i) {
            sum = _mm512_add_pd(sum, _mm512_abs_pd(*(__m512d*)(A.data + i * A.rows + j)));
        }
        max = _mm512_max_pd(max, sum);
    }

    sum = _mm512_setzero_pd();
    for (i = 0; i < A.cols; ++i) {
        sum = _mm512_add_pd(sum, _mm512_abs_pd(*(__m512d*)(A.data + i * A.rows + n)));
    }
    max = _mm512_max_pd(max, _mm512_mask_blend_pd((1 << (A.rows - n)) - 1, _mm512_setzero_pd(), sum));

    return _mm512_reduce_max_pd(max);
}

float Norm_inf(const MAT<float>& A) {
    int n = (A.rows / 16) * 16;
    __m512 max = _mm512_setzero_ps();
    __m512 sum;

    for (int j = 0; j < n; j += 16) {
        sum = _mm512_setzero_ps();
        for (int i = 0; i < A.cols; ++i) {
            sum = _mm512_add_ps(sum, _mm512_abs_ps(*(__m512*)(A.data + i * A.rows + j)));
        }
        max = _mm512_max_ps(max, sum);
    }

    sum = _mm512_setzero_ps();
    for (int i = 0; i < A.cols; ++i) {
        sum = _mm512_add_ps(sum, _mm512_abs_ps(*(__m512*)(A.data + i * A.rows + n)));
    }
    max = _mm512_max_ps(max, _mm512_mask_blend_ps((1 << (A.rows - n)) - 1, _mm512_setzero_ps(), sum));

    return _mm512_reduce_max_ps(max);
}

int32_t Norm_inf(const MAT<int32_t>& A) {
    int n = (A.rows / 16) * 16;
    __m512i max = _mm512_setzero_si512();
    __m512i sum;

    for (int j = 0; j < n; j += 16) {
        sum = _mm512_setzero_si512();
        for (int i = 0; i < A.cols; ++i) {
            sum = _mm512_add_epi32(sum, _mm512_abs_epi32(*(__m512i*)(A.data + i * A.rows + j)));
        }
        max = _mm512_max_epi32(max, sum);
    }

    sum = _mm512_setzero_si512();
    for (int i = 0; i < A.cols; ++i) {
        sum = _mm512_add_epi32(sum, _mm512_abs_epi32(*(__m512i*)(A.data + i * A.rows + n)));
    }
    max = _mm512_max_epi32(max,
        _mm512_mask_blend_epi32((1 << (A.rows - n)) - 1, _mm512_setzero_si512(), sum));

    return _mm512_reduce_max_epi32(max);
}

int64_t Norm_inf(const MAT<int64_t>& A) {
    int n = (A.rows / 8) * 8;
    __m512i max = _mm512_setzero_si512();
    __m512i sum;

    for (int j = 0; j < n; j += 8) {
        sum = _mm512_setzero_si512();
        for (int i = 0; i < A.cols; ++i) {
            sum = _mm512_add_epi64(sum, _mm512_abs_epi64(*(__m512i*)(A.data + i * A.rows + j)));
        }
        max = _mm512_max_epi64(max, sum);
    }

    sum = _mm512_setzero_si512();
    for (int i = 0; i < A.cols; ++i) {
        sum = _mm512_add_epi64(sum, _mm512_abs_epi64(*(__m512i*)(A.data + i * A.rows + n)));
    }
    max = _mm512_max_epi64(max,
        _mm512_mask_blend_epi64((1 << (A.rows - n)) - 1, _mm512_setzero_si512(), sum));

    return _mm512_reduce_max_epi64(max);
}

#elif defined(__AVX2__)

double Norm_inf(const MAT<double>& A) {
    int n = (A.rows / 4) * 4;
    __m256d max = _mm256_setzero_pd();
    __m256d sum;

    for (int j = 0; j < n; j += 4) {
        sum = _mm256_setzero_pd();
        for (int i = 0; i < A.cols; ++i) {
            __m256d val = *(__m256d*)(A.data + i * A.rows + j);
            val = _mm256_andnot_pd(_mm256_set1_pd(-0.0), val);
            sum = _mm256_add_pd(sum, val);
        }
        max = _mm256_max_pd(max, sum);
    }

    if (A.rows - n > 0) {
        __m256d mask = _mm256_castsi256_pd(_mm256_setr_epi64x(
            A.rows - n > 0 ? -1LL : 0LL,
            A.rows - n > 1 ? -1LL : 0LL,
            A.rows - n > 2 ? -1LL : 0LL,
            A.rows - n > 3 ? -1LL : 0LL
        ));
        sum = _mm256_setzero_pd();
        for (int i = 0; i < A.cols; ++i) {
            __m256d val = *(__m256d*)(A.data + i * A.rows + n);
            val = _mm256_andnot_pd(_mm256_set1_pd(-0.0), val);
            val = _mm256_blendv_pd(_mm256_setzero_pd(), val, mask);
            sum = _mm256_add_pd(sum, val);
        }
        max = _mm256_max_pd(max, sum);
    }

    __m128d low = _mm256_castpd256_pd128(max);
    __m128d high = _mm256_extractf128_pd(max, 1);
    __m128d max128 = _mm_max_pd(low, high);
    max128 = _mm_max_pd(max128, _mm_permute_pd(max128, 0x1));
    return _mm_cvtsd_f64(max128);
}

float Norm_inf(const MAT<float>& A) {
    int n = (A.rows / 8) * 8;
    __m256 max = _mm256_setzero_ps();
    __m256 sum;

    for (int j = 0; j < n; j += 8) {
        sum = _mm256_setzero_ps();
        for (int i = 0; i < A.cols; ++i) {
            __m256 val = *(__m256*)(A.data + i * A.rows + j);
            val = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), val);
            sum = _mm256_add_ps(sum, val);
        }
        max = _mm256_max_ps(max, sum);
    }

    if (A.rows - n > 0) {
        __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(
            A.rows - n > 0 ? -1 : 0,
            A.rows - n > 1 ? -1 : 0,
            A.rows - n > 2 ? -1 : 0,
            A.rows - n > 3 ? -1 : 0,
            A.rows - n > 4 ? -1 : 0,
            A.rows - n > 5 ? -1 : 0,
            A.rows - n > 6 ? -1 : 0,
            A.rows - n > 7 ? -1 : 0
        ));
        sum = _mm256_setzero_ps();
        for (int i = 0; i < A.cols; ++i) {
            __m256 val = *(__m256*)(A.data + i * A.rows + n);
            val = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), val);
            val = _mm256_blendv_ps(_mm256_setzero_ps(), val, mask);
            sum = _mm256_add_ps(sum, val);
        }
        max = _mm256_max_ps(max, sum);
    }

    __m128 low = _mm256_castps256_ps128(max);
    __m128 high = _mm256_extractf128_ps(max, 1);
    __m128 max128 = _mm_max_ps(low, high);
    max128 = _mm_max_ps(max128, _mm_permute_ps(max128, _MM_SHUFFLE(2, 3, 0, 1)));
    max128 = _mm_max_ps(max128, _mm_permute_ps(max128, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max128);
}

int32_t Norm_inf(const MAT<int32_t>& A) {
    int n = (A.rows / 8) * 8;
    __m256i max = _mm256_setzero_si256();
    __m256i sum;

    for (int j = 0; j < n; j += 8) {
        sum = _mm256_setzero_si256();
        for (int i = 0; i < A.cols; ++i) {
            __m256i val = *(__m256i*)(A.data + i * A.rows + j);
            val = _mm256_abs_epi32(val);
            sum = _mm256_add_epi32(sum, val);
        }
        max = _mm256_max_epi32(max, sum);
    }

    if (A.rows - n > 0) {
        __m256i mask = _mm256_setr_epi32(
            A.rows - n > 0 ? -1 : 0,
            A.rows - n > 1 ? -1 : 0,
            A.rows - n > 2 ? -1 : 0,
            A.rows - n > 3 ? -1 : 0,
            A.rows - n > 4 ? -1 : 0,
            A.rows - n > 5 ? -1 : 0,
            A.rows - n > 6 ? -1 : 0,
            A.rows - n > 7 ? -1 : 0
        );
        sum = _mm256_setzero_si256();
        for (int i = 0; i < A.cols; ++i) {
            __m256i val = *(__m256i*)(A.data + i * A.rows + n);
            val = _mm256_abs_epi32(val);
            val = _mm256_blendv_epi8(_mm256_setzero_si256(), val, mask);
            sum = _mm256_add_epi32(sum, val);
        }
        max = _mm256_max_epi32(max, sum);
    }

    __m128i low = _mm256_castsi256_si128(max);
    __m128i high = _mm256_extracti128_si256(max, 1);
    __m128i max128 = _mm_max_epi32(low, high);
    max128 = _mm_max_epi32(max128, _mm_shuffle_epi32(max128, _MM_SHUFFLE(2, 3, 0, 1)));
    max128 = _mm_max_epi32(max128, _mm_shuffle_epi32(max128, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(max128);
}

#endif

#endif
