Skip to content
This repository  
Pull requests
Issues
Gist
 @romeric
 Watch 176
  Star 2,028
  Fork 1,044
numpy/numpy
 tree: eaaa9313b3  numpy/numpy/core/src/multiarray/einsum.c.src
@sebergseberg on 6 Jun BUG: einsums bool_sum_of_products_contig incorrect for small arrays
10 contributors @mwiebe @charris @juliantaylor @pv @seberg @rgommers @njsmith @insertinterestingnamehere @hpaulj @tovrstra
RawBlameHistory    3006 lines (2712 sloc)  91.428 kB
/*
 * This file contains the implementation of the 'einsum' function,
 * which provides an einstein-summation operation.
 *
 * Copyright (c) 2011 by Mark Wiebe (mwwiebe@gmail.com)
 * The Univerity of British Columbia
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>
#include <numpy/halffloat.h>
#include <npy_pycompat.h>

#include <ctype.h>

#include "convert.h"
#include "common.h"

#ifdef NPY_HAVE_SSE_INTRINSICS
#define EINSUM_USE_SSE1 1
#else
#define EINSUM_USE_SSE1 0
#endif

/*
 * TODO: Only some SSE2 for float64 is implemented.
 */
#ifdef NPY_HAVE_SSE2_INTRINSICS
#define EINSUM_USE_SSE2 1
#else
#define EINSUM_USE_SSE2 0
#endif

#if EINSUM_USE_SSE1
#include <xmmintrin.h>
#endif

#if EINSUM_USE_SSE2
#include <emmintrin.h>
#endif

#define EINSUM_IS_SSE_ALIGNED(x) ((((npy_intp)x)&0xf) == 0)

/********** PRINTF DEBUG TRACING **************/
#define NPY_EINSUM_DBG_TRACING 0

#if NPY_EINSUM_DBG_TRACING
#define NPY_EINSUM_DBG_PRINT(s) printf("%s", s);
#define NPY_EINSUM_DBG_PRINT1(s, p1) printf(s, p1);
#define NPY_EINSUM_DBG_PRINT2(s, p1, p2) printf(s, p1, p2);
#define NPY_EINSUM_DBG_PRINT3(s, p1, p2, p3) printf(s);
#else
#define NPY_EINSUM_DBG_PRINT(s)
#define NPY_EINSUM_DBG_PRINT1(s, p1)
#define NPY_EINSUM_DBG_PRINT2(s, p1, p2)
#define NPY_EINSUM_DBG_PRINT3(s, p1, p2, p3)
#endif
/**********************************************/

/**begin repeat
 * #name = byte, short, int, long, longlong,
 *         ubyte, ushort, uint, ulong, ulonglong,
 *         half, float, double, longdouble,
 *         cfloat, cdouble, clongdouble#
 * #type = npy_byte, npy_short, npy_int, npy_long, npy_longlong,
 *         npy_ubyte, npy_ushort, npy_uint, npy_ulong, npy_ulonglong,
 *         npy_half, npy_float, npy_double, npy_longdouble,
 *         npy_cfloat, npy_cdouble, npy_clongdouble#
 * #temptype = npy_byte, npy_short, npy_int, npy_long, npy_longlong,
 *             npy_ubyte, npy_ushort, npy_uint, npy_ulong, npy_ulonglong,
 *             npy_float, npy_float, npy_double, npy_longdouble,
 *             npy_float, npy_double, npy_longdouble#
 * #to = ,,,,,
 *       ,,,,,
 *       npy_float_to_half,,,,
 *       ,,#
 * #from = ,,,,,
 *         ,,,,,
 *         npy_half_to_float,,,,
 *         ,,#
 * #complex = 0*5,
 *            0*5,
 *            0*4,
 *            1*3#
 * #float32 = 0*5,
 *            0*5,
 *            0,1,0,0,
 *            0*3#
 * #float64 = 0*5,
 *            0*5,
 *            0,0,1,0,
 *            0*3#
 */

/**begin repeat1
 * #nop = 1, 2, 3, 1000#
 * #noplabel = one, two, three, any#
 */
static void
@name@_sum_of_products_@noplabel@(int nop, char **dataptr,
                                npy_intp *strides, npy_intp count)
{
#if (@nop@ == 1) || (@nop@ <= 3 && !@complex@)
    char *data0 = dataptr[0];
    npy_intp stride0 = strides[0];
#endif
#if (@nop@ == 2 || @nop@ == 3) && !@complex@
    char *data1 = dataptr[1];
    npy_intp stride1 = strides[1];
#endif
#if (@nop@ == 3) && !@complex@
    char *data2 = dataptr[2];
    npy_intp stride2 = strides[2];
#endif
#if (@nop@ == 1) || (@nop@ <= 3 && !@complex@)
    char *data_out = dataptr[@nop@];
    npy_intp stride_out = strides[@nop@];
#endif

    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_@noplabel@ (%d)\n", (int)count);

    while (count--) {
#if !@complex@
#  if @nop@ == 1
        *(@type@ *)data_out = @to@(@from@(*(@type@ *)data0) +
                                         @from@(*(@type@ *)data_out));
        data0 += stride0;
        data_out += stride_out;
#  elif @nop@ == 2
        *(@type@ *)data_out = @to@(@from@(*(@type@ *)data0) *
                                         @from@(*(@type@ *)data1) +
                                         @from@(*(@type@ *)data_out));
        data0 += stride0;
        data1 += stride1;
        data_out += stride_out;
#  elif @nop@ == 3
        *(@type@ *)data_out = @to@(@from@(*(@type@ *)data0) *
                                         @from@(*(@type@ *)data1) *
                                         @from@(*(@type@ *)data2) +
                                         @from@(*(@type@ *)data_out));
        data0 += stride0;
        data1 += stride1;
        data2 += stride2;
        data_out += stride_out;
#  else
        @temptype@ temp = @from@(*(@type@ *)dataptr[0]);
        int i;
        for (i = 1; i < nop; ++i) {
            temp *= @from@(*(@type@ *)dataptr[i]);
        }
        *(@type@ *)dataptr[nop] = @to@(temp +
                                           @from@(*(@type@ *)dataptr[i]));
        for (i = 0; i <= nop; ++i) {
            dataptr[i] += strides[i];
        }
#  endif
#else /* complex */
#  if @nop@ == 1
        ((@temptype@ *)data_out)[0] = ((@temptype@ *)data0)[0] +
                                         ((@temptype@ *)data_out)[0];
        ((@temptype@ *)data_out)[1] = ((@temptype@ *)data0)[1] +
                                         ((@temptype@ *)data_out)[1];
        data0 += stride0;
        data_out += stride_out;
#  else
#    if @nop@ <= 3
#define _SUMPROD_NOP @nop@
#    else
#define _SUMPROD_NOP nop
#    endif
        @temptype@ re, im, tmp;
        int i;
        re = ((@temptype@ *)dataptr[0])[0];
        im = ((@temptype@ *)dataptr[0])[1];
        for (i = 1; i < _SUMPROD_NOP; ++i) {
            tmp = re * ((@temptype@ *)dataptr[i])[0] -
                  im * ((@temptype@ *)dataptr[i])[1];
            im = re * ((@temptype@ *)dataptr[i])[1] +
                 im * ((@temptype@ *)dataptr[i])[0];
            re = tmp;
        }
        ((@temptype@ *)dataptr[_SUMPROD_NOP])[0] = re +
                                     ((@temptype@ *)dataptr[_SUMPROD_NOP])[0];
        ((@temptype@ *)dataptr[_SUMPROD_NOP])[1] = im +
                                     ((@temptype@ *)dataptr[_SUMPROD_NOP])[1];

        for (i = 0; i <= _SUMPROD_NOP; ++i) {
            dataptr[i] += strides[i];
        }
#undef _SUMPROD_NOP
#  endif
#endif
    }
}

#if @nop@ == 1

static void
@name@_sum_of_products_contig_one(int nop, char **dataptr,
                                npy_intp *NPY_UNUSED(strides), npy_intp count)
{
    @type@ *data0 = (@type@ *)dataptr[0];
    @type@ *data_out = (@type@ *)dataptr[1];

    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_contig_one (%d)\n",
                                                            (int)count);

/* This is placed before the main loop to make small counts faster */
finish_after_unrolled_loop:
    switch (count) {
/**begin repeat2
 * #i = 6, 5, 4, 3, 2, 1, 0#
 */
        case @i@+1:
#if !@complex@
            data_out[@i@] = @to@(@from@(data0[@i@]) +
                                 @from@(data_out[@i@]));
#else
            ((@temptype@ *)data_out + 2*@i@)[0] =
                                    ((@temptype@ *)data0 + 2*@i@)[0] +
                                    ((@temptype@ *)data_out + 2*@i@)[0];
            ((@temptype@ *)data_out + 2*@i@)[1] =
                                    ((@temptype@ *)data0 + 2*@i@)[1] +
                                    ((@temptype@ *)data_out + 2*@i@)[1];
#endif
/**end repeat2**/
        case 0:
            return;
    }

    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;

/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
#if !@complex@
        data_out[@i@] = @to@(@from@(data0[@i@]) +
                             @from@(data_out[@i@]));
#else /* complex */
        ((@temptype@ *)data_out + 2*@i@)[0] =
                                ((@temptype@ *)data0 + 2*@i@)[0] +
                                ((@temptype@ *)data_out + 2*@i@)[0];
        ((@temptype@ *)data_out + 2*@i@)[1] =
                                ((@temptype@ *)data0 + 2*@i@)[1] +
                                ((@temptype@ *)data_out + 2*@i@)[1];
#endif
/**end repeat2**/
        data0 += 8;
        data_out += 8;
    }

    /* Finish off the loop */
    goto finish_after_unrolled_loop;
}

#elif @nop@ == 2 && !@complex@

static void
@name@_sum_of_products_contig_two(int nop, char **dataptr,
                                npy_intp *NPY_UNUSED(strides), npy_intp count)
{
    @type@ *data0 = (@type@ *)dataptr[0];
    @type@ *data1 = (@type@ *)dataptr[1];
    @type@ *data_out = (@type@ *)dataptr[2];

#if EINSUM_USE_SSE1 && @float32@
    __m128 a, b;
#endif

    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_contig_two (%d)\n",
                                                            (int)count);

/* This is placed before the main loop to make small counts faster */
finish_after_unrolled_loop:
    switch (count) {
/**begin repeat2
 * #i = 6, 5, 4, 3, 2, 1, 0#
 */
        case @i@+1:
            data_out[@i@] = @to@(@from@(data0[@i@]) *
                                 @from@(data1[@i@]) +
                                 @from@(data_out[@i@]));
/**end repeat2**/
        case 0:
            return;
    }

#if EINSUM_USE_SSE1 && @float32@
    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data0) && EINSUM_IS_SSE_ALIGNED(data1) &&
        EINSUM_IS_SSE_ALIGNED(data_out)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

/**begin repeat2
 * #i = 0, 4#
 */
            a = _mm_mul_ps(_mm_load_ps(data0+@i@), _mm_load_ps(data1+@i@));
            b = _mm_add_ps(a, _mm_load_ps(data_out+@i@));
            _mm_store_ps(data_out+@i@, b);
/**end repeat2**/
            data0 += 8;
            data1 += 8;
            data_out += 8;
        }

        /* Finish off the loop */
        goto finish_after_unrolled_loop;
    }
#endif

    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;

#if EINSUM_USE_SSE1 && @float32@
/**begin repeat2
 * #i = 0, 4#
 */
        a = _mm_mul_ps(_mm_loadu_ps(data0+@i@), _mm_loadu_ps(data1+@i@));
        b = _mm_add_ps(a, _mm_loadu_ps(data_out+@i@));
        _mm_storeu_ps(data_out+@i@, b);
/**end repeat2**/
#else
/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        data_out[@i@] = @to@(@from@(data0[@i@]) *
                             @from@(data1[@i@]) +
                             @from@(data_out[@i@]));
/**end repeat2**/
#endif
        data0 += 8;
        data1 += 8;
        data_out += 8;
    }

    /* Finish off the loop */
    goto finish_after_unrolled_loop;
}

/* Some extra specializations for the two operand case */
static void
@name@_sum_of_products_stride0_contig_outcontig_two(int nop, char **dataptr,
                                npy_intp *NPY_UNUSED(strides), npy_intp count)
{
    @temptype@ value0 = @from@(*(@type@ *)dataptr[0]);
    @type@ *data1 = (@type@ *)dataptr[1];
    @type@ *data_out = (@type@ *)dataptr[2];

#if EINSUM_USE_SSE1 && @float32@
    __m128 a, b, value0_sse;
#elif EINSUM_USE_SSE2 && @float64@
    __m128d a, b, value0_sse;
#endif

    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_stride0_contig_outcontig_two (%d)\n",
                                                    (int)count);

/* This is placed before the main loop to make small counts faster */
finish_after_unrolled_loop:
    switch (count) {
/**begin repeat2
 * #i = 6, 5, 4, 3, 2, 1, 0#
 */
        case @i@+1:
            data_out[@i@] = @to@(value0 *
                                 @from@(data1[@i@]) +
                                 @from@(data_out[@i@]));
/**end repeat2**/
        case 0:
            return;
    }

#if EINSUM_USE_SSE1 && @float32@
    value0_sse = _mm_set_ps1(value0);

    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data1) && EINSUM_IS_SSE_ALIGNED(data_out)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

/**begin repeat2
 * #i = 0, 4#
 */
            a = _mm_mul_ps(value0_sse, _mm_load_ps(data1+@i@));
            b = _mm_add_ps(a, _mm_load_ps(data_out+@i@));
            _mm_store_ps(data_out+@i@, b);
/**end repeat2**/
            data1 += 8;
            data_out += 8;
        }

        /* Finish off the loop */
        if (count > 0) {
            goto finish_after_unrolled_loop;
        }
        else {
            return;
        }
    }
#elif EINSUM_USE_SSE2 && @float64@
    value0_sse = _mm_set1_pd(value0);

    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data1) && EINSUM_IS_SSE_ALIGNED(data_out)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

/**begin repeat2
 * #i = 0, 2, 4, 6#
 */
            a = _mm_mul_pd(value0_sse, _mm_load_pd(data1+@i@));
            b = _mm_add_pd(a, _mm_load_pd(data_out+@i@));
            _mm_store_pd(data_out+@i@, b);
/**end repeat2**/
            data1 += 8;
            data_out += 8;
        }

        /* Finish off the loop */
        if (count > 0) {
            goto finish_after_unrolled_loop;
        }
        else {
            return;
        }
    }
#endif

    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;

#if EINSUM_USE_SSE1 && @float32@
/**begin repeat2
 * #i = 0, 4#
 */
        a = _mm_mul_ps(value0_sse, _mm_loadu_ps(data1+@i@));
        b = _mm_add_ps(a, _mm_loadu_ps(data_out+@i@));
        _mm_storeu_ps(data_out+@i@, b);
/**end repeat2**/
#elif EINSUM_USE_SSE2 && @float64@
/**begin repeat2
 * #i = 0, 2, 4, 6#
 */
        a = _mm_mul_pd(value0_sse, _mm_loadu_pd(data1+@i@));
        b = _mm_add_pd(a, _mm_loadu_pd(data_out+@i@));
        _mm_storeu_pd(data_out+@i@, b);
/**end repeat2**/
#else
/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        data_out[@i@] = @to@(value0 *
                             @from@(data1[@i@]) +
                             @from@(data_out[@i@]));
/**end repeat2**/
#endif
        data1 += 8;
        data_out += 8;
    }

    /* Finish off the loop */
    if (count > 0) {
        goto finish_after_unrolled_loop;
    }
}

static void
@name@_sum_of_products_contig_stride0_outcontig_two(int nop, char **dataptr,
                                npy_intp *NPY_UNUSED(strides), npy_intp count)
{
    @type@ *data0 = (@type@ *)dataptr[0];
    @temptype@ value1 = @from@(*(@type@ *)dataptr[1]);
    @type@ *data_out = (@type@ *)dataptr[2];

#if EINSUM_USE_SSE1 && @float32@
    __m128 a, b, value1_sse;
#endif

    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_contig_stride0_outcontig_two (%d)\n",
                                                    (int)count);

/* This is placed before the main loop to make small counts faster */
finish_after_unrolled_loop:
    switch (count) {
/**begin repeat2
 * #i = 6, 5, 4, 3, 2, 1, 0#
 */
        case @i@+1:
            data_out[@i@] = @to@(@from@(data0[@i@])*
                                 value1  +
                                 @from@(data_out[@i@]));
/**end repeat2**/
        case 0:
            return;
    }

#if EINSUM_USE_SSE1 && @float32@
    value1_sse = _mm_set_ps1(value1);

    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data0) && EINSUM_IS_SSE_ALIGNED(data_out)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

/**begin repeat2
 * #i = 0, 4#
 */
            a = _mm_mul_ps(_mm_load_ps(data0+@i@), value1_sse);
            b = _mm_add_ps(a, _mm_load_ps(data_out+@i@));
            _mm_store_ps(data_out+@i@, b);
/**end repeat2**/
            data0 += 8;
            data_out += 8;
        }

        /* Finish off the loop */
        goto finish_after_unrolled_loop;
    }
#endif

    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;

#if EINSUM_USE_SSE1 && @float32@
/**begin repeat2
 * #i = 0, 4#
 */
        a = _mm_mul_ps(_mm_loadu_ps(data0+@i@), value1_sse);
        b = _mm_add_ps(a, _mm_loadu_ps(data_out+@i@));
        _mm_storeu_ps(data_out+@i@, b);
/**end repeat2**/
#else
/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        data_out[@i@] = @to@(@from@(data0[@i@])*
                             value1  +
                             @from@(data_out[@i@]));
/**end repeat2**/
#endif
        data0 += 8;
        data_out += 8;
    }

    /* Finish off the loop */
    goto finish_after_unrolled_loop;
}

static void
@name@_sum_of_products_contig_contig_outstride0_two(int nop, char **dataptr,
                                npy_intp *NPY_UNUSED(strides), npy_intp count)
{
    @type@ *data0 = (@type@ *)dataptr[0];
    @type@ *data1 = (@type@ *)dataptr[1];
    @temptype@ accum = 0;

#if EINSUM_USE_SSE1 && @float32@
    __m128 a, accum_sse = _mm_setzero_ps();
#elif EINSUM_USE_SSE2 && @float64@
    __m128d a, accum_sse = _mm_setzero_pd();
#endif

    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_contig_contig_outstride0_two (%d)\n",
                                                    (int)count);

/* This is placed before the main loop to make small counts faster */
finish_after_unrolled_loop:
    switch (count) {
/**begin repeat2
 * #i = 6, 5, 4, 3, 2, 1, 0#
 */
        case @i@+1:
            accum += @from@(data0[@i@]) * @from@(data1[@i@]);
/**end repeat2**/
        case 0:
            *(@type@ *)dataptr[2] += @to@(accum);
            return;
    }

#if EINSUM_USE_SSE1 && @float32@
    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data0) && EINSUM_IS_SSE_ALIGNED(data1)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

            _mm_prefetch(data0 + 512, _MM_HINT_T0);
            _mm_prefetch(data1 + 512, _MM_HINT_T0);

/**begin repeat2
 * #i = 0, 4#
 */
            /*
             * NOTE: This accumulation changes the order, so will likely
             *       produce slightly different results.
             */
            a = _mm_mul_ps(_mm_load_ps(data0+@i@), _mm_load_ps(data1+@i@));
            accum_sse = _mm_add_ps(accum_sse, a);
/**end repeat2**/
            data0 += 8;
            data1 += 8;
        }

        /* Add the four SSE values and put in accum */
        a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(2,3,0,1));
        accum_sse = _mm_add_ps(a, accum_sse);
        a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(1,0,3,2));
        accum_sse = _mm_add_ps(a, accum_sse);
        _mm_store_ss(&accum, accum_sse);

        /* Finish off the loop */
        goto finish_after_unrolled_loop;
    }
#elif EINSUM_USE_SSE2 && @float64@
    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data0) && EINSUM_IS_SSE_ALIGNED(data1)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

            _mm_prefetch(data0 + 512, _MM_HINT_T0);
            _mm_prefetch(data1 + 512, _MM_HINT_T0);

/**begin repeat2
 * #i = 0, 2, 4, 6#
 */
            /*
             * NOTE: This accumulation changes the order, so will likely
             *       produce slightly different results.
             */
            a = _mm_mul_pd(_mm_load_pd(data0+@i@), _mm_load_pd(data1+@i@));
            accum_sse = _mm_add_pd(accum_sse, a);
/**end repeat2**/
            data0 += 8;
            data1 += 8;
        }

        /* Add the two SSE2 values and put in accum */
        a = _mm_shuffle_pd(accum_sse, accum_sse, _MM_SHUFFLE2(0,1));
        accum_sse = _mm_add_pd(a, accum_sse);
        _mm_store_sd(&accum, accum_sse);

        /* Finish off the loop */
        goto finish_after_unrolled_loop;
    }
#endif

    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;

#if EINSUM_USE_SSE1 && @float32@
        _mm_prefetch(data0 + 512, _MM_HINT_T0);
        _mm_prefetch(data1 + 512, _MM_HINT_T0);

/**begin repeat2
 * #i = 0, 4#
 */
        /*
         * NOTE: This accumulation changes the order, so will likely
         *       produce slightly different results.
         */
        a = _mm_mul_ps(_mm_loadu_ps(data0+@i@), _mm_loadu_ps(data1+@i@));
        accum_sse = _mm_add_ps(accum_sse, a);
/**end repeat2**/
#elif EINSUM_USE_SSE2 && @float64@
        _mm_prefetch(data0 + 512, _MM_HINT_T0);
        _mm_prefetch(data1 + 512, _MM_HINT_T0);

/**begin repeat2
 * #i = 0, 2, 4, 6#
 */
        /*
         * NOTE: This accumulation changes the order, so will likely
         *       produce slightly different results.
         */
        a = _mm_mul_pd(_mm_loadu_pd(data0+@i@), _mm_loadu_pd(data1+@i@));
        accum_sse = _mm_add_pd(accum_sse, a);
/**end repeat2**/
#else
/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        accum += @from@(data0[@i@]) * @from@(data1[@i@]);
/**end repeat2**/
#endif
        data0 += 8;
        data1 += 8;
    }

#if EINSUM_USE_SSE1 && @float32@
    /* Add the four SSE values and put in accum */
    a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(2,3,0,1));
    accum_sse = _mm_add_ps(a, accum_sse);
    a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(1,0,3,2));
    accum_sse = _mm_add_ps(a, accum_sse);
    _mm_store_ss(&accum, accum_sse);
#elif EINSUM_USE_SSE2 && @float64@
    /* Add the two SSE2 values and put in accum */
    a = _mm_shuffle_pd(accum_sse, accum_sse, _MM_SHUFFLE2(0,1));
    accum_sse = _mm_add_pd(a, accum_sse);
    _mm_store_sd(&accum, accum_sse);
#endif

    /* Finish off the loop */
    goto finish_after_unrolled_loop;
}

static void
@name@_sum_of_products_stride0_contig_outstride0_two(int nop, char **dataptr,
                                npy_intp *NPY_UNUSED(strides), npy_intp count)
{
    @temptype@ value0 = @from@(*(@type@ *)dataptr[0]);
    @type@ *data1 = (@type@ *)dataptr[1];
    @temptype@ accum = 0;

#if EINSUM_USE_SSE1 && @float32@
    __m128 a, accum_sse = _mm_setzero_ps();
#endif

    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_stride0_contig_outstride0_two (%d)\n",
                                                    (int)count);

/* This is placed before the main loop to make small counts faster */
finish_after_unrolled_loop:
    switch (count) {
/**begin repeat2
 * #i = 6, 5, 4, 3, 2, 1, 0#
 */
        case @i@+1:
            accum += @from@(data1[@i@]);
/**end repeat2**/
        case 0:
            *(@type@ *)dataptr[2] += @to@(value0 * accum);
            return;
    }

#if EINSUM_USE_SSE1 && @float32@
    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data1)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

/**begin repeat2
 * #i = 0, 4#
 */
            /*
             * NOTE: This accumulation changes the order, so will likely
             *       produce slightly different results.
             */
            accum_sse = _mm_add_ps(accum_sse, _mm_load_ps(data1+@i@));
/**end repeat2**/
            data1 += 8;
        }

#if EINSUM_USE_SSE1 && @float32@
        /* Add the four SSE values and put in accum */
        a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(2,3,0,1));
        accum_sse = _mm_add_ps(a, accum_sse);
        a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(1,0,3,2));
        accum_sse = _mm_add_ps(a, accum_sse);
        _mm_store_ss(&accum, accum_sse);
#endif

        /* Finish off the loop */
        goto finish_after_unrolled_loop;
    }
#endif

    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;

#if EINSUM_USE_SSE1 && @float32@
/**begin repeat2
 * #i = 0, 4#
 */
        /*
         * NOTE: This accumulation changes the order, so will likely
         *       produce slightly different results.
         */
        accum_sse = _mm_add_ps(accum_sse, _mm_loadu_ps(data1+@i@));
/**end repeat2**/
#else
/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        accum += @from@(data1[@i@]);
/**end repeat2**/
#endif
        data1 += 8;
    }

#if EINSUM_USE_SSE1 && @float32@
    /* Add the four SSE values and put in accum */
    a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(2,3,0,1));
    accum_sse = _mm_add_ps(a, accum_sse);
    a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(1,0,3,2));
    accum_sse = _mm_add_ps(a, accum_sse);
    _mm_store_ss(&accum, accum_sse);
#endif

    /* Finish off the loop */
    goto finish_after_unrolled_loop;
}

static void
@name@_sum_of_products_contig_stride0_outstride0_two(int nop, char **dataptr,
                                npy_intp *NPY_UNUSED(strides), npy_intp count)
{
    @type@ *data0 = (@type@ *)dataptr[0];
    @temptype@ value1 = @from@(*(@type@ *)dataptr[1]);
    @temptype@ accum = 0;

#if EINSUM_USE_SSE1 && @float32@
    __m128 a, accum_sse = _mm_setzero_ps();
#endif

    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_contig_stride0_outstride0_two (%d)\n",
                                                    (int)count);

/* This is placed before the main loop to make small counts faster */
finish_after_unrolled_loop:
    switch (count) {
/**begin repeat2
 * #i = 6, 5, 4, 3, 2, 1, 0#
 */
        case @i@+1:
            accum += @from@(data0[@i@]);
/**end repeat2**/
        case 0:
            *(@type@ *)dataptr[2] += @to@(accum * value1);
            return;
    }

#if EINSUM_USE_SSE1 && @float32@
    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data0)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

/**begin repeat2
 * #i = 0, 4#
 */
            /*
             * NOTE: This accumulation changes the order, so will likely
             *       produce slightly different results.
             */
            accum_sse = _mm_add_ps(accum_sse, _mm_load_ps(data0+@i@));
/**end repeat2**/
            data0 += 8;
        }

#if EINSUM_USE_SSE1 && @float32@
        /* Add the four SSE values and put in accum */
        a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(2,3,0,1));
        accum_sse = _mm_add_ps(a, accum_sse);
        a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(1,0,3,2));
        accum_sse = _mm_add_ps(a, accum_sse);
        _mm_store_ss(&accum, accum_sse);
#endif

        /* Finish off the loop */
        goto finish_after_unrolled_loop;
    }
#endif

    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;

#if EINSUM_USE_SSE1 && @float32@
/**begin repeat2
 * #i = 0, 4#
 */
        /*
         * NOTE: This accumulation changes the order, so will likely
         *       produce slightly different results.
         */
        accum_sse = _mm_add_ps(accum_sse, _mm_loadu_ps(data0+@i@));
/**end repeat2**/
#else
/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        accum += @from@(data0[@i@]);
/**end repeat2**/
#endif
        data0 += 8;
    }

#if EINSUM_USE_SSE1 && @float32@
    /* Add the four SSE values and put in accum */
    a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(2,3,0,1));
    accum_sse = _mm_add_ps(a, accum_sse);
    a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(1,0,3,2));
    accum_sse = _mm_add_ps(a, accum_sse);
    _mm_store_ss(&accum, accum_sse);
#endif

    /* Finish off the loop */
    goto finish_after_unrolled_loop;
}

#elif @nop@ == 3 && !@complex@

static void
@name@_sum_of_products_contig_three(int nop, char **dataptr,
                                npy_intp *NPY_UNUSED(strides), npy_intp count)
{
    @type@ *data0 = (@type@ *)dataptr[0];
    @type@ *data1 = (@type@ *)dataptr[1];
    @type@ *data2 = (@type@ *)dataptr[2];
    @type@ *data_out = (@type@ *)dataptr[3];

    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;

/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        data_out[@i@] = @to@(@from@(data0[@i@]) *
                             @from@(data1[@i@]) *
                             @from@(data2[@i@]) +
                             @from@(data_out[@i@]));
/**end repeat2**/
        data0 += 8;
        data1 += 8;
        data2 += 8;
        data_out += 8;
    }

    /* Finish off the loop */

/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
    if (count-- == 0) {
        return;
    }
    data_out[@i@] = @to@(@from@(data0[@i@]) *
                         @from@(data1[@i@]) *
                         @from@(data2[@i@]) +
                         @from@(data_out[@i@]));
/**end repeat2**/
}

#else /* @nop@ > 3 || @complex */

static void
@name@_sum_of_products_contig_@noplabel@(int nop, char **dataptr,
                                npy_intp *NPY_UNUSED(strides), npy_intp count)
{
    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_contig_@noplabel@ (%d)\n",
                                                    (int)count);

    while (count--) {
#if !@complex@
        @temptype@ temp = @from@(*(@type@ *)dataptr[0]);
        int i;
        for (i = 1; i < nop; ++i) {
            temp *= @from@(*(@type@ *)dataptr[i]);
        }
        *(@type@ *)dataptr[nop] = @to@(temp +
                                           @from@(*(@type@ *)dataptr[i]));
        for (i = 0; i <= nop; ++i) {
            dataptr[i] += sizeof(@type@);
        }
#else /* complex */
#  if @nop@ <= 3
#    define _SUMPROD_NOP @nop@
#  else
#    define _SUMPROD_NOP nop
#  endif
        @temptype@ re, im, tmp;
        int i;
        re = ((@temptype@ *)dataptr[0])[0];
        im = ((@temptype@ *)dataptr[0])[1];
        for (i = 1; i < _SUMPROD_NOP; ++i) {
            tmp = re * ((@temptype@ *)dataptr[i])[0] -
                  im * ((@temptype@ *)dataptr[i])[1];
            im = re * ((@temptype@ *)dataptr[i])[1] +
                 im * ((@temptype@ *)dataptr[i])[0];
            re = tmp;
        }
        ((@temptype@ *)dataptr[_SUMPROD_NOP])[0] = re +
                                     ((@temptype@ *)dataptr[_SUMPROD_NOP])[0];
        ((@temptype@ *)dataptr[_SUMPROD_NOP])[1] = im +
                                     ((@temptype@ *)dataptr[_SUMPROD_NOP])[1];

        for (i = 0; i <= _SUMPROD_NOP; ++i) {
            dataptr[i] += sizeof(@type@);
        }
#  undef _SUMPROD_NOP
#endif
    }
}

#endif /* functions for various @nop@ */

#if @nop@ == 1

static void
@name@_sum_of_products_contig_outstride0_one(int nop, char **dataptr,
                                npy_intp *strides, npy_intp count)
{
#if @complex@
    @temptype@ accum_re = 0, accum_im = 0;
    @temptype@ *data0 = (@temptype@ *)dataptr[0];
#else
    @temptype@ accum = 0;
    @type@ *data0 = (@type@ *)dataptr[0];
#endif

#if EINSUM_USE_SSE1 && @float32@
    __m128 a, accum_sse = _mm_setzero_ps();
#elif EINSUM_USE_SSE2 && @float64@
    __m128d a, accum_sse = _mm_setzero_pd();
#endif


    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_contig_outstride0_one (%d)\n",
                                                    (int)count);

/* This is placed before the main loop to make small counts faster */
finish_after_unrolled_loop:
    switch (count) {
/**begin repeat2
 * #i = 6, 5, 4, 3, 2, 1, 0#
 */
        case @i@+1:
#if !@complex@
            accum += @from@(data0[@i@]);
#else /* complex */
            accum_re += data0[2*@i@+0];
            accum_im += data0[2*@i@+1];
#endif
/**end repeat2**/
        case 0:
#if @complex@
            ((@temptype@ *)dataptr[1])[0] += accum_re;
            ((@temptype@ *)dataptr[1])[1] += accum_im;
#else
            *((@type@ *)dataptr[1]) = @to@(accum +
                                    @from@(*((@type@ *)dataptr[1])));
#endif
            return;
    }

#if EINSUM_USE_SSE1 && @float32@
    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data0)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

            _mm_prefetch(data0 + 512, _MM_HINT_T0);

/**begin repeat2
 * #i = 0, 4#
 */
            /*
             * NOTE: This accumulation changes the order, so will likely
             *       produce slightly different results.
             */
            accum_sse = _mm_add_ps(accum_sse, _mm_load_ps(data0+@i@));
/**end repeat2**/
            data0 += 8;
        }

        /* Add the four SSE values and put in accum */
        a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(2,3,0,1));
        accum_sse = _mm_add_ps(a, accum_sse);
        a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(1,0,3,2));
        accum_sse = _mm_add_ps(a, accum_sse);
        _mm_store_ss(&accum, accum_sse);

        /* Finish off the loop */
        goto finish_after_unrolled_loop;
    }
#elif EINSUM_USE_SSE2 && @float64@
    /* Use aligned instructions if possible */
    if (EINSUM_IS_SSE_ALIGNED(data0)) {
        /* Unroll the loop by 8 */
        while (count >= 8) {
            count -= 8;

            _mm_prefetch(data0 + 512, _MM_HINT_T0);

/**begin repeat2
 * #i = 0, 2, 4, 6#
 */
            /*
             * NOTE: This accumulation changes the order, so will likely
             *       produce slightly different results.
             */
            accum_sse = _mm_add_pd(accum_sse, _mm_load_pd(data0+@i@));
/**end repeat2**/
            data0 += 8;
        }

        /* Add the two SSE2 values and put in accum */
        a = _mm_shuffle_pd(accum_sse, accum_sse, _MM_SHUFFLE2(0,1));
        accum_sse = _mm_add_pd(a, accum_sse);
        _mm_store_sd(&accum, accum_sse);

        /* Finish off the loop */
        goto finish_after_unrolled_loop;
    }
#endif

    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;

#if EINSUM_USE_SSE1 && @float32@
        _mm_prefetch(data0 + 512, _MM_HINT_T0);

/**begin repeat2
 * #i = 0, 4#
 */
        /*
         * NOTE: This accumulation changes the order, so will likely
         *       produce slightly different results.
         */
        accum_sse = _mm_add_ps(accum_sse, _mm_loadu_ps(data0+@i@));
/**end repeat2**/
#elif EINSUM_USE_SSE2 && @float64@
        _mm_prefetch(data0 + 512, _MM_HINT_T0);

/**begin repeat2
 * #i = 0, 2, 4, 6#
 */
        /*
         * NOTE: This accumulation changes the order, so will likely
         *       produce slightly different results.
         */
        accum_sse = _mm_add_pd(accum_sse, _mm_loadu_pd(data0+@i@));
/**end repeat2**/
#else
/**begin repeat2
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
#  if !@complex@
        accum += @from@(data0[@i@]);
#  else /* complex */
        accum_re += data0[2*@i@+0];
        accum_im += data0[2*@i@+1];
#  endif
/**end repeat2**/
#endif

#if !@complex@
        data0 += 8;
#else
        data0 += 8*2;
#endif
    }

#if EINSUM_USE_SSE1 && @float32@
    /* Add the four SSE values and put in accum */
    a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(2,3,0,1));
    accum_sse = _mm_add_ps(a, accum_sse);
    a = _mm_shuffle_ps(accum_sse, accum_sse, _MM_SHUFFLE(1,0,3,2));
    accum_sse = _mm_add_ps(a, accum_sse);
    _mm_store_ss(&accum, accum_sse);
#elif EINSUM_USE_SSE2 && @float64@
    /* Add the two SSE2 values and put in accum */
    a = _mm_shuffle_pd(accum_sse, accum_sse, _MM_SHUFFLE2(0,1));
    accum_sse = _mm_add_pd(a, accum_sse);
    _mm_store_sd(&accum, accum_sse);
#endif

    /* Finish off the loop */
    goto finish_after_unrolled_loop;
}

#endif /* @nop@ == 1 */

static void
@name@_sum_of_products_outstride0_@noplabel@(int nop, char **dataptr,
                                npy_intp *strides, npy_intp count)
{
#if @complex@
    @temptype@ accum_re = 0, accum_im = 0;
#else
    @temptype@ accum = 0;
#endif

#if (@nop@ == 1) || (@nop@ <= 3 && !@complex@)
    char *data0 = dataptr[0];
    npy_intp stride0 = strides[0];
#endif
#if (@nop@ == 2 || @nop@ == 3) && !@complex@
    char *data1 = dataptr[1];
    npy_intp stride1 = strides[1];
#endif
#if (@nop@ == 3) && !@complex@
    char *data2 = dataptr[2];
    npy_intp stride2 = strides[2];
#endif

    NPY_EINSUM_DBG_PRINT1("@name@_sum_of_products_outstride0_@noplabel@ (%d)\n",
                                                    (int)count);

    while (count--) {
#if !@complex@
#  if @nop@ == 1
        accum += @from@(*(@type@ *)data0);
        data0 += stride0;
#  elif @nop@ == 2
        accum += @from@(*(@type@ *)data0) *
                 @from@(*(@type@ *)data1);
        data0 += stride0;
        data1 += stride1;
#  elif @nop@ == 3
        accum += @from@(*(@type@ *)data0) *
                 @from@(*(@type@ *)data1) *
                 @from@(*(@type@ *)data2);
        data0 += stride0;
        data1 += stride1;
        data2 += stride2;
#  else
        @temptype@ temp = @from@(*(@type@ *)dataptr[0]);
        int i;
        for (i = 1; i < nop; ++i) {
            temp *= @from@(*(@type@ *)dataptr[i]);
        }
        accum += temp;
        for (i = 0; i < nop; ++i) {
            dataptr[i] += strides[i];
        }
#  endif
#else /* complex */
#  if @nop@ == 1
        accum_re += ((@temptype@ *)data0)[0];
        accum_im += ((@temptype@ *)data0)[1];
        data0 += stride0;
#  else
#    if @nop@ <= 3
#define _SUMPROD_NOP @nop@
#    else
#define _SUMPROD_NOP nop
#    endif
        @temptype@ re, im, tmp;
        int i;
        re = ((@temptype@ *)dataptr[0])[0];
        im = ((@temptype@ *)dataptr[0])[1];
        for (i = 1; i < _SUMPROD_NOP; ++i) {
            tmp = re * ((@temptype@ *)dataptr[i])[0] -
                  im * ((@temptype@ *)dataptr[i])[1];
            im = re * ((@temptype@ *)dataptr[i])[1] +
                 im * ((@temptype@ *)dataptr[i])[0];
            re = tmp;
        }
        accum_re += re;
        accum_im += im;
        for (i = 0; i < _SUMPROD_NOP; ++i) {
            dataptr[i] += strides[i];
        }
#undef _SUMPROD_NOP
#  endif
#endif
    }

#if @complex@
#  if @nop@ <= 3
    ((@temptype@ *)dataptr[@nop@])[0] += accum_re;
    ((@temptype@ *)dataptr[@nop@])[1] += accum_im;
#  else
    ((@temptype@ *)dataptr[nop])[0] += accum_re;
    ((@temptype@ *)dataptr[nop])[1] += accum_im;
#  endif
#else
#  if @nop@ <= 3
    *((@type@ *)dataptr[@nop@]) = @to@(accum +
                                    @from@(*((@type@ *)dataptr[@nop@])));
#  else
    *((@type@ *)dataptr[nop]) = @to@(accum +
                                    @from@(*((@type@ *)dataptr[nop])));
#  endif
#endif

}

/**end repeat1**/

/**end repeat**/


/* Do OR of ANDs for the boolean type */

/**begin repeat
 * #nop = 1, 2, 3, 1000#
 * #noplabel = one, two, three, any#
 */

static void
bool_sum_of_products_@noplabel@(int nop, char **dataptr,
                                npy_intp *strides, npy_intp count)
{
#if (@nop@ <= 3)
    char *data0 = dataptr[0];
    npy_intp stride0 = strides[0];
#endif
#if (@nop@ == 2 || @nop@ == 3)
    char *data1 = dataptr[1];
    npy_intp stride1 = strides[1];
#endif
#if (@nop@ == 3)
    char *data2 = dataptr[2];
    npy_intp stride2 = strides[2];
#endif
#if (@nop@ <= 3)
    char *data_out = dataptr[@nop@];
    npy_intp stride_out = strides[@nop@];
#endif

    while (count--) {
#if @nop@ == 1
        *(npy_bool *)data_out = *(npy_bool *)data0 ||
                                  *(npy_bool *)data_out;
        data0 += stride0;
        data_out += stride_out;
#elif @nop@ == 2
        *(npy_bool *)data_out = (*(npy_bool *)data0 &&
                                   *(npy_bool *)data1) ||
                                   *(npy_bool *)data_out;
        data0 += stride0;
        data1 += stride1;
        data_out += stride_out;
#elif @nop@ == 3
        *(npy_bool *)data_out = (*(npy_bool *)data0 &&
                                   *(npy_bool *)data1 &&
                                   *(npy_bool *)data2) ||
                                   *(npy_bool *)data_out;
        data0 += stride0;
        data1 += stride1;
        data2 += stride2;
        data_out += stride_out;
#else
        npy_bool temp = *(npy_bool *)dataptr[0];
        int i;
        for (i = 1; i < nop; ++i) {
            temp = temp && *(npy_bool *)dataptr[i];
        }
        *(npy_bool *)dataptr[nop] = temp || *(npy_bool *)dataptr[i];
        for (i = 0; i <= nop; ++i) {
            dataptr[i] += strides[i];
        }
#endif
    }
}

static void
bool_sum_of_products_contig_@noplabel@(int nop, char **dataptr,
                                npy_intp *strides, npy_intp count)
{
#if (@nop@ <= 3)
    char *data0 = dataptr[0];
#endif
#if (@nop@ == 2 || @nop@ == 3)
    char *data1 = dataptr[1];
#endif
#if (@nop@ == 3)
    char *data2 = dataptr[2];
#endif
#if (@nop@ <= 3)
    char *data_out = dataptr[@nop@];
#endif

#if (@nop@ <= 3)
/* This is placed before the main loop to make small counts faster */
finish_after_unrolled_loop:
    switch (count) {
/**begin repeat1
 * #i = 6, 5, 4, 3, 2, 1, 0#
 */
        case @i@+1:
#  if @nop@ == 1
            ((npy_bool *)data_out)[@i@] = ((npy_bool *)data0)[@i@] ||
                                            ((npy_bool *)data_out)[@i@];
#  elif @nop@ == 2
            ((npy_bool *)data_out)[@i@] =
                            (((npy_bool *)data0)[@i@] &&
                             ((npy_bool *)data1)[@i@]) ||
                                ((npy_bool *)data_out)[@i@];
#  elif @nop@ == 3
            ((npy_bool *)data_out)[@i@] =
                           (((npy_bool *)data0)[@i@] &&
                            ((npy_bool *)data1)[@i@] &&
                            ((npy_bool *)data2)[@i@]) ||
                                ((npy_bool *)data_out)[@i@];
#  endif
/**end repeat1**/
        case 0:
            return;
    }
#endif

/* Unroll the loop by 8 for fixed-size nop */
#if (@nop@ <= 3)
    while (count >= 8) {
        count -= 8;
#else
    while (count--) {
#endif

#  if @nop@ == 1
/**begin repeat1
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        *((npy_bool *)data_out + @i@) = (*((npy_bool *)data0 + @i@)) ||
                                        (*((npy_bool *)data_out + @i@));
/**end repeat1**/
        data0 += 8*sizeof(npy_bool);
        data_out += 8*sizeof(npy_bool);
#  elif @nop@ == 2
/**begin repeat1
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        *((npy_bool *)data_out + @i@) =
                        ((*((npy_bool *)data0 + @i@)) &&
                         (*((npy_bool *)data1 + @i@))) ||
                            (*((npy_bool *)data_out + @i@));
/**end repeat1**/
        data0 += 8*sizeof(npy_bool);
        data1 += 8*sizeof(npy_bool);
        data_out += 8*sizeof(npy_bool);
#  elif @nop@ == 3
/**begin repeat1
 * #i = 0, 1, 2, 3, 4, 5, 6, 7#
 */
        *((npy_bool *)data_out + @i@) =
                       ((*((npy_bool *)data0 + @i@)) &&
                        (*((npy_bool *)data1 + @i@)) &&
                        (*((npy_bool *)data2 + @i@))) ||
                            (*((npy_bool *)data_out + @i@));
/**end repeat1**/
        data0 += 8*sizeof(npy_bool);
        data1 += 8*sizeof(npy_bool);
        data2 += 8*sizeof(npy_bool);
        data_out += 8*sizeof(npy_bool);
#  else
        npy_bool temp = *(npy_bool *)dataptr[0];
        int i;
        for (i = 1; i < nop; ++i) {
            temp = temp && *(npy_bool *)dataptr[i];
        }
        *(npy_bool *)dataptr[nop] = temp || *(npy_bool *)dataptr[i];
        for (i = 0; i <= nop; ++i) {
            dataptr[i] += sizeof(npy_bool);
        }
#  endif
    }

    /* If the loop was unrolled, we need to finish it off */
#if (@nop@ <= 3)
    goto finish_after_unrolled_loop;
#endif
}

static void
bool_sum_of_products_outstride0_@noplabel@(int nop, char **dataptr,
                                npy_intp *strides, npy_intp count)
{
    npy_bool accum = 0;

#if (@nop@ <= 3)
    char *data0 = dataptr[0];
    npy_intp stride0 = strides[0];
#endif
#if (@nop@ == 2 || @nop@ == 3)
    char *data1 = dataptr[1];
    npy_intp stride1 = strides[1];
#endif
#if (@nop@ == 3)
    char *data2 = dataptr[2];
    npy_intp stride2 = strides[2];
#endif

    while (count--) {
#if @nop@ == 1
        accum = *(npy_bool *)data0 || accum;
        data0 += stride0;
#elif @nop@ == 2
        accum = (*(npy_bool *)data0 && *(npy_bool *)data1) || accum;
        data0 += stride0;
        data1 += stride1;
#elif @nop@ == 3
        accum = (*(npy_bool *)data0 &&
                 *(npy_bool *)data1 &&
                 *(npy_bool *)data2) || accum;
        data0 += stride0;
        data1 += stride1;
        data2 += stride2;
#else
        npy_bool temp = *(npy_bool *)dataptr[0];
        int i;
        for (i = 1; i < nop; ++i) {
            temp = temp && *(npy_bool *)dataptr[i];
        }
        accum = temp || accum;
        for (i = 0; i <= nop; ++i) {
            dataptr[i] += strides[i];
        }
#endif
    }

#  if @nop@ <= 3
    *((npy_bool *)dataptr[@nop@]) = accum || *((npy_bool *)dataptr[@nop@]);
#  else
    *((npy_bool *)dataptr[nop]) = accum || *((npy_bool *)dataptr[nop]);
#  endif
}

/**end repeat**/

typedef void (*sum_of_products_fn)(int, char **, npy_intp *, npy_intp);

/* These tables need to match up with the type enum */
static sum_of_products_fn
_contig_outstride0_unary_specialization_table[NPY_NTYPES] = {
/**begin repeat
 * #name = bool,
 *         byte, ubyte,
 *         short, ushort,
 *         int, uint,
 *         long, ulong,
 *         longlong, ulonglong,
 *         float, double, longdouble,
 *         cfloat, cdouble, clongdouble,
 *         object, string, unicode, void,
 *         datetime, timedelta, half#
 * #use = 0,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1, 1,
 *        1, 1, 1,
 *        0, 0, 0, 0,
 *        0, 0, 1#
 */
#if @use@
    &@name@_sum_of_products_contig_outstride0_one,
#else
    NULL,
#endif
/**end repeat**/
}; /* End of _contig_outstride0_unary_specialization_table */

static sum_of_products_fn _binary_specialization_table[NPY_NTYPES][5] = {
/**begin repeat
 * #name = bool,
 *         byte, ubyte,
 *         short, ushort,
 *         int, uint,
 *         long, ulong,
 *         longlong, ulonglong,
 *         float, double, longdouble,
 *         cfloat, cdouble, clongdouble,
 *         object, string, unicode, void,
 *         datetime, timedelta, half#
 * #use = 0,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1, 1,
 *        0, 0, 0,
 *        0, 0, 0, 0,
 *        0, 0, 1#
 */
#if @use@
{
    &@name@_sum_of_products_stride0_contig_outstride0_two,
    &@name@_sum_of_products_stride0_contig_outcontig_two,
    &@name@_sum_of_products_contig_stride0_outstride0_two,
    &@name@_sum_of_products_contig_stride0_outcontig_two,
    &@name@_sum_of_products_contig_contig_outstride0_two,
},
#else
    {NULL, NULL, NULL, NULL, NULL},
#endif
/**end repeat**/
}; /* End of _binary_specialization_table */

static sum_of_products_fn _outstride0_specialized_table[NPY_NTYPES][4] = {
/**begin repeat
 * #name = bool,
 *         byte, ubyte,
 *         short, ushort,
 *         int, uint,
 *         long, ulong,
 *         longlong, ulonglong,
 *         float, double, longdouble,
 *         cfloat, cdouble, clongdouble,
 *         object, string, unicode, void,
 *         datetime, timedelta, half#
 * #use = 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1, 1,
 *        1, 1, 1,
 *        0, 0, 0, 0,
 *        0, 0, 1#
 */
#if @use@
{
    &@name@_sum_of_products_outstride0_any,
    &@name@_sum_of_products_outstride0_one,
    &@name@_sum_of_products_outstride0_two,
    &@name@_sum_of_products_outstride0_three
},
#else
    {NULL, NULL, NULL, NULL},
#endif
/**end repeat**/
}; /* End of _outstride0_specialized_table */

static sum_of_products_fn _allcontig_specialized_table[NPY_NTYPES][4] = {
/**begin repeat
 * #name = bool,
 *         byte, ubyte,
 *         short, ushort,
 *         int, uint,
 *         long, ulong,
 *         longlong, ulonglong,
 *         float, double, longdouble,
 *         cfloat, cdouble, clongdouble,
 *         object, string, unicode, void,
 *         datetime, timedelta, half#
 * #use = 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1, 1,
 *        1, 1, 1,
 *        0, 0, 0, 0,
 *        0, 0, 1#
 */
#if @use@
{
    &@name@_sum_of_products_contig_any,
    &@name@_sum_of_products_contig_one,
    &@name@_sum_of_products_contig_two,
    &@name@_sum_of_products_contig_three
},
#else
    {NULL, NULL, NULL, NULL},
#endif
/**end repeat**/
}; /* End of _allcontig_specialized_table */

static sum_of_products_fn _unspecialized_table[NPY_NTYPES][4] = {
/**begin repeat
 * #name = bool,
 *         byte, ubyte,
 *         short, ushort,
 *         int, uint,
 *         long, ulong,
 *         longlong, ulonglong,
 *         float, double, longdouble,
 *         cfloat, cdouble, clongdouble,
 *         object, string, unicode, void,
 *         datetime, timedelta, half#
 * #use = 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1,
 *        1, 1, 1,
 *        1, 1, 1,
 *        0, 0, 0, 0,
 *        0, 0, 1#
 */
#if @use@
{
    &@name@_sum_of_products_any,
    &@name@_sum_of_products_one,
    &@name@_sum_of_products_two,
    &@name@_sum_of_products_three
},
#else
    {NULL, NULL, NULL, NULL},
#endif
/**end repeat**/
}; /* End of _unnspecialized_table */

static sum_of_products_fn
get_sum_of_products_function(int nop, int type_num,
                             npy_intp itemsize, npy_intp *fixed_strides)
{
    int iop;

    if (type_num >= NPY_NTYPES) {
        return NULL;
    }

    /* contiguous reduction */
    if (nop == 1 && fixed_strides[0] == itemsize && fixed_strides[1] == 0) {
        sum_of_products_fn ret =
            _contig_outstride0_unary_specialization_table[type_num];
        if (ret != NULL) {
            return ret;
        }
    }

    /* nop of 2 has more specializations */
    if (nop == 2) {
        /* Encode the zero/contiguous strides */
        int code;
        code = (fixed_strides[0] == 0) ? 0 :
                    (fixed_strides[0] == itemsize) ? 2*2*1 : 8;
        code += (fixed_strides[1] == 0) ? 0 :
                    (fixed_strides[1] == itemsize) ? 2*1 : 8;
        code += (fixed_strides[2] == 0) ? 0 :
                    (fixed_strides[2] == itemsize) ? 1 : 8;
        if (code >= 2 && code < 7) {
            sum_of_products_fn ret =
                        _binary_specialization_table[type_num][code-2];
            if (ret != NULL) {
                return ret;
            }
        }
    }

    /* Inner loop with an output stride of 0 */
    if (fixed_strides[nop] == 0) {
        return _outstride0_specialized_table[type_num][nop <= 3 ? nop : 0];
    }

    /* Check for all contiguous */
    for (iop = 0; iop < nop + 1; ++iop) {
        if (fixed_strides[iop] != itemsize) {
            break;
        }
    }

    /* Contiguous loop */
    if (iop == nop + 1) {
        return _allcontig_specialized_table[type_num][nop <= 3 ? nop : 0];
    }

    /* None of the above specializations caught it, general loops */
    return _unspecialized_table[type_num][nop <= 3 ? nop : 0];
}

/*
 * Parses the subscripts for one operand into an output
 * of 'ndim' labels
 */
static int
parse_operand_subscripts(char *subscripts, int length,
                        int ndim,
                        int iop, char *out_labels,
                        char *out_label_counts,
                        int *out_min_label,
                        int *out_max_label,
                        int *out_num_labels)
{
    int i, idim, ndim_left, label;
    int ellipsis = 0;

    /* Process the labels from the end until the ellipsis */
    idim = ndim-1;
    for (i = length-1; i >= 0; --i) {
        label = subscripts[i];
        /* A label for an axis */
        if (label > 0 && isalpha(label)) {
            if (idim >= 0) {
                out_labels[idim--] = label;
                /* Calculate the min and max labels */
                if (label < *out_min_label) {
                    *out_min_label = label;
                }
                if (label > *out_max_label) {
                    *out_max_label = label;
                }
                /* If it's the first time we see this label, count it */
                if (out_label_counts[label] == 0) {
                    (*out_num_labels)++;
                }
                out_label_counts[label]++;
            }
            else {
                PyErr_Format(PyExc_ValueError,
                            "einstein sum subscripts string contains "
                            "too many subscripts for operand %d", iop);
                return 0;
            }
        }
        /* The end of the ellipsis */
        else if (label == '.') {
            /* A valid ellipsis */
            if (i >= 2 && subscripts[i-1] == '.' && subscripts[i-2] == '.') {
                ellipsis = 1;
                length = i-2;
                break;
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                            "einstein sum subscripts string contains a "
                            "'.' that is not part of an ellipsis ('...')");
                return 0;

            }
        }
        else if (label != ' ') {
            PyErr_Format(PyExc_ValueError,
                        "invalid subscript '%c' in einstein sum "
                        "subscripts string, subscripts must "
                        "be letters", (char)label);
            return 0;
        }
    }

    if (!ellipsis && idim != -1) {
        PyErr_Format(PyExc_ValueError,
                    "operand has more dimensions than subscripts "
                    "given in einstein sum, but no '...' ellipsis "
                    "provided to broadcast the extra dimensions.");
        return 0;
    }

    /* Reduce ndim to just the dimensions left to fill at the beginning */
    ndim_left = idim+1;
    idim = 0;

    /*
     * If we stopped because of an ellipsis, start again from the beginning.
     * The length was truncated to end at the ellipsis in this case.
     */
    if (i > 0) {
        for (i = 0; i < length; ++i) {
            label = subscripts[i];
            /* A label for an axis */
            if (label > 0 && isalnum(label)) {
                if (idim < ndim_left) {
                    out_labels[idim++] = label;
                    /* Calculate the min and max labels */
                    if (label < *out_min_label) {
                        *out_min_label = label;
                    }
                    if (label > *out_max_label) {
                        *out_max_label = label;
                    }
                    /* If it's the first time we see this label, count it */
                    if (out_label_counts[label] == 0) {
                        (*out_num_labels)++;
                    }
                    out_label_counts[label]++;
                }
                else {
                    PyErr_Format(PyExc_ValueError,
                                "einstein sum subscripts string contains "
                                "too many subscripts for operand %d", iop);
                    return 0;
                }
            }
            else if (label != ' ') {
                PyErr_Format(PyExc_ValueError,
                            "invalid subscript '%c' in einstein sum "
                            "subscripts string, subscripts must "
                            "be letters", (char)label);
                return 0;
            }
        }
    }

    /* Set the remaining labels to 0 */
    while (idim < ndim_left) {
        out_labels[idim++] = 0;
    }

    /*
     * Find any labels duplicated for this operand, and turn them
     * into negative offets to the axis to merge with.
     *
     * In C, the char type may be signed or unsigned, but with
     * twos complement arithmetic the char is ok either way here, and
     * later where it matters the char is cast to a signed char.
     */
    for (idim = 0; idim  < ndim-1; ++idim) {
        char *next;
        /* If this is a proper label, find any duplicates of it */
        label = out_labels[idim];
        if (label > 0) {
            /* Search for the next matching label */
            next = (char *)memchr(out_labels+idim+1, label,
                                    ndim-idim-1);
            while (next != NULL) {
                /* The offset from next to out_labels[idim] (negative) */
                *next = (char)((out_labels+idim)-next);
                /* Search for the next matching label */
                next = (char *)memchr(next+1, label,
                                        out_labels+ndim-1-next);
            }
        }
    }

    return 1;
}

/*
 * Parses the subscripts for the output operand into an output
 * that requires 'ndim_broadcast' unlabeled dimensions, returning
 * the number of output dimensions.  Returns -1 if there is an error.
 */
static int
parse_output_subscripts(char *subscripts, int length,
                        int ndim_broadcast,
                        const char *label_counts,
                        char *out_labels)
{
    int i, nlabels, label, idim, ndim, ndim_left;
    int ellipsis = 0;

    /* Count the labels, making sure they're all unique and valid */
    nlabels = 0;
    for (i = 0; i < length; ++i) {
        label = subscripts[i];
        if (label > 0 && isalpha(label)) {
            /* Check if it occurs again */
            if (memchr(subscripts+i+1, label, length-i-1) == NULL) {
                /* Check that it was used in the inputs */
                if (label_counts[label] == 0) {
                    PyErr_Format(PyExc_ValueError,
                            "einstein sum subscripts string included "
                            "output subscript '%c' which never appeared "
                            "in an input", (char)label);
                    return -1;
                }

                nlabels++;
            }
            else {
                PyErr_Format(PyExc_ValueError,
                        "einstein sum subscripts string includes "
                        "output subscript '%c' multiple times",
                        (char)label);
                return -1;
            }
        }
        else if (label != '.' && label != ' ') {
            PyErr_Format(PyExc_ValueError,
                        "invalid subscript '%c' in einstein sum "
                        "subscripts string, subscripts must "
                        "be letters", (char)label);
            return -1;
        }
    }

    /* The number of output dimensions */
    ndim = ndim_broadcast + nlabels;

    /* Process the labels from the end until the ellipsis */
    idim = ndim-1;
    for (i = length-1; i >= 0; --i) {
        label = subscripts[i];
        /* A label for an axis */
        if (label != '.' && label != ' ') {
            if (idim >= 0) {
                out_labels[idim--] = label;
            }
            else {
                PyErr_Format(PyExc_ValueError,
                            "einstein sum subscripts string contains "
                            "too many output subscripts");
                return -1;
            }
        }
        /* The end of the ellipsis */
        else if (label == '.') {
            /* A valid ellipsis */
            if (i >= 2 && subscripts[i-1] == '.' && subscripts[i-2] == '.') {
                ellipsis = 1;
                length = i-2;
                break;
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                            "einstein sum subscripts string contains a "
                            "'.' that is not part of an ellipsis ('...')");
                return -1;

            }
        }
    }

    if (!ellipsis && idim != -1) {
        PyErr_SetString(PyExc_ValueError,
                    "output has more dimensions than subscripts "
                    "given in einstein sum, but no '...' ellipsis "
                    "provided to broadcast the extra dimensions.");
        return 0;
    }

    /* Reduce ndim to just the dimensions left to fill at the beginning */
    ndim_left = idim+1;
    idim = 0;

    /*
     * If we stopped because of an ellipsis, start again from the beginning.
     * The length was truncated to end at the ellipsis in this case.
     */
    if (i > 0) {
        for (i = 0; i < length; ++i) {
            label = subscripts[i];
            /* A label for an axis */
            if (label != '.' && label != ' ') {
                if (idim < ndim_left) {
                    out_labels[idim++] = label;
                }
                else {
                    PyErr_Format(PyExc_ValueError,
                                "einstein sum subscripts string contains "
                                "too many subscripts for the output");
                    return -1;
                }
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                            "einstein sum subscripts string contains a "
                            "'.' that is not part of an ellipsis ('...')");
                return -1;
            }
        }
    }

    /* Set the remaining output labels to 0 */
    while (idim < ndim_left) {
        out_labels[idim++] = 0;
    }

    return ndim;
}


/*
 * When there's just one operand and no reduction, we
 * can return a view into op.  This calculates the view
 * if possible.
 */
static int
get_single_op_view(PyArrayObject *op, int  iop, char *labels,
                   int ndim_output, char *output_labels,
                   PyArrayObject **ret)
{
    npy_intp new_strides[NPY_MAXDIMS];
    npy_intp new_dims[NPY_MAXDIMS];
    char *out_label;
    int label, i, idim, ndim, ibroadcast = 0;

    ndim = PyArray_NDIM(op);

    /* Initialize the dimensions and strides to zero */
    for (idim = 0; idim < ndim_output; ++idim) {
        new_dims[idim] = 0;
        new_strides[idim] = 0;
    }

    /* Match the labels in the operand with the output labels */
    for (idim = 0; idim < ndim; ++idim) {
        /*
         * The char type may be either signed or unsigned, we
         * need it to be signed here.
         */
        label = (signed char)labels[idim];
        /* If this label says to merge axes, get the actual label */
        if (label < 0) {
            label = labels[idim+label];
        }
        /* If the label is 0, it's an unlabeled broadcast dimension */
        if (label == 0) {
            /* The next output label that's a broadcast dimension */
            for (; ibroadcast < ndim_output; ++ibroadcast) {
                if (output_labels[ibroadcast] == 0) {
                    break;
                }
            }
            if (ibroadcast == ndim_output) {
                PyErr_SetString(PyExc_ValueError,
                        "output had too few broadcast dimensions");
                return 0;
            }
            new_dims[ibroadcast] = PyArray_DIM(op, idim);
            new_strides[ibroadcast] = PyArray_STRIDE(op, idim);
            ++ibroadcast;
        }
        else {
            /* Find the position for this dimension in the output */
            out_label = (char *)memchr(output_labels, label,
                                                    ndim_output);
            /* If it's not found, reduction -> can't return a view */
            if (out_label == NULL) {
                break;
            }
            /* Update the dimensions and strides of the output */
            i = out_label - output_labels;
            if (new_dims[i] != 0 &&
                    new_dims[i] != PyArray_DIM(op, idim)) {
                PyErr_Format(PyExc_ValueError,
                        "dimensions in operand %d for collapsing "
                        "index '%c' don't match (%d != %d)",
                        iop, label, (int)new_dims[i],
                        (int)PyArray_DIM(op, idim));
                return 0;
            }
            new_dims[i] = PyArray_DIM(op, idim);
            new_strides[i] += PyArray_STRIDE(op, idim);
        }
    }
    /* If we processed all the input axes, return a view */
    if (idim == ndim) {
        Py_INCREF(PyArray_DESCR(op));
        *ret = (PyArrayObject *)PyArray_NewFromDescr(
                                Py_TYPE(op),
                                PyArray_DESCR(op),
                                ndim_output, new_dims, new_strides,
                                PyArray_DATA(op),
                                PyArray_ISWRITEABLE(op) ? NPY_ARRAY_WRITEABLE : 0,
                                (PyObject *)op);

        if (*ret == NULL) {
            return 0;
        }
        if (!PyArray_Check(*ret)) {
            Py_DECREF(*ret);
            *ret = NULL;
            PyErr_SetString(PyExc_RuntimeError,
                        "NewFromDescr failed to return an array");
            return 0;
        }
        PyArray_UpdateFlags(*ret,
                    NPY_ARRAY_C_CONTIGUOUS|
                    NPY_ARRAY_ALIGNED|
                    NPY_ARRAY_F_CONTIGUOUS);
        Py_INCREF(op);
        if (PyArray_SetBaseObject(*ret, (PyObject *)op) < 0) {
            Py_DECREF(*ret);
            *ret = NULL;
            return 0;
        }
        return 1;
    }

    /* Return success, but that we couldn't make a view */
    *ret = NULL;
    return 1;
}

static PyArrayObject *
get_combined_dims_view(PyArrayObject *op, int iop, char *labels)
{
    npy_intp new_strides[NPY_MAXDIMS];
    npy_intp new_dims[NPY_MAXDIMS];
    int i, idim, ndim, icombine, combineoffset, label;
    int icombinemap[NPY_MAXDIMS];

    PyArrayObject *ret = NULL;

    ndim = PyArray_NDIM(op);

    /* Initialize the dimensions and strides to zero */
    for (idim = 0; idim < ndim; ++idim) {
        new_dims[idim] = 0;
        new_strides[idim] = 0;
    }

    /* Copy the dimensions and strides, except when collapsing */
    icombine = 0;
    for (idim = 0; idim < ndim; ++idim) {
        /*
         * The char type may be either signed or unsigned, we
         * need it to be signed here.
         */
        label = (signed char)labels[idim];
        /* If this label says to merge axes, get the actual label */
        if (label < 0) {
            combineoffset = label;
            label = labels[idim+label];
        }
        else {
            combineoffset = 0;
            if (icombine != idim) {
                labels[icombine] = labels[idim];
            }
            icombinemap[idim] = icombine;
        }
        /* If the label is 0, it's an unlabeled broadcast dimension */
        if (label == 0) {
            new_dims[icombine] = PyArray_DIM(op, idim);
            new_strides[icombine] = PyArray_STRIDE(op, idim);
        }
        else {
            /* Update the combined axis dimensions and strides */
            i = idim + combineoffset;
            if (combineoffset < 0 && new_dims[i] != 0 &&
                        new_dims[i] != PyArray_DIM(op, idim)) {
                PyErr_Format(PyExc_ValueError,
                        "dimensions in operand %d for collapsing "
                        "index '%c' don't match (%d != %d)",
                        iop, label, (int)new_dims[i],
                        (int)PyArray_DIM(op, idim));
                return NULL;
            }
            i = icombinemap[i];
            new_dims[i] = PyArray_DIM(op, idim);
            new_strides[i] += PyArray_STRIDE(op, idim);
        }

        /* If the label didn't say to combine axes, increment dest i */
        if (combineoffset == 0) {
            icombine++;
        }
    }

    /* The compressed number of dimensions */
    ndim = icombine;

    Py_INCREF(PyArray_DESCR(op));
    ret = (PyArrayObject *)PyArray_NewFromDescr(
                            Py_TYPE(op),
                            PyArray_DESCR(op),
                            ndim, new_dims, new_strides,
                            PyArray_DATA(op),
                            PyArray_ISWRITEABLE(op) ? NPY_ARRAY_WRITEABLE : 0,
                            (PyObject *)op);

    if (ret == NULL) {
        return NULL;
    }
    if (!PyArray_Check(ret)) {
        Py_DECREF(ret);
        PyErr_SetString(PyExc_RuntimeError,
                    "NewFromDescr failed to return an array");
        return NULL;
    }
    PyArray_UpdateFlags(ret,
                NPY_ARRAY_C_CONTIGUOUS|
                NPY_ARRAY_ALIGNED|
                NPY_ARRAY_F_CONTIGUOUS);
    Py_INCREF(op);
    if (PyArray_SetBaseObject(ret, (PyObject *)op) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    return ret;
}

static int
prepare_op_axes(int ndim, int iop, char *labels, int *axes,
            int ndim_iter, char *iter_labels)
{
    int i, label, ibroadcast;

    ibroadcast = ndim-1;
    for (i = ndim_iter-1; i >= 0; --i) {
        label = iter_labels[i];
        /*
         * If it's an unlabeled broadcast dimension, choose
         * the next broadcast dimension from the operand.
         */
        if (label == 0) {
            while (ibroadcast >= 0 && labels[ibroadcast] != 0) {
                --ibroadcast;
            }
            /*
             * If we used up all the operand broadcast dimensions,
             * extend it with a "newaxis"
             */
            if (ibroadcast < 0) {
                axes[i] = -1;
            }
            /* Otherwise map to the broadcast axis */
            else {
                axes[i] = ibroadcast;
                --ibroadcast;
            }
        }
        /* It's a labeled dimension, find the matching one */
        else {
            char *match = memchr(labels, label, ndim);
            /* If the op doesn't have the label, broadcast it */
            if (match == NULL) {
                axes[i] = -1;
            }
            /* Otherwise use it */
            else {
                axes[i] = match - labels;
            }
        }
    }

    return 1;
}

static int
unbuffered_loop_nop1_ndim2(NpyIter *iter)
{
    npy_intp coord, shape[2], strides[2][2];
    char *ptrs[2][2], *ptr;
    sum_of_products_fn sop;

#if NPY_EINSUM_DBG_TRACING
    NpyIter_DebugPrint(iter);
#endif
    NPY_EINSUM_DBG_PRINT("running hand-coded 1-op 2-dim loop\n");

    NpyIter_GetShape(iter, shape);
    memcpy(strides[0], NpyIter_GetAxisStrideArray(iter, 0),
                                            2*sizeof(npy_intp));
    memcpy(strides[1], NpyIter_GetAxisStrideArray(iter, 1),
                                            2*sizeof(npy_intp));
    memcpy(ptrs[0], NpyIter_GetInitialDataPtrArray(iter),
                                            2*sizeof(char *));
    memcpy(ptrs[1], ptrs[0], 2*sizeof(char*));

    sop = get_sum_of_products_function(1,
                    NpyIter_GetDescrArray(iter)[0]->type_num,
                    NpyIter_GetDescrArray(iter)[0]->elsize,
                    strides[0]);

    if (sop == NULL) {
        PyErr_SetString(PyExc_TypeError,
                    "invalid data type for einsum");
        return -1;
    }

    /*
     * Since the iterator wasn't tracking coordinates, the
     * loop provided by the iterator is in Fortran-order.
     */
    for (coord = shape[1]; coord > 0; --coord) {
        sop(1, ptrs[0], strides[0], shape[0]);

        ptr = ptrs[1][0] + strides[1][0];
        ptrs[0][0] = ptrs[1][0] = ptr;
        ptr = ptrs[1][1] + strides[1][1];
        ptrs[0][1] = ptrs[1][1] = ptr;
    }

    return 0;
}

static int
unbuffered_loop_nop1_ndim3(NpyIter *iter)
{
    npy_intp coords[2], shape[3], strides[3][2];
    char *ptrs[3][2], *ptr;
    sum_of_products_fn sop;

#if NPY_EINSUM_DBG_TRACING
    NpyIter_DebugPrint(iter);
#endif
    NPY_EINSUM_DBG_PRINT("running hand-coded 1-op 3-dim loop\n");

    NpyIter_GetShape(iter, shape);
    memcpy(strides[0], NpyIter_GetAxisStrideArray(iter, 0),
                                            2*sizeof(npy_intp));
    memcpy(strides[1], NpyIter_GetAxisStrideArray(iter, 1),
                                            2*sizeof(npy_intp));
    memcpy(strides[2], NpyIter_GetAxisStrideArray(iter, 2),
                                            2*sizeof(npy_intp));
    memcpy(ptrs[0], NpyIter_GetInitialDataPtrArray(iter),
                                            2*sizeof(char *));
    memcpy(ptrs[1], ptrs[0], 2*sizeof(char*));
    memcpy(ptrs[2], ptrs[0], 2*sizeof(char*));

    sop = get_sum_of_products_function(1,
                    NpyIter_GetDescrArray(iter)[0]->type_num,
                    NpyIter_GetDescrArray(iter)[0]->elsize,
                    strides[0]);

    if (sop == NULL) {
        PyErr_SetString(PyExc_TypeError,
                    "invalid data type for einsum");
        return -1;
    }

    /*
     * Since the iterator wasn't tracking coordinates, the
     * loop provided by the iterator is in Fortran-order.
     */
    for (coords[1] = shape[2]; coords[1] > 0; --coords[1]) {
        for (coords[0] = shape[1]; coords[0] > 0; --coords[0]) {
            sop(1, ptrs[0], strides[0], shape[0]);

            ptr = ptrs[1][0] + strides[1][0];
            ptrs[0][0] = ptrs[1][0] = ptr;
            ptr = ptrs[1][1] + strides[1][1];
            ptrs[0][1] = ptrs[1][1] = ptr;
        }
        ptr = ptrs[2][0] + strides[2][0];
        ptrs[0][0] = ptrs[1][0] = ptrs[2][0] = ptr;
        ptr = ptrs[2][1] + strides[2][1];
        ptrs[0][1] = ptrs[1][1] = ptrs[2][1] = ptr;
    }

    return 0;
}

static int
unbuffered_loop_nop2_ndim2(NpyIter *iter)
{
    npy_intp coord, shape[2], strides[2][3];
    char *ptrs[2][3], *ptr;
    sum_of_products_fn sop;

#if NPY_EINSUM_DBG_TRACING
    NpyIter_DebugPrint(iter);
#endif
    NPY_EINSUM_DBG_PRINT("running hand-coded 2-op 2-dim loop\n");

    NpyIter_GetShape(iter, shape);
    memcpy(strides[0], NpyIter_GetAxisStrideArray(iter, 0),
                                            3*sizeof(npy_intp));
    memcpy(strides[1], NpyIter_GetAxisStrideArray(iter, 1),
                                            3*sizeof(npy_intp));
    memcpy(ptrs[0], NpyIter_GetInitialDataPtrArray(iter),
                                            3*sizeof(char *));
    memcpy(ptrs[1], ptrs[0], 3*sizeof(char*));

    sop = get_sum_of_products_function(2,
                    NpyIter_GetDescrArray(iter)[0]->type_num,
                    NpyIter_GetDescrArray(iter)[0]->elsize,
                    strides[0]);

    if (sop == NULL) {
        PyErr_SetString(PyExc_TypeError,
                    "invalid data type for einsum");
        return -1;
    }

    /*
     * Since the iterator wasn't tracking coordinates, the
     * loop provided by the iterator is in Fortran-order.
     */
    for (coord = shape[1]; coord > 0; --coord) {
        sop(2, ptrs[0], strides[0], shape[0]);

        ptr = ptrs[1][0] + strides[1][0];
        ptrs[0][0] = ptrs[1][0] = ptr;
        ptr = ptrs[1][1] + strides[1][1];
        ptrs[0][1] = ptrs[1][1] = ptr;
        ptr = ptrs[1][2] + strides[1][2];
        ptrs[0][2] = ptrs[1][2] = ptr;
    }

    return 0;
}

static int
unbuffered_loop_nop2_ndim3(NpyIter *iter)
{
    npy_intp coords[2], shape[3], strides[3][3];
    char *ptrs[3][3], *ptr;
    sum_of_products_fn sop;

#if NPY_EINSUM_DBG_TRACING
    NpyIter_DebugPrint(iter);
#endif
    NPY_EINSUM_DBG_PRINT("running hand-coded 2-op 3-dim loop\n");

    NpyIter_GetShape(iter, shape);
    memcpy(strides[0], NpyIter_GetAxisStrideArray(iter, 0),
                                            3*sizeof(npy_intp));
    memcpy(strides[1], NpyIter_GetAxisStrideArray(iter, 1),
                                            3*sizeof(npy_intp));
    memcpy(strides[2], NpyIter_GetAxisStrideArray(iter, 2),
                                            3*sizeof(npy_intp));
    memcpy(ptrs[0], NpyIter_GetInitialDataPtrArray(iter),
                                            3*sizeof(char *));
    memcpy(ptrs[1], ptrs[0], 3*sizeof(char*));
    memcpy(ptrs[2], ptrs[0], 3*sizeof(char*));

    sop = get_sum_of_products_function(2,
                    NpyIter_GetDescrArray(iter)[0]->type_num,
                    NpyIter_GetDescrArray(iter)[0]->elsize,
                    strides[0]);

    if (sop == NULL) {
        PyErr_SetString(PyExc_TypeError,
                    "invalid data type for einsum");
        return -1;
    }

    /*
     * Since the iterator wasn't tracking coordinates, the
     * loop provided by the iterator is in Fortran-order.
     */
    for (coords[1] = shape[2]; coords[1] > 0; --coords[1]) {
        for (coords[0] = shape[1]; coords[0] > 0; --coords[0]) {
            sop(2, ptrs[0], strides[0], shape[0]);

            ptr = ptrs[1][0] + strides[1][0];
            ptrs[0][0] = ptrs[1][0] = ptr;
            ptr = ptrs[1][1] + strides[1][1];
            ptrs[0][1] = ptrs[1][1] = ptr;
            ptr = ptrs[1][2] + strides[1][2];
            ptrs[0][2] = ptrs[1][2] = ptr;
        }
        ptr = ptrs[2][0] + strides[2][0];
        ptrs[0][0] = ptrs[1][0] = ptrs[2][0] = ptr;
        ptr = ptrs[2][1] + strides[2][1];
        ptrs[0][1] = ptrs[1][1] = ptrs[2][1] = ptr;
        ptr = ptrs[2][2] + strides[2][2];
        ptrs[0][2] = ptrs[1][2] = ptrs[2][2] = ptr;
    }

    return 0;
}


/*NUMPY_API
 * This function provides summation of array elements according to
 * the Einstein summation convention.  For example:
 *  - trace(a)        -> einsum("ii", a)
 *  - transpose(a)    -> einsum("ji", a)
 *  - multiply(a,b)   -> einsum(",", a, b)
 *  - inner(a,b)      -> einsum("i,i", a, b)
 *  - outer(a,b)      -> einsum("i,j", a, b)
 *  - matvec(a,b)     -> einsum("ij,j", a, b)
 *  - matmat(a,b)     -> einsum("ij,jk", a, b)
 *
 * subscripts: The string of subscripts for einstein summation.
 * nop:        The number of operands
 * op_in:      The array of operands
 * dtype:      Either NULL, or the data type to force the calculation as.
 * order:      The order for the calculation/the output axes.
 * casting:    What kind of casts should be permitted.
 * out:        Either NULL, or an array into which the output should be placed.
 *
 * By default, the labels get placed in alphabetical order
 * at the end of the output. So, if c = einsum("i,j", a, b)
 * then c[i,j] == a[i]*b[j], but if c = einsum("j,i", a, b)
 * then c[i,j] = a[j]*b[i].
 *
 * Alternatively, you can control the output order or prevent
 * an axis from being summed/force an axis to be summed by providing
 * indices for the output. This allows us to turn 'trace' into
 * 'diag', for example.
 *  - diag(a)         -> einsum("ii->i", a)
 *  - sum(a, axis=0)  -> einsum("i...->", a)
 *
 * Subscripts at the beginning and end may be specified by
 * putting an ellipsis "..." in the middle.  For example,
 * the function einsum("i...i", a) takes the diagonal of
 * the first and last dimensions of the operand, and
 * einsum("ij...,jk...->ik...") takes the matrix product using
 * the first two indices of each operand instead of the last two.
 *
 * When there is only one operand, no axes being summed, and
 * no output parameter, this function returns a view
 * into the operand instead of making a copy.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_EinsteinSum(char *subscripts, npy_intp nop,
                    PyArrayObject **op_in,
                    PyArray_Descr *dtype,
                    NPY_ORDER order, NPY_CASTING casting,
                    PyArrayObject *out)
{
    int iop, label, min_label = 127, max_label = 0, num_labels;
    char label_counts[128];
    char op_labels[NPY_MAXARGS][NPY_MAXDIMS];
    char output_labels[NPY_MAXDIMS], *iter_labels;
    int idim, ndim_output, ndim_broadcast, ndim_iter;

    PyArrayObject *op[NPY_MAXARGS], *ret = NULL;
    PyArray_Descr *op_dtypes_array[NPY_MAXARGS], **op_dtypes;

    int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
    int *op_axes[NPY_MAXARGS];
    npy_uint32 op_flags[NPY_MAXARGS];

    NpyIter *iter;
    sum_of_products_fn sop;
    npy_intp fixed_strides[NPY_MAXARGS];

    /* nop+1 (+1 is for the output) must fit in NPY_MAXARGS */
    if (nop >= NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError,
                    "too many operands provided to einstein sum function");
        return NULL;
    }
    else if (nop < 1) {
        PyErr_SetString(PyExc_ValueError,
                    "not enough operands provided to einstein sum function");
        return NULL;
    }

    /* Parse the subscripts string into label_counts and op_labels */
    memset(label_counts, 0, sizeof(label_counts));
    num_labels = 0;
    for (iop = 0; iop < nop; ++iop) {
        int length = (int)strcspn(subscripts, ",-");

        if (iop == nop-1 && subscripts[length] == ',') {
            PyErr_SetString(PyExc_ValueError,
                        "more operands provided to einstein sum function "
                        "than specified in the subscripts string");
            return NULL;
        }
        else if(iop < nop-1 && subscripts[length] != ',') {
            PyErr_SetString(PyExc_ValueError,
                        "fewer operands provided to einstein sum function "
                        "than specified in the subscripts string");
            return NULL;
        }

        if (!parse_operand_subscripts(subscripts, length,
                        PyArray_NDIM(op_in[iop]),
                        iop, op_labels[iop], label_counts,
                        &min_label, &max_label, &num_labels)) {
            return NULL;
        }

        /* Move subscripts to the start of the labels for the next op */
        subscripts += length;
        if (iop < nop-1) {
            subscripts++;
        }
    }

    /*
     * Find the number of broadcast dimensions, which is the maximum
     * number of labels == 0 in an op_labels array.
     */
    ndim_broadcast = 0;
    for (iop = 0; iop < nop; ++iop) {
        npy_intp count_zeros = 0;
        int ndim;
        char *labels = op_labels[iop];

        ndim = PyArray_NDIM(op_in[iop]);
        for (idim = 0; idim < ndim; ++idim) {
            if (labels[idim] == 0) {
                ++count_zeros;
            }
        }

        if (count_zeros > ndim_broadcast) {
            ndim_broadcast = count_zeros;
        }
    }

    /*
     * If there is no output signature, create one using each label
     * that appeared once, in alphabetical order
     */
    if (subscripts[0] == '\0') {
        char outsubscripts[NPY_MAXDIMS + 3];
        int length;
        /* If no output was specified, always broadcast left (like normal) */
        outsubscripts[0] = '.';
        outsubscripts[1] = '.';
        outsubscripts[2] = '.';
        length = 3;
        for (label = min_label; label <= max_label; ++label) {
            if (label_counts[label] == 1) {
                if (length < NPY_MAXDIMS-1) {
                    outsubscripts[length++] = label;
                }
                else {
                    PyErr_SetString(PyExc_ValueError,
                                "einstein sum subscript string has too many "
                                "distinct labels");
                    return NULL;
                }
            }
        }
        /* Parse the output subscript string */
        ndim_output = parse_output_subscripts(outsubscripts, length,
                                        ndim_broadcast, label_counts,
                                        output_labels);
    }
    else {
        if (subscripts[0] != '-' || subscripts[1] != '>') {
            PyErr_SetString(PyExc_ValueError,
                        "einstein sum subscript string does not "
                        "contain proper '->' output specified");
            return NULL;
        }
        subscripts += 2;

        /* Parse the output subscript string */
        ndim_output = parse_output_subscripts(subscripts, strlen(subscripts),
                                        ndim_broadcast, label_counts,
                                        output_labels);
    }
    if (ndim_output < 0) {
        return NULL;
    }

    if (out != NULL && PyArray_NDIM(out) != ndim_output) {
        PyErr_Format(PyExc_ValueError,
                "out parameter does not have the correct number of "
                "dimensions, has %d but should have %d",
                (int)PyArray_NDIM(out), (int)ndim_output);
        return NULL;
    }

    /* Set all the op references to NULL */
    for (iop = 0; iop < nop; ++iop) {
        op[iop] = NULL;
    }

    /*
     * Process all the input ops, combining dimensions into their
     * diagonal where specified.
     */
    for (iop = 0; iop < nop; ++iop) {
        char *labels = op_labels[iop];
        int combine, ndim;

        ndim = PyArray_NDIM(op_in[iop]);

        /*
         * If there's just one operand and no output parameter,
         * first try remapping the axes to the output to return
         * a view instead of a copy.
         */
        if (iop == 0 && nop == 1 && out == NULL) {
            ret = NULL;

            if (!get_single_op_view(op_in[iop], iop, labels,
                                    ndim_output, output_labels,
                                    &ret)) {
                return NULL;
            }

            if (ret != NULL) {
                return ret;
            }
        }

        /*
         * Check whether any dimensions need to be combined
         *
         * The char type may be either signed or unsigned, we
         * need it to be signed here.
         */
        combine = 0;
        for (idim = 0; idim < ndim; ++idim) {
            if ((signed char)labels[idim] < 0) {
                combine = 1;
            }
        }

        /* If any dimensions are combined, create a view which combines them */
        if (combine) {
            op[iop] = get_combined_dims_view(op_in[iop], iop, labels);
            if (op[iop] == NULL) {
                goto fail;
            }
        }
        /* No combining needed */
        else {
            Py_INCREF(op_in[iop]);
            op[iop] = op_in[iop];
        }
    }

    /* Set the output op */
    op[nop] = out;

    /*
     * Set up the labels for the iterator (output + combined labels).
     * Can just share the output_labels memory, because iter_labels
     * is output_labels with some more labels appended.
     */
    iter_labels = output_labels;
    ndim_iter = ndim_output;
    for (label = min_label; label <= max_label; ++label) {
        if (label_counts[label] > 0 &&
                memchr(output_labels, label, ndim_output) == NULL) {
            if (ndim_iter >= NPY_MAXDIMS) {
                PyErr_SetString(PyExc_ValueError,
                            "too many subscripts in einsum");
                goto fail;
            }
            iter_labels[ndim_iter++] = label;
        }
    }

    /* Set up the op_axes for the iterator */
    for (iop = 0; iop < nop; ++iop) {
        op_axes[iop] = op_axes_arrays[iop];

        if (!prepare_op_axes(PyArray_NDIM(op[iop]), iop, op_labels[iop],
                    op_axes[iop], ndim_iter, iter_labels)) {
            goto fail;
        }
    }

    /* Set up the op_dtypes if dtype was provided */
    if (dtype == NULL) {
        op_dtypes = NULL;
    }
    else {
        op_dtypes = op_dtypes_array;
        for (iop = 0; iop <= nop; ++iop) {
            op_dtypes[iop] = dtype;
        }
    }

    /* Set the op_axes for the output */
    op_axes[nop] = op_axes_arrays[nop];
    for (idim = 0; idim < ndim_output; ++idim) {
        op_axes[nop][idim] = idim;
    }
    for (idim = ndim_output; idim < ndim_iter; ++idim) {
        op_axes[nop][idim] = -1;
    }

    /* Set the iterator per-op flags */

    for (iop = 0; iop < nop; ++iop) {
        op_flags[iop] = NPY_ITER_READONLY|
                        NPY_ITER_NBO|
                        NPY_ITER_ALIGNED;
    }
    op_flags[nop] = NPY_ITER_READWRITE|
                    NPY_ITER_NBO|
                    NPY_ITER_ALIGNED|
                    NPY_ITER_ALLOCATE|
                    NPY_ITER_NO_BROADCAST;

    /* Allocate the iterator */
    iter = NpyIter_AdvancedNew(nop+1, op, NPY_ITER_EXTERNAL_LOOP|
                ((dtype != NULL) ? 0 : NPY_ITER_COMMON_DTYPE)|
                                       NPY_ITER_BUFFERED|
                                       NPY_ITER_DELAY_BUFALLOC|
                                       NPY_ITER_GROWINNER|
                                       NPY_ITER_REDUCE_OK|
                                       NPY_ITER_REFS_OK|
                                       NPY_ITER_ZEROSIZE_OK,
                                       order, casting,
                                       op_flags, op_dtypes,
                                       ndim_iter, op_axes, NULL, 0);

    if (iter == NULL) {
        goto fail;
    }

    /* Initialize the output to all zeros and reset the iterator */
    ret = NpyIter_GetOperandArray(iter)[nop];
    Py_INCREF(ret);
    PyArray_AssignZero(ret, NULL);


    /***************************/
    /*
     * Acceleration for some specific loop structures. Note
     * that with axis coalescing, inputs with more dimensions can
     * be reduced to fit into these patterns.
     */
    if (!NpyIter_RequiresBuffering(iter)) {
        int ndim = NpyIter_GetNDim(iter);
        switch (nop) {
            case 1:
                if (ndim == 2) {
                    if (unbuffered_loop_nop1_ndim2(iter) < 0) {
                        Py_DECREF(ret);
                        ret = NULL;
                        goto fail;
                    }
                    goto finish;
                }
                else if (ndim == 3) {
                    if (unbuffered_loop_nop1_ndim3(iter) < 0) {
                        Py_DECREF(ret);
                        ret = NULL;
                        goto fail;
                    }
                    goto finish;
                }
                break;
            case 2:
                if (ndim == 2) {
                    if (unbuffered_loop_nop2_ndim2(iter) < 0) {
                        Py_DECREF(ret);
                        ret = NULL;
                        goto fail;
                    }
                    goto finish;
                }
                else if (ndim == 3) {
                    if (unbuffered_loop_nop2_ndim3(iter) < 0) {
                        Py_DECREF(ret);
                        ret = NULL;
                        goto fail;
                    }
                    goto finish;
                }
                break;
        }
    }
    /***************************/

    if (NpyIter_Reset(iter, NULL) != NPY_SUCCEED) {
        Py_DECREF(ret);
        goto fail;
    }

    /*
     * Get an inner loop function, specializing it based on
     * the strides that are fixed for the whole loop.
     */
    NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
    sop = get_sum_of_products_function(nop,
                        NpyIter_GetDescrArray(iter)[0]->type_num,
                        NpyIter_GetDescrArray(iter)[0]->elsize,
                        fixed_strides);

#if NPY_EINSUM_DBG_TRACING
    NpyIter_DebugPrint(iter);
#endif

    /* Finally, the main loop */
    if (sop == NULL) {
        PyErr_SetString(PyExc_TypeError,
                    "invalid data type for einsum");
        Py_DECREF(ret);
        ret = NULL;
    }
    else if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *stride;
        npy_intp *countptr;
        NPY_BEGIN_THREADS_DEF;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(ret);
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        stride = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NPY_BEGIN_THREADS_NDITER(iter);
        NPY_EINSUM_DBG_PRINT("Einsum loop\n");
        do {
            sop(nop, dataptr, stride, *countptr);
        } while(iternext(iter));
        NPY_END_THREADS;

        /* If the API was needed, it may have thrown an error */
        if (NpyIter_IterationNeedsAPI(iter) && PyErr_Occurred()) {
            Py_DECREF(ret);
            ret = NULL;
        }
    }

finish:
    NpyIter_Deallocate(iter);
    for (iop = 0; iop < nop; ++iop) {
        Py_DECREF(op[iop]);
    }

    return ret;

fail:
    for (iop = 0; iop < nop; ++iop) {
        Py_XDECREF(op[iop]);
    }

    return NULL;
}
Status API Training Shop Blog About Help
© 2015 GitHub, Inc. Terms Privacy Security Contact