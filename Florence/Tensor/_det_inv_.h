#ifndef __DET_INV__H
#define __DET_INV__H

typedef double Real;

inline void inv2x2(const Real *src, Real *dst)
{
    Real det;

    /* Compute adjoint: */

    dst[0] = + src[3];
    dst[1] = - src[1];
    dst[2] = - src[2];
    dst[3] = + src[0];

    /* Compute determinant: */

    det = src[0] * dst[0] + src[1] * dst[2];

    /* Multiply adjoint with reciprocal of determinant: */

    det = 1.0 / det;

    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
}


inline void inv3x3(const Real * src, Real * dst)
{
    Real det;

    /* Compute adjoint: */

    dst[0] = + src[4] * src[8] - src[5] * src[7];
    dst[1] = - src[1] * src[8] + src[2] * src[7];
    dst[2] = + src[1] * src[5] - src[2] * src[4];
    dst[3] = - src[3] * src[8] + src[5] * src[6];
    dst[4] = + src[0] * src[8] - src[2] * src[6];
    dst[5] = - src[0] * src[5] + src[2] * src[3];
    dst[6] = + src[3] * src[7] - src[4] * src[6];
    dst[7] = - src[0] * src[7] + src[1] * src[6];
    dst[8] = + src[0] * src[4] - src[1] * src[3];

    /* Compute determinant: */

    det = src[0] * dst[0] + src[1] * dst[3] + src[2] * dst[6];

    /* Multiply adjoint with reciprocal of determinant: */

    det = 1.0 / det;

    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
    dst[4] *= det;
    dst[5] *= det;
    dst[6] *= det;
    dst[7] *= det;
    dst[8] *= det;
}

inline void _inv_(size_t ndim, const Real * src, Real * dst)
{
    if (ndim==3) {
        inv3x3(src, dst);
    }
    else {
        inv2x2(src, dst);
    }
}



inline Real det2x2(const Real *src)
{
    Real dst0 = + src[3];
    Real dst2 = - src[2];
    Real det = src[0] * dst0 + src[1] * dst2;
    return det;
}


inline Real det3x3(const Real * src)
{
    Real dst0 = + src[4] * src[8] - src[5] * src[7];
    Real dst3 = - src[3] * src[8] + src[5] * src[6];
    Real dst6 = + src[3] * src[7] - src[4] * src[6];

    Real det = src[0] * dst0 + src[1] * dst3 + src[2] * dst6;
    return det;
}


inline Real _det_(size_t ndim, const Real * src)
{
    if (ndim==3) {
        return det3x3(src);
    }
    else {
        return det2x2(src);
    }
}












inline Real invdet2x2(const Real *src, Real *dst)
{
    Real det;

    Real src0 = src[0];
    Real src1 = src[1];
    Real src2 = src[2];
    Real src3 = src[3];

    /* Compute adjoint: */
    dst[0] = + src3;
    dst[1] = - src1;
    dst[2] = - src2;
    dst[3] = + src0;

    /* Compute determinant: */
    det = src0 * dst[0] + src1 * dst[2];

    /* Multiply adjoint with reciprocal of determinant: */
    Real det2 = static_cast<Real>(1.0) / det;
    dst[0] *= det2;
    dst[1] *= det2;
    dst[2] *= det2;
    dst[3] *= det2;

    return det;
}


inline Real invdet3x3(const Real *src, Real *dst)
{
    Real det;

    Real src0 = src[0];
    Real src1 = src[1];
    Real src2 = src[2];
    Real src3 = src[3];
    Real src4 = src[4];
    Real src5 = src[5];
    Real src6 = src[6];
    Real src7 = src[7];
    Real src8 = src[8];

    /* Compute adjoint: */
    dst[0] = + src4 * src8 - src5 * src7;
    dst[1] = - src1 * src8 + src2 * src7;
    dst[2] = + src1 * src5 - src2 * src4;
    dst[3] = - src3 * src8 + src5 * src6;
    dst[4] = + src0 * src8 - src2 * src6;
    dst[5] = - src0 * src5 + src2 * src3;
    dst[6] = + src3 * src7 - src4 * src6;
    dst[7] = - src0 * src7 + src1 * src6;
    dst[8] = + src0 * src4 - src1 * src3;

    /* Compute determinant: */
    det = src0 * dst[0] + src1 * dst[3] + src2 * dst[6];

    /* Multiply adjoint with reciprocal of determinant: */
    Real det2 = static_cast<Real>(1.0) / det;

    dst[0] *= det2;
    dst[1] *= det2;
    dst[2] *= det2;
    dst[3] *= det2;
    dst[4] *= det2;
    dst[5] *= det2;
    dst[6] *= det2;
    dst[7] *= det2;
    dst[8] *= det2;

    return det;
}


inline Real _invdet_(size_t ndim, const Real * src, Real * dst)
{
    if (ndim==3) {
        return invdet3x3(src, dst);
    }
    else {
        return invdet2x2(src, dst);
    }
}

#endif