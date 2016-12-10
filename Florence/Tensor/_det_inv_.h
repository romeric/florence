#ifndef __DET_INV__H
#define __DET_INV__H

typedef double Real;

static inline void inv2x2(const Real *src, Real *dst)
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


static inline void inv3x3(const Real * src, Real * dst)
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



static inline Real det2x2(const Real *src)
{
    Real dst0 = + src[3];
    Real dst2 = - src[2];
    Real det = src[0] * dst0 + src[1] * dst2;
    return det;
}


static inline Real det3x3(const Real * src)
{
    Real dst0 = + src[4] * src[8] - src[5] * src[7];
    Real dst3 = - src[3] * src[8] + src[5] * src[6];
    Real dst6 = + src[3] * src[7] - src[4] * src[6];

    Real det = src[0] * dst0 + src[1] * dst3 + src[2] * dst6;
    return det;
}

#endif