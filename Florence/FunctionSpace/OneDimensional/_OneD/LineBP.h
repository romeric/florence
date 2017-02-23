#include <vector>
#include "_fast_power_.h"


template<typename T>
inline void _LagrangeBP(int n, T xi, T *N, T *dN) {

    int nsize = n-1;
    T ndiv = 2.0/nsize;
    std::vector<T> eps(n);
    eps[0]=-1.; eps[n-1]= 1.;

    for (int i=0; i<nsize; ++i) {
        eps[i+1] = eps[i]+ndiv;
    }

    std::vector<T> xis(n);
    for (int incr=0; incr<n; ++incr) 
        xis[incr] = fast_pow(xi,incr);

    for (int ishape=0; ishape<n; ++ishape) {

        std::vector<T> d(n);
        d[ishape] = 1.;
        
        // Find the Newton Divided Difference
        for (int k=0; k<n; ++k) {
            for (int j=n-1; j>k; --j) {
                d[j] = (d[j]-d[j-1])/(eps[j]-eps[j-k-1]);
            }
        }

        // Convert to Monomials
        for (int k=n-1; k>=0; --k) {
            for (int j=k; j<n-1; ++j) {
                d[j] -= eps[k]*d[j+1];
            }
        }
        
        // // Build shape functions 
        // for (int incr=0; incr<n; ++incr) 
        //     N[ishape] += d[incr]*fast_pow(xi,incr);

        // // Build derivate of shape functions
        // for (int incr=0; incr<n-1; ++incr)
        //     dN[ishape] += (incr+1)*d[incr+1]*fast_pow(xi,incr);

        // Build shape functions 
        for (int incr=0; incr<n; ++incr) 
            N[ishape] += d[incr]*xis[incr];

        // Build derivate of shape functions
        for (int incr=0; incr<n-1; ++incr)
            dN[ishape] += (incr+1)*d[incr+1]*xis[incr];        
    }
}



template<typename T>
inline void _LagrangeGaussLobattoBP(int n, T xi, T *eps, T *N, T *dN) {

    std::vector<T> xis(n);
    for (int incr=0; incr<n; ++incr) 
        xis[incr] = fast_pow(xi,incr);

    for (int ishape=0; ishape<n; ++ishape) {

        std::vector<T> d(n);
        d[ishape] = 1.;
        
        // Find the Newton Divided Difference
        for (int k=0; k<n; ++k) {
            for (int j=n-1; j>k; --j) {
                d[j] = (d[j]-d[j-1])/(eps[j]-eps[j-k-1]);
            }
        }

        // Convert to Monomials
        for (int k=n-1; k>=0; --k) {
            for (int j=k; j<n-1; ++j) {
                d[j] -= eps[k]*d[j+1];
            }
        }
        
        // Build shape functions 
        for (int incr=0; incr<n; ++incr) 
            N[ishape] += d[incr]*xis[incr];

        // Build derivate of shape functions
        for (int incr=0; incr<n-1; ++incr)
            dN[ishape] += (incr+1)*d[incr+1]*xis[incr];        
    }
}
