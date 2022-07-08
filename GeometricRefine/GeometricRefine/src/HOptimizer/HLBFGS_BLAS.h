#pragma once

#include <cstddef>

//! return \f$ \sum_{i=0}^{n-1} x_iy_i \f$
double HLBFGS_DDOT(const size_t n, const double *x, const double *y);
//!  \f$ y_i += \alpha x_i \f$
void HLBFGS_DAXPY(const size_t n, const double alpha, const double *x, double *y);
//! return \f$ \sqrt{\sum_{i=0}^{n-1} x_i^2} \f$
double HLBFGS_DNRM2(const size_t n, const double *x);
//!  \f$ x_i *= a \f$
void HLBFGS_DSCAL(const size_t n, const double a, double *x);
