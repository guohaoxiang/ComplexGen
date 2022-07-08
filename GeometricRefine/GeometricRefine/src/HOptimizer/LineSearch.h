#pragma once

#include "HLBFGS_BLAS.h"

#include <cstddef>

//The following functions are from LBFGS,VA35,TNPACK,CG+.
//for details, see lbfgs.f and va35.f

/*
 The license of LBFGS:

 This software is freely available for educational or commercial purposes.
 This software is released under the GNU Public License (GPL)
 */

 /*
  MCSRCH is modified a little for Preconditioned CG.
  */

  //!LINE SEARCH ROUTINE
int MCSRCH(size_t *n, double *x, double *f, double *g, double *s, double *stp,
	double *ftol, double *gtol, double *xtol, double *stpmin,
	double * stpmax, size_t *maxfev, ptrdiff_t *info, size_t *nfev,
	double *wa, size_t *keep, double *rkeep, double *cg_dginit = 0, bool weak_wolf = false);

//!   MCSTEP ROUTINE
/*!
 *   COMPUTE A SAFEGUARDED STEP FOR A LINESEARCH AND TO
 *   UPDATE AN INTERVAL OF UNCERTAINTY FOR  A MINIMIZER OF THE FUNCTION
 */
int MCSTEP(double *stx, double *fx, double *dx, double *sty, double *fy,
	double *dy, double *stp, double *fp, double *dp, bool *brackt,
	double *stpmin, double *stpmax, ptrdiff_t *info);
