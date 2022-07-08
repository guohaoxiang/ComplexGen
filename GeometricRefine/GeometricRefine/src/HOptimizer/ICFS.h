/*****************************************************************

 COPYRIGHT NOTIFICATION

 This program discloses material protectable under copyright laws of
 the United States. Permission to copy and modify this software and its
 documentation for internal research use is hereby granted, provided
 that this notice is retained thereon and on all copies or modifications.
 The University of Chicago makes no representations as to the suitability
 and operability of this software for any purpose.
 It is provided "as is" without express or implied warranty.

 Use of this software for commercial purposes is expressly prohibited
 without contacting

 Jorge J. More'
 Mathematics and Computer Science Division
 Argonne National Laboratory
 9700 S. Cass Ave.
 Argonne, Illinois 60439-4844
 e-mail: more@mcs.anl.gov

 Argonne National Laboratory with facilities in the states of
 Illinois and Idaho, is owned by The United States Government, and
 operated by the University of Chicago under provision of a contract
 with the Department of Energy.

 *****************************************************************/

#pragma once

#include <cstddef>

int dicfs_(size_t *n, size_t *nnz, double *a, double *adiag,
	size_t *acol_ptr__, size_t *arow_ind__, double *l, double *ldiag,
	size_t *lcol_ptr__, size_t * lrow_ind__, size_t *p, double *alpha,
	size_t *iwa, double * wa1, double *wa2);

int dicf_(size_t *n, size_t *nnz, double *a, double *diag, size_t *col_ptr__,
	size_t *row_ind__, size_t *p, ptrdiff_t *info, size_t *indr,
	size_t *indf, size_t *list, double *w);

int dsel2_(size_t *n, double *x, size_t *keys, size_t *k);

int insort_(size_t *n, size_t *keys);

int ihsort_(size_t *n, size_t *keys);

int dstrsol_(size_t *n, double *l, double *ldiag, size_t *jptr, size_t *indr,
	double *r__, char *task);
