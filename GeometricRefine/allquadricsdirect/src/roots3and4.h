#ifndef ROOTS_34_H
#define ROOTS_34_H

// Functions to find the roots of polynomials of order 1-4 directly
// from "Graphics Gems"
// see cpp file for version history
int SolveQuadric(double *c, double *s);
int SolveCubic(double *c, double *s);
int SolveQuartic(double *c, double *s);


// evaluate polynomials 
inline double poly2(double *c, double t) {
    return c[0]+c[1]*t+c[2]*t*t;
}
inline double poly3(double *c, double t) {
    return c[0]+c[1]*t+c[2]*t*t+c[3]*t*t*t;
}
// evaluate polynomial derivatives
inline double dpoly2(double *c, double t) {
    return c[1]+2*c[2]*t;
}
inline double dpoly3(double *c, double t) {
    return c[1]+2*c[2]*t+3*c[3]*t*t;
}

#endif

