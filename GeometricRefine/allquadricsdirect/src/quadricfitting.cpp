
#include <fstream>
#include <string>
#include <sstream>


#include "quadricfitting.h"


#include "jama_eig.h"
#include "jama_svd.h"


#include "f2c.h"
#include "blaswrap.h"
#include "clapack.h"


#include "roots3and4.h"

#include "float.h"

using namespace std;
using namespace TNT;


namespace allquadrics {


// default work size for lapack to use when solving eigenvalue problems
#define QUADFITWORKSIZE 200

vec3 findOrtho(vec3 v) {
	const vec3 x(1,0,0), y(0,1,0);
	vec3 toRet = v % x;
	if (toRet.length2() < .000001) { // ... doesn't work if v is just very small, so pls normalize v first.
		toRet = v % y;
	}
	toRet.normalize();
	return toRet;
}

struct sortBySecond {
    bool operator()(const pair<int,double> &a, const pair<int,double> &b)
    { return a.second < b.second; }
};


// algebra3.h omits mat2, so I add it here
struct mat2 {
	vec2 v[2];

	mat2() {
		v[0] = vec2(0,0);
		v[1] = vec2(0,0);
	}

	mat2(const vec2 &r1, const vec2 &r2) {
		v[0] = r1;
		v[1] = r2;
	}

	inline mat2& operator *=(const double d) {
		v[0] *= d;
		v[1] *= d;
		return *this;
	}
	inline mat2& operator +=(const mat2& b) {
		v[0] += b[0];
		v[1] += b[1];
		return *this;
	}
	inline vec2& operator [](int i) {
		return v[i];
	}
	inline const vec2& operator [](int i) const {
		return v[i];
	}

    inline double det() const {
        return v[0][0]*v[1][1] - v[0][1]*v[1][0];
    }
};
inline mat2 operator *(const mat2& a, const double d) {
    return mat2(a[0]*d, a[1]*d);
}
inline mat2 operator *(const double d, const mat2& a) {
    return mat2(a[0]*d, a[1]*d);
}
inline vec2 operator *(const mat2& a, const vec2& v) {
	return vec2(a.v[0]*v,a.v[1]*v);
}
inline mat2 outer(const vec2 &a, const vec2 &b) {
	return mat2(vec2(a[0]*b[0], a[0]*b[1]), vec2(a[1]*b[0], a[1]*b[1]));
}

inline void arraytomat(Array2D<double> &a, mat3 &m) {
	for (int i = 0; i < 3; i++) { for ( int j = 0; j < 3; j++) { m[i][j] = a[i][j]; } }
}
inline void mattoarray(mat3 &m, Array2D<double> &a) {
	for (int i = 0; i < 3; i++) { for ( int j = 0; j < 3; j++) { a[i][j] = m[i][j]; } }
}
namespace {
vec3 MultArray2dTransposeByVec3(Array2D<double> &A, vec3 &v) {
	vec3 toret(0,0,0);
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			toret[j] += A[i][j]*v[i];
		}
	}
	return toret;
}
vec3 MultArray2dByVec3(Array2D<double> &A, vec3 &v) {

	vec3 toret(0,0,0);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			toret[i] += A[i][j]*v[j];
		}
	}
	return toret;
}

void lapackSVD(const mat3 &M, mat3 &Um, mat3 &Vtm, vec3 &D) {
    char jobu = 'A';
    char jobvt = 'A';
    integer m = 3, n = 3;
    double A[9];
    int ind = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[ind++] = M[i][j];
        }
    }
    integer lda = 3;
    double S[3];
    double U[9];
    integer ldu = 3;
    double VT[9];
    integer ldvt = 3;
    double work[100];
    integer lwork = 100;
    integer info;
    dgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu,
        VT, &ldvt, work, &lwork, &info);
    ind = 0;
    

    for (int i = 0; i < 3; i++) {
        D[i]=S[i];
        for (int j = 0; j < 3; j++) {
            Um[i][j] = U[ind];
            Vtm[i][j] = VT[ind];
            ind++;
        }
    }
}
vec3 getNullSpace(const mat3 &M) {
    mat3 Um, Vtm;
    vec3 D;
    lapackSVD(M, Um, Vtm, D);
    return vec3(Vtm[0][2], Vtm[1][2], Vtm[2][2]);
}
}

vec3 getMat3Eigenvalues(const mat3 &m) {
    Array2D<double> M(3,3);
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { M[i][j] = m[i][j]; } }
    JAMA::Eigenvalue<double> eig(M);
    Array1D<double> S;
    eig.getRealEigenvalues(S);
    return vec3(S[0], S[1], S[2]);
}

void getSmallestEvec(Array2D<double> &M, double *tofill) {
    JAMA::Eigenvalue<double> eig(M);
    Array1D<double> S;
    eig.getRealEigenvalues(S);
    double minVal = fabs(S[0]);
    int minInd = 0;

    int dim = M.dim1();

    for (int i = 1; i < dim; ++i) {
        if (fabs(S[i]) < minVal) {
            minVal = S[i];
            minInd = i;
        }
    }

    Array2D<double> V;
    eig.getV(V);

    for (int i = 0; i < dim; i++) {
        tofill[i] = V[i][minInd];
    }
}



// scalar triple product
inline double stp(vec3 &a0, vec3 &a1, vec3 &a2) {
    return a0*(a1%a2);
}
inline double det(vec2 &a0, vec2 &a1) {
    return a0[0]*a1[1]-a0[1]*a1[0];
}
void buildDetPoly3(mat3 &A, mat3 &B, double *a) {
    a[0] = stp(A[0], A[1], A[2]);
    a[1] = stp(B[0], A[1], A[2]) + stp(A[0], B[1], A[2]) + stp(A[0], A[1], B[2]);
    a[2] = stp(A[0], B[1], B[2]) + stp(B[0], A[1], B[2]) + stp(B[0], B[1], A[2]);
    a[3] = stp(B[0], B[1], B[2]);
}
void buildDetPoly2(mat2 &A, mat2 &B, double *a) {
    a[0] = det(A[0], A[1]);
    a[1] = det(B[0], A[1]) + det(A[0], B[1]);
    a[2] = det(B[0], B[1]);
}
mat2 getMinor2(mat3 &M) {
    return mat2(vec2(M[0][0], M[0][1]), vec2(M[1][0], M[1][1]));
}

namespace {
    inline double atBc(double *p1, double *p2, double *m) {
        double msum = 0, nsum = 0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                msum += m[i*10+j]*p1[j]*p2[i];
            }
        }
        return msum;
    }
}

struct sortByFirst {
    bool operator()(const pair<double,int> &a, const pair<double,int> &b)
    { return a.second < b.second; }
};

// returns:
// -1 => negative definite
// 1 => positive definite
// 0 => neither
// tries to return 0 ("neither") for near-singular matrices
int classifyDefiniteness(mat3 &q, double epsilon = 0){//2e-6) {
    if (q[0][0]*q[1][1] - q[0][1]*q[1][0] <= epsilon) return 0;
    double det = q.det();
    if (q[0][0]*det <= epsilon) return 0;
    if (det <= epsilon) return -1;
    else return 1;
}
int classifyDefiniteness(double *dets, double epsilon = 0){//2e-6) {
    if (dets[0]*dets[2] <= epsilon || dets[1] <= epsilon) return 0;
    return dets[0] < 0 ? -1 : 1;
}

int isHyperbolic(const mat3 &q, double epsilon = 2e-4) {
    vec3 eigs = getMat3Eigenvalues(q);
    eigs.normalize();
    
    double e01 = eigs[0]*eigs[1], e02 = eigs[0]*eigs[2], e12 = eigs[1]*eigs[2];
    if (eigs[0]*eigs[1] < -epsilon || eigs[0]*eigs[2] < -epsilon || eigs[1]*eigs[2] < -epsilon) return 1;
    int numZ = (fabs(eigs[0]) < epsilon) + (fabs(eigs[1]) < epsilon) + (fabs(eigs[2]) < epsilon);
    if (numZ > 1) return -1;
    return 0;
}
bool isHyperbolic2(const mat2 &q, double epsilon = 2e-4) {
    return q.det() < 0;
}
void minorDets(mat3 &q, double *dets) {
    dets[0] = q[0][0];
    dets[1] = q[0][0]*q[1][1] - q[0][1]*q[1][0];
    dets[2] = q.det();
}

template<typename T>
mat3 getQmat(T *q) {
    mat3 A(0);
    A[0][0] = q[4];
    A[1][0] = A[0][1] = q[5]*.5;
    A[2][0] = A[0][2] = q[6]*.5;
    A[1][1] = q[7];
    A[1][2] = A[2][1] = q[8]*.5;
    A[2][2] = q[9];
    return A;
}





namespace {

    void getParams_Ellipse_Hyperbola(vector<double> &tofill_el, vector<double> &tofill_hyp, vector<double> &M, vector<double> &N, int numparams) {
        double work[QUADFITWORKSIZE];
        double *Mtemp = new double[M.size()], *Ntemp = new double[N.size()];
        for (size_t i = 0; i < M.size(); i++) { Mtemp[i] = M[i]; Ntemp[i] = 0.0; }
        Ntemp[6*5+3] = 4;
        Ntemp[6*4+4] = -1;
        Ntemp[6*3+5] = 4;

        integer itype = 1;
        char jobvl = 'N'; // skip the left eigenvectors
        char jobvr = 'V'; // compute the right eigenvectors
        integer n = numparams;
        // A = Mtemp, will be overwritten
        integer lda = n;
        // B = Ntemp, will be overwritten
        integer ldb = n;
        double *alphar = new double[n], *alphai = new double[n], *beta = new double[n]; // real, imaginary and denom of eigs
        double VL[1]; // dummy var, we don't want left eigvecs
        integer ldvl = 1; // must be >= 1 even though we don't use VL
        double *VR = new double[n*n]; // right eigvecs
        integer ldvr = n;
        integer lwork = QUADFITWORKSIZE;
		integer info;
        
        dggev_(&jobvl, &jobvr, &n, Mtemp, &lda, Ntemp, &ldb,
                alphar, alphai, beta, VL, &ldvl, VR, &ldvr, 
                work, &lwork, &info);

        tofill_el.resize(n); tofill_hyp.resize(n);

        
        int mincol_e = -1, mincol_h = -1;


        vector< pair<int, double> > solnOrder_e, solnOrder_h;

        double bestErr_e = -1, bestErr_h;
        for (int col = 0; col < n; col++) {
            double merr = 0, nerr = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    merr += M[i+n*j]*VR[i+n*col]*VR[j+n*col];
                    nerr += N[i+n*j]*VR[i+n*col]*VR[j+n*col];
                }
            }
            double err = fabs(merr/nerr);

            double det = 5*VR[3+n*col]*VR[5+n*col]-VR[4+n*col]*VR[4+n*col];

            if (det >= 0) {
                if (col == 0 || err < bestErr_e) {
                    bestErr_e = err;
                    mincol_e = col;
                }
            }
            if (det <= 0) {
                if (col == 0 || err < bestErr_h) {
                    bestErr_h = err;
                    mincol_h = col;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            tofill_el[i] = VR[i+n*mincol_e];
            tofill_hyp[i] = VR[i+n*mincol_h];
        }


        delete [] alphar;
        delete [] alphai;
        delete [] beta;
        delete [] VR;
        delete [] Mtemp;
        delete [] Ntemp;
    }

    void getParamsHelper(vector<double> &tofill, vector<double> &M, vector<double> &N, int numparams, 
                    vector< pair<double, vector<double> > > *orderedSet = NULL, bool recomputeErrors = true) {
        double work[QUADFITWORKSIZE];
        double *Mtemp = new double[M.size()], *Ntemp = new double[N.size()];
        for (size_t i = 0; i < M.size(); i++) { Mtemp[i] = M[i]; Ntemp[i] = N[i]; }
        integer itype = 1;
        char jobvl = 'N'; // skip the left eigenvectors
        char jobvr = 'V'; // compute the right eigenvectors
        integer n = numparams;
        // A = Mtemp, will be overwritten
        integer lda = n;
        // B = Ntemp, will be overwritten
        integer ldb = n;
        double *alphar = new double[n], *alphai = new double[n], *beta = new double[n]; // real, imaginary and denom of eigs
        double VL[1]; // dummy var, we don't want left eigvecs
        integer ldvl = 1; // must be >= 1 even though we don't use VL
        double *VR = new double[n*n]; // right eigvecs
        integer ldvr = n;
        integer lwork = QUADFITWORKSIZE;
        integer info;
        
        dggev_(&jobvl, &jobvr, &n, Mtemp, &lda, Ntemp, &ldb,
                alphar, alphai, beta, VL, &ldvl, VR, &ldvr,
                work, &lwork, &info);
        
        tofill.resize(n);

        
        int mincol = -1;


        vector< pair<int, double> > solnOrder;
        if (orderedSet) {
            orderedSet->clear();
        }

        if (recomputeErrors) {
            double bestErr = -1;
            for (int col = 0; col < n; col++) {
                double merr = 0, nerr = 0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        merr += M[i+n*j]*VR[i+n*col]*VR[j+n*col];
                        nerr += N[i+n*j]*VR[i+n*col]*VR[j+n*col];
                    }
                }
                double err = fabs(merr/nerr);
                if (orderedSet) {
                    solnOrder.push_back(pair<int,double>(col, err));
                }
                if (col == 0 || err < bestErr) {
                    bestErr = err;
                    mincol = col;
                }
            }
        } else {
            double mineig = -1;
            for (int i = 0; i < n; i++) {
                if (fabs(alphai[i]) < 0.00001 && beta[i] > 0) {
                    double eigval = fabs(alphar[i])/beta[i];
                    if (mincol == -1 || eigval < mineig) {
                        mincol = i;
                        mineig = eigval;
                    }
                    if (orderedSet) {
                        solnOrder.push_back(pair<int,double>(i, eigval));
                    }
                }
            }
        }

        if (orderedSet) {
            sort(solnOrder.begin(), solnOrder.end(), sortBySecond());
            for (int i = 0; i < (int)solnOrder.size(); i++) {
                int col = solnOrder[i].first;
                orderedSet->push_back(pair<double, vector<double> > (solnOrder[i].second, vector<double>(n,0)));
                pair<double, vector<double> > &toFill = orderedSet->back();
                for (int i = 0; i < n; i++) {
                    toFill.second[i] = VR[i+n*col];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            tofill[i] = VR[i+n*mincol]; 
        }


        delete [] alphar;
        delete [] alphai;
        delete [] beta;
        delete [] VR;
        delete [] Mtemp;
        delete [] Ntemp;
    }
}




struct QuadricPlaneFieldFitter {
    double M[16], N[16];
	QuadricPlaneFieldFitter() { clear(0); }
	void clear(int sizehint) {
		for (int i = 0; i < 16; i++) { M[i]=N[i]=0; } 
	}

	void addEl(double scale, const vec3 &p, const vec3 &n) {
		addPosition(scale, p);
        addNormal(scale, p, n);
	}

    void addPosition(double scale, const vec3 &p) {
        double c[4];
        double x = p[0], y = p[1], z = p[2];

        // coefficient vector
        c[0] = x;
        c[1] = y;
        c[2] = z;
        c[3] = 1;
        
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                M[i*4+j] += c[i]*c[j] * scale;
            }
        }
    }
    inline void addNormal(double scale, const vec3 &p, const vec3 &n) {
        // intentionally left blank: I'm not using the normals for plane fitting currently
    }

	void getParams(std::vector<double> &tofill) {
		double work[QUADFITWORKSIZE];
		double Mtemp[16], Ntemp[16];
		for (int i = 0; i < 16; i++) { Mtemp[i] = M[i]; Ntemp[i] = 0;}
		for (int i = 0; i < 3; i++) { Ntemp[i*4+i] = 1; } // first 3 diagonal entries are 1
		integer itype = 1;
		char jobvl = 'N'; // skip the left eigenvectors
		char jobvr = 'V'; // compute the right eigenvectors
		integer n = 4;
		// A = Mtemp, will be overwritten
		integer lda = 4;
		// B = Ntemp, will be overwritten
		integer ldb = 4;
		double alphar[4], alphai[4], beta[4]; // real, imaginary and denom of eigs
		double VL[1]; // dummy var, we don't want left eigvecs
		integer ldvl = 1; // must be >= 1 even though we don't use VL
		double VR[16]; // right eigvecs
		integer ldvr = 4;
		integer lwork = QUADFITWORKSIZE;
		integer info;
	    
		dggev_(&jobvl, &jobvr, &n, Mtemp, &lda, Ntemp, &ldb,
				alphar, alphai, beta, VL, &ldvl, VR, &ldvr, 
				work, &lwork, &info);

	    
		int mincol = -1;
		double mineig = -1;
		for (int i = 0; i < 4; i++) {
			if (fabs(alphai[i]) < 0.00001 && beta[i] > 0) {
				double eigval = fabs(alphar[i])/beta[i];
				if (mincol == -1 || eigval < mineig) {
					mincol = i;
					mineig = eigval;
				}
			}
		}


		tofill.resize(10);
		tofill[0] = VR[3+4*mincol];
		for (int i = 0; i < 3; i++) {
			tofill[i+1] = VR[i+4*mincol];
		}
		for (int i = 4; i < 10; i++) {
			tofill[i] = 0;
		}
	}
};


struct SphereFieldEval {
    static int numparams() {
        return 5;
    }
    static void evalFuncVecs(const double *p, double *c) {
        double x = p[0], y = p[1], z = p[2];
        c[0] = 1; c[1] = x; c[2] = y; c[3] = z; c[4] = x*x+y*y+z*z;
    }
    static void evalGradVecs(const double *p, double *cx, double *cy, double *cz) {
        double x = p[0], y = p[1], z = p[2];
        for (int i = 0; i < numparams(); i++) {
            cx[i] = cy[i] = cz[i] = 0;
        }
        cx[1] = 1; cx[4] = 2*x;
        cy[2] = 1; cy[4] = 2*y;
        cz[3] = 1; cz[4] = 2*z;
    }
    static void convertToQuadric(double *sphereparams, double *quadricparams) {
        for (int i = 0; i < 4; i++) {
            quadricparams[i] = sphereparams[i];
        }
        for (int i = 5; i < 9; i++) {
            quadricparams[i] = 0;
        }
        quadricparams[4] = sphereparams[4];
        quadricparams[7] = sphereparams[4];
        quadricparams[9] = sphereparams[4];
    }
};

struct RotSymQuadricFieldEval {
    static int numparams() {
        return 4;
    }

    static void evalFuncVecs(const double *p, double *c) {
        double x = p[0], y = p[1], z = p[2];

        c[0] = x*x + y*y;
        c[1] = z*z;
        c[2] = z;
        c[3] = 1;
    }

    static void evalGradVecs(const double *p, double *cx, double *cy, double *cz) {
        double x = p[0], y = p[1], z = p[2];
        for (int i = 0; i < numparams(); i++) {
            cx[i] = cy[i] = cz[i] = 0;
        }
        cx[0] = 2*x;
        cy[0] = 2*y;
        cz[1] = 2*z;   cz[2] = 1;
    }
    static void convertToQuadric(double *coneparams, double *quadricparams) {
        for (int i = 0; i < 10; i++) { quadricparams[i] = 0; }
        quadricparams[0] = coneparams[3];
        quadricparams[3] = coneparams[2];
        quadricparams[9] = coneparams[1];
        quadricparams[4] = coneparams[0];
        quadricparams[7] = coneparams[0];
    }
};

// fitting a circular cone based on points which have been transformed to 2D polar coordinates around an axis
// and fitting a line to those transformed points
struct CircCone_Line2DEval {
    static int numparams() {
        return 3;
    }
    static void evalFuncVecs(const double *p, double *c) {
        double x = p[0], y = p[1], z = p[2];
        
        // coefficient vector
        c[0] = x;
        c[1] = y;
        c[2] = 1;
    }
    static void evalGradVecs(const double *p, double *cx, double *cy, double *cz) {
        double x = p[0], y = p[1], z = p[2];
        for (int i = 0; i < numparams(); i++) {
            cx[i] = cy[i] = cz[i] = 0;
        }
        cx[0] = 1;
        cy[1] = 1;
    }
    static void convertToQuadric(double *coneparams, double *quadricparams) {
        Quadric qf;
        qf.q[4] = qf.q[7] = coneparams[0]*coneparams[0];
        qf.q[9] = -coneparams[1]*coneparams[1];
        
        if (fabs(coneparams[1]) > 0) { // if we're not a cylinder, transform apex from origin to correct center.
            vec3 offset(0,0,-coneparams[2]/coneparams[1]);
            qf.transformQuadric(offset);
        }
        
        for (int i = 0; i < 10; i++) {
            quadricparams[i] = qf.q[i];
        }
    }
};

struct ConeFieldEval {
    static int numparams() {
        return 6;
    }
    static void evalFuncVecs(const double *p, double *c) {
        double x = p[0], y = p[1], z = p[2];
        
        // coefficient vector
        c[0] = x*x;
        c[1] = x*y;
        c[2] = x*z;
        c[3] = y*y;
        c[4] = y*z;
        c[5] = z*z;
    }
    static void evalGradVecs(const double *p, double *cx, double *cy, double *cz) {
        double x = p[0], y = p[1], z = p[2];
        for (int i = 0; i < numparams(); i++) {
            cx[i] = cy[i] = cz[i] = 0;
        }
        cx[0] = 2*x; cx[1] = y;   cx[2] = z;
        cy[1] = x;   cy[3] = 2*y; cy[4] = z;
        cz[2] = x;   cz[4] = y;   cz[5] = 2*z;
    }
    static void convertToQuadric(double *coneparams, double *quadricparams) {
        for (int i = 0; i < 4; i++) {
            quadricparams[i] = 0;
        }
        for (int i = 0; i < 6; i++) {
            quadricparams[i+4] = coneparams[i];
        }
    }
};


struct ZAlignedCylindricalQuadricFieldEval {
    static int numparams() {
        return 6;
    }
    static void evalFuncVecs(const double *p, double *c) {
        double x = p[0], y = p[1];
        c[0] = 1; c[1] = x; c[2] = y; c[3] = x*x; c[4] = x*y; c[5] = y*y;
    }
    static void evalGradVecs(const double *p, double *cx, double *cy, double *cz) {
        double x = p[0], y = p[1];
        for (int i = 0; i < numparams(); i++) {
            cx[i] = cy[i] = cz[i] = 0;
        }
        cx[1] = 1; cx[3] = 2*x; cx[4] = y; 
        cy[2] = 1; cy[5] = 2*y; cy[4] = x;
    }
    static void convertToQuadric(double *cylinderparams, double *quadricparams) {
        for (int i = 0; i < 10; i++) {
            quadricparams[i] = 0;
        }
        for (int i = 0; i < 3; i++) {
            quadricparams[i] = cylinderparams[i];
        }
        quadricparams[4] = cylinderparams[3];
        quadricparams[5] = cylinderparams[4];
        quadricparams[7] = cylinderparams[5];
    }
};

struct ZAlignedCircularCylindricalQuadricFieldEval {
    static int numparams() {
        return 4;
    }
    static void evalFuncVecs(const double *p, double *c) {
        double x = p[0], y = p[1];
        c[0] = 1; c[1] = x; c[2] = y; c[3] = x*x+y*y;
    }
    static void evalGradVecs(const double *p, double *cx, double *cy, double *cz) {
        double x = p[0], y = p[1];
        for (int i = 0; i < numparams(); i++) {
            cx[i] = cy[i] = cz[i] = 0;
        }
        cx[1] = 1; cx[3] = 2*x;
        cy[2] = 1; cy[3] = 2*y;
    }
    static void convertToQuadric(double *cylinderparams, double *quadricparams) {
        for (int i = 0; i < 10; i++) {
            quadricparams[i] = 0;
        }
        for (int i = 0; i < 3; i++) {
            quadricparams[i] = cylinderparams[i];
        }
        quadricparams[4] = cylinderparams[3];
        quadricparams[7] = cylinderparams[3];
    }
};

struct ConeKinFieldEval {
    static int numparams() {
        return 4;
    }
    static void evalFuncVecs(const double *p, const double *n, double *c) {
        double x = p[0], y = p[1], z = p[2];
        double nx = n[0], ny = n[1], nz = n[2];
        c[0] = x*nx+y*ny+z*nz; c[1] = nx; c[2] = ny; c[3] = nz;
    }
    static void evalGradVecs(const double *p, const double *n, double *cx, double *cy, double *cz) {
        double x = p[0], y = p[1], z = p[2];
        double nx = n[0], ny = n[1], nz = n[2];
        for (int i = 0; i < numparams(); i++) {
            cx[i] = cy[i] = cz[i] = 0;
        }
        cx[0] = x; cx[1] = 1;
        cy[0] = y; cy[2] = 1;
        cz[0] = z; cz[3] = 1;
    }
};

struct RotKinFieldEval {
    static int numparams() {
        return 6;
    }
    static void evalFuncVecs(const double *p, const double *n, double *c) {
        double x = p[0], y = p[1], z = p[2];
        double nx = n[0], ny = n[1], nz = n[2];
        vec3 pv(x,y,z), nv(nx,ny,nz);
        vec3 pxn = (pv % nv);
        c[0] = pxn[0]; c[1] = pxn[1]; c[2] = pxn[2]; 
        c[3] = nx; c[4] = ny; c[5] = nz;
    }
    static void evalGradVecs(const double *p, const double *n, double *cx, double *cy, double *cz) {
        double x = p[0], y = p[1], z = p[2];
        double nx = n[0], ny = n[1], nz = n[2];
        for (int i = 0; i < numparams(); i++) {
            cx[i] = cy[i] = cz[i] = 0;
        }
        cx[1] =  z; cx[2] = -y; cx[3] = 1;
        cy[0] = -z; cy[2] =  x; cy[4] = 1;
        cz[0] =  y; cz[1] = -x; cz[5] = 1;
    }
};


template<typename LinFieldEval>
struct LinParamKinematicFieldFitter {
    vector<double>  c, cx, cy, cz;
    vector<double> M, N;
    
    void debugPrintMatrices() {
        cout << "M:" << endl;
        int n = LinFieldEval::numparams();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << M[i*n+j] << " ";
            }
            cout << endl;
        }
        cout << "N:" << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << N[i*n+j] << " ";
            }
            cout << endl;
        }
    }
    
    LinParamKinematicFieldFitter() { clear(0); }
    void clear(int sizehint) {
        const int n = LinFieldEval::numparams();
        M.clear(); N.clear(); M.resize(n*n); N.resize(n*n);
    }

    void addElement(double scale, const vec3 &p, const vec3 &n) {
        const int numpar = LinFieldEval::numparams();
        c.resize(numpar); cx.resize(numpar); cy.resize(numpar); cz.resize(numpar);
        LinFieldEval::evalFuncVecs(p.n, n.n, &c[0]);
        LinFieldEval::evalGradVecs(p.n, n.n, &cx[0], &cy[0], &cz[0]);

        // add cross products to accumulator matrices
        for (int i = 0; i < numpar; i++) {
            for (int j = 0; j < numpar; j++) {
                M[i*numpar+j] += c[i]*c[j] * scale;
                N[i*numpar+j] += (cx[i]*cx[j] + cy[i]*cy[j] + cz[i]*cz[j]) * scale;
            }
        }
    }

    void getParams(vector<double> &tofill) 
	{
	   getParamsHelper(tofill, M, N, LinFieldEval::numparams());
	}
    void getParams(vector< pair<double, vector<double> > > &tofill) 
	{
	   vector<double> dummy;
	   getParamsHelper(dummy, M, N, LinFieldEval::numparams(), &tofill);
	}

};


template<typename LinFieldEval>
struct LinParamImplicitFieldFitter {
    vector<double> M, N;
    vector<double>  c, cx, cy, cz;
    LinParamImplicitFieldFitter() { clear(0); }
    void clear(int sizehint) {
        const int n = LinFieldEval::numparams();
        M.clear(); N.clear(); M.resize(n*n); N.resize(n*n);
    }
    
    void debugPrintMatrices() {
        cout << "M:" << endl;
        int n = LinFieldEval::numparams();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << M[i*n+j] << " ";
            }
            cout << endl;
        }
        cout << "N:" << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << N[i*n+j] << " ";
            }
            cout << endl;
        }
    }

    void addPosition(double scale, const vec3 &p) {
        const int n = LinFieldEval::numparams();
        c.resize(n); cx.resize(n); cy.resize(n); cz.resize(n);
        LinFieldEval::evalFuncVecs(p.n, &c[0]);
        LinFieldEval::evalGradVecs(p.n, &cx[0], &cy[0], &cz[0]);

        // add cross products to accumulator matrices
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                M[i*n+j] += c[i]*c[j] * scale;
                N[i*n+j] += (cx[i]*cx[j] + cy[i]*cy[j] + cz[i]*cz[j]) * scale;
            }
        }
    }
    void addNormal(double scale, const vec3 &p, const vec3 &no) {
		// commented out: I'm not using normals for implicit fitting currently; there wasn't a benefit to doing so.  Below is the code I used before to test the idea.
        /*if (g_optk.gradientErrorWt > 0) {
            const int n = LinFieldEval::numparams();
            //vector<double> c(n), cx(n), cy(n), cz(n);
            LinFieldEval::evalFuncVecs(p.n, &c[0]);
            LinFieldEval::evalGradVecs(p.n, &cx[0], &cy[0], &cz[0]);
        
            vec3 t = findOrtho(no);
            vec3 b = t%no;
            b.normalize();

            // square vectors:
            vector<double> ct(n,0), cb(n,0);
            for (int i = 0; i < n; i++) {
                ct[i] = cx[i]*t[0]+cy[i]*t[1]+cz[i]*t[2];
                cb[i] = cx[i]*b[0]+cy[i]*b[1]+cz[i]*b[2];
            }

            double gradwt = g_optk.gradientErrorWt;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    M[i*n+j] += ct[i]*ct[j] * scale * gradwt;
                    M[i*n+j] += cb[i]*cb[j] * scale * gradwt;
                }
            }
        }*/
    }

    void addEl(double scale, const vec3 &p, const vec3 &n) {
        addPosition(scale, p);
        addNormal(scale, p, n);
    }

    void getParams(vector<double> &tofill) 
	{
	   getParamsHelper(tofill, M, N, LinFieldEval::numparams());
	}
    void getQuadric(Quadric &qf) {
        vector<double> tofill;
        getParams(tofill);
        qf.clear();
        LinFieldEval::convertToQuadric(&tofill[0], qf.q);
    }
};



struct QuadricFieldFitter {
    double M[100], N[100];
    QuadricFieldFitter() { clear(0); }

    void clear(int sizehint) {
        for (int i = 0; i < 100; i++) { M[i]=N[i]=0; } 
    }

    // find the range of parameters for which interpolating the two param. vectors gives ellipsoids and hyperboloids
    // also, the best (by M,N metric) interpolation t's for each type
    void lineSearch(double *p1, double *p2,  
                                     double &bestHyper, double &bestEll, double &bestPara, double &bestParaH, double &bestParaE,
                                      double &t_hyper, double &t_ell, double &t_para, double &t_hpara, double &t_epara);
    void lineSearch2D(double *p1, double *p2, 
                                      double &bestHyper, double &bestEll, double &bestPara,
                                      double &t_hyper, double &t_ell, double &t_para);

    template<typename T>
    T getTaubinError(T *params) {
        T msum = 0, nsum = 0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                msum += M[i*10+j]*params[j]*params[i];
                nsum += N[i*10+j]*params[j]*params[i];
            }
        }
        return msum / nsum;
    }
    template<typename T>
    T getTaubinError_interp(T *params1, T *params2, double t) {
        T params[10];
        for (int i = 0; i < 10; i++) {
            params[i] = params1[i] + t*params2[i];
        }
        T msum = 0, nsum = 0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                msum += M[i*10+j]*params[j]*params[i];
                nsum += N[i*10+j]*params[j]*params[i];
            }
        }
        return msum / nsum;
    }

    void addPosition(double scale, const vec3 &p) {
        double c[10];
        double cx[10],cy[10],cz[10];
        double x = p[0], y = p[1], z = p[2];

        // coefficient vector
        c[0] = 1;
        c[1] = x;
        c[2] = y;
        c[3] = z;
        c[4] = x*x;
        c[5] = x*y;
        c[6] = x*z;
        c[7] = y*y;
        c[8] = y*z;
        c[9] = z*z;

        // partial derivative vector's coefficient vectors
        for (int i = 0; i < 10; i++) { cx[i] = cy[i] = cz[i] = 0; }
        cx[1] = 1; cx[4] = 2*x; cx[5] = y; cx[6] = z;
        cy[2] = 1; cy[5] = x; cy[7] = 2*y; cy[8] = z;
        cz[3] = 1; cz[6] = x; cz[8] = y; cz[9] = 2*z;

        // add cross products to accumulator matrices
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                M[i*10+j] += c[i]*c[j] * scale;
                N[i*10+j] += (cx[i]*cx[j] + cy[i]*cy[j] + cz[i]*cz[j]) * scale;
            }
        }
    }
    void addNormal(double scale, const vec3 &p, const vec3 &n) {
        double cx[10],cy[10],cz[10];
        double x = p[0], y = p[1], z = p[2];

        // partial derivative vector's coefficient vectors
        for (int i = 0; i < 10; i++) { cx[i] = cy[i] = cz[i] = 0; }
        cx[1] = 1; cx[4] = 2*x; cx[5] = y; cx[6] = z;
        cy[2] = 1; cy[5] = x; cy[7] = 2*y; cy[8] = z;
        cz[3] = 1; cz[6] = x; cz[8] = y; cz[9] = 2*z;

        // add cross products to accumulator matrices
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                N[i*10+j] += (cx[i]*cx[j] + cy[i]*cy[j] + cz[i]*cz[j]) * scale;
            }
        }

		/*if (g_optk.gradientErrorWt > 0) {
            vec3 t = findOrtho(n);
            vec3 b = t%n;
            b.normalize();

            // square vectors:
            double ct[10], cb[10];
            for (int i = 0; i < 10; i++) {
                ct[i] = cx[i]*t[0]+cy[i]*t[1]+cz[i]*t[2];
                cb[i] = cx[i]*b[0]+cy[i]*b[1]+cz[i]*b[2];
            }

            double gradwt = g_optk.gradientErrorWt;
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    M[i*10+j] += ct[i]*ct[j] * scale * gradwt;
                    M[i*10+j] += cb[i]*cb[j] * scale * gradwt;
                }
            }
        }*/
    }

    void addEl(double scale, const vec3 &p, const vec3 &n) {
        double c[10];
        double cx[10],cy[10],cz[10];
        double x = p[0], y = p[1], z = p[2];

        // coefficient vector
        c[0] = 1;
        c[1] = x;
        c[2] = y;
        c[3] = z;
        c[4] = x*x;
        c[5] = x*y;
        c[6] = x*z;
        c[7] = y*y;
        c[8] = y*z;
        c[9] = z*z;

        // partial derivative vector's coefficient vectors
        for (int i = 0; i < 10; i++) { cx[i] = cy[i] = cz[i] = 0; }
        cx[1] = 1; cx[4] = 2*x; cx[5] = y; cx[6] = z;
        cy[2] = 1; cy[5] = x; cy[7] = 2*y; cy[8] = z;
        cz[3] = 1; cz[6] = x; cz[8] = y; cz[9] = 2*z;

        // add cross products to accumulator matrices
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                M[i*10+j] += c[i]*c[j] * scale;
                N[i*10+j] += (cx[i]*cx[j] + cy[i]*cy[j] + cz[i]*cz[j]) * scale;
            }
        }

		// code to account for error in the normals as well; commented out for release (b/c it required a new parameter and didn't seem to have a huge effect in practice)
/*        if (g_optk.gradientErrorWt > 0) {
            vec3 t = findOrtho(n);
            vec3 b = t%n;
            b.normalize();

            // square vectors:
            double ct[10], cb[10];
            for (int i = 0; i < 10; i++) {
                ct[i] = cx[i]*t[0]+cy[i]*t[1]+cz[i]*t[2];
                cb[i] = cx[i]*b[0]+cy[i]*b[1]+cz[i]*b[2];
            }

            double gradwt = g_optk.gradientErrorWt;
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    M[i*10+j] += ct[i]*ct[j] * scale * gradwt;
                    M[i*10+j] += cb[i]*cb[j] * scale * gradwt;
                }
            }
        }*/
    }

    void getParams(vector<double> &tofill, int forceType = -1); // uses lapack stuff that is included in the cpp file
    void getParamBasis(vector<double> &sortedFabsEigvals, vector<double> &sortedEigvecs);
};


// given: p1 and p2 are 10-dim parameter vectors for quadrics with all Z params = 0, i.e. Z-aligned cylinders
// p1 should be the taubin optimal parameter vector
// outputs: interpolation parameters of the best hyperbolic, elliptical, and parabolic cylinders
void QuadricFieldFitter::lineSearch2D(double *p1, double *p2, 
                                      double &bestHyper, double &bestEll, double &bestPara,
									  double &t_hyper, double &t_ell, double &t_para) 
{
    bestHyper = bestEll = bestPara = -1; // negative errors indicate unassigned / no results found yet

    mat3 q31f = getQmat(p1);
    mat3 q32f = getQmat(p2);
    mat2 q21 = getMinor2(q31f), q22 = getMinor2(q32f);
    double det2poly[3];
    buildDetPoly2(q21, q22, det2poly);
    double roots2[3];
    int nr2 = SolveQuadric(det2poly, roots2);
    vector< pair<int,double> > allRoots;

    // assign the t=0 case as best ell or hyper based on the type
    if (isHyperbolic2(q21)) {
        t_hyper = 0;
        bestHyper = getTaubinError_interp(p1, p2, 0);
    } else {
        t_ell = 0;
        bestEll = getTaubinError_interp(p1, p2, 0);
    }

    // evaluate the error at the roots of the determinant 
    for (int ri = 0; ri < nr2; ri++) {
        double t = roots2[ri];
        double err = getTaubinError_interp(p1, p2, t);

        if (bestPara < 0 || err < bestPara) {
            t_para = t;
            bestPara = err;
        }
        if (bestHyper < 0 || err < bestHyper) {
            t_hyper = t;
            bestHyper = err;
        }
        if ((bestEll < 0 || err < bestEll)) {
            t_ell = t;
            bestEll = err;
        }
    }

}

// given: p1 and p2 are general quadric parameter vectors
// p1 should be the taubin optimal parameter vector
// outputs: errors and interpolation parameters of the best hyperboloid, ellipsoid, and paraboloid
void QuadricFieldFitter::lineSearch(double *p1, double *p2,  
                                     double &bestHyper, double &bestEll, double &bestPara, double &bestParaH, double &bestParaE,
                                      double &t_hyper, double &t_ell, double &t_para, double &t_hpara, double &t_epara)
{
    bestHyper = bestEll = bestPara = bestParaE = bestParaH = -2; // negative errors indicate unassigned / no results found yet

    mat3 q31 = getQmat(p1);
    mat3 q32 = getQmat(p2);
    double det3poly[4];
    buildDetPoly3(q31, q32, det3poly);
    double roots3[3];
    int nr3;
    nr3 = SolveCubic(det3poly, roots3);
    vector< pair<int,double> > allRoots;

    // assign the t=0 case as best ell or hyper based on the type
    int hyperT = isHyperbolic(q31);
    if (hyperT != 0) {
        t_hyper = 0;
        bestHyper = getTaubinError_interp(p1, p2, 0);
    }
    if (hyperT < 1) {
        t_ell = 0;
        bestEll = getTaubinError_interp(p1, p2, 0);
    }

    // evaluate the error at the roots of the determinant 
    for (int ri = 0; ri < nr3; ri++) {
        double t = roots3[ri];
        double err = getTaubinError_interp(p1, p2, t);

        if (bestPara < 0 || err < bestPara) {
            t_para = t;
            bestPara = err;
        }
        if (bestHyper < 0 || err < bestHyper) {
            t_hyper = t;
            bestHyper = err;
        }

        int hyperType = isHyperbolic(q31+t*q32);
        if (hyperType != 0) {
            if (bestParaH < 0 || err < bestParaH) {
                t_hpara = t;
                bestParaH = err;
            }
        }
        if (hyperType < 1) {
            if (bestEll < 0 || err < bestEll) {
                t_ell = t;
                bestEll = err;
            }
            if (bestParaE < 0 || err < bestParaE) {
                t_epara = t;
                bestParaE = err;
            }
        }

        
    }
}



void QuadricFieldFitter::getParamBasis(vector<double> &sortedFabsEigvals, vector<double> &sortedEigvecs) {
    sortedFabsEigvals.clear(); sortedEigvecs.clear();

    double work[QUADFITWORKSIZE];
    double Mtemp[100], Ntemp[100];
    for (int i = 0; i < 100; i++) { Mtemp[i] = M[i]; Ntemp[i] = N[i]; }
    integer itype = 1;
    char jobvl = 'N'; // skip the left eigenvectors
    char jobvr = 'V'; // compute the right eigenvectors
    integer n = 10;
    integer lda = 10;
    integer ldb = 10;
    double alphar[10], alphai[10], beta[10]; // real, imaginary and denom of eigs
    double VL[1]; // dummy var, we don't want left eigvecs
    integer ldvl = 1; // must be >= 1 even though we don't use VL
    double VR[100]; // right eigvecs
    integer ldvr = 10;
    integer lwork = QUADFITWORKSIZE;
    integer info;
    
    dggev_(&jobvl, &jobvr, &n, Mtemp, &lda, Ntemp, &ldb,
            alphar, alphai, beta, VL, &ldvl, VR, &ldvr, 
            work, &lwork, &info);



    vector<pair<int, double> > fabsEigvalColumns;

    int mincol = -1;
    double mineig = -1;
    for (int i = 0; i < 10; i++) {
        if (beta[i] > 0) {
            double eigval = fabs(alphar[i])/beta[i];
            fabsEigvalColumns.push_back(pair<int,double>(i, eigval));
            if (mincol == -1 || eigval < mineig) {
                mincol = i;
                mineig = eigval;
            }
        }
    }
    sort(fabsEigvalColumns.begin(), fabsEigvalColumns.end(), sortBySecond());

    for (size_t i = 0; i < fabsEigvalColumns.size(); i++) {
        int col = fabsEigvalColumns[i].first;
        sortedFabsEigvals.push_back(fabsEigvalColumns[i].second);
        for (int j = 0; j < 10; j++) {
            sortedEigvecs.push_back(VR[j+10*col]);
        }
    }
}

void QuadricFieldFitter::getParams(std::vector<double> &tofill, int forceType) {
    double work[QUADFITWORKSIZE];
    double Mtemp[100], Ntemp[100];
    for (int i = 0; i < 100; i++) { Mtemp[i] = M[i]; Ntemp[i] = N[i]; }

    enum {TYPE_ELLIPSOID = 0, TYPE_HYPERBOLOID, TYPE_MAX};

    if (forceType > -1) {
        for (int i = 0; i < 100; i++) { Ntemp[i] = 0; }
        
        if (forceType == TYPE_ELLIPSOID) {
            Ntemp[44] = Ntemp[77] = Ntemp[99] = -1;
            Ntemp[47] = Ntemp[74] = Ntemp[49] = Ntemp[94] = Ntemp[79] = Ntemp[97] = 1;
            Ntemp[55] = Ntemp[66] = Ntemp[88] = -1;
        } else if (forceType == TYPE_HYPERBOLOID) {
            Ntemp[44] = Ntemp[77] = Ntemp[99] = 0;
            Ntemp[47] = Ntemp[74] = Ntemp[49] = Ntemp[94] = Ntemp[79] = Ntemp[97] = 1;
            Ntemp[55] = Ntemp[66] = Ntemp[88] = -.5;
        }
    }

    integer itype = 1;
    char jobvl = 'N'; // skip the left eigenvectors
    char jobvr = 'V'; // compute the right eigenvectors
    integer n = 10;
    // A = Mtemp, will be overwritten
    integer lda = 10;
    // B = Ntemp, will be overwritten
    integer ldb = 10;
    double alphar[10], alphai[10], beta[10]; // real, imaginary and denom of eigs
    double VL[1]; // dummy var, we don't want left eigvecs
    integer ldvl = 1; // must be >= 1 even though we don't use VL
    double VR[100]; // right eigvecs
    integer ldvr = 10;
    integer lwork = QUADFITWORKSIZE;
    integer info;
    
    dggev_(&jobvl, &jobvr, &n, Mtemp, &lda, Ntemp, &ldb,
            alphar, alphai, beta, VL, &ldvl, VR, &ldvr, 
            work, &lwork, &info);

    tofill.resize(10);
    int mincol = -1;
    double mineig = -1;
    for (int i = 0; i < 10; i++) {
        //if (fabs(alphai[i]) < 0.00001 && beta[i] > 0) {
        if (fabs(beta[i]) > 0.000001) {
            double eigvalErr = fabs(alphar[i])/beta[i];
            if (forceType > -1) { // also check that the parameters form the target type
                mat3 Qmat = getQmat(VR+10*i);
                double det2 = Qmat[0][0]*Qmat[1][1] - Qmat[1][0]*Qmat[0][1];
                double det3 = Qmat.det();
                double det1 = Qmat[0][0];

                bool ellipsoid = det2 > 0 && ( (det1 > 0 && det3 > 0) || (det1 < 0 && det3 < 0) );
                //if (g_optk.useEigenvalueSignForQuadricTypeCheck) ellipsoid = alphar[i]/beta[i] > 0; 
                if (forceType == TYPE_ELLIPSOID ) {
                    if (!ellipsoid)
                        continue; // skip it; it's not an ellipsoid
                    eigvalErr = getTaubinError(VR+10*i);
                }
                if (forceType == TYPE_HYPERBOLOID && ellipsoid) {
                    continue; // skip it; not a hyperboloid
                }
                if (forceType == TYPE_HYPERBOLOID) {
                    eigvalErr = getTaubinError(VR+10*i);
                    vec3 eigs = getMat3Eigenvalues(Qmat);
                    for (int i = 0; i < 3; i++) {
                        if (fabs(eigs[i]) < .0001) eigs[i] = 0; 
                    }
                }
            }
            
            if (mincol == -1 || eigvalErr < mineig) {
                mincol = i;
                mineig = eigvalErr;
            }
        }
    }
    if (mincol == -1) {
        for (int i = 0; i < 10; i++) {
            tofill[i] = 9999; // no valid solutions; make a degenerate failquadric
        }
    } else {
        for (int i = 0; i < 10; i++) {
            tofill[i] = VR[i+10*mincol]; // grab the best column
        }
    }

}



template <class DataIterator>
void fitEllipsoidHelper(DataIterator first, DataIterator last, Quadric &field) {
    QuadricFieldFitter qff;
    
	for (DataIterator i = first; i != last; i++) {
		qff.addEl(i->w, i->p, i->n);
	}
    
    vector<double> sortedFabsEigvals, sortedEigvecs;
    qff.getParamBasis(sortedFabsEigvals, sortedEigvecs);
    
    double th, te, tp, tpe, tph;
    double eh, ee, ep, epe, eph;
    double *q0 = &sortedEigvecs[0], *q1 = &sortedEigvecs[10];
    
    mat3 qmat = getQmat(q0);
    int hyper = isHyperbolic(qmat);

    if (hyper < 1) {
        field.init(q0);
    } else {
        qff.lineSearch(&sortedEigvecs[0], &sortedEigvecs[10], eh, ee, ep, eph, epe, th, te, tp, tph, tpe);
        
        if (ee < 0) { // no ellipsoid fit
            vector<double> biasedEllipsoidParams;
            qff.getParams(biasedEllipsoidParams, 0); // (biased) fit that is forced to be an ellipsoid

            q1 = &biasedEllipsoidParams[0];
            qff.lineSearch(&sortedEigvecs[0], &biasedEllipsoidParams[0], eh, ee, ep, eph, epe, th, te, tp, tph, tpe);
        }
        
        vector<double> optEllParams(10, 0);

        for (int i = 0; i < 10; i++) {
            //optHyperParams[i] = q0[i] + th*q1[i];
            optEllParams[i] = q0[i] + te*q1[i];
        }
        
        field.init(optEllParams);
    }
}
template <class DataIterator>
void fitHyperboloidHelper(DataIterator first, DataIterator last, Quadric &field) {
    QuadricFieldFitter qff;
    
    for (DataIterator i = first; i != last; i++) {
        qff.addEl(i->w, i->p, i->n);
    }
    
    vector<double> sortedFabsEigvals, sortedEigvecs;
    qff.getParamBasis(sortedFabsEigvals, sortedEigvecs);
    
    double th, te, tp, tpe, tph;
    double eh, ee, ep, epe, eph;
    double *q0 = &sortedEigvecs[0], *q1 = &sortedEigvecs[10];
    
    mat3 qmat = getQmat(q0);
    int hyper = isHyperbolic(qmat);
    
    if (hyper != 0) {
        field.init(q0);
    } else {
        qff.lineSearch(&sortedEigvecs[0], &sortedEigvecs[10], eh, ee, ep, eph, epe, th, te, tp, tph, tpe);
        
        if (eh < 0) { // no hyperboloid fit
            vector<double> biasedHyperboloidParams;
            qff.getParams(biasedHyperboloidParams, 1); // (biased) fit that is forced to be a hyperboloid
            
            q1 = &biasedHyperboloidParams[0];
            qff.lineSearch(&sortedEigvecs[0], &biasedHyperboloidParams[0], eh, ee, ep, eph, epe, th, te, tp, tph, tpe);
        }
        
        vector<double> optHyperParams(10, 0);
        
        for (int i = 0; i < 10; i++) {
            optHyperParams[i] = q0[i] + th*q1[i];
        }
        
        field.init(optHyperParams);
    }
}

template <class DataIterator>
double fitParaboloidOrEllipsoidHelper(DataIterator first, DataIterator last, Quadric &field, bool &isEllipsoid) {
    QuadricFieldFitter qff;
    
    for (DataIterator i = first; i != last; i++) {
        qff.addEl(i->w, i->p, i->n);
    }
    
    vector<double> sortedFabsEigvals, sortedEigvecs;
    qff.getParamBasis(sortedFabsEigvals, sortedEigvecs);
    
    double th, te, tp, tpe, tph;
    double eh, ee, ep, epe, eph;
    double *q0 = &sortedEigvecs[0], *q1 = &sortedEigvecs[10];
    
    mat3 qmat = getQmat(q0);
    int hyper = isHyperbolic(qmat);

    if (hyper == 0) { // best fit is an ellipsoid
        isEllipsoid = true;
        field.init(q0);
    } else if (hyper == -1) { // the best fit returned a paraboloid directly
        isEllipsoid = false;
        field.init(q0);
    } else { // best fit was a hyperboloid; we will line-search for the best paraboloid
        isEllipsoid = false;
        qff.lineSearch(&sortedEigvecs[0], &sortedEigvecs[10], eh, ee, ep, eph, epe, th, te, tp, tph, tpe);
        
        vector<double> optParaParams(10, 0);
        
        for (int i = 0; i < 10; i++) {
            optParaParams[i] = q0[i] + tp*q1[i];
        }
        
        field.init(optParaParams);
    }

    return qff.getTaubinError(field.q);
}
template <class DataIterator>
void fitParaboloidQuadricHelper(DataIterator first, DataIterator last, Quadric &field) {
    QuadricFieldFitter qff;
    
    for (DataIterator i = first; i != last; i++) {
        qff.addEl(i->w, i->p, i->n);
    }
    
    vector<double> sortedFabsEigvals, sortedEigvecs;
    qff.getParamBasis(sortedFabsEigvals, sortedEigvecs);
    
    double th, te, tp, tpe, tph;
    double eh, ee, ep, epe, eph;
    double *q0 = &sortedEigvecs[0], *q1 = &sortedEigvecs[10];
    
    mat3 qmat = getQmat(q0);
    int hyper = isHyperbolic(qmat);
    
    if (hyper == -1) {
        field.init(q0);
    } else {
        qff.lineSearch(&sortedEigvecs[0], &sortedEigvecs[10], eh, ee, ep, eph, epe, th, te, tp, tph, tpe);
        
        vector<double> optParaParams(10, 0);
        
        for (int i = 0; i < 10; i++) {
            optParaParams[i] = q0[i] + tp*q1[i];
        }
        
        field.init(optParaParams);
    }
}
template <class DataIterator>
void fitRotSymQuadricHelper(DataIterator first, DataIterator last, Quadric &field) {
    
    // first fit a kinematic field to find the axis of rotation
    LinParamKinematicFieldFitter<RotKinFieldEval> rkfe;

    for (DataIterator i = first; i != last; i++) {
        vec3 &p = i->p;
        vec3 &n = i->n;
        double s = i->w;

        rkfe.addElement(s, p, n);
    }
    
    vector<pair<double, vector<double> > > allParams;
    rkfe.getParams(allParams);
    vector<double> rotparams(6,0);
    for (int i = 0; i < (int)allParams.size(); i++) { // take the first field with non-zero rotation in it
        vec3 r(allParams[i].second[0],allParams[i].second[1],allParams[i].second[2]);
        if (r*r > .00001) {
            rotparams = allParams[i].second;
            break;
        }
    }
    
    // extract the rotation axis from the field
    vec3 rotaxis(rotparams[0], rotparams[1], rotparams[2]);
    vec3 rotpt(0);
    {
        vec3 c(rotparams[3], rotparams[4], rotparams[5]);
        
        vec3 &r = rotaxis;
        double rlen = r.length();
        c /= rlen;
        r /= rlen;
        rotpt = r%(c-(c*r)*r);
    }
    
    mat3 Rrot;
    {
        vec3 b = findOrtho(rotaxis);
        vec3 t = rotaxis % b;
        Rrot = mat3(b,t,rotaxis);
    }
    
    // now fit a rotationally symmetric quadric about that axis
    // (we will fit this with data transformed such that the Z axis is the rotation axis)
    LinParamImplicitFieldFitter<RotSymQuadricFieldEval> rsqfe;
    
    {
		for (DataIterator i = first; i != last; i++) {
			double s = i->w;
			vec3 &p = i->p;
			vec3 &n = i->n;
            
            rsqfe.addNormal(s, Rrot*(p-rotpt), Rrot*n);
            rsqfe.addPosition(s, Rrot*(p-rotpt));
		}
    }
    
    rsqfe.getQuadric(field);
    
    
    // transform the quadric back to global coordinates
    field.transformQuadric(Rrot.transpose());
    field.transformQuadric(rotpt);
}

template <class DataIterator>
void fitGenConeQuadricHelper(DataIterator first, DataIterator last, Quadric &field, bool acceptCylinderAsCone) {
    LinParamKinematicFieldFitter<ConeKinFieldEval> ckfe;
    Array2D<double> cylDirMat(3,3,0.0);
    
    
    for (DataIterator i = first; i != last; i++) {
        vec3 &p = i->p;
        vec3 &n = i->n;
        double s = i->w;
        
        
        ckfe.addElement(s, p, n);
        for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                cylDirMat[ii][jj] += n[ii]*n[jj] * s;
            }
        }
    }
    
    
    
    vector<pair<double, vector<double> > > allParams;
    ckfe.getParams(allParams);
    vector<double> ckparams(4,0);
    
    if (acceptCylinderAsCone) {
		ckparams = allParams[0].second;
	} else {
		for (int i = 0; i < (int)allParams.size(); i++) {
			if (fabs(allParams[i].second[0]) > .0001) {
				ckparams = allParams[i].second;
				break;
			}
		}
	}
    
    
    
    if (acceptCylinderAsCone && fabs(ckparams[0]) < .00001) {
    // if it's a cylinder AND we accept cylinders as cones, find the best elliptical cylinder as our fit
    // note this is a rather involved special case; it seems actually more efficient to reject cylinders as not being cones
    // (in both cases we have a somewhat arbitrary threshold for deciding when something should be considered a cylinder)
    // (essentially it is considered a cylinder if the cone center would be almost infinitely far from the origin;
    //  which seems reasonable assuming that the origin is the centroid of the data, as it should be since
    //  I expect users to center and re-scale their data before fitting)
        
        vec3 cyldir(0);
        getSmallestEvec(cylDirMat, &cyldir[0]);
        vec3 b = findOrtho(cyldir);
        vec3 t = cyldir % b;
        mat3 Rcyl(b,t,cyldir);
        
        LinParamImplicitFieldFitter<ZAlignedCylindricalQuadricFieldEval> zacfe;
        LinParamImplicitFieldFitter<ConeFieldEval> cfe;
        
        
        for (DataIterator i = first; i != last; i++) {
            double s = i->w;
            vec3 &p = i->p;
            vec3 &n = i->n;
            
            vec3 px = Rcyl * p, nx = Rcyl * n;
            zacfe.addNormal(s, px, nx);
            zacfe.addPosition(s, px);
        }

        
        // we fit an elliptical cylinder specifically as our 'cylinder cone'
        vector<double> par_e(6,0), par_h(6,0), par_opt(6,0);  // biased fit parameters for ellipse and hyperbola
        zacfe.getParams(par_opt);
        getParams_Ellipse_Hyperbola(par_e, par_h, zacfe.M, zacfe.N, 6);
        vector<double> fpare(10,0), fparh(10,0), fparo(10,0);
        int xform[6] = {0,1,2,4,5,7};
        for (int i = 0; i < 6; i++) {
            fpare[xform[i]] = par_e[i];
            fparh[xform[i]] = par_h[i];
            fparo[xform[i]] = par_opt[i];
        }
        QuadricFieldFitter qfz;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                qfz.M[xform[i]*10+xform[j]] = zacfe.M[i*6+j];
                qfz.N[xform[i]*10+xform[j]] = zacfe.N[i*6+j];
            }
        }
        
        double *bestpar = &fparo[0];
        double *otherpar = &fparh[0];
        bool ellIsBest = true;
        if (4*fparo[7]*fparo[4]-fparo[5]*fparo[5] < 0) {
            otherpar = &fpare[0];
            ellIsBest = false;
        }
        
        double bestHyper, bestEll, bestPara, t_hyper, t_ell, t_para;
        qfz.lineSearch2D(bestpar, otherpar, bestHyper, bestEll, bestPara, t_hyper, t_ell, t_para);
        
        for (int i = 0; i < 10; i++) {
            field.q[i] = bestpar[i]+t_ell*otherpar[i];
        }
        
        field.transformQuadric(Rcyl.transpose());
        
    } else { // it's not a cylinder (or we don't accept cylinders as cones) so fit a cone
        vec3 coneCenter(ckparams[1], ckparams[2], ckparams[3]);
        coneCenter /= -ckparams[0];
        
        LinParamImplicitFieldFitter<ConeFieldEval> cfe;
        
        {
            for (DataIterator i = first; i != last; i++) {
                double s = i->w;
                vec3 &p = i->p;
                vec3 &n = i->n;
                
                cfe.addNormal(s, p-coneCenter, n);
                cfe.addPosition(s, p-coneCenter);
            }
        }
        
        
        cfe.getQuadric(field);
        
        field.transformQuadric(coneCenter);
    }
}

template <class DataIterator>
void fitCircConeQuadricHelper(DataIterator first, DataIterator last, Quadric &field) {
    
    LinParamKinematicFieldFitter<RotKinFieldEval> rkfe;
    Array2D<double> cylDirMat(3,3,0.0);
    
    for (DataIterator i = first; i != last; i++) {
        vec3 &p = i->p;
        vec3 &n = i->n;
        double s = i->w;
        
        rkfe.addElement(s, p, n);
    }
    
    vector<pair<double, vector<double> > > allParams;
    rkfe.getParams(allParams);
    vector<double> rotparams(6,0);
    for (int i = 0; i < (int)allParams.size(); i++) {
        vec3 r(allParams[i].second[0],allParams[i].second[1],allParams[i].second[2]);
        if (r*r > .00001) {
            rotparams = allParams[i].second;
            break;
        }
    }
    vec3 rotaxis(rotparams[0], rotparams[1], rotparams[2]);
    vec3 rotpt(0);
    {
        
        vec3 c(rotparams[3], rotparams[4], rotparams[5]);
        
        
        vec3 &r = rotaxis;
        double rlen = r.length();
        c /= rlen;
        r /= rlen;
        rotpt = r%(c-(c*r)*r);
    }
    
    mat3 Rrot;
    {
        vec3 b = findOrtho(rotaxis);
        vec3 t = rotaxis % b;
        Rrot = mat3(b,t,rotaxis);
    }
    
    vec3 cyldir(0);
    getSmallestEvec(cylDirMat, &cyldir[0]);
    vec3 b = findOrtho(cyldir);
    vec3 t = cyldir % b;
    mat3 Rcyl(b,t,cyldir);
    
    
    LinParamImplicitFieldFitter<CircCone_Line2DEval> ccle;
    
    {
		for (DataIterator i = first; i != last; i++) {
			double s = i->w;
			vec3 &p = i->p;
			vec3 &n = i->n;
            
            double along = (p-rotpt)*rotaxis;
            ccle.addPosition(s, vec3((p-along*rotaxis-rotpt).length(), along, 0));
		}
    }
    
    ccle.getQuadric(field);
    
    field.transformQuadric(Rrot.transpose());
    field.transformQuadric(rotpt);
}


template <class DataIterator>
void fitGenCylinderQuadricHelper(DataIterator first, DataIterator last, Quadric &field) {
    Array2D<double> cylDirMat(3,3,0.0);
    
    for (DataIterator i = first; i != last; i++) {
        vec3 &p = i->p;
        vec3 &n = i->n;
        double s = i->w;
        
        for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                cylDirMat[ii][jj] += n[ii]*n[jj] * s;
            }
        }
    }
    
    
    vec3 cyldir(0);
    getSmallestEvec(cylDirMat, &cyldir[0]);
    vec3 b = findOrtho(cyldir);
    vec3 t = cyldir % b;
    mat3 Rcyl(b,t,cyldir);
    
    LinParamImplicitFieldFitter<ZAlignedCylindricalQuadricFieldEval> zacfe;
    
    {
		for (DataIterator i = first; i != last; i++) {
			double s = i->w;
			vec3 &p = i->p;
			vec3 &n = i->n;
            
			vec3 px = Rcyl * p, nx = Rcyl * n;
            zacfe.addNormal(s, px, nx);
            zacfe.addPosition(s, px);
		}
    }
    
    
    zacfe.getQuadric(field);
    
    field.transformQuadric(Rcyl.transpose());
}
template <class DataIterator>
void fitCircCylinderQuadricHelper(DataIterator first, DataIterator last, Quadric &field) {
    Array2D<double> cylDirMat(3,3,0.0);
    
    for (DataIterator i = first; i != last; i++) {
        vec3 &p = i->p;
        vec3 &n = i->n;
        double s = i->w;
        
        for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                cylDirMat[ii][jj] += n[ii]*n[jj] * s;
            }
        }
    }
    
    
    vec3 cyldir(0);
    getSmallestEvec(cylDirMat, &cyldir[0]);
    vec3 b = findOrtho(cyldir);
    vec3 t = cyldir % b;
    mat3 Rcyl(b,t,cyldir);
    
    //Rcyl = Rcyl;
    LinParamImplicitFieldFitter<ZAlignedCircularCylindricalQuadricFieldEval> zaccfe;
    
    {
		for (DataIterator i = first; i != last; i++) {
			double s = i->w;
			vec3 &p = i->p;
			vec3 &n = i->n;
            
			vec3 px = Rcyl * p, nx = Rcyl * n;
            zaccfe.addNormal(s, px, nx);
            zaccfe.addPosition(s, px);
		}
    }
    
    zaccfe.getQuadric(field);
    
    field.transformQuadric(Rcyl.transpose());
}
template <class DataIterator>
void fitSphereHelper(DataIterator first, DataIterator last, Quadric &field) {
    
    LinParamImplicitFieldFitter<SphereFieldEval> sfe;
    
    {
		for (DataIterator i = first; i != last; i++) {
			double s = i->w;
			vec3 &p = i->p;
			vec3 &n = i->n;
            
            sfe.addNormal(s, p, n);
            
            sfe.addPosition(s, p);
		}
    }
    
    
    sfe.getQuadric(field);
    

}
    
template <class DataIterator>
void fitGeneralQuadricHelper(DataIterator first, DataIterator last, Quadric &field) {
    QuadricFieldFitter qff;
    
    for (DataIterator i = first; i != last; i++) {
        qff.addEl(i->w, i->p, i->n);
    }
    
    // fit different quadric types separately
    vector<double> generalQuadricParams;
    qff.getParams(generalQuadricParams);
    
    field.init(generalQuadricParams);
}



// the massive "do all the fits!" function
// if you want to use a *lot* of the fits, then this is a bit more efficient than calling different fitting functions separately because it reuses work
// (for example, you may want this if you want to bias towards various specialized fits, but use more general fits if the specialized fits would have excessive error)
// this function also performs a few more obscure fits that the functions above do not bother with
template <class DataIterator>
void fitQuadricHelper(DataIterator first, DataIterator last, vector<Quadric> &fields) {
	QuadricFieldFitter qff;
	QuadricPlaneFieldFitter qpff;

	for (DataIterator i = first; i != last; i++) {
		qff.addEl(i->w, i->p, i->n);
		qpff.addEl(i->w, i->p, i->n);
	}
	

    // fit different quadric types separately
    vector<double> generalQuadricParams;
    vector<double> generalEllipsoidParams, generalHyperboloidParams;
    qff.getParams(generalQuadricParams);
    qff.getParams(generalEllipsoidParams, 0);
    qff.getParams(generalHyperboloidParams, 1);
    vector<double> planeParams;
    qpff.getParams(planeParams);

	fields.clear();
	fields.resize(NUM_QUADRIC_TYPES);
    //Quadric fields[NUM_QUADRIC_TYPES];
    
    fields[TYPE_GENERAL_QUADRIC].init(generalQuadricParams);
    fields[TYPE_PLANE].init(planeParams);
    fields[TYPE_ELLIPSOID_BIASED].init(generalEllipsoidParams);
    fields[TYPE_HYPERBOLOID_BIASED].init(generalHyperboloidParams);

    {
        
        vector<double> sortedFabsEigvals, sortedEigvecs;
        qff.getParamBasis(sortedFabsEigvals, sortedEigvecs);

        vector<double> optEllParams(10, 0), optHyperParams(10, 0), optParaParams(10, 0), optParaParamsH(10, 0), optParaParamsE(10, 0);
        
        double th, te, tp, tpe, tph;
        double eh, ee, ep, epe, eph;
        double *q0 = &sortedEigvecs[0], *q1 = &sortedEigvecs[10];
        qff.lineSearch(&sortedEigvecs[0], &sortedEigvecs[10], eh, ee, ep, eph, epe, th, te, tp, tph, tpe);

        bool foundParaboloid = false, foundParaboloidH = false, foundParaboloidE = false;;
        if (ep > -1) {        
            foundParaboloid = true;
            for (int i = 0; i < 10; i++) {
                optParaParams[i] = q0[i] + tp*q1[i];
            }
            fields[TYPE_PARABOLOID].init(optParaParams);
        }
        if (eph > -1) {
            foundParaboloidH = true;
            for (int i = 0; i < 10; i++) {
                optParaParamsH[i] = q0[i] + tph*q1[i];
            }
            fields[TYPE_PARABOLOID_HYPERBOLIC].init(optParaParamsH);
        }
        if (epe > -1) {
            foundParaboloidE = true;
            for (int i = 0; i < 10; i++) {
                optParaParamsE[i] = q0[i] + tpe*q1[i];
            }
            fields[TYPE_PARABOLOID_ELLIPTICAL].init(optParaParamsE);
        }

        if (eh < 0) { // no hyperboloid fit
            q1 = fields[TYPE_HYPERBOLOID_BIASED].q;
            qff.lineSearch(&sortedEigvecs[0], fields[TYPE_HYPERBOLOID_BIASED].q, eh, ee, ep, eph, epe, th, te, tp, tph, tpe);
        } else if (ee < 0) { // no ellipsoid fit
            q1 = fields[TYPE_ELLIPSOID_BIASED].q;
            qff.lineSearch(&sortedEigvecs[0], fields[TYPE_ELLIPSOID_BIASED].q, eh, ee, ep, eph, epe, th, te, tp, tph, tpe);
        }

        if (!foundParaboloid) {        
            foundParaboloid = true;
            for (int i = 0; i < 10; i++) {
                optParaParams[i] = q0[i] + tp*q1[i];
            }
            fields[TYPE_PARABOLOID].init(optParaParams);
        }
        if (!foundParaboloidH && eph > -1) {
            foundParaboloidH = true;
            for (int i = 0; i < 10; i++) {
                optParaParamsH[i] = q0[i] + tph*q1[i];
            }
            fields[TYPE_PARABOLOID_HYPERBOLIC].init(optParaParamsH);
        }
        if (!foundParaboloidE && epe > -1) {
            foundParaboloidE = true;
            for (int i = 0; i < 10; i++) {
                optParaParamsE[i] = q0[i] + tpe*q1[i];
            }
            fields[TYPE_PARABOLOID_ELLIPTICAL].init(optParaParamsE);
        }

        for (int i = 0; i < 10; i++) {
            optHyperParams[i] = q0[i] + th*q1[i];
            optEllParams[i] = q0[i] + te*q1[i];
        }

        fields[TYPE_HYPERBOLOID_OPT].init(optHyperParams);
        fields[TYPE_ELLIPSOID_OPT].init(optEllParams);

        if (!foundParaboloidE) {
            q1 = fields[TYPE_ELLIPSOID_BIASED].q;
            qff.lineSearch(&sortedEigvecs[0], fields[TYPE_ELLIPSOID_BIASED].q, eh, ee, ep, eph, epe, th, te, tp, tph, tpe);
            if (!foundParaboloidE && epe > -1) {
                foundParaboloidE = true;
                for (int i = 0; i < 10; i++) {
                    optParaParamsE[i] = q0[i] + tpe*q1[i];
                }
                fields[TYPE_PARABOLOID_ELLIPTICAL].init(optParaParamsE);
            }
        }
    }



    
    LinParamKinematicFieldFitter<ConeKinFieldEval> ckfe;
    LinParamKinematicFieldFitter<RotKinFieldEval> rkfe;
    Array2D<double> cylDirMat(3,3,0.0);


    vec3 centroid(0); double wts = 0;
    for (DataIterator i = first; i != last; i++) {
        vec3 &p = i->p;
        vec3 &n = i->n;
        double s = i->w;

        centroid += p*s; wts += s;

        ckfe.addElement(s, p, n);
        rkfe.addElement(s, p, n);
        for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                cylDirMat[ii][jj] += n[ii]*n[jj] * s;
            }
        }
    }
    

    centroid /= wts;
   
        

    vector<pair<double, vector<double> > > allParams; 
    ckfe.getParams(allParams);
    vector<double> ckparams(4,0);

	bool acceptCylinderAsCone = true; // set to false if you don't accept cylinders as cones
	if (acceptCylinderAsCone) {
		ckparams = allParams[0].second;
	} else {
		for (int i = 0; i < (int)allParams.size(); i++) {
			if (fabs(allParams[i].second[0]) > .0001) {
				ckparams = allParams[i].second;
				break;
			}
		}
	}

    vec3 coneCenter(ckparams[1], ckparams[2], ckparams[3]);
    coneCenter /= -ckparams[0];

    allParams.clear();
    rkfe.getParams(allParams);
    vector<double> rotparams(6,0);
    for (int i = 0; i < (int)allParams.size(); i++) {
        vec3 r(allParams[i].second[0],allParams[i].second[1],allParams[i].second[2]);
        if (r*r > .00001) {
            rotparams = allParams[i].second;
            break;
        }
    }
    vec3 rotaxis(rotparams[0], rotparams[1], rotparams[2]);
    vec3 rotpt(0);
    {
        
        vec3 c(rotparams[3], rotparams[4], rotparams[5]);

        
        vec3 &r = rotaxis;
        double rlen = r.length();
        c /= rlen;
        r /= rlen;
        rotpt = r%(c-(c*r)*r);
    }

    mat3 Rrot;
    {    
        vec3 b = findOrtho(rotaxis);
        vec3 t = rotaxis % b;
        Rrot = mat3(b,t,rotaxis);
    }

    vec3 cyldir(0);
    getSmallestEvec(cylDirMat, &cyldir[0]);
    vec3 b = findOrtho(cyldir);
    vec3 t = cyldir % b;
    mat3 Rcyl(b,t,cyldir);

    //Rcyl = Rcyl;
    LinParamImplicitFieldFitter<ZAlignedCircularCylindricalQuadricFieldEval> zaccfe;
    LinParamImplicitFieldFitter<ZAlignedCylindricalQuadricFieldEval> zacfe;
    LinParamImplicitFieldFitter<ConeFieldEval> cfe;
    LinParamImplicitFieldFitter<RotSymQuadricFieldEval> rsqfe;
    LinParamImplicitFieldFitter<SphereFieldEval> sfe;
    LinParamImplicitFieldFitter<CircCone_Line2DEval> ccle;
    
    {
		for (DataIterator i = first; i != last; i++) {
			double s = i->w;
			vec3 &p = i->p;
			vec3 &n = i->n;

			vec3 px = Rcyl * p, nx = Rcyl * n;
            zaccfe.addNormal(s, px, nx);
            zacfe.addNormal(s, px, nx);
            cfe.addNormal(s, p-coneCenter, n);
            rsqfe.addNormal(s, Rrot*(p-rotpt), Rrot*n);
            // ccle.addNormal (not implemented)
            sfe.addNormal(s, p, n);

			//vec3 px = Rcyl * p;
            zaccfe.addPosition(s, px);
            zacfe.addPosition(s, px);
            cfe.addPosition(s, p-coneCenter);
            rsqfe.addPosition(s, Rrot*(p-rotpt));
            double along = (p-rotpt)*rotaxis;
            ccle.addPosition(s, vec3((p-along*rotaxis-rotpt).length(), along, 0));
            sfe.addPosition(s, p);
		}
    }

    {
        vector<double> par_e(6,0), par_h(6,0), par_opt(6,0);  // biased fit parameters for ellipse and hyperbola
        zacfe.getParams(par_opt);
        getParams_Ellipse_Hyperbola(par_e, par_h, zacfe.M, zacfe.N, 6);
        vector<double> fpare(10,0), fparh(10,0), fparo(10,0);
        int xform[6] = {0,1,2,4,5,7};
        for (int i = 0; i < 6; i++) {
            fpare[xform[i]] = par_e[i];
            fparh[xform[i]] = par_h[i];
            fparo[xform[i]] = par_opt[i];
        }
        QuadricFieldFitter qfz;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                qfz.M[xform[i]*10+xform[j]] = zacfe.M[i*6+j];
                qfz.N[xform[i]*10+xform[j]] = zacfe.N[i*6+j];
            }
        }

        double *bestpar = &fparo[0];
        double *otherpar = &fparh[0];
        bool ellIsBest = true;
        if (4*fparo[7]*fparo[4]-fparo[5]*fparo[5] < 0) {
            otherpar = &fpare[0];
            ellIsBest = false;
        }

        double bestHyper, bestEll, bestPara, t_hyper, t_ell, t_para;
        qfz.lineSearch2D(bestpar, otherpar, bestHyper, bestEll, bestPara, t_hyper, t_ell, t_para);

        for (int i = 0; i < 10; i++) {
            fields[TYPE_ELL_CYL].q[i] = bestpar[i]+t_ell*otherpar[i];
            fields[TYPE_HYPER_CYL].q[i] = bestpar[i]+t_hyper*otherpar[i];
            fields[TYPE_PARA_CYL].q[i] = bestpar[i]+t_para*otherpar[i];
        }
    }

    //TYPE_GEN_CYL, TYPE_CIRC_CYL
    sfe.getQuadric(fields[TYPE_SPHERE]);
    zaccfe.getQuadric(fields[TYPE_CIRC_CYL]);
    zacfe.getQuadric(fields[TYPE_GEN_CYL]);
    cfe.getQuadric(fields[TYPE_CONE]);
    rsqfe.getQuadric(fields[TYPE_ROTSYM_QUADRIC]);
    ccle.getQuadric(fields[TYPE_CIRC_CONE]);



    fields[TYPE_GEN_CYL].transformQuadric(Rcyl.transpose());
    fields[TYPE_CIRC_CYL].transformQuadric(Rcyl.transpose());
    fields[TYPE_CONE].transformQuadric(coneCenter);
    fields[TYPE_ROTSYM_QUADRIC].transformQuadric(Rrot.transpose());
    fields[TYPE_ROTSYM_QUADRIC].transformQuadric(rotpt);
    fields[TYPE_CIRC_CONE].transformQuadric(Rrot.transpose());
    fields[TYPE_CIRC_CONE].transformQuadric(rotpt);
    fields[TYPE_ELL_CYL].transformQuadric(Rcyl.transpose());
    fields[TYPE_HYPER_CYL].transformQuadric(Rcyl.transpose());
    fields[TYPE_PARA_CYL].transformQuadric(Rcyl.transpose());

    // if gamma was zero, use ell. cyl. instead of cone
    if (acceptCylinderAsCone && fabs(ckparams[0]) < .00001) {
        for (int i = 0; i < 10; i++) {
            fields[TYPE_CONE].q[i] = fields[TYPE_ELL_CYL].q[i];
        }
    }


	// now fit hyperboloids with specific num sheets
	// first I need to figure out how many sheets the already-fit hyperboloid has;
	// currently I do this in a brute force way
	{
		mat3 Q = getQmat(fields[TYPE_HYPERBOLOID_OPT].q);
		Array2D<double> A(3,3);
		for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { A[i][j] = Q[i][j]; } }
		JAMA::Eigenvalue<double> eig(A);
		Array1D<double> S(3);
		eig.getRealEigenvalues(S);
		// first check if we're essentially at a paraboloid
		int zeroeigind = -1;
		for (int i = 0; i < 3; i++) {
			if (fabs(S[i]) < .000000001) {
				zeroeigind = i;break;
			}
		}
		int numSheets = 0;
		if (zeroeigind > -1) {
			double mul = 1;
			for (int i = 0; i < 3; i++) { if (i != zeroeigind) mul *= S[i]; }
			if (mul < 0) { // hyperbolic paraboloid -> treat as a 1 sheet hyperboloid
				numSheets = 1;
			} else { // elliptical paraboloid -> treat as a 2 sheet hyperboloid
				numSheets = 2;
			}
		} else { // not a paraboloid; check number of sheets the hard way
			Array2D<double> Rt, Dt; 
			eig.getV(Rt);
			eig.getD(Dt);
			mat3 A0, R, D;
			arraytomat(A, A0); 
			arraytomat(Rt, R); 
			arraytomat(Dt, D);
			mat3 RT = R.transpose();
			mat3 DP = D;
			for (int i = 0; i < 3; i++) {
				if (fabs(D[i][i])>.000001) {
					DP[i][i] = 1.0f/D[i][i];
				} else {
					DP[i][i] = 0;
				}
			}
			
			vec3 b( fields[TYPE_HYPERBOLOID_OPT].q[1],
					fields[TYPE_HYPERBOLOID_OPT].q[2],
					fields[TYPE_HYPERBOLOID_OPT].q[3]);
			vec3 trans = -.5f*DP*RT*b;
			vec3 rotb = RT*b;
			vec3 bnew = rotb-DP*D*rotb;
			double cnew = (trans * (D*trans)) + (rotb*trans) + fields[TYPE_HYPERBOLOID_OPT].q[0];

			int oppsignind = -1;
			int AA = 0, AB = 1, AC = 2; // axis ids; AC is always the oppositely-signed axis
			if (D[0][0]*D[1][1] < 0 || D[0][0]*D[2][2] < 0 || D[1][1]*D[2][2] < 0) { // if one of these has a different sign than the others
				for (int i = 0; i < 3; i++) {
					int j = (i+1)%3;
					if (D[i][i]*D[j][j] >= 0) { // look for the two with the same sign
						oppsignind = (j+1)%3; // choose the min axis as the other
						AA = i, AB = j, AC = oppsignind;
						break;
					}
				}
			}

			
			if (cnew*D[AC][AC] <= 0) { // best hyperboloid had two sheets
				numSheets = 2;
			} else {
				numSheets = 1;
			}
		}
		if (numSheets == 2) {
			fields[TYPE_HYPERBOLOID_2SHEET] = fields[TYPE_HYPERBOLOID_OPT];
			double coneErr = qff.getTaubinError(fields[TYPE_CONE].q);
			if (coneErr < qff.getTaubinError(fields[TYPE_PARABOLOID_HYPERBOLIC].q)) {
				fields[TYPE_HYPERBOLOID_1SHEET] = fields[TYPE_CONE];
			} else {
				fields[TYPE_HYPERBOLOID_1SHEET] = fields[TYPE_PARABOLOID_HYPERBOLIC];
			}
		} else { // best had one sheet
			fields[TYPE_HYPERBOLOID_1SHEET] = fields[TYPE_HYPERBOLOID_OPT];
			double coneErr = qff.getTaubinError(fields[TYPE_CONE].q);
			if (coneErr < qff.getTaubinError(fields[TYPE_PARABOLOID_ELLIPTICAL].q)) {
				fields[TYPE_HYPERBOLOID_2SHEET] = fields[TYPE_CONE];
			} else {
				fields[TYPE_HYPERBOLOID_2SHEET] = fields[TYPE_PARABOLOID_ELLIPTICAL];
			}
		}

	}


}


// helper function: computes the six order-4 dunavant points and weights
// a, b, c are the triangle vertex positions
// i is the index of the dunavant point (must be in range [0,5])
// out is the i'th dunavant point, wt is the corresponding dunavant weight
inline void computeIthDunavant(vec3 &a, vec3 &b, vec3 &c, int i, vec3 &out, double &wt) {
	// dunavant weights and coordinates (for polynomials of order 4)
	double w[6] = {0.223381589678011, 0.223381589678011, 0.223381589678011, 0.109951743655322, 0.109951743655322, 0.109951743655322};
	double x[6] = {0.10810301816807, 0.445948490915965, 0.445948490915965, 0.81684757298045896, 0.091576213509771007, 0.091576213509771007};
	double y[6] = {0.445948490915965, 0.445948490915965, 0.10810301816807, 0.091576213509771007, 0.091576213509771007, 0.81684757298045896};
	out = a*(1.0-x[i]-y[i]) + b*(x[i]) + c*(y[i]);
	wt = w[i];
}

// iterator to traverse dunavant points of a triangle mesh
class dunavantIterator
{
    public:
        dunavantIterator(TriangleMesh &m) : mesh(&m), triInd(0), subInd(0) { findStartTri(); computeData(); }
        dunavantIterator operator++() { increment(); return *this; }
        dunavantIterator operator++(int junk) { increment(); return *this; }
        data_pnw& operator*() { return data; }
        data_pnw* operator->() { return &data; }
		bool operator==(const dunavantIterator& rhs) { 
			return mesh == rhs.mesh && triInd == rhs.triInd && subInd == rhs.subInd; 
		}
        bool operator!=(const dunavantIterator& rhs) { 
			return mesh != rhs.mesh || triInd != rhs.triInd || subInd != rhs.subInd;  
		}
		static dunavantIterator end(TriangleMesh &m) {
			dunavantIterator it(m);
			it.triInd = (int)it.mesh->triangles.size();
			return it;
		}
    private:
        TriangleMesh *mesh;
		int triInd, subInd;
		data_pnw data;
    
        void findStartTri() {
            while (!mesh->triangleTags.empty() && mesh->triangleTags.size()==mesh->triangles.size()
                   && triInd < mesh->triangles.size()
                   && mesh->triangleTags[triInd] != mesh->activeTag) {
                triInd++;
            }
        }

		inline void increment() {
			if (done()) return;

			subInd++;
			if (subInd > 5) {
				subInd = 0;
				triInd++;
                
                //if the triangleTags vector is set up, skip over non-active triangles 
                while (!mesh->triangleTags.empty() && mesh->triangleTags.size()==mesh->triangles.size()
                       && triInd < mesh->triangles.size()
                       && mesh->triangleTags[triInd] != mesh->activeTag) {
                    triInd++;
                }
			}
			computeData();
		}
		inline bool done() {
			return triInd >= (int)mesh->triangles.size();
		}
		inline void computeData() {
			if (done()) {
				return;
			}

			computeIthDunavant(
				mesh->vertices[mesh->triangles[triInd].ind[0]], 
				mesh->vertices[mesh->triangles[triInd].ind[1]],
				mesh->vertices[mesh->triangles[triInd].ind[2]],
				subInd, data.p, data.w
				);
			data.n = mesh->triangleNormals[triInd];
		}
};	

//Fits all quadric types
void fitAllQuadricTypes(TriangleMesh &mesh, vector<Quadric> &allQuadrics) {
	fitQuadricHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), allQuadrics);
}
void fitAllQuadricTypes(vector<data_pnw> &data, vector<Quadric> &allQuadrics) {
	fitQuadricHelper(data.begin(), data.end(), allQuadrics);
}

//Fits a quadric of unconstrained type (via Taubin's method)
void fitGeneralQuadric(std::vector<data_pnw> &data, Quadric &quadric) {
    fitGeneralQuadricHelper(data.begin(), data.end(), quadric);
}
void fitGeneralQuadric(TriangleMesh &mesh, Quadric &quadric) {
    fitGeneralQuadricHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric);
}

//Fits a sphere
void fitSphere(std::vector<data_pnw> &data, Quadric &quadric) {
    fitSphereHelper(data.begin(), data.end(), quadric);
}
void fitSphere(TriangleMesh &mesh, Quadric &quadric) {
    fitSphereHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric);
}

//Fits an ellipsoid
void fitEllipsoid(std::vector<data_pnw> &data, Quadric &quadric) {
    fitEllipsoidHelper(data.begin(), data.end(), quadric);
}
void fitEllipsoid(TriangleMesh &mesh, Quadric &quadric) {
    fitEllipsoidHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric);
}

// fit a paraboloid
void fitParaboloid(std::vector<data_pnw> &data, Quadric &quadric) {
    fitParaboloidQuadricHelper(data.begin(), data.end(), quadric);
}
void fitParaboloid(TriangleMesh &mesh, Quadric &quadric) {
    fitParaboloidQuadricHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric);
}

//Fits a hyperboloid
void fitHyperboloid(std::vector<data_pnw> &data, Quadric &quadric) {
    fitHyperboloidHelper(data.begin(), data.end(), quadric);
}
void fitHyperboloid(TriangleMesh &mesh, Quadric &quadric) {
    fitHyperboloidHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric);
}

//Fits a general cylinder (invariant in one direction, cross section may be any 2D conic)
void fitGeneralCylinder(std::vector<data_pnw> &data, Quadric &quadric) {
    fitGenCylinderQuadricHelper(data.begin(), data.end(), quadric);
}
void fitGeneralCylinder(TriangleMesh &mesh, Quadric &quadric) {
    fitGenCylinderQuadricHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric);
}

//Fits a circular cylinder (cylinder with circular cross section)
void fitCircularCylinder(std::vector<data_pnw> &data, Quadric &quadric){
    fitCircCylinderQuadricHelper(data.begin(), data.end(), quadric);
}
void fitCircularCylinder(TriangleMesh &mesh, Quadric &quadric) {
    fitCircCylinderQuadricHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric);
}

//Fits a general cone (cross sections perpendicular to axis may be ellipses, not circles)
// if center would be off at infinity, we can allow a cylinder as the fit
// not allowing cylinders forces the next best cone fit to be used in this case (with center not off at infinity)
void fitGeneralCone(std::vector<data_pnw> &data, Quadric &quadric, bool allowCylinder) {
    fitGenConeQuadricHelper(data.begin(), data.end(), quadric, allowCylinder);
}
void fitGeneralCone(TriangleMesh &mesh, Quadric &quadric, bool allowCylinder) {
    fitGenConeQuadricHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric, allowCylinder);
}

//Fits a circular cone (cross section perpendicular to axis is a circle)
void fitCircularCone(std::vector<data_pnw> &data, Quadric &quadric) {
    fitCircConeQuadricHelper(data.begin(), data.end(), quadric);
}
void fitCircularCone(TriangleMesh &mesh, Quadric &quadric) {
    fitCircConeQuadricHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric);
}

double fitParaboloidOrEllipsoid(std::vector<data_pnw> &data, Quadric &quadric, bool &isEllipsoid) {
    return fitParaboloidOrEllipsoidHelper(data.begin(), data.end(), quadric, isEllipsoid);
}
double fitParaboloidOrEllipsoid(TriangleMesh &mesh, Quadric &quadric, bool &isEllipsoid) {
    return fitParaboloidOrEllipsoidHelper(dunavantIterator(mesh), dunavantIterator::end(mesh), quadric, isEllipsoid);
}


// helper functions to build a triangle mesh of a quadric surface:
// ------------------------------------------------------------------

// makes elliptical rings of vertices
// returns vertex at start of ellipse
int makeEllipse(TriangleMesh &m, double w, double h, int AA, int AB, int AC, double z, int numpts) {
    vec3 X(0), Y(0), Z(0);
    X[AA] = w; Y[AB] = h; Z[AC] = z;
    double step = 2*M_PI / double(numpts);
    int elstart = (int)m.vertices.size();
    for (double th = 0; th < 2*M_PI; th += step) {
        m.vertices.push_back(Z + X*cos(th) + Y*sin(th));
    }
    return elstart;
}

// make triangles between ellipses and pts ...
void connectEllipses(TriangleMesh &m, int starta, int startb, int numpts) {
    for (int i = 0; i < numpts; i++) {
        int j = (i+1)%numpts;
		m.triangles.push_back(Tri(starta+i, startb+i, starta+j));
        m.triangles.push_back(Tri(startb+i, startb+j, starta+j));
    }
}
void connectPtEllipse(TriangleMesh &m, int startpt, int startel, int side, int numpts) {
    for (int i = 0; i < numpts; i++) {
        int j = (i+1)%numpts;
        if (!side)
            m.triangles.push_back(Tri(startpt, startel+i, startel+j));
        else
            m.triangles.push_back(Tri(startel+i, startpt, startel+j));
    }
}
void closeEllipse(TriangleMesh &m, int startel, int side, int numpts) {
    for (int i = 1; i < numpts-1; i++) {
        int j = i+1;
        if (side)
            m.triangles.push_back(Tri(startel, startel+i, startel+j));
        else
            m.triangles.push_back(Tri(startel+i, startel, startel+j));
    }
}

vec2 getRangeForVec_postx(vec3 &bmin, vec3 &bmax, vec3 p, vec3 dir) {
    vec3 b[2]; b[0] = bmin; b[1] = bmax;
    vec2 range(FLT_MAX, -FLT_MAX);
    for (int ii = 0; ii < 2; ii++) {
        vec3 pt(0); pt[0] = b[ii][0];
        for (int jj = 0; jj < 2; jj++) {
            pt[1] = b[jj][1];
            for (int kk = 0; kk < 2; kk++) {
                pt[2] = b[kk][2];
                double along = (pt-p) * dir;
                if (along < range[0]) range[0] = along;
                if (along > range[1]) range[1] = along;
            }
        }
    }
    return range;
}

vec2 getRangeForDim(vec3 &bmin, vec3 &bmax, int dim, mat3& R, vec3& offset) {
    vec3 dir(0); dir[dim] = 1;
    dir = R*dir;
    dir.normalize();
    vec3 p = R*offset;
    return getRangeForVec_postx(bmin, bmax, p, dir);
}
vec2 getRangeForDim(vec3 &bmin, vec3 &bmax, vec3 p, vec3 dir, mat3& R, vec3& offset) {
    dir = R*dir;
    dir.normalize();
    p = R*(p+offset);
    return getRangeForVec_postx(bmin, bmax, p, dir);
}


void Quadric::buildMeshFromQuadric(TriangleMesh &mesh, vec3 &bmin, vec3 &bmax, 
								   double planethresh, double conethresh, double cylinderthresh, bool snapParams, 
								   int ptsPerEllipse, double zStepSize) {
    mesh.clear();

    if (q[0] != q[0]) { // nan quadrics are dumb
		return;
	}

    bool closeQuadrics = false; // if set to true, I try to close off all unbounded quadric meshes to make watertight volumes; not sure this is implemented fully yet though.

    Array2D<double> A(3,3);
    A[0][0] = q[4];
    A[1][0] = A[0][1] = q[5]*.5;
    A[2][0] = A[0][2] = q[6]*.5;
    A[1][1] = q[7];
    A[1][2] = A[2][1] = q[8]*.5;
    A[2][2] = q[9];
    vec3 b(q[1],q[2],q[3]);

	JAMA::Eigenvalue<double> eig(A);
	Array2D<double> Rt, Dt; 
    eig.getV(Rt);
    eig.getD(Dt);
    mat3 A0, R, D;
    arraytomat(A, A0); 
    arraytomat(Rt, R); 
    arraytomat(Dt, D);
    double rdet = R.det();
    if (rdet < 0) {
        mat3 F = mat3(vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));
        
        mat3 RT = R.transpose();
        mat3 A0d = A0-R*F*D*F*RT;
        mat3 A0d2 = A0-R*D*RT;
        mat3 A0d22 = A0-R*D*RT;
        R = R*F;
        RT = R.transpose();
        mat3 A0d3 = A0-R*D*RT;
        //cout << A0d << " " << A0d2 << A0d22 << " " << A0d3 << endl;
    }
    double rdet2 = R.det();
    mat3 RT = R.transpose();
    int nonzeros = 0;
    int anonzero = -1;

    // make psuedo-inverse of D
    mat3 DP = D;
    bool nonzero[3];
    for (int i = 0; i < 3; i++) {
        if (fabs(D[i][i])>cylinderthresh) {
            nonzeros++;
            DP[i][i] = 1.0f/D[i][i];
            nonzero[i] = true;
            anonzero = i;
        } else {
            DP[i][i] = 0;
            nonzero[i] = false;
        }
    }

    vec3 trans = -.5f*DP*RT*b;

    vec3 rotb = RT*b;
    vec3 bnew = rotb-DP*D*rotb;
    double cnew = (trans * (D*trans)) + (rotb*trans) + q[0];




    // transform the quadric parameters and rescale them
    Quadric qftemp = *this;
    qftemp.transformQuadric(RT);
    qftemp.transformQuadric(-trans);
    double maxq = .001f;
    for (int i = 4; i < 10; i++) { // find max of squared terms for normalization
        double qi = fabs(qftemp.q[i]);
        if (qi > maxq) {
            maxq = qi;
        }
    }
    if (maxq < .00101f) { // normalize via linear terms if squared terms were all too small
        for (int i = 1; i < 4; i++) { 
            double qi = fabs(qftemp.q[i]);
            if (qi > maxq) {
                maxq = qi;
            }
        }
    } // (don't bother with constant term; if linear and squared terms were all vanishing, quadric likely degenerate.)
    double scaleparams = 1.0f/maxq;
    D *= scaleparams;
    cnew *= scaleparams;
    rotb *= scaleparams;
    bnew *= scaleparams;



    // recompute the pseudo inverse with the rescaled parameters; use the rescaled params for any subsequent thresholding
    DP = D;
    nonzeros = 0;
    for (int i = 0; i < 3; i++) {
        if (nonzero[i]) {
            nonzeros++;
            DP[i][i] = 1.0f/D[i][i];
        } else {
            // transfer excess into cnew
            vec2 range = getRangeForDim(bmin, bmax, i, R, trans);
            double v = .5*(range[0]+range[1]);
            cnew += v*v*D[i][i];

            // snap to zero
            D[i][i] = 0;
            DP[i][i] = 0;
        }
    }
    for (int ii = 0; ii < 3; ii++) {
        if (fabs(bnew[ii]) < cylinderthresh) {
            // transfer the value of this dim of bnew into the constant term, for the midpoint of the bounding box
            vec2 range = getRangeForDim(bmin, bmax, ii, R, trans);
            double v = .5*(range[0]+range[1]);
            cnew += bnew[ii]*v;

            // snap this dim of bnew to zero
            bnew[ii] = 0;

        }
    }




    int oppsignind = -1;
    int AA = 0, AB = 1, AC = 2; // axis ids; AC is always the oppositely-signed axis
    if (D[0][0]*D[1][1] < 0 || D[0][0]*D[2][2] < 0 || D[1][1]*D[2][2] < 0) { // if one of these has a different sign than the others
        for (int i = 0; i < 3; i++) {
            int j = (i+1)%3;
            if (D[i][i]*D[j][j] >= 0) { // look for the two with the same sign
                oppsignind = (j+1)%3; // choose the min axis as the other
                AA = i, AB = j, AC = oppsignind;
                break;
            }
        }
    }

    
    if (snapParams) {
        if (fabs(cnew) < conethresh) q[0] = 0.0f;
        else q[0] = cnew;
        for (int i = 0; i < 3; i++) {
            q[1+i] = bnew[i];
        }
        for (int i=4; i<10; i++) {
            q[i] = 0;
        }
        q[4] = D[0][0]; q[7] = D[1][1]; q[9] = D[2][2];
        transformQuadric(trans);
        transformQuadric(R);
    }

    

    // fully non-degen case: all squared coeffs are non-zero
    if (nonzeros == 3) {
        int axis = AC;
        if (oppsignind > -1) {
            axis = oppsignind; // wait, isn't AC already going to be oppsignind in this case?
        }

        if (oppsignind > -1) {
            if (fabs(cnew) < conethresh) {
                // we have a cone!
                int ptvert = (int)mesh.vertices.size();
                mesh.vertices.push_back(vec3(0));
                //for (int side = 0; side < 2; side++) { // handle both sides of the cone
                    vec2 size = getRangeForDim(bmin, bmax, AC, R, trans);
                    vec2 coff(cnew + D[axis][axis]*size[0]*size[0], cnew + D[axis][axis]*size[1]*size[1]);
                    int starta = makeEllipse(mesh, sqrt(fabs(coff[1]/D[AA][AA])), sqrt(fabs(coff[1]/D[AB][AB])), AA, AB, AC, size[1], ptsPerEllipse);
                    int startb = makeEllipse(mesh, sqrt(fabs(coff[0]/D[AA][AA])), sqrt(fabs(coff[0]/D[AB][AB])), AA, AB, AC, size[0], ptsPerEllipse);
                    connectPtEllipse(mesh, ptvert, starta, 1, ptsPerEllipse);
                    connectPtEllipse(mesh, ptvert, startb, 0, ptsPerEllipse);
                    if (closeQuadrics) {
                        closeEllipse(mesh, starta, 1, ptsPerEllipse);
                        closeEllipse(mesh, startb, 0, ptsPerEllipse);
                    }
                //}
            } else if (cnew*D[AC][AC] < 0) { // hyperboloid of two sheets
                double size = sqrt(-cnew/D[AC][AC]);
                vec2 range = getRangeForDim(bmin, bmax, AC, R, trans);
                double zstep = zStepSize*(range[1]-range[0]);
                for (int side = 0; side < 2; side++) {
                    //if (g_optk.quadricMeshingSkipSide == side+1) continue;
                    int sideSign = -1+2*side;

                    vec3 Ztip(0); Ztip[AC] = sideSign*size;
                    if (sideSign*Ztip[AC] > sideSign*range[side]) {
                        continue;
                    }

                    int lastEl = -1;
                    for (double z = range[side]; z*sideSign > Ztip[AC]*sideSign; z += -1*zstep*sideSign) {
                        double coff = cnew + D[AC][AC]*z*z;
                        int nextEl = makeEllipse(mesh, sqrt(fabs(coff/D[AA][AA])), sqrt(fabs(coff/D[AB][AB])), AA, AB, AC, z, ptsPerEllipse);
                        if (lastEl > -1) {
                            if (!side) connectEllipses(mesh, lastEl, nextEl, ptsPerEllipse);
                            else connectEllipses(mesh, nextEl, lastEl, ptsPerEllipse);
                        }
                        lastEl = nextEl;
                    }
                    int tipVert = (int)mesh.vertices.size();
                    mesh.vertices.push_back(Ztip);
                    connectPtEllipse(mesh, tipVert, lastEl, !side, ptsPerEllipse);
                }
            } else { // hyperboloid of one sheet

                vec2 size = getRangeForDim(bmin, bmax, AC, R, trans);
                double zstep = zStepSize*(size[1]-size[0]);
                int lastEl = -1;
                for (double z = size[0]; z < size[1]; z += zstep) {
                    double coff = cnew + D[AC][AC]*z*z;
                    int nextEl = makeEllipse(mesh, sqrt(fabs(coff/D[AA][AA])), sqrt(fabs(coff/D[AB][AB])), AA, AB, AC, z, ptsPerEllipse);
                    if (lastEl > -1) connectEllipses(mesh, lastEl, nextEl, ptsPerEllipse);
                    lastEl = nextEl;
                }
            }
        } else { // ellipsoid 
            double size = sqrt(-cnew/D[AC][AC]);
            int centere = makeEllipse(mesh, sqrt(fabs(cnew/D[AA][AA])), sqrt(fabs(cnew/D[AB][AB])), AA, AB, AC, 0, ptsPerEllipse);
            for (int side = 0; side < 2; side++) {
                int sideSign = -1+2*side;
                int lastEl = centere;
                double zstep = zStepSize*size;
                for (double z = zstep; z < size-zstep*.5; z += zstep) {
                    double coff = cnew + D[AC][AC]*z*z;
                    int nextEl = makeEllipse(mesh, sqrt(fabs(coff/D[AA][AA])), sqrt(fabs(coff/D[AB][AB])), AA, AB, AC, sideSign*z, ptsPerEllipse);
                    if (!side) connectEllipses(mesh, lastEl, nextEl, ptsPerEllipse);
                    else connectEllipses(mesh, nextEl, lastEl, ptsPerEllipse);
                    lastEl = nextEl;
                }
                vec3 Z(0); Z[AC] = sideSign*size;
                int tipVert = (int)mesh.vertices.size();
                mesh.vertices.push_back(Z);
                connectPtEllipse(mesh, tipVert, lastEl, !side, ptsPerEllipse);
            }
        }
    } else if (nonzeros == 0) { // it's a bird ... no, a plane!
        vec3 n = bnew;
        vec3 p = (-cnew/n.length2()) * n;
        n.normalize();
        vec3 X = findOrtho(n); X.normalize();
        vec3 Y = X % n; Y.normalize();
        vec2 Xrange = getRangeForDim(bmin, bmax, p, X, R, trans);
        vec2 Yrange = getRangeForDim(bmin, bmax, p, Y, R, trans);
        vec2 Zrange = getRangeForDim(bmin, bmax, p, n, R, trans);
        int v0 = (int)mesh.vertices.size();
        int v1 = v0+1, v2 = v0+2, v3 = v0+3;
        int v4 = v0+4, v5 = v0+5, v6 = v0+6, v7 = v0+7;
        mesh.vertices.push_back((p + X*Xrange[1] + Y*Yrange[1]));
        mesh.vertices.push_back((p + X*Xrange[0] + Y*Yrange[1]));
        mesh.vertices.push_back((p + X*Xrange[0] + Y*Yrange[0]));
        mesh.vertices.push_back((p + X*Xrange[1] + Y*Yrange[0]));

        // the plane
        mesh.triangles.push_back(Tri(v1,v0,v2));
        mesh.triangles.push_back(Tri(v2,v0,v3));

        if (closeQuadrics) {
            mesh.vertices.push_back((p + X*Xrange[1] + Y*Yrange[1] + n*Zrange[0]));
            mesh.vertices.push_back((p + X*Xrange[0] + Y*Yrange[1] + n*Zrange[0]));
            mesh.vertices.push_back((p + X*Xrange[0] + Y*Yrange[0] + n*Zrange[0]));
            mesh.vertices.push_back((p + X*Xrange[1] + Y*Yrange[0] + n*Zrange[0]));

            // top
            mesh.triangles.push_back(Tri(v6,v4,v5));
            mesh.triangles.push_back(Tri(v4,v6,v7));
            // cube sides
            mesh.triangles.push_back(Tri(v0,v1,v5));
            mesh.triangles.push_back(Tri(v0,v5,v4));
            mesh.triangles.push_back(Tri(v1,v2,v5));
            mesh.triangles.push_back(Tri(v5,v2,v6));
            mesh.triangles.push_back(Tri(v2,v3,v7));
            mesh.triangles.push_back(Tri(v2,v7,v6));
            mesh.triangles.push_back(Tri(v3,v0,v4));
            mesh.triangles.push_back(Tri(v3,v4,v7));
        }

    } else {
        // check for cylinder case: bnew[i] == 0 and D[i][i] == 0
        int zerod = -1;
        int bnonzeros = 0;
        int bnonzerod = -1;
        int bnon_dzero = -1;
        for (int ii = 0; ii < 3; ii++) {
            if (bnew[ii] != 0) { bnonzeros++; bnonzerod = ii; }
            if (bnew[ii] == 0 && D[ii][ii] == 0) {
                zerod = ii;
            }
            if (bnew[ii] != 0 && D[ii][ii] == 0) {
                bnon_dzero = ii;
            }
        }
        if (zerod > -1) { // it's a cylinder of some type
            
            int AX = (zerod+1)%3, AY = (zerod+2)%3;
            int AZ = zerod;

            int numcurves = 0;
            if (nonzeros == 1) {
                if (bnonzeros == 0) { // two plane quadric
                    AZ = anonzero;
                    AX = (anonzero+1)%3;
                    AY = (anonzero+2)%3;

                    vec3 n(0); n[AZ] = 1;
                    vec3 p(0); p[AZ] = sqrt(-cnew/D[AZ][AZ]);
                    n.normalize();
                    vec3 X = findOrtho(n); X.normalize();
                    vec3 Y = X % n; Y.normalize();
                    vec2 Xrange = getRangeForDim(bmin, bmax, p, X, R, trans);
                    vec2 Yrange = getRangeForDim(bmin, bmax, p, Y, R, trans);
                    //vec2 Zrange = getRangeForDim(bmin, bmax, p, n, R, trans);
                    int v0 = (int)mesh.vertices.size();
                    int v1 = v0+1, v2 = v0+2, v3 = v0+3;
                    int v4 = v0+4, v5 = v0+5, v6 = v0+6, v7 = v0+7;
                    mesh.vertices.push_back((p + X*Xrange[1] + Y*Yrange[1]));
                    mesh.vertices.push_back((p + X*Xrange[0] + Y*Yrange[1]));
                    mesh.vertices.push_back((p + X*Xrange[0] + Y*Yrange[0]));
                    mesh.vertices.push_back((p + X*Xrange[1] + Y*Yrange[0]));

                    p = -p;
                    mesh.vertices.push_back((p + X*Xrange[1] + Y*Yrange[1]));
                    mesh.vertices.push_back((p + X*Xrange[0] + Y*Yrange[1]));
                    mesh.vertices.push_back((p + X*Xrange[0] + Y*Yrange[0]));
                    mesh.vertices.push_back((p + X*Xrange[1] + Y*Yrange[0]));

                    // positive plane
                    mesh.triangles.push_back(Tri(v1,v0,v2));
                    mesh.triangles.push_back(Tri(v2,v0,v3));
                    // negative plane
                    mesh.triangles.push_back(Tri(v6,v4,v5));
                    mesh.triangles.push_back(Tri(v4,v6,v7));

                    if (closeQuadrics) {
                        // box sides
                        mesh.triangles.push_back(Tri(v0,v1,v5));
                        mesh.triangles.push_back(Tri(v0,v5,v4));
                        mesh.triangles.push_back(Tri(v1,v2,v5));
                        mesh.triangles.push_back(Tri(v5,v2,v6));
                        mesh.triangles.push_back(Tri(v2,v3,v7));
                        mesh.triangles.push_back(Tri(v2,v7,v6));
                        mesh.triangles.push_back(Tri(v3,v0,v4));
                        mesh.triangles.push_back(Tri(v3,v4,v7));
                    }

                    
                } else {
                    AY = bnon_dzero;
                    AX = (bnon_dzero+1)%3;
                    if (AX == AZ) AX = (AZ+1)%3;
                    vec2 size = getRangeForDim(bmin, bmax, AX, R, trans);
                    vec2 zsize = getRangeForDim(bmin, bmax, AZ, R, trans);
                    double xstep = zStepSize*(size[1]-size[0]);
                    int npts = 0;
                    for (double x = size[0]; x < size[1]; x+= xstep) {
                        vec3 p(0);
                        p[AX] = x;
                        p[AY] = -(D[AX][AX]*x*x+bnew[AX]*x+cnew)/bnew[AY];
                        p[AZ] = zsize[0];
                        mesh.vertices.push_back(p);
                        p[AZ] = zsize[1];
                        mesh.vertices.push_back(p);
                        npts++;
                    }
                    for (int i = 0; i+1 < npts; i++) {
                        mesh.triangles.push_back(Tri(i*2, i*2+1, (i+1)*2));
                        mesh.triangles.push_back(Tri((i+1)*2, i*2+1, (i+1)*2+1));
                    }
                    
                }
            } else if (oppsignind == -1) { // elliptical cylinder
                
                numcurves = 1;
                double w = sqrt(-cnew/D[AX][AX]);
                double h = sqrt(-cnew/D[AY][AY]);
                vec2 size = getRangeForDim(bmin, bmax, AZ, R, trans);
                int etop = makeEllipse(mesh, w, h, AX, AY, AZ, size[1], ptsPerEllipse);
                int ebot = makeEllipse(mesh, w, h, AX, AY, AZ, size[0], ptsPerEllipse);
                vec3 X(0), Y(0), Z(0);
                X[AX] = w; Y[AY] = h; Z[AZ] = size[0];
                

                connectEllipses(mesh, etop, ebot, ptsPerEllipse);
                if (closeQuadrics) {
                    closeEllipse(mesh, etop, 1, ptsPerEllipse);
                    closeEllipse(mesh, ebot, 0, ptsPerEllipse);
                }
            } else { // hyperbolic cylinder
                if (fabs(cnew) < planethresh) { // two plane quadric
                    // I don't actually handle this as a special case currently; the hyperbolic cylinder case will handle it, albeit with more facets than needed
                }
                if (cnew / D[AX][AX] < 0) {
                    int temp = AX;
                    AX = AY;
                    AY = temp;
                }
                vec2 size = getRangeForDim(bmin, bmax, AX, R, trans);
                vec2 zsize = getRangeForDim(bmin, bmax, AZ, R, trans);
                double xstep = zStepSize*(size[1]-size[0]);
                int npts = 0;
                for (double x = size[0]; x < size[1]; x+= xstep) {
                    vec3 p(0);
                    for (int side = 0; side < 2; side++) {
                        int sidesign = -1+2*side;
                        
                        p[AX] = x;
                        p[AY] = sidesign*sqrt((-D[AX][AX]*x*x-cnew)/D[AY][AY]);
                        p[AZ] = zsize[0];
                        mesh.vertices.push_back(p);
                        p[AZ] = zsize[1];
                        mesh.vertices.push_back(p);
                    }
                    npts++;
                }
                for (int i = 0; i+1 < npts; i++) {
                    mesh.triangles.push_back(Tri(i*4, i*4+1, (i+1)*4));
                    mesh.triangles.push_back(Tri((i+1)*4, i*4+1, (i+1)*4+1));
                    mesh.triangles.push_back(Tri(i*4 +2, i*4+1 +2, (i+1)*4 +2));
                    mesh.triangles.push_back(Tri((i+1)*4 +2, i*4+1 +2, (i+1)*4+1 +2));
                }
            }
        } else { 
            int AX = (bnon_dzero+1)%3, AY = (bnon_dzero+2)%3;
            int AZ = bnon_dzero;
            
            // there's some dim for which bnew =/= 0 (implies also: D *is* zero for that dim)
            // -> rephrase eqn as a fn of that dim and render as height field

            vec2 sizex = getRangeForDim(bmin, bmax, AX, R, trans);
            vec2 sizey = getRangeForDim(bmin, bmax, AY, R, trans);

            int v0 = (int)mesh.vertices.size();

            double xstep = zStepSize * (sizex[1]-sizex[0]);
            double ystep = zStepSize * (sizey[1]-sizey[0]);
            int nx = 0, ny = 0;
            for (double x = sizex[0]; x < sizex[1]; x += xstep) {
                nx++; ny = 0;
                for (double y = sizey[0]; y < sizey[1]; y+= ystep) {
                    double zval = -(x*x*D[AX][AX]+y*y*D[AY][AY]+x*bnew[AX]+y*bnew[AY]+cnew) / bnew[AZ];
                    vec3 p(0); p[AX] = x; p[AY] = y; p[AZ] = zval;
                    mesh.vertices.push_back((p));
                    ny++;
                }
            }

            for (int xi = 0; xi+1 < nx; xi++) {
                for (int yi = 0; yi+1 < ny; yi++) {
                    mesh.triangles.push_back(Tri(v0+xi*ny+yi, v0+(xi+1)*ny+yi, v0+xi*ny+yi+1));
                    mesh.triangles.push_back(Tri(v0+xi*ny+yi+1, v0+(xi+1)*ny+yi, v0+(xi+1)*ny+yi+1));
                }
            }
        }
    }

    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        mesh.vertices[i] += trans;
        mesh.vertices[i] = R * mesh.vertices[i];
    }

    mesh.computeNormals();
}


namespace { 
    int getNValues(stringstream &ss, vector<int> &values, char delim = '/') {
	    values.clear();
	    string sblock;
	    if (ss >> sblock) {
		    stringstream block(sblock);
		    string s;
		    int value;
		    while (getline(block, s, delim)) {
			    stringstream valuestream(s);
			    if (valuestream >> value)
				    values.push_back(value);
                else
                    values.push_back(-1);
		    }
	    }
	    return (int)values.size();
    }
}


bool TriangleMesh::loadObj(const char *file) {
	vertices.clear(); triangles.clear(); triangleNormals.clear();

	ifstream f(file);
	if (!f) {
		cerr << "Couldn't open file: " << file << endl;
		return false;
	}
	string line;
	vector<int> first, second, third;
	vector<int> orig;
	while (getline(f,line)) {
		if (line.empty())
			continue;
		stringstream ss(line);
		string op;
		ss >> op;
		if (op.empty() || op[0] == '#')
			continue;
		if (op == "v") {
			vec3 v;
			ss >> v;
			vertices.push_back(v);
		}
		if (op == "f")
		{
            if (!getNValues(ss, first))
				continue;
            if (!getNValues(ss, second))
                continue;
			while (getNValues(ss, third)) {
                triangles.push_back(Tri(first[0]-1, second[0]-1, third[0]-1));
                second = third;
			}
		}
	}

	computeNormals();

	return true;
}



} // namespace allquadrics

