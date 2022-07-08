// type-specific quadric fitting library
// by James Andrews (zaphos@gmail.com)

// this is 'research code': it's not thoroughly tested and it's not optimized for performance
// report bugs and feature requests to: zaphos@gmail.com

// overview: call one of the fit*() functions on your data to get direct fits of all quadric types
// currently your data must be a vector of data_pnw objects, or a TriangleMesh

// for best results, please center and scale your data before calling the fitting functions

#ifndef QUADRICFITTING_H
#define QUADRICFITTING_H



#include "algebra3.h"

#include <vector>

namespace allquadrics {
    
// THE DATA STRUCTURES (pre-declarations)

// structure containing data points (with point, normal, and a 'weight' indicating how much it contributes to the fit (e.g. the dunavant weights go here)
struct data_pnw;
// structure for representing a triangle mesh; used to fit quadric surfaces to a mesh and to export quadric surfaces as meshes
// the fitting functions use dunavant quadrature on the triangles of the mesh to integrate error over the surface
// note: you can optionally use triangleTags to filter which triangles are used for the fit (e.g. if you want to fit selections on a given mesh, use this to indicate the selection.)
struct TriangleMesh;

// structure defining a quadric surface, with helpful member functions for evaluating, transforming and meshing that surface
struct Quadric;


// THE FITTING FUNCTIONS!
    
//Fits all quadric types, and returns them in a giant vector
void fitAllQuadricTypes(std::vector<data_pnw> &data, std::vector<Quadric> &quadrics);
void fitAllQuadricTypes(TriangleMesh &mesh, std::vector<Quadric> &quadrics);

//Fits a quadric of unconstrained type (via Taubin's method)
void fitGeneralQuadric(std::vector<data_pnw> &data, Quadric &quadric);
void fitGeneralQuadric(TriangleMesh &mesh, Quadric &quadric);

//Fits a sphere
void fitSphere(std::vector<data_pnw> &data, Quadric &quadric);
void fitSphere(TriangleMesh &mesh, Quadric &quadric);

//Fits an ellipsoid
void fitEllipsoid(std::vector<data_pnw> &data, Quadric &quadric);
void fitEllipsoid(TriangleMesh &mesh, Quadric &quadric);

//Fits a paraboloid
void fitParaboloid(std::vector<data_pnw> &data, Quadric &quadric);
void fitParaboloid(TriangleMesh &mesh, Quadric &quadric);

//Fits a paraboloid or an ellipsoid, whichever is best, and returns the Taubin error and (by reference) whether the fit was an ellipsoid
double fitParaboloidOrEllipsoid(std::vector<data_pnw> &data, Quadric &quadric, bool &isEllipsoid);
double fitParaboloidOrEllipsoid(TriangleMesh &mesh, Quadric &quadric, bool &isEllipsoid);

//Fits a hyperboloid
void fitHyperboloid(std::vector<data_pnw> &data, Quadric &quadric);
void fitHyperboloid(TriangleMesh &mesh, Quadric &quadric);

//Fits a general cylinder (invariant in one direction, cross section may be any 2D conic)
void fitGeneralCylinder(std::vector<data_pnw> &data, Quadric &quadric);
void fitGeneralCylinder(TriangleMesh &mesh, Quadric &quadric);

//Fits a circular cylinder (cylinder with circular cross section)
void fitCircularCylinder(std::vector<data_pnw> &data, Quadric &quadric);
void fitCircularCylinder(TriangleMesh &mesh, Quadric &quadric);

//Fits a general cone (cross sections perpendicular to axis may be ellipses, not circles)
// if center would be off at infinity, we can allow a cylinder as the fit
// not allowing cylinders forces the next best cone fit to be used in this case (with center not off at infinity)
void fitGeneralCone(std::vector<data_pnw> &data, Quadric &quadric, bool allowCylinder = false);
void fitGeneralCone(TriangleMesh &mesh, Quadric &quadric, bool allowCylinder = false);

//Fits a circular cone (cross section perpendicular to axis is a circle)
void fitCircularCone(std::vector<data_pnw> &data, Quadric &quadric);
void fitCircularCone(TriangleMesh &mesh, Quadric &quadric);


// IDs for all quadric types we fit; the fitAllQuadrics functions return the quadrics in this order
enum { TYPE_GENERAL_QUADRIC = 0, TYPE_ROTSYM_QUADRIC, TYPE_PLANE, TYPE_SPHERE,                  // 0-3
    TYPE_GEN_CYL, TYPE_CIRC_CYL, TYPE_CONE, TYPE_CIRC_CONE,                                     // 4-7
    TYPE_ELLIPSOID_BIASED, TYPE_HYPERBOLOID_BIASED, TYPE_ELLIPSOID_OPT, TYPE_HYPERBOLOID_OPT,   // 8-11
    TYPE_HYPERBOLOID_1SHEET, TYPE_HYPERBOLOID_2SHEET, TYPE_PARABOLOID,                          // 12-14
    TYPE_PARABOLOID_ELLIPTICAL, TYPE_PARABOLOID_HYPERBOLIC,                                     // 15-16
    TYPE_ELL_CYL, TYPE_HYPER_CYL, TYPE_PARA_CYL,                                                // 17-19
    NUM_QUADRIC_TYPES };


    

// THE DATA STRUCTURES (full definitions)


// a data point with position, normal and weight
struct data_pnw {
	vec3 p, n;
	double w;

	data_pnw(vec3 &p, vec3 &n, double w) : p(p), n(n), w(w) {}
	data_pnw() {}
};

// stores triangles
struct Tri {
	int ind[3];
	Tri(int a, int b, int c) { ind[0] = a; ind[1] = b; ind[2] = c; }
	Tri() { ind[0] = ind[1] = ind[2] = 0; }
};

// stores triangle meshes
struct TriangleMesh {
	
	std::vector<Tri> triangles;
	std::vector<vec3> triangleNormals;
	std::vector<vec3> vertices;
    
    // use the tags to filter what gets fit
    // if triangleTag.size()!=triangles.size(), we fit everything
    // otherwise, we only fit the triangles that have a tag matching activeTag
    std::vector<int> triangleTags;
    int activeTag; 

	void computeNormals() {
		triangleNormals.resize(triangles.size());
		for (size_t i = 0; i < triangles.size(); i++) {
			vec3 edge1 = vertices[triangles[i].ind[1]] - vertices[triangles[i].ind[0]];
			vec3 edge2 = vertices[triangles[i].ind[2]] - vertices[triangles[i].ind[1]];
			triangleNormals[i] = edge1 % edge2;
			triangleNormals[i].normalize();
		}
	}
	void clear() {
		triangles.clear(); triangleNormals.clear(); vertices.clear();
	}
	bool loadObj(const char *file);

	void centerAndScale(double scale) {
		if (vertices.empty())
			return;

		vec3 maxp = vertices[0], minp = vertices[0];
		for (vector<vec3>::iterator it = vertices.begin(); it != vertices.end(); ++it) {
			maxp = max(*it, maxp); // max and min def'd in algebra3.h take the max or min componentwise from each vector
			minp = min(*it, minp);
		}
		vec3 center = (maxp+minp)*.5;
		vec3 size = maxp-minp;
		double maxSizeInv = MAX(size[0],MAX(size[1],size[2]));
		if (maxSizeInv == 0) // mesh is just one point
			return;
		maxSizeInv = 1.0/maxSizeInv;
		for (vector<vec3>::iterator it = vertices.begin(); it != vertices.end(); ++it) {
			*it = (*it-center)*maxSizeInv*scale;
		}
	}

};

// Class to represent a quadric in general form, using 10 parameters
// includes helpful functions to transform the quadric shape, and to produce a mesh of the quadric surface
struct Quadric {
    double q[10];

	Quadric()  { clear(); }
	void clear() { for (int i = 0; i < 10; i++) { q[i] = 0; } }


	// build's a triangle mesh describing the quadric surface; useful for visualization of your results
	// min and max define the range that the quadric should fill, if it's an unbounded quadric surface
	// (for bounded surfaces, i.e. ellipsoids, we build the whole surface; for unbounded surfaces this is of course not possible!)
	// the *thresh and snap* parameters dictate how close the parameters need to be to various quadric subtypes to snap to those subtypes
	void buildMeshFromQuadric(TriangleMesh &mesh, vec3 &min, vec3 &max, 
		double planethresh = .0001, double conethresh = .0001, double cylinderthresh = .000001, bool snapParams = true,
		int ptsPerEllipse = 60, double zStepSize = .02);

	// rescale the quadric parameters
	void normalizeField() {
		double maxq = 0;
        for (int i = 0; i < 10; i++) {
            maxq = MAX(fabs(q[i]), maxq);
        }
        for (int i = 0; i < 10; i++) {
            q[i] /= maxq;
        }
    }

    void init(const double *params) {
        for (int i = 0; i < 10; i++) {
            q[i] = params[i];
        }
    }
	void init(const std::vector<double> &params) {
        for (int i = 0; i < 10; i++) {
            q[i] = params[i];
        }
    }
    void fill(std::vector<double> &params) {
        params.resize(10);
        for (int i = 0; i < 10; i++) {
            params[i] = q[i];
        }
    }
    double f(const vec3 &p) const { // value of quadric function
        double x = p[0], y = p[1], z = p[2];
        return q[0] + q[1]*x + q[2]*y + q[3]*z 
            + q[4]*x*x + q[5]*x*y + q[6]*x*z
            + q[7]*y*y + q[8]*y*z + q[9]*z*z;
    }
    vec3 df(const vec3 &p) const { // gradient of quadric function
        double x = p[0], y = p[1], z = p[2];
        return vec3(q[1]+2*x*q[4]+y*q[5]+z*q[6],
                    q[2]+x*q[5]+2*y*q[7]+z*q[8],
                    q[3]+x*q[6]+y*q[8]+2*z*q[9]);
    }

    // approx distance due to [taubin, 93]
    double approxDist(const vec3 &p) {
        vec3 dfp = df(p);
        // quadric coefficients, following quadric surface extraction by ... paper's text (b, a reversed as in errata)
        double b = -sqrt(dfp*dfp);
        double a = -sqrt( (q[5]*q[5]+q[6]*q[6]+q[8]*q[8])/2 
                        + q[4]*q[4]+q[7]*q[7]+q[9]*q[9]);
        double c = fabs(f(p));
        double num = (-b-sqrt(b*b-4*a*c));
        double denom = 2*a;
		if (fabs(a) < .0000001) { // special case: it's a plane ...
            if (b > -.0000001) { // b == -|grad|
                return c / -b;
            } else { // it's a totally degen quadric
                return 0;
            }
        } else {
            double d = num/denom;
            return fabs(d);
        }
    }
	
    void transformQuadric(const mat4 &xf) {
        mat3 xf3(vec3(xf[0],VW), vec3(xf[1],VW), vec3(xf[2],VW));
        vec3 off(xf[0][3],xf[1][3],xf[2][3]);
        transformQuadric(xf3);
        transformQuadric(off);
    }
    void transformQuadric(const vec3 offset) {
        vec3 m = -offset;

        mat3 A;
        A[0][0] = q[4];
        A[1][0] = A[0][1] = q[5]*.5f;
        A[2][0] = A[0][2] = q[6]*.5f;
        A[1][1] = q[7];
        A[1][2] = A[2][1] = q[8]*.5f;
        A[2][2] = q[9];
        vec3 b(q[1], q[2], q[3]);

        q[0] += m*(A*m) + b*m;

        b += A.transpose() * m + A*m;
        q[1] = b[0];
        q[2] = b[1];
        q[3] = b[2];
    }
    void transformQuadric(const mat3 &xf) {
        // transform params by the inverse of xf
        mat3 X = xf.inverse();
        mat3 Xt = X.transpose();

        // build matrix version of quadric params, s.t. q(p) = p^T A p + b^T p + q[0]
        mat3 A;
        A[0][0] = q[4];
        A[1][0] = A[0][1] = q[5]*.5f;
        A[2][0] = A[0][2] = q[6]*.5f;
        A[1][1] = q[7];
        A[1][2] = A[2][1] = q[8]*.5f;
        A[2][2] = q[9];
        vec3 b(q[1], q[2], q[3]);

        // transform matrix version of parameters
        A = Xt*A*X;
        b = Xt*b;

        // copy back in to quadric parameters
        q[1] = b[0];
        q[2] = b[1];
        q[3] = b[2];
        q[4] = A[0][0];
        q[5] = A[0][1] * 2.0f;
        q[6] = A[0][2] * 2.0f;
        q[7] = A[1][1];
        q[8] = A[1][2] * 2.0f;
        q[9] = A[2][2];
    }

};


// useful for evaluating error of fit: a (slow) function to find the closest point on a quadric surface
// note: THIS CAN FAIL to find a good point -- "closestPt" can be pretty far from the level set
// Result of projection is returned by reference through closestPt.
// returns: true if abs value of quadric function, at closestPt, is below a tolerance; false otherwise
bool projectToQuadric_Slow(vec3 point, const Quadric &quadric, vec3 &closestPt);

} // namespace allquadrics


#endif




