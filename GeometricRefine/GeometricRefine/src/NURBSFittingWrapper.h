#pragma once
#include <igl/writeOBJ.h>
#include <igl/AABB.h>
#include "NURBSFitting.h"
#include "SurfFitter.h"
#include "CurveFitter.h"


class NURBSCurveFitter : public CurveFitter
{
public:
    NURBSCurveFitter()
    {
        cf = NULL;
        iter = 20;
        degree = 3;
        num_ctrl = 5;
        sample = 50; //larger than 34, for self projection
    }

    ~NURBSCurveFitter()
    {
        if (cf)
            delete cf;
    }

    vec3d GetPosition(double t)
    {
        vec3 p = cf->get_curve()->mycurve->GetPosition(t);
        return vec3d(p[0], p[1], p[2]);
    }

    void fitting();

    void get_seg_pts();

    void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t);
    CurveFitting* cf;
    int iter, degree, num_ctrl, sample;
    std::vector<vec3d> nurbs_pts;
};

class NURBSSurfFitter : public SurfFitter
{
public:
    NURBSSurfFitter()
    {
        aabb_tree = NULL;
        sf = NULL;
        iter = 20;
        u_degree = 3;
        v_degree = 3;
        num_u_ctrl = 5;
        num_v_ctrl = 5;
        sample = 20; //per-dim samples for projection
        sample_rate = 100;
        smoothness = 0.0;
    }

    NURBSSurfFitter(int u_deg, int v_deg, int u_ctrl, int v_ctrl)
    {
        aabb_tree = NULL;
        sf = NULL;
        iter = 20;
        u_degree = u_deg;
        v_degree = v_deg;
        num_u_ctrl = u_ctrl;
        num_v_ctrl = v_ctrl;
        sample = 20; //per-dim samples for constructing AABB tree
        sample_rate = 100;
        smoothness = 0.0;
    }

    NURBSSurfFitter(int u_deg, int v_deg, int u_ctrl, int v_ctrl, double s)
    {
        aabb_tree = NULL;
        sf = NULL;
        iter = 20;
        u_degree = u_deg;
        v_degree = v_deg;
        num_u_ctrl = u_ctrl;
        num_v_ctrl = v_ctrl;
        sample = 20; //per-dim samples for constructing AABB tree
        sample_rate = 100;
        smoothness = s;
    }

    ~NURBSSurfFitter()
    {
        if (sf)
            delete sf;
    }

    void set_smooth(double s)
    {
        smoothness = s;
    }

    vec3d GetPosition(double u, double v)
    {
        vec3 p = sf->get_surf()->mysurf->GetPosition(u, v);
        return vec3d(p[0], p[1], p[2]);
    }

    vec3d GetNormal(double u, double v)
    {
        vec3 ut = sf->get_surf()->mysurf->GetUTangent(u, v);
        vec3 vt = sf->get_surf()->mysurf->GetVTangent(u, v);
        vec3d utd(ut[0], ut[1], ut[2]), vtd(vt[0], vt[1], vt[2]);
        vec3d normal = utd.Cross(vtd);
        normal.Normalize();
        return normal;
    }

    void build_aabb_tree();

    void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false);
    void projection_with_normal(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, std::vector<vec3d>& tgt_normal, bool save_uv = false);
    void fitting();

    SurfFitting* sf;
    int iter, u_degree, v_degree, num_u_ctrl, num_v_ctrl;
    int sample;
    std::vector<vec3d> nurbs_pts;

    //aabb tree part
    igl::AABB<Eigen::MatrixXd, 3>* aabb_tree;
    Eigen::MatrixXd aabb_V;
    Eigen::MatrixXi aabb_F;
    int sample_rate;
    double smoothness;
};