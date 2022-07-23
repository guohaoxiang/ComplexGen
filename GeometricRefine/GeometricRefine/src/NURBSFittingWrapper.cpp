#include <igl/per_face_normals.h>
#include "NURBSFittingWrapper.h"

void NURBSCurveFitter::fitting()
{
    if (cf)
        delete cf;
    std::vector<vec3> points(input_pts.size());
    for (size_t i = 0; i < points.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            points[i][j] = input_pts[i][j];
        }
    }

   
    std::vector<vec3> addon_points;
    std::vector<double> weights(points.size(), 1.0), addon_weights;

    if (!nongrid_pts.empty())
    {
        assert(nongrid_pts.size() == nongrid_weights.size());
        addon_points.resize(nongrid_pts.size());
        for (size_t i = 0; i < nongrid_pts.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                addon_points[i][j] = nongrid_pts[i][j];
            }
        }
        cf = new CurveFitting(points, weights, addon_points, nongrid_weights, is_closed(), degree, num_ctrl, iter);
    }
    else
    {
        cf = new CurveFitting(points, weights, addon_points, addon_weights, is_closed(), degree, num_ctrl, iter);
    }



    nurbs_pts.clear();
    double split = is_closed() ? 1.0 / sample : 1.0 / (sample - 1);
    for (size_t i = 0; i < sample; i++)
    {
        nurbs_pts.push_back(GetPosition(split * i));
    }

}

void NURBSCurveFitter::get_seg_pts()
{
    
    //equal distance sample
    if (cf && nurbs_pts.empty())
    {
        double split = is_closed() ? 1.0 / sample : 1.0 / (sample - 1);
        for (size_t i = 0; i < sample; i++)
        {
            nurbs_pts.push_back(GetPosition(split * i));
        }
    }
    
    std::vector<double> curvelen;
    double len = 0;
    for (size_t i = 0; i < nurbs_pts.size() - 1; i++)
    {
        curvelen.push_back(len);
        len += (nurbs_pts[i + 1] - nurbs_pts[i]).Length();
    }
    curvelen.push_back(len);
    if (closed)
    {
        len += (nurbs_pts.front() - nurbs_pts.back()).Length();
        curvelen.push_back(len);
    }

    //set seg pts
    seg_pts.clear();
    seg_pts.resize(t_split);
    int k = 0;
    for (size_t i = 0; i < t_split; i++)
    {
        double t = i * len / (closed ? t_split : t_split - 1);
        while (t >= curvelen[k] - 1.0e-6 && k + 1 < curvelen.size())
            k++;
        double shortlen = curvelen[k] - curvelen[k - 1] + 1.0e-12;
        seg_pts[i] = (t - curvelen[k - 1]) / shortlen * nurbs_pts[k % nurbs_pts.size()] + (curvelen[k] - t) / shortlen * nurbs_pts[k - 1];
    }
}


void NURBSCurveFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t)
{
    //impl as nearest neighbors
    if (cf && nurbs_pts.empty())
    {
        double split = is_closed() ? 1.0 / sample : 1.0 / (sample - 1);
        for (size_t i = 0; i < sample; i++)
        {
            nurbs_pts.push_back(GetPosition(split * i));
        }
    }
    tgt.clear();
    assert(nurbs_pts.size() > 0);
    for (size_t i = 0; i < src.size(); i++)
    {
        double min_dist = -1.0;
        size_t min_id = 0;
        for (size_t j = 0; j < nurbs_pts.size(); j++)
        {
            double tmp_dist = (nurbs_pts[j] - src[i]).Length();
            if (min_dist < 0.0 || min_dist > tmp_dist)
            {
                min_dist = tmp_dist;
                min_id = j;
            }
        }
        assert(min_dist > -0.5);
        tgt.push_back(nurbs_pts[min_id]);
    }
}

void NURBSSurfFitter::build_aabb_tree()
{
    bool flag_init_face = true;
    if (aabb_tree)
    {
        delete aabb_tree;
        flag_init_face = false;
    }
    else
    {
        aabb_V.resize(sample * sample, 3);
    }

    //set V
    int denom_u = sample - 1, denom_v = sample - 1;
    if (u_closed)
        denom_u = sample;
    for (size_t i = 0; i < sample; i++)
    {
        double du = u_min + i * (u_max - u_min) / denom_u;
        for (size_t j = 0; j < sample; j++)
        {
            double dv = v_min + j * (v_max - v_min) / denom_v;
            //nurbs_pts.push_back(GetPosition(du, dv));
            vec3d cur = GetPosition(du, dv);
            aabb_V(i * sample + j, 0) = cur[0];
            aabb_V(i * sample + j, 1) = cur[1];
            aabb_V(i * sample + j, 2) = cur[2];
        }
    }

    if (flag_init_face)
    {
        std::vector<std::vector<size_t>> faces;
        size_t counter = 0;
        for (size_t j = 0; j < sample - 1; j++)
        {
            for (size_t k = 0; k < sample - 1; k++)
            {
                faces.push_back(std::vector<size_t>({ counter + j * sample + k, counter + (j + 1) * sample + k, counter + (j + 1) * sample + k + 1 }));
                faces.push_back(std::vector<size_t>({ counter + j * sample + k, counter + (j + 1) * sample + k + 1, counter + j * sample + k + 1 }));

            }
        }

        if (u_closed)
        {
            for (size_t j = 0; j < sample - 1; j++)
            {
                faces.push_back(std::vector<size_t>({ counter + (sample - 1) * sample + j, counter + j, counter + j + 1 }));
                faces.push_back(std::vector<size_t>({ counter + (sample - 1) * sample + j,  counter + j + 1, counter + (sample - 1) * sample + j + 1 }));
            }
        }
        aabb_F.resize(faces.size(), 3);
        for (size_t i = 0; i < faces.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                aabb_F(i, j) = faces[i][j];
            }
        }
    }

    aabb_tree = new igl::AABB<Eigen::MatrixXd, 3>();
    aabb_tree->init(aabb_V, aabb_F);
}

void NURBSSurfFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv)
{
    //impl as AABB tree
    assert(sf != NULL && aabb_tree != NULL);
    tgt.clear();

    Eigen::VectorXi I;
    Eigen::MatrixXd C;
    Eigen::VectorXd sqrD;
    Eigen::MatrixXd P(src.size(), 3);
    for (size_t i = 0; i < src.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            P(i, j) = src[i][j];
        }
    }
    aabb_tree->squared_distance(aabb_V, aabb_F, P, sqrD, I, C);
    tgt.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            tgt[i][j] = C(i, j);
        }
    }

    return;
    //impl as nearest neighbors, no longer used
    if (sf && nurbs_pts.empty())
    {
        int denom_u = sample - 1, denom_v = sample - 1;
        if (u_closed)
            denom_u = sample;
        for (size_t i = 0; i < sample; i++)
        {
            double du = u_min + i * (u_max - u_min) / denom_u;
            for (size_t j = 0; j < sample; j++)
            {
                double dv = v_min + j * (v_max - v_min) / denom_v;
                nurbs_pts.push_back(GetPosition(du, dv));
            }
        }
    }
    tgt.clear();
    assert(nurbs_pts.size() > 0);
    for (size_t i = 0; i < src.size(); i++)
    {
        double min_dist = -1.0;
        size_t min_id = 0;
        for (size_t j = 0; j < nurbs_pts.size(); j++)
        {
            double tmp_dist = (nurbs_pts[j] - src[i]).Length();
            if (min_dist < 0.0 || min_dist > tmp_dist)
            {
                min_dist = tmp_dist;
                min_id = j;
            }
        }
        assert(min_dist > -0.5);
        tgt.push_back(nurbs_pts[min_id]);
    }
}

void NURBSSurfFitter::projection_with_normal(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, std::vector<vec3d>& tgt_normal, bool save_uv)
{
    //impl as AABB tree
    assert(sf != NULL && aabb_tree != NULL);
    Eigen::MatrixXd aabb_F_normals;
    igl::per_face_normals(aabb_V, aabb_F, aabb_F_normals);
    tgt.clear();
    tgt_normal.clear();

    Eigen::VectorXi I;
    Eigen::MatrixXd C;
    Eigen::VectorXd sqrD;
    Eigen::MatrixXd P(src.size(), 3);
    for (size_t i = 0; i < src.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            P(i, j) = src[i][j];
        }
    }
    aabb_tree->squared_distance(aabb_V, aabb_F, P, sqrD, I, C);
    tgt.resize(src.size());
    tgt_normal.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            tgt[i][j] = C(i, j);
            tgt_normal[i][j] = aabb_F_normals(I(i), j);
        }
    }

    return;
    //impl as nearest neighbors, no longer used
    if (sf && nurbs_pts.empty())
    {
        int denom_u = sample - 1, denom_v = sample - 1;
        if (u_closed)
            denom_u = sample;
        for (size_t i = 0; i < sample; i++)
        {
            double du = u_min + i * (u_max - u_min) / denom_u;
            for (size_t j = 0; j < sample; j++)
            {
                double dv = v_min + j * (v_max - v_min) / denom_v;
                nurbs_pts.push_back(GetPosition(du, dv));
            }
        }
    }
    tgt.clear();
    assert(nurbs_pts.size() > 0);
    for (size_t i = 0; i < src.size(); i++)
    {
        double min_dist = -1.0;
        size_t min_id = 0;
        for (size_t j = 0; j < nurbs_pts.size(); j++)
        {
            double tmp_dist = (nurbs_pts[j] - src[i]).Length();
            if (min_dist < 0.0 || min_dist > tmp_dist)
            {
                min_dist = tmp_dist;
                min_id = j;
            }
        }
        assert(min_dist > -0.5);
        tgt.push_back(nurbs_pts[min_id]);
    }
}


void NURBSSurfFitter::fitting()
{
    //input_pts length should be square
    if (sf)
        delete sf;
    std::vector<std::vector<vec3>> gridpoints(dim_u_input); // 20-u-direction
    for (auto& e : gridpoints)
        e.resize(dim_v_input); // 20-v-direction
    for (size_t i = 0; i < gridpoints.size(); i++)
    {
        for (size_t j = 0; j < gridpoints[i].size(); j++)
        {
            gridpoints[i][j][0] = input_pts[i * dim_v_input + j][0];
            gridpoints[i][j][1] = input_pts[i * dim_v_input + j][1];
            gridpoints[i][j][2] = input_pts[i * dim_v_input + j][2];
        }
    }

    //all together version
    std::vector<std::vector<double>> grid_weights;
    std::vector<vec3> addon_pts;
    if (!input_pts_weight.empty())
    {
        grid_weights.resize(dim_u_input);
        for (auto& e : grid_weights)
        {
            e.resize(dim_v_input);
        }
        for (size_t i = 0; i < grid_weights.size(); i++)
        {
            for (size_t j = 0; j < grid_weights[i].size(); j++)
            {
                grid_weights[i][j] = input_pts_weight[i * dim_v_input + j];
            }
        }
    }

    //addon points
    if (!nongrid_pts.empty())
    {
        assert(nongrid_pts.size() == nongrid_weights.size());
        addon_pts.resize(nongrid_pts.size());
        for (size_t i = 0; i < nongrid_pts.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                addon_pts[i][j] = nongrid_pts[i][j];
            }
        }

        sf = new SurfFitting(gridpoints, grid_weights, addon_pts, nongrid_weights, u_closed, v_closed, u_degree, v_degree, num_u_ctrl, num_v_ctrl, iter, sample_rate, smoothness);
    }
    else
    {
        std::vector<double> addon_weights;
        sf = new SurfFitting(gridpoints, grid_weights, addon_pts, addon_weights, u_closed, v_closed, u_degree, v_degree, num_u_ctrl, num_v_ctrl, iter, sample_rate, smoothness);
    }

    //constructing AABB tree
    build_aabb_tree();

    return;
}