#include <iostream>
#include <fstream>
#include "NURBSFitting.h"

void MyGeometry::set_points(const std::vector<vec3>& points, const std::vector<double>& weights)
{
    if (points.empty())
        return;
    input_points.assign(points.begin(), points.end());
    if (weights.size() == input_points.size())
        input_point_weights.assign(weights.begin(), weights.end());
    else
        input_point_weights.assign(input_points.size(), 1.0);
}

void MyGeometry::set_points(const std::vector<std::vector<vec3>>& points, const std::vector<std::vector<double>>& weights)
{
    if (points.empty() || points.front().empty())
        return;
    input_grid_points.assign(points.begin(), points.end());
    if (weights.size() == points.size())
        input_grid_point_weights.assign(weights.begin(), weights.end());
    else
    {
        input_grid_point_weights.resize(input_grid_points.size());
        for (auto& e : input_grid_point_weights)
            e.assign(points.front().size(), 1.0);
    }
}

void MyGeometry::set_addon(const std::vector<vec3>& points, const std::vector<double>& weights)
{
    if (points.empty())
        return;
    additional_points.assign(points.begin(), points.end());
    if (weights.size() == additional_points.size())
        additional_point_weights.assign(weights.begin(), weights.end());
    else
        additional_point_weights.assign(additional_points.size(), 1.0);
}

void MyFitBase::optimize(MyGeometry& mygeometry, const std::vector<vec3>& points, int max_iter)
{
    mygeometry.set_points(points);
    std::vector<double> variables;
    mygeometry.initialize(variables);
    IGOpt::LBFGS_Optimizer(variables, max_iter, eval_funcgrad, newiteration, &mygeometry, false);
    mygeometry.update_solution(variables);
}

void MyFitBase::optimize(MyGeometry& mygeometry, const std::vector<vec3>& points, const std::vector<double>& point_weights, int max_iter)
{
    mygeometry.set_points(points, point_weights);
    std::vector<double> variables;
    mygeometry.initialize(variables);
    IGOpt::LBFGS_Optimizer(variables, max_iter, eval_funcgrad, newiteration, &mygeometry, false);
    mygeometry.update_solution(variables);
}

void MyFitBase::optimize(MyGeometry& mygeometry, const std::vector<vec3>& points, const std::vector<double>& point_weights,
    const std::vector<vec3>& addon_points, const std::vector<double>& addon_weights, int max_iter)
{
    mygeometry.set_points(points, point_weights);
    mygeometry.set_addon(addon_points, addon_weights);

    std::vector<double> variables;
    mygeometry.initialize(variables);
    IGOpt::LBFGS_Optimizer(variables, max_iter, eval_funcgrad, newiteration, &mygeometry, false);
    mygeometry.update_solution(variables);
}

void MyFitBase::optimize(MyGeometry& mygeometry, const std::vector<std::vector<vec3>>& points, int max_iter)
{
    mygeometry.set_points(points);
    std::vector<double> variables;
    mygeometry.initialize(variables);
    IGOpt::LBFGS_Optimizer(variables, max_iter, eval_funcgrad, newiteration, &mygeometry, false);
    mygeometry.update_solution(variables);
}

void MyFitBase::optimize(MyGeometry& mygeometry, const std::vector<std::vector<vec3>>& points, const std::vector<std::vector<double>>& weights, int max_iter)
{
    mygeometry.set_points(points, weights);
    std::vector<double> variables;
    mygeometry.initialize(variables);
    IGOpt::LBFGS_Optimizer(variables, max_iter, eval_funcgrad, newiteration, &mygeometry, false);
    mygeometry.update_solution(variables);
}

void MyFitBase::optimize(MyGeometry& mygeometry, const std::vector<std::vector<vec3>>& points, const std::vector<std::vector<double>>& weights,
    const std::vector<vec3>& addon_points, const std::vector<double>& addon_weights, int max_iter)
{
    mygeometry.set_points(points, weights);
    mygeometry.set_addon(addon_points, addon_weights);
    std::vector<double> variables;
    mygeometry.initialize(variables);
    IGOpt::LBFGS_Optimizer(variables, max_iter, eval_funcgrad, newiteration, &mygeometry, false);
    mygeometry.update_solution(variables);
}

MyNURBSCurve::MyNURBSCurve(int degree, bool closed, const std::vector<vec3>& ctl_points)
    : is_closed(closed), opt_weight(false), opt_addon(false), sample_rate(100)
{
    gte::BasisFunctionInput<double> basis;
    std::vector<vec3> controls(ctl_points);
    if (is_closed)
        controls.insert(controls.end(), ctl_points.begin(), ctl_points.begin() + degree);

    std::vector<double> weights(controls.size(), 1);
    if (closed)
    {
        basis.numControls = (int)controls.size();
        basis.degree = degree;
        basis.uniform = true;
        basis.periodic = false;
        basis.numUniqueKnots = basis.numControls + degree + 1;
        basis.uniqueKnots.resize(basis.numUniqueKnots);

        for (int i = 0; i < basis.numUniqueKnots; i++)
        {
            basis.uniqueKnots[i].t = (i - degree) / (double)(basis.numControls - degree);
            basis.uniqueKnots[i].multiplicity = 1;
        }
    }
    else
    {
        basis = gte::BasisFunctionInput<double>((int)controls.size(), degree);
    }

    mycurve = new gte::NURBSCurve<3, double>(basis, &controls[0], weights.data());
}

void MyNURBSCurve::set_opt_weight(bool opt)
{
    opt_weight = opt;
}

void MyNURBSCurve::set_opt_addon(bool opt, int _sample_rate)
{
    opt_addon = opt, sample_rate = _sample_rate;
}

void MyNURBSCurve::initialize(std::vector<double>& variables)
{
    int num_free_controls = is_closed ? mycurve->GetNumControls() - mycurve->GetBasisFunction().GetDegree() : mycurve->GetNumControls();
    variables.resize(num_free_controls * 4);
    for (int i = 0; i < num_free_controls; i++)
    {
        const auto& p = mycurve->GetControl(i);
        variables[4 * i] = p[0], variables[4 * i + 1] = p[1], variables[4 * i + 2] = p[2], variables[4 * i + 3] = mycurve->GetWeight(i);
    }
}
void MyNURBSCurve::update_solution(const std::vector<double>& variables)
{
    int num_free_controls = is_closed ? mycurve->GetNumControls() - mycurve->GetBasisFunction().GetDegree() : mycurve->GetNumControls();
    vec3 v;
    for (int i = 0; i < mycurve->GetNumControls(); i++)
    {
        int id = i % num_free_controls;
        v[0] = variables[4 * id], v[1] = variables[4 * id + 1], v[2] = variables[4 * id + 2];
        mycurve->SetControl(i, v);
        mycurve->SetWeight(i, variables[4 * id + 3]);
    }
}
void MyNURBSCurve::funcgrad(const std::vector<double>& variables, double& func_value, std::vector<double>& gradient)
{
    if (!mycurve)
        return;

    const auto& basis = mycurve->GetBasisFunction();
    const int degree = basis.GetDegree();
    double u_min = mycurve->GetTMin();
    double u_max = mycurve->GetTMax();
    int numControls = mycurve->GetNumControls();
    int num_free_controls = is_closed ? mycurve->GetNumControls() - mycurve->GetBasisFunction().GetDegree() : mycurve->GetNumControls();

    int size = (int)input_points.size();
    double denom_u = is_closed ? size : size - 1;
    int iumin, iumax;
    vec3 X, Y;

    func_value = 0;
    std::vector<int> idvec;
    std::vector<double> coeff;
    for (int i = 0; i < size; i++)
    {
        double u = u_min + (u_max - u_min) * i / denom_u;

        basis.Evaluate(u, 0, iumin, iumax);

        X.MakeZero();
        double w = 0;
        idvec.resize(iumax - iumin + 1);
        coeff.resize(iumax - iumin + 1);
        for (int i = iumin; i <= iumax; ++i)
        {
            int j = (i >= numControls ? i - numControls : i);
            if (is_closed)
                j %= num_free_controls;
            idvec[i - iumin] = j;
            double ci = basis.GetValue(0, i);
            double tmp = ci * variables[4 * j + 3];
            X[0] += tmp * variables[4 * j];
            X[1] += tmp * variables[4 * j + 1];
            X[2] += tmp * variables[4 * j + 2];
            coeff[i - iumin] = ci;
            w += tmp;
        }
        X /= w;

        const auto weight = input_point_weights[i];
        Y = X - input_points[i];

        func_value += weight * gte::Dot(Y, Y);

        for (int k = 0; k < (int)idvec.size(); k++)
        {
            double t0 = Y[0] * coeff[k] / w;
            double t1 = Y[1] * coeff[k] / w;
            double t2 = Y[2] * coeff[k] / w;

            gradient[4 * idvec[k]] += weight * t0;
            gradient[4 * idvec[k] + 1] += weight * t1;
            gradient[4 * idvec[k] + 2] += weight * t2;
            if (opt_weight)
            {
                gradient[4 * idvec[k] + 3] += weight * (t0 * (variables[4 * idvec[k]] - X[0]) + t1 * (variables[4 * idvec[k] + 1] - X[1]) + t2 * (variables[4 * idvec[k] + 2] - X[2]));
            }
        }
    }

    if (opt_addon && additional_points.empty() == false)
    {
        std::vector<double> sample_u;
        sample_u.reserve(sample_rate);
        sample_points.reserve(sample_rate);
        sample_u.resize(0);
        sample_points.resize(0);

        for (int i = 0; i < sample_rate; i++)
        {
            double du = u_min + (u_max - u_min) * i / sample_rate;
            basis.Evaluate(du, 0, iumin, iumax);
            X.MakeZero();
            double w = 0;
            for (int i = iumin; i <= iumax; ++i)
            {
                int j = (i >= numControls ? i - numControls : i);
                if (is_closed)
                    j %= num_free_controls;
                idvec[i - iumin] = j;
                double ci = basis.GetValue(0, i);
                double tmp = ci * variables[4 * j + 3];
                X[0] += tmp * variables[4 * j];
                X[1] += tmp * variables[4 * j + 1];
                X[2] += tmp * variables[4 * j + 2];
                w += tmp;
            }
            X /= w;
            sample_u.push_back(du);
            sample_points.push_back(X);
        }

        typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, MyNURBSCurve>, MyNURBSCurve, 3> my_kd_tree_t;
        my_kd_tree_t m_kdtree(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        m_kdtree.buildIndex();

        for (size_t i = 0; i < additional_points.size(); i++)
        {
            nanoflann::KNNResultSet<float> resultSet(1);
            size_t nearest_point_id;
            float distance;
            resultSet.init(&nearest_point_id, &distance);
            m_kdtree.findNeighbors(resultSet, &additional_points[i][0], nanoflann::SearchParams(10));

            double du = sample_u[nearest_point_id];
            basis.Evaluate(du, 0, iumin, iumax);

            X.MakeZero();
            double w = 0;
            idvec.resize(iumax - iumin + 1);
            coeff.resize(iumax - iumin + 1);
            for (int i = iumin; i <= iumax; ++i)
            {
                int j = (i >= numControls ? i - numControls : i);
                if (is_closed)
                    j %= num_free_controls;
                idvec[i - iumin] = j;
                double ci = basis.GetValue(0, i);
                double tmp = ci * variables[4 * j + 3];
                X[0] += tmp * variables[4 * j];
                X[1] += tmp * variables[4 * j + 1];
                X[2] += tmp * variables[4 * j + 2];
                coeff[i - iumin] = ci;
                w += tmp;
            }
            X /= w;

            const auto weight = additional_point_weights[i];
            Y = X - additional_points[i];

            func_value += weight * gte::Dot(Y, Y);

            for (int k = 0; k < (int)idvec.size(); k++)
            {
                double t0 = Y[0] * coeff[k] / w;
                double t1 = Y[1] * coeff[k] / w;
                double t2 = Y[2] * coeff[k] / w;

                gradient[4 * idvec[k]] += weight * t0;
                gradient[4 * idvec[k] + 1] += weight * t1;
                gradient[4 * idvec[k] + 2] += weight * t2;
                if (opt_weight)
                {
                    gradient[4 * idvec[k] + 3] += weight * (t0 * (variables[4 * idvec[k]] - X[0]) + t1 * (variables[4 * idvec[k] + 1] - X[1]) + t2 * (variables[4 * idvec[k] + 2] - X[2]));
                }
            }
        }
    }

    if (smoothness_weight > 0)
    {
        // int msize = 2*num_free_controls;
        // double du[3];
        // std::vector<std::vector<int>> midvec(3);
        // std::vector<std::vector<double>> mcoeff(3);
        // vec3 mX[3], mY;
        // double mw[3];
        // for (int s = 1; s < msize; s++)
        // {
        //     du[0] = u_min + (u_max - u_min) * (s - 1) / msize;
        //     du[1] = u_min + (u_max - u_min) * s / msize;
        //     du[2] = u_min + (u_max - u_min) * (s + 1) / msize;

        //     for (int k = 0; k < 3; k++)
        //     {
        //         basis.Evaluate(du[k], 0, iumin, iumax);
        //         mX[k].MakeZero();
        //         mw[k] = 0;
        //         midvec[k].resize(iumax - iumin + 1);
        //         mcoeff[k].resize(iumax - iumin + 1);
        //         for (int i = iumin; i <= iumax; ++i)
        //         {
        //             int j = (i >= numControls ? i - numControls : i);
        //             if (is_closed)
        //                 j %= num_free_controls;
        //             idvec[i - iumin] = j;
        //             double ci = basis.GetValue(0, i);
        //             double tmp = ci * variables[4 * j + 3];
        //             mX[k][0] += tmp * variables[4 * j];
        //             mX[k][1] += tmp * variables[4 * j + 1];
        //             mX[k][2] += tmp * variables[4 * j + 2];
        //             mcoeff[k][i - iumin] = ci;
        //             mw[k] += tmp;
        //         }
        //         mX[k] /= mw[k];
        //     }
        //     mY = 0.5 * (mX[0] + mX[2]) - mX[1];
        //     func_value += smoothness_weight * gte::Dot(mY, mY);

        //     for (int l = 0; l < 3; l++)
        //     {
        //         double factor = (l == 1 ? -smoothness_weight : 0.5 * smoothness_weight);
        //         for (int k = 0; k < (int)midvec[l].size(); k++)
        //         {
        //             double t0 = Y[0] * mcoeff[l][k] / mw[l];
        //             double t1 = Y[1] * mcoeff[l][k] / mw[l];
        //             double t2 = Y[2] * mcoeff[l][k] / mw[l];

        //             gradient[4 * midvec[l][k]] += factor * t0;
        //             gradient[4 * midvec[l][k] + 1] += factor * t1;
        //             gradient[4 * midvec[l][k] + 2] += factor * t2;
        //             if (opt_weight)
        //             {
        //                 gradient[4 * midvec[l][k] + 3] += factor * (t0 * (variables[4 * midvec[l][k]] - mX[l][0]) + t1 * (variables[4 * midvec[l][k] + 1] - mX[l][1]) + t2 * (variables[4 * midvec[l][k] + 2] - mX[l][2]));
        //             }
        //         }
        //     }
        // }
        for (int i = 1; i + 1 < numControls; i++)
        {
            int l_id = (i - 1) % num_free_controls;
            int id = i % num_free_controls;
            int r_id = (i + 1) % num_free_controls;

            for (int j = 0; j < 3; j++)
            {
                double v = 0.5 * (variables[4 * l_id + j] + variables[4 * r_id + j]) - variables[4 * id + j];
                func_value += smoothness_weight * v * v;
                gradient[4 * l_id + j] += smoothness_weight * v * 0.5;
                gradient[4 * r_id + j] += smoothness_weight * v * 0.5;
                gradient[4 * id + j] -= smoothness_weight * v;
            }
        }
    }

    func_value *= 0.5;
}

void MyNURBSCurve::write_obj(const std::string& filename, int u_samplerate, int v_samplerate)
{

    std::ofstream objfile(filename);

    const auto& basis = mycurve->GetBasisFunction();

    double denom = is_closed ? u_samplerate : u_samplerate - 1;

    vec3 p;

    double u_min = mycurve->GetTMin();
    double u_max = mycurve->GetTMax();

    for (int i = 0; i < u_samplerate; i++)
    {
        double t = u_min + (u_max - u_min) * i / denom;
        p = mycurve->GetPosition(t);
        objfile << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << std::endl;
    }

    for (int i = 0; i < u_samplerate - 1; i++)
        objfile << "l " << i + 1 << ' ' << i + 2 << std::endl;

    if (is_closed)
        objfile << "l " << u_samplerate << ' ' << 1 << std::endl;

    objfile.close();
}

void MyNURBSCurve::write_nurbs(const std::string& filename)
{
    std::ofstream objfile(filename);
    objfile << "#NURBS curve" << std::endl;
    objfile << "g curve" << std::endl;
    for (int i = 0; i < (is_closed ? mycurve->GetNumControls() - mycurve->GetBasisFunction().GetDegree() : mycurve->GetNumControls()); i++)
    {
        const auto& p = mycurve->GetControl(i);
        objfile << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << mycurve->GetWeight(i) << std::endl;
    }
    objfile << "cstype rat bspline" << std::endl;
    objfile << "deg " << mycurve->GetBasisFunction().GetDegree() << std::endl;
    objfile << "curv " << mycurve->GetTMin() << ' ' << mycurve->GetTMax();

    if (!is_closed)
    {
        for (int i = 0; i < mycurve->GetNumControls(); i++)
            objfile << ' ' << i + 1;
    }
    else
    {
        for (int i = 0; i < mycurve->GetNumControls(); i++)
            objfile << ' ' << i % (mycurve->GetNumControls() - mycurve->GetBasisFunction().GetDegree()) + 1;
    }

    objfile << std::endl
        << "parm u";
    for (int i = 0; i < mycurve->GetBasisFunction().GetNumKnots(); i++)
    {
        objfile << ' ' << mycurve->GetBasisFunction().GetKnots()[i];
    }
    objfile << std::endl
        << "end" << std::endl;

    objfile.close();
}

CurveFitting::CurveFitting(const std::vector<vec3>& points, const std::vector<double>& weights, const std::vector<vec3>& addon_points,
    const std::vector<double>& addon_weights, bool closed, int degree, int num_ctrl, int num_iter, int sample_rate,
    double smoothness_weight)
{
    if (points.size() < num_ctrl + degree + 1 || degree < 1 || num_ctrl < degree + 1)
        return;

    std::vector<double> curvelen;
    double len = 0;
    for (size_t i = 0; i < points.size() - 1; i++)
    {
        curvelen.push_back(len);
        len += gte::Length(points[i + 1] - points[i]);
    }

    curvelen.push_back(len);

    std::vector<vec3> controls(num_ctrl);
    int k = 0;
    for (int i = 0; i < num_ctrl; i++)
    {
        double t = i * len / num_ctrl;
        //while (t >= curvelen[k] && k < (int)points.size())
        while (k < (int)points.size() - 1 && t >= curvelen[k])
            k++;
        double shortlen = curvelen[k] - curvelen[k - 1] + 1.0e-12;
        controls[i] = (t - curvelen[k - 1]) / shortlen * points[k - 1] + (curvelen[k] - t) / shortlen * points[k];
    }
    mycurve = new MyNURBSCurve(degree, closed, controls);
    MyFitBase fit;

    mycurve->smoothness() = smoothness_weight;

    mycurve->set_opt_weight(false);
    fit.optimize(*mycurve, points, weights, addon_points, addon_weights, num_iter);
    mycurve->set_opt_weight(true);
    fit.optimize(*mycurve, points, weights, addon_points, addon_weights, num_iter);
    if (!addon_points.empty())
    {
        mycurve->set_opt_addon(true, sample_rate);
        fit.optimize(*mycurve, points, weights, addon_points, addon_weights, num_iter);
    }
}


MyNURBSSurf::MyNURBSSurf(int u_degree, bool u_closed, int v_degree, bool v_closed, const std::vector<std::vector<vec3>>& ctl_points)
    : is_u_closed(u_closed), is_v_closed(v_closed), opt_weight(false), opt_addon(false), sample_rate(100)
{
    gte::BasisFunctionInput<double> u_basis, v_basis;
    std::vector<std::vector<vec3>> controls(ctl_points);
    if (is_v_closed)
    {
        for (auto& e : controls)
        {
            e.insert(e.end(), e.begin(), e.begin() + v_degree);
        }
    }
    if (is_u_closed)
    {
        controls.insert(controls.end(), controls.begin(), controls.begin() + u_degree);
    }

    std::vector<std::vector<double>> weights(controls.size());
    for (auto& e : weights)
        e.assign(controls.front().size(), 1);

    if (is_u_closed)
    {
        u_basis.numControls = (int)controls.size();
        u_basis.degree = u_degree;
        u_basis.uniform = true;
        u_basis.periodic = false;
        u_basis.numUniqueKnots = u_basis.numControls + u_degree + 1;
        u_basis.uniqueKnots.resize(u_basis.numUniqueKnots);

        for (int i = 0; i < u_basis.numUniqueKnots; i++)
        {
            u_basis.uniqueKnots[i].t = (i - u_degree) / (double)(u_basis.numControls - u_degree);
            u_basis.uniqueKnots[i].multiplicity = 1;
        }
    }
    else
    {
        u_basis = gte::BasisFunctionInput<double>((int)controls.size(), u_degree);
    }

    if (is_v_closed)
    {
        v_basis.numControls = (int)controls.front().size();
        v_basis.degree = v_degree;
        v_basis.uniform = true;
        v_basis.periodic = false;
        v_basis.numUniqueKnots = v_basis.numControls + v_degree + 1;
        v_basis.uniqueKnots.resize(v_basis.numUniqueKnots);

        for (int i = 0; i < v_basis.numUniqueKnots; i++)
        {
            v_basis.uniqueKnots[i].t = (i - v_degree) / (double)(v_basis.numControls - v_degree);
            v_basis.uniqueKnots[i].multiplicity = 1;
        }
    }
    else
    {
        v_basis = gte::BasisFunctionInput<double>((int)controls.front().size(), v_degree);
    }
    std::vector<vec3> mcontrols;
    for (const auto& e : controls)
    {
        mcontrols.insert(mcontrols.end(), e.begin(), e.end());
    }
    std::vector<double> mweights;
    for (const auto& e : weights)
    {
        mweights.insert(mweights.end(), e.begin(), e.end());
    }
    mysurf = new gte::NURBSSurface<3, double>(u_basis, v_basis, mcontrols.data(), mweights.data());
}

void MyNURBSSurf::set_opt_weight(bool opt)
{
    opt_weight = opt;
}

void MyNURBSSurf::set_opt_addon(bool opt, int _sample_rate)
{
    opt_addon = opt, sample_rate = _sample_rate;
}

void MyNURBSSurf::initialize(std::vector<double>& variables)
{
    int num_free_u_controls = is_u_closed ? mysurf->GetNumControls(0) - mysurf->GetBasisFunction(0).GetDegree() : mysurf->GetNumControls(0);
    int num_free_v_controls = is_v_closed ? mysurf->GetNumControls(1) - mysurf->GetBasisFunction(1).GetDegree() : mysurf->GetNumControls(1);
    variables.resize(num_free_u_controls * num_free_v_controls * 4);
    for (int i = 0; i < num_free_u_controls; i++)
        for (int j = 0; j < num_free_v_controls; j++)
        {
            const auto& p = mysurf->GetControl(i, j);
            auto var = 4 * (i * num_free_v_controls + j);
            variables[var] = p[0], variables[var + 1] = p[1], variables[var + 2] = p[2], variables[var + 3] = mysurf->GetWeight(i, j);
        }
}
void MyNURBSSurf::update_solution(const std::vector<double>& variables)
{
    int num_free_u_controls = is_u_closed ? mysurf->GetNumControls(0) - mysurf->GetBasisFunction(0).GetDegree() : mysurf->GetNumControls(0);
    int num_free_v_controls = is_v_closed ? mysurf->GetNumControls(1) - mysurf->GetBasisFunction(1).GetDegree() : mysurf->GetNumControls(1);
    for (int i = 0; i < mysurf->GetNumControls(0); i++)
        for (int j = 0; j < mysurf->GetNumControls(1); j++)
        {
            auto var = 4 * ((i % num_free_u_controls) * num_free_v_controls + (j % num_free_v_controls));
            vec3 p;
            p[0] = variables[var], p[1] = variables[var + 1], p[2] = variables[var + 2];
            mysurf->SetControl(i, j, p);
            mysurf->SetWeight(i, j, variables[var + 3]);
        }
}
void MyNURBSSurf::funcgrad(const std::vector<double>& variables, double& func_value, std::vector<double>& gradient)
{
    if (!mysurf)
        return;

    const auto& u_basis = mysurf->GetBasisFunction(0);
    const auto& v_basis = mysurf->GetBasisFunction(1);
    int num_free_u_controls = is_u_closed ? mysurf->GetNumControls(0) - mysurf->GetBasisFunction(0).GetDegree() : mysurf->GetNumControls(0);
    int num_free_v_controls = is_v_closed ? mysurf->GetNumControls(1) - mysurf->GetBasisFunction(1).GetDegree() : mysurf->GetNumControls(1);
    double u_min = mysurf->GetUMin(), u_max = mysurf->GetUMax();
    double v_min = mysurf->GetVMin(), v_max = mysurf->GetVMax();
    int denom_u = is_u_closed ? (int)input_grid_points.size() : (int)input_grid_points.size() - 1,
        denom_v = is_v_closed ? (int)input_grid_points.front().size() : (int)input_grid_points.front().size() - 1;
    auto numControls0 = mysurf->GetNumControls(0);
    auto numControls1 = mysurf->GetNumControls(1);

    int iumin, iumax, ivmin, ivmax;
    vec3 X, Y;
    std::vector<int> idvec;
    std::vector<double> coeff;
    for (int i = 0; i < (int)input_grid_points.size(); i++)
    {
        double du = u_min + (u_max - u_min) * i / denom_u;
        u_basis.Evaluate(du, 0, iumin, iumax);
        for (int j = 0; j < (int)input_grid_points.front().size(); j++)
        {
            double dv = v_min + (v_max - v_min) * j / denom_v;
            double w = 0;
            v_basis.Evaluate(dv, 0, ivmin, ivmax);
            X.MakeZero();
            idvec.resize((ivmax - ivmin + 1) * (iumax - iumin + 1));
            coeff.resize(idvec.size());
            for (int iv = ivmin; iv <= ivmax; ++iv)
            {
                double tmpv = v_basis.GetValue(0, iv);
                int jv = (iv >= numControls1 ? iv - numControls1 : iv);
                if (is_v_closed)
                    jv %= num_free_v_controls;

                for (int iu = iumin; iu <= iumax; ++iu)
                {
                    double tmpu = u_basis.GetValue(0, iu);
                    int ju = (iu >= numControls0 ? iu - numControls0 : iu);
                    if (is_u_closed)
                        ju %= num_free_u_controls;
                    int index = 4 * (ju * num_free_v_controls + jv);
                    double tmp = tmpu * tmpv * variables[index + 3];
                    X[0] += tmp * variables[index];
                    X[1] += tmp * variables[index + 1];
                    X[2] += tmp * variables[index + 2];
                    w += tmp;
                    idvec[(iu - iumin) * (ivmax - ivmin + 1) + (iv - ivmin)] = index;
                    coeff[(iu - iumin) * (ivmax - ivmin + 1) + (iv - ivmin)] = tmpu * tmpv;
                }
            }
            X /= w;

            const auto weight = input_grid_point_weights[i][j];

            Y = X - input_grid_points[i][j];
            func_value += weight * gte::Dot(Y, Y);

            for (int k = 0; k < (int)idvec.size(); k++)
            {
                double t0 = Y[0] * coeff[k] / w;
                double t1 = Y[1] * coeff[k] / w;
                double t2 = Y[2] * coeff[k] / w;

                gradient[idvec[k]] += weight * t0;
                gradient[idvec[k] + 1] += weight * t1;
                gradient[idvec[k] + 2] += weight * t2;
                if (opt_weight)
                {
                    gradient[idvec[k] + 3] += weight * (t0 * (variables[idvec[k]] - X[0]) + t1 * (variables[idvec[k] + 1] - X[1]) + t2 * (variables[idvec[k] + 2] - X[2]));
                }
            }
        }
    }

    if (opt_addon && additional_points.empty() == false)
    {
        std::vector<std::pair<double, double>> sample_uv;
        sample_uv.reserve(sample_rate * sample_rate);
        sample_points.reserve(sample_rate * sample_rate);
        sample_uv.resize(0);
        sample_points.resize(0);

        for (int i = 0; i < sample_rate; i++)
        {
            double du = u_min + (u_max - u_min) * i / sample_rate;
            u_basis.Evaluate(du, 0, iumin, iumax);
            for (int j = 0; j < sample_rate; j++)
            {
                double dv = v_min + (v_max - v_min) * j / sample_rate;
                double w = 0;
                v_basis.Evaluate(dv, 0, ivmin, ivmax);
                X.MakeZero();
                for (int iv = ivmin; iv <= ivmax; ++iv)
                {
                    double tmpv = v_basis.GetValue(0, iv);
                    int jv = (iv >= numControls1 ? iv - numControls1 : iv);
                    if (is_v_closed)
                        jv %= num_free_v_controls;
                    for (int iu = iumin; iu <= iumax; ++iu)
                    {
                        double tmpu = u_basis.GetValue(0, iu);
                        int ju = (iu >= numControls0 ? iu - numControls0 : iu);
                        if (is_u_closed)
                            ju %= num_free_u_controls;
                        int index = 4 * (ju * num_free_v_controls + jv);
                        double tmp = tmpu * tmpv * variables[index + 3];
                        X[0] += tmp * variables[index];
                        X[1] += tmp * variables[index + 1];
                        X[2] += tmp * variables[index + 2];
                        w += tmp;
                    }
                }
                X /= w;
                sample_uv.push_back(std::make_pair(du, dv));
                sample_points.push_back(X);
            }
        }

        typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, MyNURBSSurf>, MyNURBSSurf, 3> my_kd_tree_t;
        my_kd_tree_t m_kdtree(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        m_kdtree.buildIndex();

        for (size_t i = 0; i < additional_points.size(); i++)
        {
            nanoflann::KNNResultSet<float> resultSet(1);
            size_t nearest_point_id;
            float distance;
            resultSet.init(&nearest_point_id, &distance);
            m_kdtree.findNeighbors(resultSet, &additional_points[i][0], nanoflann::SearchParams(10));

            double du = sample_uv[nearest_point_id].first;
            double dv = sample_uv[nearest_point_id].second;
            double w = 0;
            u_basis.Evaluate(du, 0, iumin, iumax);
            v_basis.Evaluate(dv, 0, ivmin, ivmax);
            X.MakeZero();
            idvec.resize((ivmax - ivmin + 1) * (iumax - iumin + 1));
            coeff.resize(idvec.size());
            for (int iv = ivmin; iv <= ivmax; ++iv)
            {
                double tmpv = v_basis.GetValue(0, iv);
                int jv = (iv >= numControls1 ? iv - numControls1 : iv);
                if (is_v_closed)
                    jv %= num_free_v_controls;

                for (int iu = iumin; iu <= iumax; ++iu)
                {
                    double tmpu = u_basis.GetValue(0, iu);
                    int ju = (iu >= numControls0 ? iu - numControls0 : iu);
                    if (is_u_closed)
                        ju %= num_free_u_controls;
                    int index = 4 * (ju * num_free_v_controls + jv);
                    double tmp = tmpu * tmpv * variables[index + 3];
                    X[0] += tmp * variables[index];
                    X[1] += tmp * variables[index + 1];
                    X[2] += tmp * variables[index + 2];
                    w += tmp;
                    idvec[(iu - iumin) * (ivmax - ivmin + 1) + (iv - ivmin)] = index;
                    coeff[(iu - iumin) * (ivmax - ivmin + 1) + (iv - ivmin)] = tmpu * tmpv;
                }
            }
            X /= w;

            const auto weight = additional_point_weights[i];

            Y = X - additional_points[i];
            func_value += weight * gte::Dot(Y, Y);

            for (int k = 0; k < (int)idvec.size(); k++)
            {
                double t0 = Y[0] * coeff[k] / w;
                double t1 = Y[1] * coeff[k] / w;
                double t2 = Y[2] * coeff[k] / w;

                gradient[idvec[k]] += weight * t0;
                gradient[idvec[k] + 1] += weight * t1;
                gradient[idvec[k] + 2] += weight * t2;
                if (opt_weight)
                {
                    gradient[idvec[k] + 3] += weight * (t0 * (variables[idvec[k]] - X[0]) + t1 * (variables[idvec[k] + 1] - X[1]) + t2 * (variables[idvec[k] + 2] - X[2]));
                }
            }
        }
    }

    if (smoothness_weight > 0)
    {
        double factor = 1; //0.1;
        for (int k = 0; k < numControls1; k++)
        {
            int k_mod = k % num_free_v_controls;
            for (int i = 1; i + 1 < numControls0; i++)
            {
                int l_id = ((i - 1) % num_free_u_controls) * num_free_v_controls + k_mod;
                int id = (i % num_free_u_controls) * num_free_v_controls + k_mod;
                int r_id = ((i + 1) % num_free_u_controls) * num_free_v_controls + k_mod;

                for (int j = 0; j < 3; j++)
                {
                    double v = 0.5 * (variables[4 * l_id + j] + variables[4 * r_id + j]) - variables[4 * id + j];
                    func_value += factor * smoothness_weight * v * v;
                    gradient[4 * l_id + j] += factor * smoothness_weight * v * 0.5;
                    gradient[4 * r_id + j] += factor * smoothness_weight * v * 0.5;
                    gradient[4 * id + j] -= factor * smoothness_weight * v;
                }
            }
            // for (int i = 0; i + 1 < numControls0; i++)
            // {
            //     int id = (i % num_free_u_controls) * num_free_v_controls + k_mod;
            //     int r_id = ((i + 1) % num_free_u_controls) * num_free_v_controls + k_mod;

            //     for (int j = 0; j < 3; j++)
            //     {
            //         double v = variables[4 * r_id + j] - variables[4 * id + j];
            //         func_value += smoothness_weight * v * v;
            //         gradient[4 * r_id + j] += smoothness_weight * v;
            //         gradient[4 * id + j] -= smoothness_weight * v;
            //     }
            // }
        }

        for (int k = 0; k < numControls0; k++)
        {
            int k_mod = k % num_free_u_controls;
            for (int i = 1; i + 1 < numControls1; i++)
            {
                int l_id = ((i - 1) % num_free_v_controls) + num_free_v_controls * k_mod;
                int id = (i % num_free_v_controls) + num_free_v_controls * k_mod;
                int r_id = ((i + 1) % num_free_v_controls) + num_free_v_controls * k_mod;

                for (int j = 0; j < 3; j++)
                {
                    double v = 0.5 * (variables[4 * l_id + j] + variables[4 * r_id + j]) - variables[4 * id + j];
                    func_value += factor * smoothness_weight * v * v;
                    gradient[4 * l_id + j] += factor * smoothness_weight * v * 0.5;
                    gradient[4 * r_id + j] += factor * smoothness_weight * v * 0.5;
                    gradient[4 * id + j] -= factor * smoothness_weight * v;
                }
            }
            // for (int i = 0; i + 1 < numControls1; i++)
            // {
            //     int id = (i % num_free_v_controls) + num_free_v_controls * k_mod;
            //     int r_id = ((i + 1) % num_free_v_controls) + num_free_v_controls * k_mod;

            //     for (int j = 0; j < 3; j++)
            //     {
            //         double v = variables[4 * r_id + j] - variables[4 * id + j];
            //         func_value += smoothness_weight * v * v;
            //         gradient[4 * r_id + j] += smoothness_weight * v;
            //         gradient[4 * id + j] -= smoothness_weight * v;
            //     }
            // }
        }
    }

    func_value *= 0.5;
}


void MyNURBSSurf::write_obj(const std::string& filename, int u_samplerate, int v_samplerate)
{

    std::ofstream objfile(filename);

    const auto& ubasis = mysurf->GetBasisFunction(0);
    const auto& vbasis = mysurf->GetBasisFunction(1);

    double u_denom = is_u_closed ? u_samplerate : u_samplerate - 1;
    double v_denom = is_v_closed ? v_samplerate : v_samplerate - 1;

    double u_min = mysurf->GetUMin(), u_max = mysurf->GetUMax();
    double v_min = mysurf->GetVMin(), v_max = mysurf->GetVMax();

    int denom_u = is_u_closed ? u_samplerate : u_samplerate - 1, denom_v = is_v_closed ? v_samplerate : v_samplerate - 1;

    for (int i = 0; i < u_samplerate; i++)
    {
        double du = u_min + i * (u_max - u_min) / denom_u;
        for (int j = 0; j < v_samplerate; j++)
        {
            double dv = v_min + j * (v_max - v_min) / denom_v;
            vec3 p = mysurf->GetPosition(du, dv);
            objfile << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << std::endl;
        }
    }

    for (int i = 0; i < u_samplerate - 1; i++)
    {
        for (int j = 0; j < v_samplerate - 1; j++)
        {
            objfile << "f "
                << 1 + i * v_samplerate + j << ' '
                << 1 + (i + 1) * v_samplerate + j << ' '
                << 1 + (i + 1) * v_samplerate + j + 1 << ' '
                << 1 + i * v_samplerate + j + 1 << std::endl;
        }
    }

    if (is_u_closed)
    {
        for (int j = 0; j < v_samplerate - 1; j++)
        {
            objfile << "f "
                << 1 + (u_samplerate - 1) * v_samplerate + j << ' '
                << 1 + j << ' '
                << 1 + j + 1 << ' '
                << 1 + (u_samplerate - 1) * v_samplerate + j + 1 << std::endl;
        }
    }
    if (is_v_closed)
    {
        for (int i = 0; i < u_samplerate - 1; i++)
        {
            objfile << "f "
                << 1 + i * v_samplerate + v_samplerate - 1 << ' '
                << 1 + (i + 1) * v_samplerate + v_samplerate - 1 << ' '
                << 1 + (i + 1) * v_samplerate << ' '
                << 1 + i * v_samplerate << std::endl;
        }
    }

    if (is_u_closed && is_v_closed)
    {
        objfile << "f "
            << 1 << ' '
            << 1 + (u_samplerate - 1) * v_samplerate << ' '
            << 1 + u_samplerate * v_samplerate - 1 << ' '
            << 1 + v_samplerate - 1 << std::endl;
    }

    objfile.close();
}

void MyNURBSSurf::write_nurbs(const std::string& filename)
{
    std::ofstream objfile(filename);

    const auto& ubasis = mysurf->GetBasisFunction(0);
    const auto& vbasis = mysurf->GetBasisFunction(1);

    double u_min = mysurf->GetUMin(), u_max = mysurf->GetUMax();
    double v_min = mysurf->GetVMin(), v_max = mysurf->GetVMax();
    int num_free_u_controls = is_u_closed ? mysurf->GetNumControls(0) - mysurf->GetBasisFunction(0).GetDegree() : mysurf->GetNumControls(0);
    int num_free_v_controls = is_v_closed ? mysurf->GetNumControls(1) - mysurf->GetBasisFunction(1).GetDegree() : mysurf->GetNumControls(1);

    objfile << "# surf" << std::endl;
    objfile << "g surf" << std::endl;

    for (int j = 0; j < num_free_v_controls; j++)
    {
        for (int i = 0; i < num_free_u_controls; i++)
        {
            const auto& p = mysurf->GetControl(i, j);
            objfile << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << mysurf->GetWeight(i, j) << std::endl;
        }
    }

    // for (int i = 0; i < mysurf->GetNumControls(0); i++)
    // {
    //     for (int j = 0; j <mysurf->GetNumControls(1); j++)
    //     {
    //         const auto &p = mysurf->GetControl(i, j);
    //         objfile << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << mysurf->GetWeight(i, j) << std::endl;
    //     }
    // }

    objfile << "cstype rat bspline" << std::endl;
    objfile << "deg " << ubasis.GetDegree() << ' ' << vbasis.GetDegree() << std::endl;
    objfile << "surf " << u_min << ' ' << u_max << ' ' << v_min << ' ' << v_max;
    for (int j = 0; j < mysurf->GetNumControls(1); j++)
    {
        int jj = j % num_free_v_controls;
        for (int i = 0; i < mysurf->GetNumControls(0); i++)
        {
            int ii = i % num_free_u_controls;
            objfile << ' ' << jj * num_free_u_controls + ii + 1;
        }
    }
    objfile << std::endl
        << "parm u";
    for (int i = 0; i < ubasis.GetNumKnots(); i++)
        objfile << ' ' << ubasis.GetKnots()[i];
    objfile << std::endl
        << "parm v";
    for (int i = 0; i < vbasis.GetNumKnots(); i++)
        objfile << ' ' << vbasis.GetKnots()[i];
    objfile << std::endl
        << "end" << std::endl;

    objfile.close();
}

SurfFitting::SurfFitting(const std::vector<std::vector<vec3>>& points, const std::vector<std::vector<double>>& weights,
    const std::vector<vec3>& addon_points, const std::vector<double>& addon_weights,
    bool u_closed, bool v_closed, int u_degree, int v_degree, int u_num_ctrl, int v_num_ctrl,
    int num_iter, int sample_rate, double smoothness_weight)
{
    if (smoothness_weight > 0.0)
    {
        if (points.empty() ||
            u_degree < 1 || u_num_ctrl < u_degree + 1 ||
            v_degree < 1 || v_num_ctrl < v_degree + 1)
            return;
    }
    else
    {
        if (points.empty() ||
            points.size() < u_num_ctrl + u_degree + 1 || u_degree < 1 || u_num_ctrl < u_degree + 1 ||
            points.front().size() < v_num_ctrl + v_degree + 1 || v_degree < 1 || v_num_ctrl < v_degree + 1)
            return;
    }


    std::vector<std::vector<double>> u_curvelen(points.front().size());
    for (size_t j = 0; j < points.front().size(); j++)
    {
        double len = 0;
        for (size_t i = 0; i < points.size() - 1; i++)
        {
            u_curvelen[j].push_back(len);
            len += gte::Length(points[i + 1][j] - points[i][j]);
        }
        u_curvelen[j].push_back(len);
    }
    std::vector<std::vector<vec3>> u_controls(points.front().size());
    for (auto& e : u_controls)
        e.resize(u_num_ctrl);

    for (int j = 0; j < points.front().size(); j++)
    {
        int k = 0;
        for (int i = 0; i < u_num_ctrl; i++)
        {
            double t = i * u_curvelen[j].back() / u_num_ctrl;
            //while (k < (int)points.size() && t >= u_curvelen[j][k])
            while (k < (int)points.size() - 1 && t >= u_curvelen[j][k])
                k++;
            double shortlen = u_curvelen[j][k] - u_curvelen[j][k - 1] + 1.0e-12;
            u_controls[j][i] = (t - u_curvelen[j][k - 1]) / shortlen * points[k - 1][j] + (u_curvelen[j][k] - t) / shortlen * points[k][j];
        }
    }

    std::vector<std::vector<double>> v_curvelen(u_num_ctrl);
    for (size_t j = 0; j < u_num_ctrl; j++)
    {
        double len = 0;
        for (size_t i = 0; i < points.front().size() - 1; i++)
        {
            v_curvelen[j].push_back(len);
            len += gte::Length(u_controls[i + 1][j] - u_controls[i][j]);
        }
        v_curvelen[j].push_back(len);
    }

    std::vector<std::vector<vec3>> controls(u_num_ctrl);
    for (auto& e : controls)
        e.resize(v_num_ctrl);

    for (int i = 0; i < u_num_ctrl; i++)
    {
        int k = 0;
        for (size_t j = 0; j < v_num_ctrl; j++)
        {
            double t = j * v_curvelen[i].back() / v_num_ctrl;
            //while (t >= v_curvelen[i][k] && k < (int)points.front().size())
            while (k < (int)points.front().size() - 1 && t >= v_curvelen[i][k])
                k++;
            double shortlen = v_curvelen[i][k] - v_curvelen[i][k - 1] + 1.0e-12;
            controls[i][j] = (t - v_curvelen[i][k - 1]) / shortlen * u_controls[k - 1][i] + (v_curvelen[i][k] - t) / shortlen * u_controls[k][i];
        }
    }

    ////////////////////////
    mysurf = new MyNURBSSurf(u_degree, u_closed, v_degree, v_closed, controls);
    MyFitBase fit;

    mysurf->smoothness() = smoothness_weight;

    mysurf->set_opt_weight(false);
    fit.optimize(*mysurf, points, weights, addon_points, addon_weights, num_iter);
    mysurf->set_opt_weight(true);
    fit.optimize(*mysurf, points, weights, addon_points, addon_weights, num_iter);
    if (!addon_points.empty())
    {
        mysurf->set_opt_addon(true, sample_rate);
        fit.optimize(*mysurf, points, weights, addon_points, addon_weights, num_iter);
    }
}