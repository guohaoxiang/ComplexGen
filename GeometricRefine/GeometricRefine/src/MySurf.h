#pragma once

#include "MyCurve.h"
#include "Mathematics/NURBSSurface.h"

class MySurf
{
public:
    MySurf()
    {
        u_min = 0, u_max = 1, v_min = 0, v_max = 1;
        u_closed = v_closed = false;
    }
    virtual vec3d GetPosition(double u, double v) = 0;
    virtual vec3d GetNormal(double u, double v)
    {
        //impl for all elements except thin-plate in surffitters, impl for all elements in this file
        return vec3d(0.0, 0.0, 0.0);
    }
    void write_obj_surf(std::ostream &out, size_t &vcounter, int u_div, int v_div)
    {
        int denom_u = u_div - 1, denom_v = v_div - 1;
        if (u_closed)
        {
            denom_u = u_div;
        }
        if (v_closed)
        {
            denom_v = v_div;
        }

        for (int i = 0; i < u_div; i++)
        {
            double du = u_min + i * (u_max - u_min) / denom_u;
            for (int j = 0; j < v_div; j++)
            {
                double dv = v_min + j * (v_max - v_min) / denom_v;
                out << "v " << GetPosition(du, dv) << std::endl;
            }
        }

        for (int i = 0; i < u_div - 1; i++)
        {
            for (int j = 0; j < v_div - 1; j++)
            {
                out << "f "
                    << vcounter + i * v_div + j << ' '
                    << vcounter + (i + 1) * v_div + j << ' '
                    << vcounter + (i + 1) * v_div + j + 1 << ' '
                    << vcounter + i * v_div + j + 1 << std::endl;
            }
        }

        if (u_closed)
        {
            for (int j = 0; j < v_div - 1; j++)
            {
                out << "f "
                    << vcounter + (u_div - 1) * v_div + j << ' '
                    << vcounter + j << ' '
                    << vcounter + j + 1 << ' '
                    << vcounter + (u_div - 1) * v_div + j + 1 << std::endl;
            }
        }

        if (v_closed)
        {
            for (int i = 0; i < u_div - 1; i++)
            {
                out << "f "
                    << vcounter + i * v_div + v_div - 1 << ' '
                    << vcounter + (i + 1) * v_div + v_div - 1 << ' '
                    << vcounter + (i + 1) * v_div << ' '
                    << vcounter + i * v_div << std::endl;
            }
        }

        if (u_closed && v_closed)
        {
            out << "f "
                << vcounter << ' '
                << vcounter + (u_div - 1) * v_div << ' '
                << vcounter + u_div * v_div - 1 << ' '
                << vcounter + v_div - 1 << std::endl;
        }

        vcounter += u_div * v_div;
    }

    

    
//protected:
    double u_min, u_max, v_min, v_max;
    bool u_closed, v_closed;
};
//////////////////////////////////////////////
class MyPlane : public MySurf
{
public:
    MyPlane()
    {
        ;
    }
    MyPlane(const vec3d &_loc, const vec3d &_xdir, const vec3d &_ydir,
            double _u_min, double _u_max, double _v_min, double _v_max)
        : loc(_loc), xdir(_xdir), ydir(_ydir)
    {
        this->u_closed = this->v_closed = false;
        this->u_min = _u_min, this->u_max = _u_max, this->v_min = _v_min, this->v_max = _v_max;
    }
    vec3d GetPosition(double u, double v)
    {
        return loc + u * xdir + v * ydir;
    }

    vec3d GetNormal(double u, double v)
    {
        vec3d normal = xdir.Cross(ydir);
        normal.Normalize();
        return normal;
    }

protected:
    vec3d loc, xdir, ydir;
};
//////////////////////////////////////////////
class MyCylinder : public MySurf
{
public:
    MyCylinder()
        : radius(0.0)
    {
        ;
    }
    MyCylinder(const vec3d &_loc, const vec3d &_xdir, const vec3d &_ydir, const vec3d &_zdir, double _radius,
               bool _u_closed, double _u_min, double _u_max, double _v_min, double _v_max)
        : loc(_loc), xdir(_xdir), ydir(_ydir), zdir(_zdir), radius(_radius)
    {
        this->u_closed = _u_closed, this->v_closed = false;
        this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max, this->v_min = _v_min, this->v_max = _v_max;
    }
    vec3d GetPosition(double u, double v)
    {
        return loc + radius * (cos(u) * xdir + sin(u) * ydir) + v * zdir;
    }

    vec3d GetNormal(double u, double v)
    {
        vec3d uvec = radius * (-sin(u) * xdir + cos(u) * ydir);
        vec3d vvec = zdir;
        vec3d normal = uvec.Cross(vvec);
        normal.Normalize();
        return normal;
    }

protected:
    vec3d loc, xdir, ydir, zdir;
    double radius;
};
//////////////////////////////////////////////
class MyCone : public MySurf
{
public:
    MyCone()
        :radius(0.0), angle(0.0)
    {}
    MyCone(const vec3d &_loc, const vec3d &_xdir, const vec3d &_ydir, const vec3d &_zdir,
           double _radius, double _angle,
           bool _u_closed, double _u_min, double _u_max, double _v_min, double _v_max)
        : loc(_loc), xdir(_xdir), ydir(_ydir), zdir(_zdir), radius(_radius), angle(_angle)
    {
        this->u_closed = _u_closed, this->v_closed = false;
        this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max, this->v_min = _v_min, this->v_max = _v_max;
    }
    vec3d GetPosition(double u, double v)
    {
        return loc + (radius + v * sin(angle)) * (cos(u) * xdir + sin(u) * ydir) + v * cos(angle) * zdir;
    }

    vec3d GetNormal(double u, double v)
    {
        vec3d uvec = (radius + v * sin(angle)) * (-sin(u) * xdir + cos(u) * ydir);
        vec3d vvec = sin(angle) * (cos(u) * xdir + sin(u) * ydir) + cos(angle) * zdir;
        vec3d normal = uvec.Cross(vvec);
        normal.Normalize();
        return normal;
    }


protected:
    vec3d loc, xdir, ydir, zdir;
    double radius, angle;
};
//////////////////////////////////////////////
class MyTorus : public MySurf
{
public:
    MyTorus(const vec3d &_loc, const vec3d &_xdir, const vec3d &_ydir, const vec3d &_zdir,
            double _max_radius, double _min_radius, bool _u_closed, bool _v_closed,
            double _u_min, double _u_max, double _v_min, double _v_max)
        : loc(_loc), xdir(_xdir), ydir(_ydir), zdir(_zdir),
          max_radius(_max_radius), min_radius(_min_radius)
    {
        this->u_closed = _u_closed, this->v_closed = _v_closed;
        this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
        this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
    }
    vec3d GetPosition(double u, double v)
    {
        return loc + (max_radius + min_radius * cos(v)) * (cos(u) * xdir + sin(u) * ydir) + min_radius * sin(v) * zdir;
    }

    vec3d GetNormal(double u, double v)
    {
        vec3d uvec = (max_radius + min_radius * cos(v)) * (-sin(u) * xdir + cos(u) * ydir);
        vec3d vvec = -min_radius * sin(v) * (cos(u) * xdir + sin(u) * ydir) + min_radius * cos(v) * zdir;
        vec3d normal = uvec.Cross(vvec);
        normal.Normalize();
        return normal;
    }

protected:
    vec3d loc, xdir, ydir, zdir;
    double max_radius, min_radius;
};
//////////////////////////////////////////////
class MySphere : public MySurf
{
public:
    MySphere(const vec3d &_loc, const vec3d &_xdir, const vec3d &_ydir, const vec3d &_zdir,
             double _radius, bool _u_closed, bool _v_closed,
             double _u_min, double _u_max, double _v_min, double _v_max)
        : loc(_loc), xdir(_xdir), ydir(_ydir), zdir(_zdir), radius(_radius)
    {
        this->u_closed = _u_closed, this->v_closed = _v_closed;
        this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
        this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? M_PI : _v_max;
    }
    vec3d GetPosition(double u, double v)
    {
        return loc + radius * (cos(v) * (cos(u) * xdir + sin(u) * ydir) + sin(v) * zdir);
    }

    vec3d GetNormal(double u, double v)
    {
        vec3d normal = GetPosition(u, v) - loc;
        normal.Normalize();
        return normal;
    }

protected:
    vec3d loc, xdir, ydir, zdir;
    double radius;
};
//////////////////////////////////////////////
//class MySplineSurf : public MySurf
//{
//public:
//    MySplineSurf(
//        int u_degree, int v_degree,
//        const std::vector<gte::Vector<3, double>> &controls,
//        const std::vector<double> &my_uknots, const std::vector<double> &my_vknots,
//        const std::vector<double> &myweights,
//        bool _u_closed, bool _v_closed,
//        double _u_min, double _u_max, double _v_min, double _v_max)
//    {
//        this->u_closed = _u_closed, this->v_closed = _v_closed;
//        this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
//        this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
//
//        gte::BasisFunctionInput<double> my_u_input;
//        my_u_input.degree = u_degree;
//        my_u_input.numControls = (int)my_uknots.size() - u_degree - 1;
//        my_u_input.periodic = _u_closed;
//        my_u_input.uniform = false;
//        std::vector<std::pair<double, int>> knots_stataus;
//        knots_stataus.push_back(std::make_pair(my_uknots[0], 1));
//        for (size_t i = 1; i < my_uknots.size(); i++)
//        {
//            if (my_uknots[i] == knots_stataus.back().first)
//                knots_stataus.back().second++;
//            else
//                knots_stataus.push_back(std::make_pair(my_uknots[i], 1));
//        }
//
//        my_u_input.numUniqueKnots = (int)knots_stataus.size();
//        my_u_input.uniqueKnots.resize(my_u_input.numUniqueKnots);
//        for (size_t i = 0; i < knots_stataus.size(); i++)
//        {
//            my_u_input.uniqueKnots[i].t = knots_stataus[i].first;
//            my_u_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
//        }
//
//        gte::BasisFunctionInput<double> my_v_input;
//        my_v_input.degree = v_degree;
//        my_v_input.numControls = (int)my_vknots.size() - v_degree - 1;
//        my_v_input.periodic = _v_closed;
//        my_v_input.uniform = false;
//
//        knots_stataus.resize(0);
//        knots_stataus.push_back(std::make_pair(my_vknots[0], 1));
//        for (size_t i = 1; i < my_vknots.size(); i++)
//        {
//            if (my_vknots[i] == knots_stataus.back().first)
//                knots_stataus.back().second++;
//            else
//                knots_stataus.push_back(std::make_pair(my_vknots[i], 1));
//        }
//
//        my_v_input.numUniqueKnots = (int)knots_stataus.size();
//        my_v_input.uniqueKnots.resize(my_v_input.numUniqueKnots);
//        for (size_t i = 0; i < knots_stataus.size(); i++)
//        {
//            my_v_input.uniqueKnots[i].t = knots_stataus[i].first;
//            my_v_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
//        }
//        mysurf = new gte::NURBSSurface<3, double>(my_u_input, my_v_input, controls.data(), myweights.data());
//    }
//
//    ~MySplineSurf()
//    {
//        if (mysurf)
//            delete mysurf;
//    }
//
//    vec3d GetPosition(double u, double v)
//    {
//        gte::Vector<3, double> V = mysurf->GetPosition(u, v);
//        return vec3d(V[0], V[1], V[2]);
//    }
//
//    vec3d GetNormal(double u, double v)
//    {
//        gte::Vector<3, double> U = mysurf->GetUTangent(u, v);
//        gte::Vector<3, double> V = mysurf->GetVTangent(u, v);
//        vec3d uvec(U[0], U[1], U[2]);
//        vec3d vvec(V[0], V[1], V[2]);
//        vec3d normal = uvec.Cross(vvec);
//        normal.Normalize();
//        return normal;
//    }
//
//protected:
//    gte::NURBSSurface<3, double> *mysurf;
//};
//

//update version
class MySplineSurf : public MySurf
{
public:
    MySplineSurf(
        int u_degree, int v_degree,
        const std::vector<gte::Vector<3, double>>& controls,
        const std::vector<double>& my_uknots, const std::vector<double>& my_vknots,
        const std::vector<double>& myweights,
        bool _u_closed, bool _v_closed,
        double _u_min, double _u_max, double _v_min, double _v_max)
    {
        this->u_closed = _u_closed, this->v_closed = _v_closed;
        // this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
        // this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
        this->u_min = _u_min, this->u_max = _u_max;
        this->v_min = _v_min, this->v_max = _v_max;

        gte::BasisFunctionInput<double> my_u_input;
        my_u_input.degree = u_degree;
        my_u_input.numControls = (int)my_uknots.size() - u_degree - 1;
        //my_u_input.periodic = _u_closed;
        my_u_input.periodic = false;
        my_u_input.uniform = false;
        std::vector<std::pair<double, int>> knots_stataus;
        knots_stataus.push_back(std::make_pair(my_uknots[0], 1));
        for (size_t i = 1; i < my_uknots.size(); i++)
        {
            if (my_uknots[i] == knots_stataus.back().first)
                knots_stataus.back().second++;
            else
                knots_stataus.push_back(std::make_pair(my_uknots[i], 1));
        }

        my_u_input.numUniqueKnots = (int)knots_stataus.size();
        my_u_input.uniqueKnots.resize(my_u_input.numUniqueKnots);
        for (size_t i = 0; i < knots_stataus.size(); i++)
        {
            my_u_input.uniqueKnots[i].t = knots_stataus[i].first;
            my_u_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
        }

        gte::BasisFunctionInput<double> my_v_input;
        my_v_input.degree = v_degree;
        my_v_input.numControls = (int)my_vknots.size() - v_degree - 1;
        //my_v_input.periodic = _v_closed;
        my_v_input.periodic = false;
        my_v_input.uniform = false;

        knots_stataus.resize(0);
        knots_stataus.push_back(std::make_pair(my_vknots[0], 1));
        for (size_t i = 1; i < my_vknots.size(); i++)
        {
            if (my_vknots[i] == knots_stataus.back().first)
                knots_stataus.back().second++;
            else
                knots_stataus.push_back(std::make_pair(my_vknots[i], 1));
        }

        my_v_input.numUniqueKnots = (int)knots_stataus.size();
        my_v_input.uniqueKnots.resize(my_v_input.numUniqueKnots);
        for (size_t i = 0; i < knots_stataus.size(); i++)
        {
            my_v_input.uniqueKnots[i].t = knots_stataus[i].first;
            my_v_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
        }
        mysurf = new gte::NURBSSurface<3, double>(my_u_input, my_v_input, controls.data(), myweights.data());
    }

    ~MySplineSurf()
    {
        if (mysurf)
            delete mysurf;
    }

    vec3d GetPosition(double u, double v)
    {
        gte::Vector<3, double> V = mysurf->GetPosition(u, v);
        return vec3d(V[0], V[1], V[2]);
    }
    
    vec3d GetNormal(double u, double v)
    {
        gte::Vector<3, double> V = mysurf->GetPosition(u, v);
        gte::Vector<3, double> PU = mysurf->GetUTangent(u, v);
        gte::Vector<3, double> PV = mysurf->GetVTangent(u, v);
        //pos[0] = V[0], pos[1] = V[1], pos[2] = V[2];
        return vec3d(PU[0], PU[1], PU[2]).UnitCross(vec3d(PV[0], PV[1], PV[2]));
    }

protected:
    gte::NURBSSurface<3, double>* mysurf;
};


//////////////////////////////////////////////
class MyExtrusionSurf : public MySurf
{
public:
    MyExtrusionSurf(
        int u_degree,
        const vec3d &_zdir,
        const std::vector<gte::Vector<3, double>> &controls,
        const std::vector<double> &my_uknots,
        const std::vector<double> &myweights,
        bool _u_closed,
        double _u_min, double _u_max, double _v_min, double _v_max)
        : zdir(_zdir)
    {
        this->u_closed = _u_closed, this->v_closed = false;
        this->u_min = _u_min, this->u_max = _u_max;
        this->v_min = _v_min, this->v_max = _v_max;
        mycurve = new MySplineCurve(u_degree, controls, my_uknots, myweights, u_min, u_max, u_closed);
    }
    MyExtrusionSurf(const vec3d &_start, const vec3d &_end,
                    const vec3d &_zdir, double _u_min, double _u_max, double _v_min, double _v_max)
        : zdir(_zdir)
    {
        this->u_closed = this->v_closed = false;
        this->u_min = _u_min, this->u_max = _u_max;
        this->v_min = _v_min, this->v_max = _v_max;
        mycurve = new MyLine(_start, _end);
    }
    MyExtrusionSurf(const vec3d &_loc, const vec3d &_dirx, const vec3d &_diry, double _radius, bool _u_closed,
                    const vec3d &_zdir, double _u_min, double _u_max, double _v_min, double _v_max)
        : zdir(_zdir)
    {
        this->u_closed = _u_closed, this->v_closed = false;
        this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
        this->v_min = _v_min, this->v_max = _v_max;
        mycurve = new MyCircle(_loc, _dirx, _diry, _radius, _u_min, _u_max, u_closed);
    }
    MyExtrusionSurf(const vec3d &_loc, const vec3d &_dirx, const vec3d &_diry, double _x_radius, double _y_radius, bool _u_closed,
                    const vec3d &_zdir, double _u_min, double _u_max, double _v_min, double _v_max)
        : zdir(_zdir)
    {
        this->u_closed = _u_closed, this->v_closed = false;
        this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
        this->v_min = _v_min, this->v_max = _v_max;
        mycurve = new MyEllipse(_loc, _dirx, _diry, _x_radius, _y_radius, _u_min, _u_max, u_closed);
    }

    ~MyExtrusionSurf()
    {
        if (mycurve)
            delete mycurve;
    }

    vec3d GetPosition(double u, double v)
    {
        return mycurve->GetPosition(u) + v * zdir;
    }

    vec3d GetNormal(double u, double v)
    {
        vec3d utang = mycurve->GetTangent(u);
        vec3d normal = utang.Cross(zdir);
        normal.Normalize();
        return normal;
    }

protected:
    MyCurve *mycurve;
    vec3d zdir;
};
//////////////////////////////////////////////
class MyRevolutionSurf : public MySurf
{
public:
    MyRevolutionSurf(
        int u_degree,
        const vec3d &_loc, const vec3d &_zdir,
        const std::vector<gte::Vector<3, double>> &controls,
        const std::vector<double> &my_uknots,
        const std::vector<double> &myweights,
        bool _u_closed, bool _v_closed,
        double _u_min, double _u_max, double _v_min, double _v_max)
        : loc(_loc), zdir(_zdir)
    {
        this->u_closed = _u_closed, this->v_closed = _v_closed;
        this->u_min = _u_min, this->u_max = _u_max;
        this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
        mycurve = new MySplineCurve(u_degree, controls, my_uknots, myweights, _u_min, _u_max, _u_closed);
    }

    MyRevolutionSurf(const vec3d &_start, const vec3d &_end,
                     const vec3d &_loc, const vec3d &_zdir, bool _v_closed, double _u_min, double _u_max, double _v_min, double _v_max)
        : loc(_loc), zdir(_zdir)
    {
        this->u_closed = false;
        this->v_closed = _v_closed;
        this->u_min = _u_min, this->u_max = _u_max;
        this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
        mycurve = new MyLine(_start, _end);
    }
    MyRevolutionSurf(const vec3d &c_loc, const vec3d &_dirx, const vec3d &_diry, double _radius, bool _u_closed, bool _v_closed,
                     const vec3d &_loc, const vec3d &_zdir, double _u_min, double _u_max, double _v_min, double _v_max)
        : loc(_loc), zdir(_zdir)
    {
        this->u_closed = _u_closed, this->v_closed = _v_closed;
        this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
        this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
        mycurve = new MyCircle(c_loc, _dirx, _diry, _radius, _u_min, _u_max, _u_closed);
    }
    MyRevolutionSurf(const vec3d &c_loc, const vec3d &_dirx, const vec3d &_diry, double _x_radius, double _y_radius, bool _u_closed, bool _v_closed,
                     const vec3d &_loc, const vec3d &_zdir, double _u_min, double _u_max, double _v_min, double _v_max)
        : loc(_loc), zdir(_zdir)
    {
        this->u_closed = _u_closed, this->v_closed = _v_closed;
        this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
        this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
        mycurve = new MyEllipse(c_loc, _dirx, _diry, _x_radius, _y_radius, _u_min, _u_max, _u_closed);
    }

    ~MyRevolutionSurf()
    {
        if (mycurve)
            delete mycurve;
    }

    vec3d GetPosition(double u, double v)
    {
        const double cos_angle = cos(v);
        const double sin_angle = sin(v);
        vec3d V = mycurve->GetPosition(u) - loc;
        return loc + cos_angle * V + (1 - cos_angle) * V.Dot(zdir) * zdir + sin_angle * zdir.Cross(V);
    }

    vec3d GetNormal(double u, double v)
    {
        const double cos_angle = cos(v);
        const double sin_angle = sin(v);
        vec3d X = mycurve->GetPosition(u) - loc;
        vec3d tang_X = mycurve->GetTangent(u);
        vec3d utang = cos_angle * tang_X + (1 - cos_angle) * tang_X.Dot(zdir) * zdir + sin_angle * zdir.Cross(tang_X);
        vec3d vtang = -sin_angle * X + sin_angle * X.Dot(zdir) * zdir + cos_angle * zdir.Cross(X);
        vec3d normal = utang.Cross(vtang);
        normal.Normalize();
        return normal;
    }

protected:
    MyCurve *mycurve;
    vec3d loc, zdir;
};