#pragma once

#include <ostream>
#include <cmath>
#include "TinyVector.h"
#include "Mathematics/NURBSCurve.h"

typedef TinyVector<double, 3> vec3d;

class MyCurve
{
public:
    MyCurve()
        : t_min(0.0), t_max(1.0), closed(false)
    {
        start_vertex_index = (size_t)-1;
        end_vertex_index = (size_t)-1;
    }

    const bool &is_closed() const { return closed; }
    bool &is_closed() { return closed; }
    const double &get_t_min() const { return t_min; }
    double &get_t_min() { return t_min; }
    const double &get_t_max() const { return t_max; }
    double &get_t_max() { return t_max; }

    const size_t &get_start_vertex_index() const { return start_vertex_index; }
    size_t &get_start_vertex_index() { return start_vertex_index; }
    const size_t &get_end_vertex_index() const { return end_vertex_index; }
    size_t &get_end_vertex_index() { return end_vertex_index; }

    virtual vec3d GetPosition(double t) = 0;

    //tangent
    virtual vec3d GetTangent(double t)
    {
        //impl for all curves withiin this class, not normalized
        return vec3d(0.0, 0.0, 0.0);
    }

    void write_obj_curve(std::ostream &out, size_t &vcounter, int div)
    {
        int denom = closed ? div : div - 1;

        for (int i = 0; i < div; i++)
        {
            double t = t_min + i * (t_max - t_min) / denom;
            out << "v " << GetPosition(t) << std::endl;
        }

        for (int i = 0; i < div - 1; i++)
            out << "l " << i + vcounter << ' ' << i + vcounter + 1 << std::endl;
        if (closed)
            out << "l " << vcounter + div - 1 << ' ' << vcounter << std::endl;

        vcounter += div;
    }
    void write_data_curve(std::ostream &out, int div)
    {
        out << "curve " << (closed ? "closed " : "open ") << div << std::endl;
        int denom = closed ? div : div - 1;

        for (int i = 0; i < div; i++)
        {
            double t = t_min + i * (t_max - t_min) / denom;
            out << GetPosition(t) << std::endl;
        }
    }

//protected:
    double t_min, t_max;
    bool closed;
    size_t start_vertex_index, end_vertex_index;
};
//////////////////////////////////////////////
class MyLine : public MyCurve
{
public:
    MyLine(const vec3d &_start, const vec3d &_end)
        : start(_start), end(_end)
    {
        this->closed = false;
        this->t_min = 0;
        this->t_max = 1;
    }
    vec3d GetPosition(double t)
    {
        return start + t * (end - start);
    }

    vec3d GetTangent(double t)
    {
        vec3d tang = end - start;
        //tang.Normalize();
        return tang;
    }

protected:
    vec3d start, end;
};
//////////////////////////////////////////////
class MyCircle : public MyCurve
{
public:
    MyCircle(const vec3d &_loc, const vec3d &_dirx, const vec3d &_diry,
             double _radius, double _t_min, double _t_max, bool _closed)
        : loc(_loc), dirx(_dirx), diry(_diry), radius(_radius)
    {
        this->t_min = _t_min;
        this->t_max = _t_max;
        this->closed = _closed;
    }
    vec3d GetPosition(double t)
    {
        return loc + radius * (cos(t) * dirx + sin(t) * diry);
    }
    
    vec3d GetTangent(double t)
    {
        return (-sin(t) * dirx + cos(t) * diry) * radius;
    }

protected:
    vec3d loc, dirx, diry;
    double radius;
};
//////////////////////////////////////////////
class MyEllipse : public MyCurve
{
public:
    MyEllipse(const vec3d &_loc, const vec3d &_dirx, const vec3d &_diry,
              double _x_radius, double _y_radius, double _t_min, double _t_max, bool _closed)
        : loc(_loc), dirx(_dirx), diry(_diry), x_radius(_x_radius), y_radius(_y_radius)
    {
        this->t_min = _t_min;
        this->t_max = _t_max;
        this->closed = _closed;
    }
    vec3d GetPosition(double t)
    {
        return loc + x_radius * cos(t) * dirx + y_radius * sin(t) * diry;
    }
    
    vec3d GetTangent(double t)
    {
        return -x_radius * sin(t) * dirx + y_radius * cos(t) * diry;
    }

protected:
    vec3d loc, dirx, diry;
    double x_radius, y_radius;
};
//////////////////////////////////////////////
//class MySplineCurve : public MyCurve
//{
//public:
//    MySplineCurve(
//        int degree,
//        const std::vector<gte::Vector<3, double>> &controls,
//        const std::vector<double> &myknots,
//        const std::vector<double> &myweights,
//        double _t_min, double _t_max, bool _closed)
//    {
//        this->t_min = _t_min;
//        this->t_max = _t_max;
//        this->closed = _closed;
//
//        gte::BasisFunctionInput<double> my_input;
//        my_input.degree = degree;
//        my_input.numControls = (int)controls.size();
//        my_input.periodic = _closed;
//        my_input.uniform = false;
//
//        std::vector<std::pair<double, int>> knots_stataus;
//        knots_stataus.push_back(std::make_pair(myknots[0], 1));
//        for (size_t i = 1; i < myknots.size(); i++)
//        {
//            if (myknots[i] == knots_stataus.back().first)
//                knots_stataus.back().second++;
//            else
//                knots_stataus.push_back(std::make_pair(myknots[i], 1));
//        }
//
//        my_input.numUniqueKnots = (int)knots_stataus.size();
//        my_input.uniqueKnots.resize(my_input.numUniqueKnots);
//        for (size_t i = 0; i < knots_stataus.size(); i++)
//        {
//            my_input.uniqueKnots[i].t = knots_stataus[i].first;
//            my_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
//        }
//        mycurve = new gte::NURBSCurve<3, double>(my_input, controls.data(), myweights.data());
//    }
//
//    ~MySplineCurve()
//    {
//        if (mycurve)
//            delete mycurve;
//    }
//
//    vec3d GetPosition(double t)
//    {
//        gte::Vector<3, double> V = mycurve->GetPosition(t);
//        return vec3d(V[0], V[1], V[2]);
//    }
//
//protected:
//    gte::NURBSCurve<3, double> *mycurve;
//};

//update version
class MySplineCurve : public MyCurve
{
public:
    MySplineCurve(
        int degree,
        const std::vector<gte::Vector<3, double>>& controls,
        const std::vector<double>& myknots,
        const std::vector<double>& myweights,
        double _t_min, double _t_max, bool _closed)
    {
        this->t_min = _t_min;
        this->t_max = _t_max;
        this->closed = _closed;

        gte::BasisFunctionInput<double> my_input;
        my_input.degree = degree;
        my_input.numControls = (int)controls.size();
        //my_input.periodic = _closed;
        my_input.periodic = false;
        my_input.uniform = false;

        std::vector<std::pair<double, int>> knots_stataus;
        knots_stataus.push_back(std::make_pair(myknots[0], 1));
        for (size_t i = 1; i < myknots.size(); i++)
        {
            if (myknots[i] == knots_stataus.back().first)
                knots_stataus.back().second++;
            else
                knots_stataus.push_back(std::make_pair(myknots[i], 1));
        }

        my_input.numUniqueKnots = (int)knots_stataus.size();
        my_input.uniqueKnots.resize(my_input.numUniqueKnots);
        for (size_t i = 0; i < knots_stataus.size(); i++)
        {
            my_input.uniqueKnots[i].t = knots_stataus[i].first;
            my_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
        }
        mycurve = new gte::NURBSCurve<3, double>(my_input, controls.data(), myweights.data());
    }

    ~MySplineCurve()
    {
        if (mycurve)
            delete mycurve;
    }

    vec3d GetPosition(double t)
    {
        gte::Vector<3, double> V = mycurve->GetPosition(t);
        return vec3d(V[0], V[1], V[2]);
    }

    vec3d GetTangent(double t)
    {
        gte::Vector<3, double> V = mycurve->GetTangent(t);
        return vec3d(V[0], V[1], V[2]);
    }

protected:
    gte::NURBSCurve<3, double>* mycurve;
};