#pragma once
#include "igl/copyleft/cgal/mesh_boolean.h"

typedef CGAL::Epeck Kernel;
typedef Kernel::FT ExactScalar;

namespace Utils {

	inline double to_double(const double &v) {
		return v;
	}



	inline double to_double(const ExactScalar &v) {
		return CGAL::to_double(v);
	}


	template <class Real>
	inline void to_double(std::vector<Real> &vi, std::vector<double> &vo) {
		vo.resize(vi.size());
		for (int i = 0; i < vi.size(); i++) {
			vo[i] = to_double(vi[i]);
		}
	}

	inline void type_conv(double &in, double&ou) {
		ou = in;
	}

	inline void type_conv(ExactScalar &in, ExactScalar &ou) {
		ou = in;
	}

	inline void type_conv(ExactScalar &in, double &ou) {
		ou = to_double(in);
	}

	inline void type_conv(double &in, ExactScalar &ou) {
		ou = ExactScalar(in);
	}

}