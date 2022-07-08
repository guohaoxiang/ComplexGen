#ifndef TINY_MATRIX_H
#define TINY_MATRIX_H

#include "TinyVector.h"

//////////////////////////////////////////////////////////////////////////
template <typename T>
class ColumnMatrix3
{
private:
	TinyVector<T, 3> V_[3];
	//////////////////////////////////////////////////////////////////////////
public:
	ColumnMatrix3()
	{
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3(const TinyVector<T, 3> &T0, const TinyVector<T, 3> &T1, const TinyVector<T, 3> &T2)
	{
		V_[0] = T0, V_[1] = T1, V_[2] = T2;
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3(const T *T0, const T *T1, const T *T2)
	{
		V_[0] = T0, V_[1] = T1, V_[2] = T2;
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3(const T *V)
	{
		V_[0] = &V[0];
		V_[1] = &V[1];
		V_[2] = &V[2];
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3(const ColumnMatrix3<T> &CM)
	{
		V_[0] = CM[0], V_[1] = CM[1], V_[2] = CM[2];
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3(const T a0, const T a1, const T a2, const T a3, const T a4, const T a5,
		const T a6, const T a7, const T a8, bool columnMajor = true)
	{
		SetEntries(a0, a1, a2, a3, a4, a5, a6, a7, a8, columnMajor);
	}
	//////////////////////////////////////////////////////////////////////////
	void SetEntries(const T a0, const T a1, const T a2, const T a3, const T a4, const T a5,
		const T a6, const T a7, const T a8, bool columnMajor = true)
	{
		if (columnMajor)
		{
			V_[0][0] = a0;
			V_[0][1] = a1;
			V_[0][2] = a2;
			V_[1][0] = a3;
			V_[1][1] = a4;
			V_[1][2] = a5;
			V_[2][0] = a6;
			V_[2][1] = a7;
			V_[2][2] = a8;
		}
		else
		{
			V_[0][0] = a0;
			V_[0][1] = a3;
			V_[0][2] = a6;
			V_[1][0] = a1;
			V_[1][1] = a4;
			V_[1][2] = a7;
			V_[2][0] = a2;
			V_[2][1] = a5;
			V_[2][2] = a8;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	TinyVector<T, 3> operator*(const TinyVector<T, 3> &R) const
	{
		return TinyVector<T, 3>(
			V_[0][0] * R[0] + V_[1][0] * R[1] + V_[2][0] * R[2],
			V_[0][1] * R[0] + V_[1][1] * R[1] + V_[2][1] * R[2],
			V_[0][2] * R[0] + V_[1][2] * R[1] + V_[2][2] * R[2]);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> operator*(const ColumnMatrix3<T> &R) const
	{
		return ColumnMatrix3<T>(
			V_[0][0] * R[0][0] + V_[1][0] * R[0][1] + V_[2][0] * R[0][2],
			V_[0][1] * R[0][0] + V_[1][1] * R[0][1] + V_[2][1] * R[0][2],
			V_[0][2] * R[0][0] + V_[1][2] * R[0][1] + V_[2][2] * R[0][2],

			V_[0][0] * R[1][0] + V_[1][0] * R[1][1] + V_[2][0] * R[1][2],
			V_[0][1] * R[1][0] + V_[1][1] * R[1][1] + V_[2][1] * R[1][2],
			V_[0][2] * R[1][0] + V_[1][2] * R[1][1] + V_[2][2] * R[1][2],

			V_[0][0] * R[2][0] + V_[1][0] * R[2][1] + V_[2][0] * R[2][2],
			V_[0][1] * R[2][0] + V_[1][1] * R[2][1] + V_[2][1] * R[2][2],
			V_[0][2] * R[2][0] + V_[1][2] * R[2][1] + V_[2][2] * R[2][2]
		);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> TransposeTimes(const ColumnMatrix3<T> &R) const
	{
		return ColumnMatrix3<T>(
			V_[0].Dot(R[0]), V_[1].Dot(R[0]), V_[2].Dot(R[0]),
			V_[0].Dot(R[1]), V_[1].Dot(R[1]), V_[2].Dot(R[1]),
			V_[0].Dot(R[2]), V_[1].Dot(R[2]), V_[2].Dot(R[2])
			);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> TimesTranspose(const ColumnMatrix3<T> &R) const
	{
		return ColumnMatrix3<T>(
			V_[0][0] * R[0][0] + V_[1][0] * R[1][0] + V_[2][0] * R[2][0],
			V_[0][1] * R[0][0] + V_[1][1] * R[1][0] + V_[2][1] * R[2][0],
			V_[0][2] * R[0][0] + V_[1][2] * R[1][0] + V_[2][2] * R[2][0],

			V_[0][0] * R[0][1] + V_[1][0] * R[1][1] + V_[2][0] * R[2][1],
			V_[0][1] * R[0][1] + V_[1][1] * R[1][1] + V_[2][1] * R[2][1],
			V_[0][2] * R[0][1] + V_[1][2] * R[1][1] + V_[2][2] * R[2][1],

			V_[0][0] * R[0][2] + V_[1][0] * R[1][2] + V_[2][0] * R[2][2],
			V_[0][1] * R[0][2] + V_[1][1] * R[1][2] + V_[2][1] * R[2][2],
			V_[0][2] * R[0][2] + V_[1][2] * R[1][2] + V_[2][2] * R[2][2]
		);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> TransposeTimesTranspose(const ColumnMatrix3<T> &R) const
	{
		return ColumnMatrix3<T>(
			R[0][0] * V_[0][0] + R[1][0] * V_[0][1] + R[2][0] * V_[0][2],
			R[0][1] * V_[0][0] + R[1][1] * V_[0][1] + R[2][1] * V_[0][2],
			R[0][2] * V_[0][0] + R[1][2] * V_[0][1] + R[2][2] * V_[0][2],

			R[0][0] * V_[1][0] + R[1][0] * V_[1][1] + R[2][0] * V_[1][2],
			R[0][1] * V_[1][0] + R[1][1] * V_[1][1] + R[2][1] * V_[1][2],
			R[0][2] * V_[1][0] + R[1][2] * V_[1][1] + R[2][2] * V_[1][2],

			R[0][0] * V_[2][0] + R[1][0] * V_[2][1] + R[2][0] * V_[2][2],
			R[0][1] * V_[2][0] + R[1][1] * V_[2][1] + R[2][1] * V_[2][2],
			R[0][2] * V_[2][0] + R[1][2] * V_[2][1] + R[2][2] * V_[2][2],
			false
			);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> TimesDiagonal(const TinyVector<T, 3> &R) const
	{
		return ColumnMatrix3<T>(R[0] * V_[0], R[1] * V_[1], R[2] * V_[2]);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> DiagonalTimes(const TinyVector<T, 3> &R) const
	{
		return ColumnMatrix3<T>(
			R[0] * V_[0][0], R[1] * V_[0][1], R[2] * V_[0][2],
			R[0] * V_[1][0], R[1] * V_[1][1], R[2] * V_[1][2],
			R[0] * V_[2][0], R[1] * V_[2][1], R[2] * V_[2][2]
		);
	}
	//////////////////////////////////////////////////////////////////////////
	const TinyVector<T, 3> &operator[](int i) const
	{
		return V_[i];
	}
	//////////////////////////////////////////////////////////////////////////
	TinyVector<T, 3> &operator[](int i)
	{
		return V_[i];
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> Inverse() const
	{
		ColumnMatrix3<T> MI;

		MI[0][0] = V_[1][1] * V_[2][2] - V_[2][1] * V_[1][2];
		MI[0][1] = V_[2][1] * V_[0][2] - V_[0][1] * V_[2][2];
		MI[0][2] = V_[0][1] * V_[1][2] - V_[1][1] * V_[0][2];

		MI[1][0] = V_[2][0] * V_[1][2] - V_[1][0] * V_[2][2];
		MI[1][1] = V_[0][0] * V_[2][2] - V_[2][0] * V_[0][2];
		MI[1][2] = V_[1][0] * V_[0][2] - V_[0][0] * V_[1][2];

		MI[2][0] = V_[1][0] * V_[2][1] - V_[2][0] * V_[1][1];
		MI[2][1] = V_[2][0] * V_[0][1] - V_[0][0] * V_[2][1];
		MI[2][2] = V_[0][0] * V_[1][1] - V_[1][0] * V_[0][1];

		T det = V_[0][0] * MI[0][0] + V_[1][0] * MI[0][1] + V_[2][0] * MI[0][2];
		if (fabs(det) > 1.0e-16)
		{
			MI[0] /= det;
			MI[1] /= det;
			MI[2] /= det;
		}
		return MI;
	}
	//////////////////////////////////////////////////////////////////////////
	T Determinant () const
	{
		return V_[0][0] * (V_[1][1] * V_[2][2] - V_[2][1] * V_[1][2]) + V_[1][0] * (V_[2][1] * V_[0][2] - V_[0][1] * V_[2][2]) + V_[2][0] * (V_[0][1] * V_[1][2] - V_[1][1] * V_[0][2]);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> &operator= (const ColumnMatrix3<T> &CM)
	{
		V_[0] = CM[0], V_[1] = CM[1], V_[2] = CM[2];
		return *this;
	}
	////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> &MakeZero()
	{
		V_[0].reset_zero();
		V_[1].reset_zero();
		V_[2].reset_zero();
		return *this;
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> &MakeIdentity()
	{
		V_[0][0] = (T)1, V_[0][1] = (T)0, V_[0][2] = (T)0;
		V_[1][0] = (T)0, V_[1][1] = (T)1, V_[1][2] = (T)0;
		V_[2][0] = (T)0, V_[2][1] = (T)0, V_[2][2] = (T)1;
		return *this;
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> operator+ (const ColumnMatrix3<T> &mat) const
	{
		return ColumnMatrix3<T>
			(
			V_[0][0] + mat[0][0], V_[0][1] + mat[0][1], V_[0][2] + mat[0][2],
			V_[1][0] + mat[1][0], V_[1][1] + mat[1][1], V_[1][2] + mat[1][2],
			V_[2][0] + mat[2][0], V_[2][1] + mat[2][1], V_[2][2] + mat[2][2]
		);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> operator- (const ColumnMatrix3<T> &mat) const
	{
		return ColumnMatrix3<T>
			(
			V_[0][0] - mat[0][0], V_[0][1] - mat[0][1], V_[0][2] - mat[0][2],
			V_[1][0] - mat[1][0], V_[1][1] - mat[1][1], V_[1][2] - mat[1][2],
			V_[2][0] - mat[2][0], V_[2][1] - mat[2][1], V_[2][2] - mat[2][2]
		);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> operator* (const T scalar) const
	{
		return ColumnMatrix3<T>
			(
			scalar * V_[0][0], scalar * V_[0][1], scalar * V_[0][2],
			scalar * V_[1][0], scalar * V_[1][1], scalar * V_[1][2],
			scalar * V_[2][0], scalar * V_[2][1], scalar * V_[2][2]
		);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> operator/ (const T scalar) const
	{
		if (scalar != (T)0)
		{
			T inv_scalar = ((T)1) / scalar;
			return ColumnMatrix3<T>
				(
				inv_scalar * V_[0][0], inv_scalar * V_[0][1], inv_scalar * V_[0][2],
				inv_scalar * V_[1][0], inv_scalar * V_[1][1], inv_scalar * V_[1][2],
				inv_scalar * V_[2][0], inv_scalar * V_[2][1], inv_scalar * V_[2][2]
			);
		}
		else
		{
			return ColumnMatrix3<T>();
		}
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> operator- () const
	{
		return ColumnMatrix3<T>
			(
			-V_[0][0], -V_[0][1], -V_[0][2],
			-V_[1][0], -V_[1][1], -V_[1][2],
			-V_[2][0], -V_[2][1], -V_[2][2]
		);
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> &operator+= (const ColumnMatrix3<T> &mat)
	{
		V_[0] += mat[0];
		V_[1] += mat[1];
		V_[2] += mat[2];
		return *this;
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> &operator-= (const ColumnMatrix3<T> &mat)
	{
		V_[0] -= mat[0];
		V_[1] -= mat[1];
		V_[2] -= mat[2];
		return *this;
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> &operator*= (const T scalar)
	{
		V_[0] *= scalar;
		V_[1] *= scalar;
		V_[2] *= scalar;
		return *this;
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> &operator/= (const T scalar)
	{
		if (scalar != (T)0)
		{
			V_[0] /= scalar;
			V_[1] /= scalar;
			V_[2] /= scalar;
		}
		return *this;
	}
	//////////////////////////////////////////////////////////////////////////
	ColumnMatrix3<T> Transpose() const
	{
		return ColumnMatrix3<T>(V_[0][0], V_[0][1], V_[0][2], V_[1][0], V_[1][1], V_[1][2], V_[2][0], V_[2][1], V_[2][2], false);
	}
	//////////////////////////////////////////////////////////////////////////
};

template <typename T>
std::ostream& operator<<(std::ostream& s, const ColumnMatrix3<T>& A)
{
	s << A[0][0] << ' ' << A[1][0] << ' ' << A[2][0] << std::endl;
	s << A[0][1] << ' ' << A[1][1] << ' ' << A[2][1] << std::endl;
	s << A[0][2] << ' ' << A[1][2] << ' ' << A[2][2] << std::endl;
	return s;
}


typedef ColumnMatrix3<double> ColumnMatrix3d;
typedef ColumnMatrix3<float> ColumnMatrix3f;

#endif
