#pragma once

#include <algorithm>
#include <cassert>

template <size_t Dim>
class MyTuple
{
public:
	MyTuple(size_t *v = 0)
	{
		if (v == 0)
		{
			std::memset(vert, 0, sizeof(size_t)*Dim);
		}
		else
		{
			std::memcpy(vert, v, sizeof(size_t)*Dim);
		}
	}
	MyTuple(size_t v0, size_t v1)
	{
		assert(Dim == 2);
		vert[0] = v0;
		vert[1] = v1;
	}
	MyTuple(size_t v0, size_t v1, size_t v2)
	{
		assert(Dim == 3);
		vert[0] = v0;
		vert[1] = v1;
		vert[2] = v2;
	}
	MyTuple(size_t v0, size_t v1, size_t v2, size_t v3)
	{
		assert(Dim == 4);
		vert[0] = v0;
		vert[1] = v1;
		vert[2] = v2;
		vert[3] = v3;
	}
	MyTuple(size_t v0, size_t v1, size_t v2, size_t v3, size_t v4, size_t v5, size_t v6, size_t v7)
	{
		assert(Dim == 8);
		vert[0] = v0;
		vert[1] = v1;
		vert[2] = v2;
		vert[3] = v3;
		vert[4] = v4;
		vert[5] = v5;
		vert[6] = v6;
		vert[7] = v7;
	}
public:
	size_t vert[Dim];
};

template <size_t Dim>
class MySortedTuple
{
public:
	MySortedTuple(size_t *v = 0)
	{
		if (v == 0)
		{
			memset(sorted_vert, 0, sizeof(size_t)*Dim);
		}
		else
		{
			memcpy(sorted_vert, v, sizeof(size_t)*Dim);
		}

		sort_array();
	}

	MySortedTuple(size_t v0, size_t v1)
	{
		assert(Dim == 2);
		sorted_vert[0] = v0;
		sorted_vert[1] = v1;
		sort_array();
	}
	MySortedTuple(size_t v0, size_t v1, size_t v2)
	{
		assert(Dim == 3);
		sorted_vert[0] = v0;
		sorted_vert[1] = v1;
		sorted_vert[2] = v2;
		sort_array();
	}
	MySortedTuple(size_t v0, size_t v1, size_t v2, size_t v3)
	{
		assert(Dim == 4);
		sorted_vert[0] = v0;
		sorted_vert[1] = v1;
		sorted_vert[2] = v2;
		sorted_vert[3] = v3;
		sort_array();
	}
	MySortedTuple(size_t v0, size_t v1, size_t v2, size_t v3, size_t v4, size_t v5, size_t v6, size_t v7)
	{
		assert(Dim == 8);
		sorted_vert[0] = v0;
		sorted_vert[1] = v1;
		sorted_vert[2] = v2;
		sorted_vert[3] = v3;
		sorted_vert[4] = v4;
		sorted_vert[5] = v5;
		sorted_vert[6] = v6;
		sorted_vert[7] = v7;
		sort_array();
	}
	inline bool operator == (const MySortedTuple &p) const
	{
		for (int i = 0; i < Dim; i++)
		{
			if (sorted_vert[i] != p.sorted_vert[i])
			{
				return false;
			}
		}
		return true;
	}
	inline bool operator != (const MySortedTuple &p) const
	{
		for (int i = 0; i < Dim; i++)
		{
			if (sorted_vert[i] != p.sorted_vert[i])
			{
				return true;
			}
		}
		return false;
	}
	inline bool operator< (const MySortedTuple &p) const
	{
		for (size_t i = 0; i < Dim; i++)
		{
			if (sorted_vert[i] < p.sorted_vert[i])
			{
				return true;
			}
			else if (sorted_vert[i] > p.sorted_vert[i])
			{
				return false;
			}
		}
		return false;
	}
private:
	void sort_array()
	{
		std::sort(&sorted_vert[0], &sorted_vert[Dim]);
	}
public:
	size_t sorted_vert[Dim];
};
