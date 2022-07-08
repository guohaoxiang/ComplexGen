#include "HLBFGS_Constraint.h"

ConstraintJacobian::ConstraintJacobian(size_t num_funcs/* = 0*/, size_t num_vars/* = 0*/)
	:m_num_funcs(num_funcs), m_num_vars(num_vars)
{
	reset();
}
void ConstraintJacobian::resize(size_t num_funcs, size_t num_vars)
{
	m_num_funcs = num_funcs, m_num_vars = num_vars;
	reset();
}

void ConstraintJacobian::reset()
{
	constraint_values_.assign(m_num_funcs, 0);
	constraint_jacobian_.resize(m_num_funcs);
	for (size_t i = 0; i < m_num_funcs; i++)
		constraint_jacobian_[i].resize(0);
}

void ConstraintJacobian::add_value(size_t func_id, double value)
{
#ifdef  _DEBUG
	assert(func_id < m_num_funcs);
#endif //  _DEBUG
	constraint_values_[func_id] = value;
}

void ConstraintJacobian::add_differential(size_t func_id, size_t var_id, double value)
{
#ifdef  _DEBUG
	assert(func_id < m_num_funcs && var_id < m_num_vars);
#endif //  _DEBUG
	constraint_jacobian_[func_id].push_back(std::pair<size_t, double>(var_id, value));
}

std::vector< std::vector<std::pair<size_t, double> > > & ConstraintJacobian::get_Jacobian() { return constraint_jacobian_; }

const std::vector< std::vector<std::pair<size_t, double> > > & ConstraintJacobian::get_Jacobian() const { return constraint_jacobian_; }

std::vector<double> & ConstraintJacobian::get_constraints() { return constraint_values_; }

const std::vector<double> & ConstraintJacobian::get_constraints() const { return constraint_values_; }

//////////////////////////////////////////////////////////////////

ConstraintHessian::ConstraintHessian(size_t num_eq, size_t num_ieq, HLBFGS_Sparse_Matrix &mat,
	const std::vector<double> & lambda, const std::vector<double> & sigma,
	size_t info, const ConstraintJacobian &jacobian)
	: n_E_(num_eq), n_IE_(num_ieq), hessian(mat),
	m_lambda(lambda), m_sigma(sigma), info14(info), m_jacobian(jacobian)
{
}
void ConstraintHessian::add_differential(size_t i, size_t var_id0, size_t var_id1, double diff_value)
{
#ifdef _DEBUG
	assert(i < n_E_ + n_IE_);
#endif
	const std::vector<double> & constraint_values_ = m_jacobian.get_constraints();

	if (info14 == 0)
	{
		if (i < n_E_)
		{
			hessian.fill_entry(var_id0, var_id1, (m_sigma[i] * constraint_values_[i] - m_lambda[i]) * diff_value);
		}
		else
		{
			if (constraint_values_[i] < m_lambda[i] / m_sigma[i])
			{
				hessian.fill_entry(var_id0, var_id1, (m_sigma[i] * constraint_values_[i] - m_lambda[i]) * diff_value);
			}
		}
	}
	else
	{
		if (i < n_E_)
		{
			hessian.fill_entry(var_id0, var_id1, (m_sigma[i] * constraint_values_[i] + m_lambda[i]) * diff_value);
		}
		else
		{
			if (constraint_values_[i] < m_lambda[i] / m_sigma[i])
			{
				hessian.fill_entry(var_id0, var_id1, (m_sigma[i] * constraint_values_[i] - m_lambda[i]) * diff_value);
			}
		}
	}
}