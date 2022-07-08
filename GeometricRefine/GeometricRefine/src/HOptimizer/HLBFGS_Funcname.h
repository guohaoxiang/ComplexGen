#pragma once

#include "HLBFGS_Constraint.h"

typedef void(*eval_funcgrad_fp)(const size_t n_vars, const std::vector<double>& variables,
	double &func_value, std::vector<double>& gradient, void *user_pointer);
typedef void(*eval_hessian_fp)(const size_t n_vars, const std::vector<double>& variables,
	HLBFGS_Sparse_Matrix& hessian, void *user_pointer);
typedef void(*eval_constraints_fp)(const size_t n_eqns, const size_t n_ieqns,
	const std::vector<double>& variables, ConstraintJacobian &jacobian,
	void *user_pointer);
typedef void(*eval_constraints_hessian_fp)(const size_t n_eqns, const size_t n_ieqns,
	const std::vector<double>& variables, ConstraintHessian &hessian,
	void *user_pointer);
typedef void(*newiteration_fp)(const size_t niter, const size_t call_iter,
	const size_t n_vars, const std::vector<double>& variables, const double &func_value,
	const std::vector<double>& gradient, const double &gnorm, void *user_pointer);
typedef void(*newiteration_constraints_fp)(const size_t niter, const size_t n_vars, const std::vector<double>& variables,
	const double f, const std::vector<double> &constraint_values, void *user_pointer);
