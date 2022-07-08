#pragma once

#include <map>
#include <cstring>
#include "HLBFGS_Funcname.h"
#include "HLBFGS_Hessian.h"

class HLBFGS
{
public:
	//////////////////////////////////////////////////////////////////////////
	HLBFGS();
	~HLBFGS();
	//////////////////////////////////////////////////////////////////////////
	void set_number_of_variables(size_t N);
	size_t get_num_of_variables();
	void set_M(size_t M);
	size_t get_M();
	void set_T(size_t T);
	size_t get_T();
	void set_max_constraint(double val);
	void set_init_sigma(double val);
	void set_number_of_equalities(size_t Ne);
	size_t get_number_of_equalities();
	void set_number_of_inequalities(size_t Ne);
	size_t get_number_of_inequalities();
	//////////////////////////////////////////////////////////////////////////
	void set_func_callback(eval_funcgrad_fp fp, eval_hessian_fp hfp = 0,
		eval_constraints_fp cfp = 0, eval_constraints_hessian_fp chfp = 0,
		newiteration_fp nfp = 0, newiteration_constraints_fp cnfp = 0
	);
	//////////////////////////////////////////////////////////////////////////
	void get_advanced_setting(double hlbfgs_parameters[], size_t hlbfgs_info[]);
	//////////////////////////////////////////////////////////////////////////
	void set_advanced_setting(double hlbfgs_parameters[], size_t hlbfgs_info[]);
	//////////////////////////////////////////////////////////////////////////
	void set_verbose(bool verbose);
	//////////////////////////////////////////////////////////////////////////
	void optimize_without_constraints(double *init_sol, size_t max_iter, void *user_pointer = 0);
	void optimize_with_constraints(double *init_sol, size_t max_iter, size_t local_iter = 30, void *user_pointer = 0, bool warm_start = false, double *init_lambda = 0, double *init_sigma = 0);
	void set_weak_wolf(bool weak_wolf);
	void copy_lambda(double *lambda_copy);
	void copy_sigma(double *sigma_copy);

	//////////////////////////////////////////////////////////////////////////
	void RNA_optimize(double *init_sol, size_t max_iter, void *user_pointer = 0);
protected:
	void mem_allocation();
	bool self_check();
	void initialize_setting();
	void print_message(bool print, size_t id);

	void update_first_step(double *q, double *s, double *y, double *rho,
		double *alpha, size_t bound, size_t curpos, size_t iter);
	void update_hessian(double *q, double *s, double *y, size_t curpos,
		double *diag);
	void update_second_step(double *q, double *s, double *y, double *rho,
		double *alpha, size_t bound, size_t curpos, size_t iter);
	void conjugate_gradient_update(double *q, double *prev_q_update,
		double *prev_q_first_stage);

	void update_q_by_inverse_hessian(double &cg_dginit);

	void my_eval_funcgrad(double &f, double *guess_sigma, void *user_pointer);

	double max_absolute_value();

	void fill_rest_constraint_hessian(HLBFGS_Sparse_Matrix *hessian_mat);

protected:
	size_t n_N_, n_M_, n_E_, n_IE_;

	eval_funcgrad_fp evalfuncgrad_callback;
	eval_hessian_fp evalhessian_callback;
	eval_constraints_fp eval_constraints_callback;
	eval_constraints_hessian_fp eval_constraints_hessian_callback;
	newiteration_fp newiteration_callback;
	newiteration_constraints_fp newiteration_constraints_callback;
	double parameters[20];
	size_t info[20];
	bool verbose_;
	bool weak_wolf_;

	std::vector<double> variables_;
	std::vector<double> gradient_;

	size_t cur_pos;
	std::vector<double> q_vec, alpha_vec, rho_vec, s_vec, y_vec,
		prev_x_vec, prev_g_vec, diag_vec, wa_vec;

	HESSIAN_MATRIX m_hessian;

	std::vector<double> prev_q_first_stage_vec, prev_q_update_vec;

	std::vector<double> lambda, d;
	double sigma;
	std::vector<double> dynamic_sigma, prev_K;
	ConstraintJacobian m_constraint_jacobian;
	double object_func_;
};
