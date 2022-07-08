#include "HLBFGS.h"

#include <omp.h>

#include <cassert>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <ctime>

#include "LineSearch.h"

//////////////////////////////////////////////////////////////////////////
HLBFGS::HLBFGS() :
	n_N_(0), n_M_(7), n_E_(0), n_IE_(0), evalfuncgrad_callback(0),
	evalhessian_callback(0), eval_constraints_callback(0),
	newiteration_callback(0), newiteration_constraints_callback(0)
{
	initialize_setting();
	verbose_ = true;
	weak_wolf_ = false;
}
//////////////////////////////////////////////////////////////////////////
HLBFGS::~HLBFGS()
{
}
void HLBFGS::set_number_of_variables(size_t N) { n_N_ = N; }
size_t HLBFGS::get_num_of_variables() { return n_N_; }
void HLBFGS::set_M(size_t M) { n_M_ = M; }
size_t HLBFGS::get_M() { return n_M_; }
void HLBFGS::set_T(size_t T) { info[6] = T; }
size_t HLBFGS::get_T() { return info[6]; }
void HLBFGS::set_max_constraint(double val) { parameters[9] = val; }
void HLBFGS::set_init_sigma(double val) { parameters[10] = val; }
void HLBFGS::set_number_of_equalities(size_t Ne) { n_E_ = Ne; }
size_t HLBFGS::get_number_of_equalities() { return n_E_; }
void HLBFGS::set_number_of_inequalities(size_t Ne) { n_IE_ = Ne; }
size_t HLBFGS::get_number_of_inequalities() { return n_IE_; }
//////////////////////////////////////////////////////////////////////////
void HLBFGS::set_func_callback(eval_funcgrad_fp fp, eval_hessian_fp hfp/* = 0*/,
	eval_constraints_fp cfp/* = 0*/, eval_constraints_hessian_fp chfp/* = 0*/,
	newiteration_fp nfp/* = 0*/, newiteration_constraints_fp cnfp/* = 0*/
)
{
	evalfuncgrad_callback = fp;
	evalhessian_callback = hfp;
	eval_constraints_callback = cfp;
	eval_constraints_hessian_callback = chfp;
	newiteration_callback = nfp;
	newiteration_constraints_callback = cnfp;
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::get_advanced_setting(double hlbfgs_parameters[], size_t hlbfgs_info[])
{
	memcpy(&hlbfgs_parameters[0], &parameters[0], sizeof(double) * 20);
	memcpy(&hlbfgs_info[0], &info[0], sizeof(size_t) * 20);
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::set_advanced_setting(double hlbfgs_parameters[], size_t hlbfgs_info[])
{
	memcpy(&parameters[0], &hlbfgs_parameters[0], sizeof(double) * 20);
	memcpy(&info[0], &hlbfgs_info[0], sizeof(size_t) * 20);
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::set_verbose(bool verbose)
{
	verbose_ = verbose;
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::set_weak_wolf(bool weak_wolf) { weak_wolf_ = weak_wolf; }
void HLBFGS::copy_lambda(double *lambda_copy) { if (lambda_copy) memcpy(lambda_copy, &lambda[0], sizeof(double) * lambda.size()); }
void HLBFGS::copy_sigma(double *sigma_copy) { if (sigma_copy) *sigma_copy = sigma; }
//////////////////////////////////////////////////////////////////////////
bool HLBFGS::self_check()
{
	if (n_N_ == 0 || n_M_ < 0 || evalfuncgrad_callback == 0)
		return false;
	return true;
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::mem_allocation()
{
	variables_.resize(n_N_);
	gradient_.resize(n_N_);
	q_vec.resize(n_N_);
	alpha_vec.resize(n_M_);
	rho_vec.resize(n_M_);
	s_vec.resize(n_M_ * n_N_);
	y_vec.resize(n_M_ * n_N_);
	prev_x_vec.resize(n_N_);
	prev_g_vec.resize(n_N_);
	wa_vec.resize(n_N_);
	if (info[3] == 1)
	{
		diag_vec.assign(n_N_, 1.0);
	}
	if (evalhessian_callback)
	{
		m_hessian.get_icfs_info().allocate_mem(n_N_);
	}
	if (info[10] == 1)
	{
		if (info[11] == 1)
		{
			prev_q_first_stage_vec.resize(n_N_);
		}
		prev_q_update_vec.resize(n_N_);
	}
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::initialize_setting()
{
	parameters[0] = 1.0e-4; //ftol
	parameters[1] = 1.0e-16; //xtol
	parameters[2] = 0.9; //gtol
	parameters[3] = 1.0e-20; //stpmin
	parameters[4] = 1.0e+20; //stpmax
	parameters[5] = 1.0e-9; // ||g||/max(1,||x||)
	parameters[6] = 1.0e-10; // ||g||
	parameters[7] = 0.25; // update criterion in Augmented Lagrangian method
	parameters[8] = 10; // multiplier of sigma
	parameters[9] = 1.0e-10; // min_K
	parameters[10] = 1; //init value of sigma
	parameters[11] = 1.0e-9; // ||x-x_prev||_max <= parameters[11]*x_max
	parameters[12] = 1.0e-9; //tiny_value

	info[0] = 20; //max_fev_in_linesearch
	info[1] = 0; //total_num_fev
	info[2] = 0; //iter
	info[3] = 1; //update strategy. 0: standard lbfgs, 1: m1qn3;
	info[4] = 100000; // max iterations
	info[5] = 1; //1: print message, 0: do nothing
	info[6] = 10; // T: update interval of Hessian
	info[7] = 0; // 0: without Hessian, 1: with accurate Hessian
	info[8] = 15; // icfs parameter
	info[9] = 0; // 0: linesearch 1: modified linesearch (do not set 1 in practice !)
	info[10] = 0; // 0: Disable preconditioned CG 1: preconditioned CG
	info[11] = 1; // different methods for choosing beta in CG.
	info[12] = 1; //internal usage. 0: only update diag in update_q_by_inverse_hessian
	info[13] = 0; // 0: standard lbfgs update, 1: Biggs's update, 2: Yuan's update; 3: Zhang and Xu's update
	info[14] = 0; // 0: standard Augmented Lagrangian method; 1: Powell-Hestenes-Rockafellar Augmented Lagrangian method
	info[15] = 0; // 1: dynamically adjust sigma in AL-method for each constraints (experimental)
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::print_message(bool print, size_t id)
{
	if (!print || verbose_ == false)
	{
		return;
	}
	switch (id)
	{
	case 0:
		std::cout << "Please check your input parameters !\n";
		break;
	case 1:
		std::cout << "Linesearch is failed !\n";
		break;
	case 2:
		std::cout << "Convergence : ||g||/max(1,||x||) <= " << parameters[5]
			<< std::endl;
		break;
	case 3:
		std::cout << "Convergence : ||g|| <=  " << parameters[6] << std::endl;
		break;
	case 4:
		std::cout << "Convergence: linesearch cannot improve anymore \n";
		break;
	case 5:
		std::cout << "Exceeds max iteration \n";
		break;
	case 6:
		std::cout << "Convergence: max element of (x-x_prev) <= " << parameters[11] << std::endl;
		break;
	default:
		break;
	}
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::update_hessian(double *q, double *s, double *y, size_t curpos,
	double *diag)
{
	if (n_M_ == 0 || info[2] == 0)
	{
		return;
	}

	size_t start = curpos * n_N_;

	double *y_start = &y[start];
	double *s_start = &s[start];

	double ys = HLBFGS_DDOT(n_N_, y_start, s_start);

	if (info[3] == 0)
	{
		double yy = HLBFGS_DDOT(n_N_, y_start, y_start);
		double factor = ys / yy;
		if (info[12] == 1)
		{
			HLBFGS_DSCAL(n_N_, factor, q);
		}
		else
		{
			diag[0] = factor;
		}
	}
	else if (info[3] == 1)
	{
		//m1qn3 update
		double dyy = 0;
		double dinvss = 0;
#pragma omp parallel for reduction(+:dinvss) reduction(+:dyy)
		for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
		{
			dinvss += s_start[i] * s_start[i] / diag[i];
			dyy += diag[i] * y_start[i] * y_start[i];
		}
#pragma omp parallel for
		for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
		{
			diag[i] = 1.0 / (dyy / (ys * diag[i]) + y_start[i] * y_start[i]
				/ ys - dyy * s_start[i] * s_start[i] / (ys * dinvss
					* diag[i] * diag[i]));
		}
		if (info[12] == 1)
		{
#pragma omp parallel for
			for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
			{
				q[i] *= diag[i];
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::update_first_step(double *q, double *s, double *y, double *rho,
	double *alpha, size_t bound, size_t curpos, size_t iter)
{
	size_t start;
	double tmp;

	for (size_t i = bound; i >= 0; i--)
	{
		start = iter <= n_M_ ? curpos + i - bound : (curpos + i + n_M_
			- bound) % n_M_;
		alpha[i] = rho[start] * HLBFGS_DDOT(n_N_, q, &s[start * n_N_]);
		tmp = -alpha[i];
		HLBFGS_DAXPY(n_N_, tmp, &y[start * n_N_], q);
		if (i == 0)
			break;
	}
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::update_second_step(double *q, double *s, double *y, double *rho,
	double *alpha, size_t bound, size_t curpos, size_t iter)
{
	size_t start;
	double tmp;

	for (size_t i = 0; i <= bound; i++)
	{
		start = iter <= n_M_ ? i : (curpos + 1 + i) % n_M_;
		tmp = alpha[i] - rho[start] * HLBFGS_DDOT(n_N_, &y[start * n_N_], q);
		HLBFGS_DAXPY(n_N_, tmp, &s[start * n_N_], q);
	}
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::conjugate_gradient_update(double *q, double *prev_q_update,
	double *prev_q_first_stage)
{
	//determine beta
	double cg_beta = 1;
	if (info[11] == 1)
	{
		if (info[2] == 0)
		{
			memcpy(prev_q_first_stage, q, sizeof(double) * n_N_);
			memcpy(prev_q_update, q, sizeof(double) * n_N_);
			return;
		}
		else
		{
			cg_beta = HLBFGS_DDOT(n_N_, q, q);
			cg_beta /= std::fabs(cg_beta - HLBFGS_DDOT(n_N_, q,
				prev_q_first_stage));
			memcpy(prev_q_first_stage, q, sizeof(double) * n_N_);
		}
	}
	else
	{
		if (info[2] == 0)
		{
			memcpy(prev_q_update, q, sizeof(double) * n_N_);
			return;
		}
	}
	//determine new q
	if (cg_beta != 1)
		HLBFGS_DSCAL(n_N_, cg_beta, prev_q_update);

#pragma omp parallel for
	for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
	{
		q[i] -= prev_q_update[i];
	}

	double quad_a = HLBFGS_DDOT(n_N_, q, q);
	double quad_b = HLBFGS_DDOT(n_N_, q, prev_q_update);
	double cg_lambda = -quad_b / quad_a;
	if (cg_lambda > 1)
		cg_lambda = 1;
	else if (cg_lambda < 0)
		cg_lambda = 0;

#pragma omp parallel for
	for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
	{
		q[i] = cg_lambda * q[i] + prev_q_update[i];
	}
	memcpy(prev_q_update, q, sizeof(double) * n_N_);
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::optimize_without_constraints(double *init_sol, size_t max_iter, void *user_pointer)
{
	if (self_check() == false)
	{
		std::cerr << "please check your input!" << std::endl;
		return;
	}
	info[5] = verbose_ ? 1 : 0;
	mem_allocation();

	info[4] = max_iter;
	memcpy(&variables_[0], init_sol, sizeof(double)*n_N_);
	//----------------
	size_t T = info[6];
	info[7] = evalhessian_callback == 0 ? 0 : 1;
	double *x = &variables_[0];
	double *q = &q_vec[0];
	double *g = &gradient_[0];
	double *rho = n_M_ == 0 ? 0 : &rho_vec[0];
	double *s = n_M_ == 0 ? 0 : &s_vec[0];
	double *y = n_M_ == 0 ? 0 : &y_vec[0];
	double *prev_x = &prev_x_vec[0];
	double *prev_g = &prev_g_vec[0];
	double *wa = &wa_vec[0];
	double update_alpha = 1;
	double cg_dginit = 0;
	info[1] = 0;
	info[2] = 0;
	double f = 0;
	cur_pos = 0;
	size_t maxfev = info[0], nfev = 0, start = 0;
	//line search parameters
	double stp, ftol = parameters[0], xtol = parameters[1], gtol =
		parameters[2], stpmin = parameters[3], stpmax = parameters[4];
	ptrdiff_t linesearch_info;
	size_t keep[20];
	double gnorm, rkeep[40];
	memset(&rkeep[0], 0, sizeof(double) * 40);
	memset(&keep[0], 0, sizeof(size_t) * 20);
	double prev_f = 0;

	bool m_true = true;
	//----------------
	//////////////////////////////////////////////////////////////////////////
	do
	{
		if (info[2] == 0)
		{
			my_eval_funcgrad(f, 0, user_pointer);
			info[1]++;
		}

		if (info[7] == 1 && ((T == 0) || (info[2] % T == 0)))
		{
			HLBFGS_Sparse_Matrix *hessian_mat = m_hessian.get_mat(n_N_);
			hessian_mat->begin_fill_entry();
			(*evalhessian_callback)(n_N_, variables_, *hessian_mat, user_pointer);

			if (n_E_ + n_IE_ > 0)
			{
				ConstraintHessian m_constraintHessian(n_E_, n_IE_, *hessian_mat, lambda,
					dynamic_sigma, info[14], m_constraint_jacobian);
				eval_constraints_hessian_callback(n_E_, n_IE_, variables_, m_constraintHessian, user_pointer);
				fill_rest_constraint_hessian(hessian_mat);
			}
			hessian_mat->end_fill_entry();
			m_hessian.build_hessian_info(info[8]);
		}

		if (info[2] > 0 && n_M_ > 0)
		{
			//compute s and y
			start = cur_pos * n_N_;
#pragma omp parallel for
			for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
			{
				s[start + i] = x[i] - prev_x[i];
				y[start + i] = g[i] - prev_g[i];
			}
			rho[cur_pos] = 1.0 / HLBFGS_DDOT(n_N_, &y[start], &s[start]);
			if (info[13] == 1)
			{
				update_alpha = 1.0 / (rho[cur_pos] * 6 * (prev_f - f
					+ HLBFGS_DDOT(n_N_, g, &s[start])) - 2.0);
			}
			else if (info[13] == 2)
			{
				update_alpha = 1.0 / (rho[cur_pos] * 2 * (prev_f - f
					+ HLBFGS_DDOT(n_N_, g, &s[start])));
			}
			else if (info[13] == 3)
			{
				update_alpha = 1.0 / (1 + rho[cur_pos] * (6 * (prev_f - f) + 3
					* (HLBFGS_DDOT(n_N_, g, &s[start]) + HLBFGS_DDOT(n_N_,
						prev_g, &s[start]))));
			}
			if (info[13] != 0)
			{
				if (update_alpha < 0.01)
				{
					update_alpha = 0.01;
				}
				else if (update_alpha > 100)
				{
					update_alpha = 100;
				}
				rho[cur_pos] *= update_alpha;
			}
		}
#pragma omp parallel for
		for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
		{
			q[i] = -g[i];
		}

		update_q_by_inverse_hessian(cg_dginit);

		if (info[2] > 0 && n_M_ > 0)
			cur_pos = (cur_pos + 1) % n_M_;

		//store g and x
		memcpy(&prev_x_vec[0], x, sizeof(double) * n_N_);
		memcpy(&prev_g_vec[0], g, sizeof(double) * n_N_);
		prev_f = f;
		//linesearch, find new x
		bool blinesearch = true;
		if (info[2] == 0)
		{
			gnorm = HLBFGS_DNRM2(n_N_, g);
			//if(gnorm > 1)
			stp = 1.0 / gnorm;
			//else
			//	stp = 1;
		}
		else
		{
			stp = 1;
		}

		linesearch_info = 0;

		do
		{
			MCSRCH(&n_N_, x, &f, g, q, &stp, &ftol, &gtol, &xtol, &stpmin,
				&stpmax, &maxfev, &linesearch_info, &nfev, wa, keep, rkeep,
				info[10] == 0 ? 0 : (&cg_dginit), weak_wolf_);
			blinesearch = (linesearch_info == -1);
			if (blinesearch)
			{
				my_eval_funcgrad(f, 0, user_pointer);
				info[1]++;
			}

			if (info[9] == 1 && prev_f > f) //modify line search to avoid too many function calls
			{
				linesearch_info = 1;
				break;
			}
		} while (blinesearch);

		gnorm = HLBFGS_DNRM2(n_N_, g);
		info[2]++;
		if (newiteration_callback)
			(*newiteration_callback)(info[2], info[1], n_N_, variables_, f, gradient_, gnorm,
				user_pointer);
		double xnorm = HLBFGS_DNRM2(n_N_, x);
		xnorm = 1 > xnorm ? 1 : xnorm;
		rkeep[2] = gnorm;
		rkeep[8] = xnorm;

		double xmax = 0, diffmax = 0;
		for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
		{
			xmax = std::max(xmax, std::fabs(x[i]));
			diffmax = std::max(diffmax, std::fabs(x[i] - prev_x[i]));
		}

		if (linesearch_info != 1)
		{
			print_message(info[5] != 0, 1);
			break;
		}
		if (gnorm / xnorm <= parameters[5])
		{
			print_message(info[5] != 0, 2);
			break;
		}
		if (gnorm < parameters[6])
		{
			print_message(info[5] != 0, 3);
			break;
		}
		if (stp < stpmin || stp > stpmax)
		{
			print_message(info[5] != 0, 4);
			break;
		}
		if (info[2] > info[4])
		{
			print_message(info[5] != 0, 5);
			break;
		}
		if (diffmax <= parameters[11] * xmax)
		{
			print_message(info[5] != 0, 6);
			break;
		}
	} while (m_true);

	memcpy(init_sol, &variables_[0], sizeof(double)*n_N_);
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::optimize_with_constraints(double *init_sol, size_t max_iter, size_t local_iter, void *user_pointer, bool warm_start, double *init_lambda, double *init_sigma)
{
	if (self_check() == false || n_E_ + n_IE_ == 0 || eval_constraints_callback == 0)
	{
		std::cerr << "please check your input!" << std::endl;
		return;
	}
	if (info[7] == 1 && (evalhessian_callback == 0 || eval_constraints_hessian_callback == 0))
	{
		std::cerr << "please provide hessian!" << std::endl;
		return;
	}

	m_constraint_jacobian.resize(n_E_ + n_IE_, n_N_);
	const std::vector< double> & constraint_values_ = m_constraint_jacobian.get_constraints();

	//evalhessian_callback = 0;
	size_t backup_info_7 = info[7];
	//info[7] = 0;
	lambda.assign(n_E_ + n_IE_, 0);
	d.resize(n_E_ + n_IE_);
	sigma = parameters[10];
	if (warm_start && init_sigma) sigma = std::max(parameters[10], *init_sigma);
	if (warm_start && init_lambda)
	{
		memcpy(&lambda[0], init_lambda, sizeof(double)*lambda.size());
	}

	//double cur_f = 0;
	mem_allocation();
	memcpy(&variables_[0], init_sol, sizeof(double)*n_N_);
	//my_eval_funcgrad(cur_f, &sigma, user_pointer);
	size_t m_iter = 0;
	double K = DBL_MAX, max_d = 0;
	double max_lambda = 1.0e20, min_lambda = -1.0e20, max_mu = 1.0e20;

	dynamic_sigma.assign(n_IE_ + n_E_, sigma);
	bool suc = true;
	do
	{
		optimize_without_constraints(init_sol, local_iter, user_pointer);

		if (info[15] == 0)
		{
			max_d = max_absolute_value();

			if (max_d <= parameters[7] * K)
			{
				//update lambda
				if (info[14] == 0)
				{
#pragma omp parallel for
					for (ptrdiff_t i = 0; i < (ptrdiff_t)(n_E_ + n_IE_); i++)
					{
						lambda[i] = std::min(std::max(min_lambda, lambda[i] - sigma * d[i]), max_lambda);
					}
				}
				else
				{
#pragma omp parallel for
					for (ptrdiff_t i = 0; i < (ptrdiff_t)n_E_; i++)
					{
						lambda[i] = std::min(std::max(min_lambda, lambda[i] + sigma * constraint_values_[i]), max_lambda);
					}
#pragma omp parallel for
					for (ptrdiff_t i = (ptrdiff_t)n_E_; i < (ptrdiff_t)(n_E_ + n_IE_); i++)
					{
						lambda[i] = std::min(std::max(0.0, lambda[i] - sigma * constraint_values_[i]), max_mu);
					}
				}
				K = max_d;
			}
			else
				sigma *= parameters[8];
			std::fill(dynamic_sigma.begin(), dynamic_sigma.end(), sigma);
			if (verbose_)
				std::cout << "K = " << K << "; sigma = " << sigma << std::endl;
		}
		else
		{
			if (m_iter == 0)
			{
				prev_K.assign(n_E_ + n_IE_, DBL_MAX);
			}
			if (info[14] == 0)
			{
#pragma omp parallel for
				for (ptrdiff_t i = 0; i < (ptrdiff_t)(n_E_ + n_IE_); i++)
				{
					const double md = std::fabs(d[i]);
					if (md < parameters[7] * prev_K[i])
					{
						lambda[i] = std::min(std::max(min_lambda, lambda[i] - dynamic_sigma[i] * d[i]), max_lambda);
						prev_K[i] = md;
					}
					else
						dynamic_sigma[i] *= parameters[8];
				}
			}
			else
			{
#pragma omp parallel for
				for (ptrdiff_t i = 0; i < (ptrdiff_t)n_E_; i++)
				{
					const double md = std::fabs(d[i]);
					if (md < parameters[7] * prev_K[i])
					{
						lambda[i] = std::min(std::max(min_lambda, lambda[i] + dynamic_sigma[i] * constraint_values_[i]), max_lambda);
						prev_K[i] = md;
					}
					else
						dynamic_sigma[i] *= parameters[8];
				}
#pragma omp parallel for
				for (ptrdiff_t i = (ptrdiff_t)n_E_; i < (ptrdiff_t)(n_E_ + n_IE_); i++)
				{
					const double md = std::fabs(std::max(-constraint_values_[i], -lambda[i] / dynamic_sigma[i]));
					if (md < parameters[7] * prev_K[i])
					{
						lambda[i] = std::min(std::max(0.0, lambda[i] - dynamic_sigma[i] * constraint_values_[i]), max_mu);
						prev_K[i] = md;
					}
					else
						dynamic_sigma[i] *= parameters[8];
				}
			}
		}

		m_iter++;
		if (newiteration_constraints_callback)
		{
			newiteration_constraints_callback(m_iter, n_N_, variables_, object_func_, constraint_values_, user_pointer);
		}
		suc = true;
		if (n_IE_ + n_E_ > 0)
		{
			for (size_t i = 0; i < n_E_; i++)
			{
				if (fabs(constraint_values_[i]) > parameters[12])
				{
					suc = false;
					break;
				}
			}
			if (suc)
			{
				for (size_t i = 0; i < n_IE_; i++)
				{
					if (constraint_values_[i + n_E_] < -parameters[12])
					{
						suc = false;
						break;
					}
				}
			}
		}
		else
			break;
	} while (m_iter < max_iter && K > parameters[9] && suc == false);
	memcpy(init_sol, &variables_[0], sizeof(double)*n_N_);
	info[7] = backup_info_7;
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::my_eval_funcgrad(double &f, double *guess_sigma, void *user_pointer)
{
	double mf = 0;
	const std::vector< std::vector<std::pair<size_t, double> > > & constraint_jacobian_ = m_constraint_jacobian.get_Jacobian();
	const std::vector< double> & constraint_values_ = m_constraint_jacobian.get_constraints();

	memset(&gradient_[0], 0, sizeof(double)*gradient_.size());
	(*evalfuncgrad_callback)(n_N_, variables_, mf, gradient_, user_pointer);
	if (eval_constraints_callback && n_E_ + n_IE_ > 0)
	{
		size_t j = 0;

		m_constraint_jacobian.resize(n_E_ + n_IE_, n_N_);

		eval_constraints_callback(n_E_, n_IE_, variables_, m_constraint_jacobian, user_pointer);
		memcpy(&d[0], &constraint_values_[0], sizeof(double)*d.size());
		object_func_ = mf;

		if (guess_sigma)
		{
			*guess_sigma = parameters[10];
			if (n_E_ + n_IE_ == 0)
				return;

			double con_sum = 0;
#pragma omp parallel for reduction(+:con_sum)
			for (ptrdiff_t i = 0; i < (ptrdiff_t)n_E_; i++)
			{
				con_sum += constraint_values_[i] * constraint_values_[i];
			}
#pragma omp parallel for reduction(+:con_sum)
			for (ptrdiff_t i = n_E_; i < (ptrdiff_t)(n_E_ + n_IE_); i++)
			{
				if (constraint_values_[i] < 0)
					con_sum += constraint_values_[i] * constraint_values_[i];
			}
			*guess_sigma = std::max(1.0e-6, std::min(10.0, 2 * std::fabs(mf) / con_sum));
			return;
		}

		if (info[14] == 0)
		{
#pragma omp parallel for reduction(+:mf)
			for (ptrdiff_t i = 0; i < (ptrdiff_t)n_E_; i++)
			{
				d[i] = constraint_values_[i];
				mf += (-lambda[i] + 0.5 * dynamic_sigma[i] * d[i]) * d[i];
			}
			for (ptrdiff_t i = 0; i < (ptrdiff_t)n_E_; i++)
			{
				for (j = 0; j < constraint_jacobian_[i].size(); j++)
				{
					gradient_[constraint_jacobian_[i][j].first] += (-lambda[i] + dynamic_sigma[i] * d[i]) * constraint_jacobian_[i][j].second;
				}
			}
#pragma omp parallel for reduction(+:mf)
			for (ptrdiff_t i = n_E_; i < (ptrdiff_t)(n_E_ + n_IE_); i++)
			{
				double tmp = lambda[i] / dynamic_sigma[i];
				d[i] = constraint_values_[i] <= tmp ? constraint_values_[i] : tmp;
				mf += (-lambda[i] + 0.5 * dynamic_sigma[i] * d[i]) * d[i];
			}
			for (ptrdiff_t i = n_E_; i < (ptrdiff_t)(n_E_ + n_IE_); i++)
			{
				double tmp = lambda[i] / dynamic_sigma[i];
				if (constraint_values_[i] <= tmp)
				{
					for (j = 0; j < constraint_jacobian_[i].size(); j++)
					{
						gradient_[constraint_jacobian_[i][j].first] += (-lambda[i] + dynamic_sigma[i] * d[i]) * constraint_jacobian_[i][j].second;
					}
				}
			}
		}
		else
		{
#pragma omp parallel for reduction(+:mf)
			for (ptrdiff_t i = 0; i < (ptrdiff_t)n_E_; i++)
			{
				d[i] = constraint_values_[i] + lambda[i] / dynamic_sigma[i];
				mf += 0.5 * dynamic_sigma[i] * d[i] * d[i];
			}
			for (ptrdiff_t i = 0; i < (ptrdiff_t)n_E_; i++)
			{
				for (j = 0; j < constraint_jacobian_[i].size(); j++)
				{
					gradient_[constraint_jacobian_[i][j].first] += dynamic_sigma[i] * d[i] * constraint_jacobian_[i][j].second;
				}
			}
#pragma omp parallel for reduction(+:mf)
			for (ptrdiff_t i = n_E_; i < (ptrdiff_t)(n_E_ + n_IE_); i++)
			{
				d[i] = std::max(0.0, -constraint_values_[i] + lambda[i] / dynamic_sigma[i]);
				mf += 0.5 * dynamic_sigma[i] * d[i] * d[i];
			}
			for (ptrdiff_t i = n_E_; i < (ptrdiff_t)(n_E_ + n_IE_); i++)
			{
				if (std::fabs(d[i]) > 1.0e-12)
				{
					for (j = 0; j < constraint_jacobian_[i].size(); j++)
					{
						gradient_[constraint_jacobian_[i][j].first] -= dynamic_sigma[i] * d[i] * constraint_jacobian_[i][j].second;
					}
				}
			}
		}
	}
	f = mf;
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::fill_rest_constraint_hessian(HLBFGS_Sparse_Matrix *hessian_mat)
{
	const std::vector< std::vector<std::pair<size_t, double> > > & constraint_jacobian_ = m_constraint_jacobian.get_Jacobian();
	const std::vector< double> & constraint_values_ = m_constraint_jacobian.get_constraints();

	for (size_t i = 0; i < n_E_; i++)
	{
		for (size_t j = 0; j < constraint_jacobian_[i].size(); j++)
		{
			for (size_t k = j; k < constraint_jacobian_[i].size(); k++)
			{
				hessian_mat->fill_entry(constraint_jacobian_[i][j].first, constraint_jacobian_[i][k].first,
					dynamic_sigma[i] * constraint_jacobian_[i][j].second * constraint_jacobian_[i][k].second);
			}
		}
	}
	for (size_t i = n_E_; i < n_E_ + n_IE_; i++)
	{
		if (constraint_values_[i] < lambda[i] / dynamic_sigma[i])
		{
			for (size_t j = 0; j < constraint_jacobian_[i].size(); j++)
			{
				for (size_t k = j; k < constraint_jacobian_[i].size(); k++)
				{
					hessian_mat->fill_entry(constraint_jacobian_[i][j].first, constraint_jacobian_[i][k].first,
						dynamic_sigma[i] * constraint_jacobian_[i][j].second * constraint_jacobian_[i][k].second);
				}
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////
double HLBFGS::max_absolute_value()
{
	const std::vector< double> & constraint_values_ = m_constraint_jacobian.get_constraints();

	double d_max = 0;
	if (info[14] == 0)
	{
		for (size_t i = 0; i < d.size(); i++)
		{
			d_max = std::max(d_max, std::fabs(d[i]));
		}
	}
	else
	{
		for (size_t i = 0; i < n_E_; i++)
		{
			d_max = std::max(d_max, std::fabs(d[i]));
		}
		for (size_t i = n_E_; i < n_E_ + n_IE_; i++)
		{
			d_max = std::max(d_max, std::fabs(std::max(-constraint_values_[i], -lambda[i] / sigma)));
		}
	}
	return d_max;
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::update_q_by_inverse_hessian(double &cg_dginit)
{
	double *q = &q_vec[0];
	double *alpha = n_M_ == 0 ? 0 : &alpha_vec[0];
	double *rho = n_M_ == 0 ? 0 : &rho_vec[0];
	double *s = n_M_ == 0 ? 0 : &s_vec[0];
	double *y = n_M_ == 0 ? 0 : &y_vec[0];
	double *prev_g = &prev_g_vec[0];
	double *diag = diag_vec.empty() ? 0 : &diag_vec[0];
	double *prev_q_first_stage = prev_q_first_stage_vec.empty() ? 0
		: &prev_q_first_stage_vec[0];
	double *prev_q_update = prev_q_update_vec.empty() ? 0
		: &prev_q_update_vec[0];
	double scale = 0.0;
	cg_dginit = 0;
	size_t bound = 0;
	char task1 = 'N';
	char task2 = 'T';

	if (info[2] > 0 && n_M_ > 0)
	{
		bound = info[2] > n_M_ ? n_M_ - 1 : info[2] - 1;
		update_first_step(q, s, y, rho, alpha, bound, cur_pos, info[2]);
	}

	if (info[10] == 0)
	{
		if (info[7] == 1)
		{
			ICFS_INFO& l_info = m_hessian.get_icfs_info();
			m_hessian.solve_task(n_N_, q, task1);
			m_hessian.solve_task(n_N_, q, task2);
		}
		else
		{
			update_hessian(q, s, y, cur_pos, diag);
		}
	}
	else
	{
		if (info[7] == 1)
		{
			ICFS_INFO& l_info = m_hessian.get_icfs_info();
			m_hessian.solve_task(n_N_, q, task1);
			conjugate_gradient_update(q, prev_q_update, prev_q_first_stage);
			cg_dginit = -HLBFGS_DDOT(n_N_, q, q);
			m_hessian.solve_task(n_N_, q, task2);
		}
		else
		{
			info[12] = 0;
			update_hessian(q, s, y, cur_pos, info[3] == 0 ? (&scale) : diag);
			if (info[3] == 0)
			{
				if (n_M_ > 0 && info[2] > 0 && scale != 1.0)
				{
					scale = std::sqrt(scale);
					HLBFGS_DSCAL(n_N_, scale, q);
				}
				conjugate_gradient_update(q, prev_q_update,
					prev_q_first_stage);
				cg_dginit = -HLBFGS_DDOT(n_N_, q, q);
				if (n_M_ > 0 && info[2] > 0 && scale != 1.0)
					HLBFGS_DSCAL(n_N_, scale, q);
			}
			else
			{
				if (n_M_ > 0 && info[2] > 0)
				{
					//use prev_g as temporary array
#pragma omp parallel for
					for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
					{
						prev_g[i] = std::sqrt(diag[i]);
						q[i] *= prev_g[i];
					}
				}
				conjugate_gradient_update(q, prev_q_update,
					prev_q_first_stage);
				cg_dginit = -HLBFGS_DDOT(n_N_, q, q);
				if (info[2] > 0 && n_M_ > 0)
				{
#pragma omp parallel for
					for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
					{
						q[i] *= prev_g[i];
					}
				}
			}
			info[12] = 1;
		}
	}

	if (info[2] > 0 && n_M_ > 0)
	{
		update_second_step(q, s, y, rho, alpha, bound, cur_pos, info[2]);
	}
}
//////////////////////////////////////////////////////////////////////////
void HLBFGS::RNA_optimize(double *init_sol, size_t max_iter, void *user_pointer)
{
	const int M = 10;
	int HalfM = 5; // < M
	const double factor = 1.0e-4;
	std::srand(unsigned(std::time(0)));
	std::vector<double> q(n_N_), wa(n_N_);

	size_t maxfev = info[0], nfev = 0, start = 0;
	//line search parameters
	double stp, ftol = parameters[0], xtol = parameters[1], gtol =
		parameters[2], stpmin = parameters[3], stpmax = parameters[4];
	ptrdiff_t linesearch_info;
	size_t keep[20];
	double gnorm, rkeep[40];
	double f = 0, prev_f = 0;

	HESSIAN_MATRIX RNA_H;
	RNA_H.get_icfs_info().allocate_mem(M - 1);

	std::vector<std::vector<double>> previous_gradient(M), previous_sol(M), grad_diff(HalfM - 1);
	std::vector<double> cur_gradient(n_N_), sol(n_N_), rightside(HalfM - 1);
	std::vector<double> prev_x(sol.begin(), sol.end());
	memcpy(&sol[0], init_sol, sizeof(double)*n_N_);

	for (int i = 0; i < M; i++)
	{
		previous_gradient[i].resize(n_N_);
		previous_sol[i].resize(n_N_);
	}
	for (int i = 0; i < HalfM - 1; i++)
	{
		grad_diff[i].resize(n_N_);
	}

	int store_pointer = -1;
	std::vector<int> idsorted(M);
	for (int i = 0; i < M; i++)
	{
		idsorted[i] = i;
	}

	for (size_t iter = 0; iter < max_iter; iter++)
	{
		prev_x.assign(sol.begin(), sol.end());

		if (iter == 0)
		{
			memset(&cur_gradient[0], 0, sizeof(double)*n_N_);
			(*evalfuncgrad_callback)(n_N_, sol, f, cur_gradient, user_pointer);
		}
		else if (iter < M)
		{
#pragma omp parallel for
			for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
			{
				q[i] = -cur_gradient[i];
			}
			prev_f = f;
			//linesearch, find new x
			bool blinesearch = true;
			if (iter == 0)
			{
				gnorm = HLBFGS_DNRM2(n_N_, &cur_gradient[0]);
				//if(gnorm > 1)
				stp = 1.0 / gnorm;
				//else
				//	stp = 1;
			}
			else
			{
				stp = 1;
			}

			linesearch_info = 0;

			do
			{
				MCSRCH(&n_N_, &sol[0], &f, &cur_gradient[0], &q[0], &stp, &ftol, &gtol, &xtol, &stpmin,
					&stpmax, &maxfev, &linesearch_info, &nfev, &wa[0], keep, rkeep, 0);
				blinesearch = (linesearch_info == -1);
				if (blinesearch)
				{
					memset(&cur_gradient[0], 0, sizeof(double)*n_N_);
					(*evalfuncgrad_callback)(n_N_, sol, f, cur_gradient, user_pointer);
					info[1]++;
				}

				if (info[9] == 1 && prev_f > f) //modify line search to avoid too many function calls
				{
					linesearch_info = 1;
					break;
				}
			} while (blinesearch);
		}
		else
		{
			int inner_iter = 0;
			do {
				std::random_shuffle(idsorted.begin(), idsorted.end());

				HLBFGS_Sparse_Matrix *hessian_mat = RNA_H.get_mat(HalfM - 1);
				hessian_mat->begin_fill_entry();

				for (int col = 0; col < HalfM - 1; col++)
				{
#pragma omp parallel for
					for (ptrdiff_t id = 0; id < (ptrdiff_t)n_N_; id++)
					{
						grad_diff[col][id] = previous_gradient[idsorted[col + 1]][id] - previous_gradient[idsorted[0]][id];
						//grad_diff[col][id] = previous_sol[idsorted[col + 1]][id] - previous_sol[idsorted[0]][id];
					}
				}

				for (int col = 0; col < HalfM - 1; col++)
				{
					for (int rcol = col; rcol < HalfM - 1; rcol++)
					{
						double val = 0;
#pragma omp parallel for reduction(+:val)
						for (ptrdiff_t id = 0; id < (ptrdiff_t)n_N_; id++)
						{
							val += grad_diff[col][id] * grad_diff[rcol][id];
						}
						hessian_mat->fill_entry(col, rcol, val);
					}
					double b = 0;
#pragma omp parallel for reduction(-:b)
					for (ptrdiff_t id = 0; id < (ptrdiff_t)n_N_; id++)
					{
						b -= grad_diff[col][id] * previous_gradient[idsorted[0]][id];
						//b -= grad_diff[col][id] * previous_sol[idsorted[0]][id];
					}
					rightside[col] = b;
				}

				double* diag = hessian_mat->get_diag();
				double trace = 0;
				for (int i = 0; i < HalfM - 1; i++)
					trace += fabs(diag[i]);
				trace /= (HalfM - 1);
				for (int i = 0; i < HalfM - 1; i++)
				{
					diag[i] += factor * trace;
				}
				hessian_mat->end_fill_entry();
				RNA_H.build_hessian_info(info[8]);
				RNA_H.solve(HalfM - 1, &rightside[0]);

#pragma omp parallel for
				for (ptrdiff_t id = 0; id < (ptrdiff_t)n_N_; id++)
				{
					sol[id] = previous_sol[idsorted[0]][id];
					for (int i = 0; i < HalfM - 1; i++)
					{
						sol[id] += rightside[i] * (previous_sol[idsorted[i + 1]][id] - previous_sol[idsorted[0]][id]);
					}
				}

				memset(&cur_gradient[0], 0, sizeof(double)*n_N_);
				(*evalfuncgrad_callback)(n_N_, sol, f, cur_gradient, user_pointer);
				inner_iter++;
			} while (f > prev_f && inner_iter < 10);
		}

		double gnorm = HLBFGS_DNRM2(sol.size(), &cur_gradient[0]);
		if (newiteration_callback)
			(*newiteration_callback)(iter, iter, sol.size(), sol, f, cur_gradient, gnorm, user_pointer);
		store_pointer++;
		store_pointer = store_pointer % M;
		std::copy(sol.begin(), sol.end(), previous_sol[store_pointer].begin());
		std::copy(cur_gradient.begin(), cur_gradient.end(), previous_gradient[store_pointer].begin());
		prev_f = f;

		double xmax = 0, diffmax = 0;
		for (ptrdiff_t i = 0; i < (ptrdiff_t)n_N_; i++)
		{
			xmax = std::max(xmax, std::fabs(sol[i]));
			diffmax = std::max(diffmax, std::fabs(sol[i] - prev_x[i]));
		}

		double xnorm = HLBFGS_DNRM2(n_N_, &sol[0]);
		xnorm = 1 > xnorm ? 1 : xnorm;
		rkeep[2] = gnorm;
		rkeep[8] = xnorm;

		if (gnorm / xnorm <= parameters[5])
		{
			print_message(info[5] != 0, 2);
			break;
		}
		if (gnorm < parameters[6])
		{
			print_message(info[5] != 0, 3);
			break;
		}
		if (iter > 0 && diffmax <= parameters[11] * xmax)
		{
			print_message(info[5] != 0, 6);
			break;
		}
	}

	memcpy(init_sol, &sol[0], sizeof(double)*n_N_);
}
//////////////////////////////////////////////////////////////////////////