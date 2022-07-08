#include "HOptimizer_Internal.h"
#include "HLBFGS.h"
#include <iostream>

namespace IGOpt
{
	namespace IGOpt_impl
	{
		void error_output(ErrorInfo error_id)
		{
			if (error_id == NO_FUNC_GRAD)
			{
				std::cerr << "eval_funcgrad is not provided ! " << std::endl;
			}
			else if (error_id == NO_HESSIAN)
			{
				std::cerr << "eval_hessian is not provided ! " << std::endl;
			}
			else if (error_id == NO_CONSTRAINT)
			{
				std::cerr << "eval_constraints is not provided ! " << std::endl;
			}
			else if (error_id == ZERO_INPUT)
			{
				std::cerr << "The input vector is zero-length ! " << std::endl;
			}
		}
		/////////////////////////////////////////////////////////////////////
		bool HOPT_ITEM Unconstrained_HOptimizer
		(
			const UnConstrainedOptMethod method,
			std::vector<double> &sol,
			int max_iter,
			eval_funcgrad_fp fp,
			eval_hessian_fp hfp,
			newiteration_fp nfp,
			void *user_pointer,
			bool HLBGS_verbose
		)
		{
			if (fp == 0)
			{
				error_output(NO_FUNC_GRAD);
				return false;
			}

			if (sol.empty())
			{
				error_output(ZERO_INPUT);
				return false;
			}

			HLBFGS m_hlbfgs;
			m_hlbfgs.set_number_of_variables(sol.size());
			m_hlbfgs.set_func_callback(fp, hfp, 0, 0, nfp, 0);
			m_hlbfgs.set_verbose(HLBGS_verbose);

			double parameters[20];
			size_t info[20];
			m_hlbfgs.get_advanced_setting(parameters, info);

			if (method == METHOD_LBFGS)
			{
				m_hlbfgs.set_M(7);
				m_hlbfgs.set_func_callback(fp, 0, 0, 0, nfp, 0);
			}
			else if (method == METHOD_CG)
			{
				info[10] = 1;
				m_hlbfgs.set_M(0);
				m_hlbfgs.set_func_callback(fp, 0, 0, 0, nfp, 0);
			}
			else if (method == METHOD_GRADIENT)
			{
				info[10] = 0;
				m_hlbfgs.set_M(0);
				m_hlbfgs.set_func_callback(fp, 0, 0, 0, nfp, 0);
			}
			else if (method == METHOD_HESSIAN)
			{
				if (hfp == 0)
				{
					error_output(NO_HESSIAN);
					return false;
				}
				m_hlbfgs.set_M(0); info[6] = 0; info[7] = 1;
				m_hlbfgs.set_func_callback(fp, hfp, 0, 0, nfp, 0);
			}
			else if (method == METHOD_PLBFGS)
			{
				if (hfp == 0)
				{
					error_output(NO_HESSIAN);
					return false;
				}
				info[6] = 10; info[7] = 1;
				m_hlbfgs.set_func_callback(fp, hfp, 0, 0, nfp, 0);
			}
			else if (method == METHOD_PCG)
			{
				if (hfp == 0)
				{
					std::cerr << "eval_hessian_fp is not provided ! " << std::endl;
					return false;
				}
				info[7] = 1; info[10] = 1;
				m_hlbfgs.set_func_callback(fp, hfp, 0, 0, nfp, 0);
			}
			else if (method == METHOD_RNA)
			{
				m_hlbfgs.set_M(7);
				m_hlbfgs.set_func_callback(fp, 0, 0, 0, nfp, 0);
			}

			m_hlbfgs.set_advanced_setting(parameters, info);

			if (method == METHOD_RNA)
				m_hlbfgs.RNA_optimize(&sol[0], max_iter, user_pointer);
			else
				m_hlbfgs.optimize_without_constraints(&sol[0], max_iter, user_pointer);

			return true;
		}
		/////////////////////////////////////////////////////////////////////
		bool HOPT_ITEM Constrained_HOptimizer
		(
			const ConstrainedOptMethod method,
			std::vector<double> &sol,
			size_t num_eq,
			size_t num_ieq,
			int max_iter,
			eval_funcgrad_fp fp,
			eval_hessian_fp hfp,
			newiteration_fp nfp,
			eval_constraints_fp cfp,
			eval_constraints_hessian_fp chfp,
			newiteration_constraints_fp ncfp,
			void *user_pointer,
			bool HLBGS_verbose
		)
		{
			if (fp == 0)
			{
				error_output(NO_FUNC_GRAD);
				return false;
			}

			if (num_eq + num_ieq > 0 && cfp == 0)
			{
				error_output(NO_CONSTRAINT);
				return false;
			}

			double parameters[20];
			size_t info[20];
			HLBFGS m_hlbfgs;
			m_hlbfgs.get_advanced_setting(parameters, info);
			m_hlbfgs.set_verbose(HLBGS_verbose);

			if (method == METHOD_LBFGS_AL)
			{
				m_hlbfgs.set_func_callback(fp, 0, cfp, 0, nfp, ncfp);
				info[7] = 0;
			}
			else if (method == METHOD_AL)
			{
				m_hlbfgs.set_T(0);
				m_hlbfgs.set_func_callback(fp, hfp, cfp, chfp, nfp, ncfp);
				info[7] = 1;
			}
			else if (method == METHOD_PAL)
			{
				m_hlbfgs.set_func_callback(fp, hfp, cfp, chfp, nfp, ncfp);
				info[7] = 1;
			}
			m_hlbfgs.set_number_of_variables(sol.size());
			m_hlbfgs.set_number_of_equalities(num_eq);
			m_hlbfgs.set_number_of_inequalities(num_ieq);
			m_hlbfgs.set_advanced_setting(parameters, info);
			m_hlbfgs.optimize_with_constraints(&sol[0], 15, max_iter, user_pointer);

			return true;
		}
	}
}