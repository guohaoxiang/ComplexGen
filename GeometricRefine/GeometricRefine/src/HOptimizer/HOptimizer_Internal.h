#pragma once

#include "HLBFGS_Funcname.h"

#if defined(HOPT_DLL_EXPORT)
// For the DLL library.
#define HOPT_ITEM __declspec(dllexport)
#elif defined(HOPT_DLL_IMPORT)
// For a client of the DLL library.
#define HOPT_ITEM __declspec(dllimport)
#else
// For the static library and for Apple/Linux.
#define HOPT_ITEM
#endif
// End Microsoft Windows DLL support.

namespace IGOpt
{
	namespace IGOpt_impl
	{
		enum UnConstrainedOptMethod { METHOD_LBFGS = 0, METHOD_CG, METHOD_GRADIENT, METHOD_HESSIAN, METHOD_PLBFGS, METHOD_PCG, METHOD_RNA };
		enum ConstrainedOptMethod { METHOD_LBFGS_AL = 0, METHOD_AL, METHOD_PAL };

		enum ErrorInfo { NO_FUNC_GRAD, NO_HESSIAN, NO_CONSTRAINT, ZERO_INPUT };
		void error_output(ErrorInfo error_id);

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
		);

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
		);
	}
}