#pragma once

#include <vector>
#include "HOptimizer_Internal.h"

namespace IGOpt
{
	//Unconstrained Optimizers
	bool HOPT_ITEM Gradient_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, newiteration_fp nfp = 0, void *user_pointer = 0, bool HLBGS_verbose = true);
	bool HOPT_ITEM CG_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, newiteration_fp nfp = 0, void *user_pointer = 0, bool HLBGS_verbose = true);
	bool HOPT_ITEM PCG_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, eval_hessian_fp hfp, newiteration_fp nfp = 0, void *user_pointer = 0, bool HLBGS_verbose = true);
	bool HOPT_ITEM LBFGS_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, newiteration_fp nfp = 0, void *user_pointer = 0, bool HLBGS_verbose = true);
	bool HOPT_ITEM Newton_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, eval_hessian_fp hfp, newiteration_fp nfp = 0, void *user_pointer = 0, bool HLBGS_verbose = true);
	bool HOPT_ITEM PLBFGS_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, eval_hessian_fp hfp, newiteration_fp nfp = 0, void *user_pointer = 0, bool HLBGS_verbose = true);
	bool HOPT_ITEM RNA_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, newiteration_fp nfp = 0, void *user_pointer = 0);
	//constrained Optimizers
	bool HOPT_ITEM AL_LBFGS_Optimizer(std::vector<double> &sol, size_t num_eq, size_t num_ieq, int max_iter,
		eval_funcgrad_fp fp, eval_constraints_fp cfp, newiteration_fp nfp = 0,
		newiteration_constraints_fp ncfp = 0, void *user_pointer = 0, bool HLBGS_verbose = true);
	bool HOPT_ITEM AL_Optimizer(std::vector<double> &sol, size_t num_eq, size_t num_ieq, int max_iter,
		eval_funcgrad_fp fp, eval_constraints_fp cfp, eval_hessian_fp hfp, eval_constraints_hessian_fp chfp, newiteration_fp nfp = 0,
		newiteration_constraints_fp ncfp = 0, void *user_pointer = 0, bool HLBGS_verbose = true);
	bool HOPT_ITEM PAL_Optimizer(std::vector<double> &sol, size_t num_eq, size_t num_ieq, int max_iter,
		eval_funcgrad_fp fp, eval_constraints_fp cfp, eval_hessian_fp hfp, eval_constraints_hessian_fp chfp, newiteration_fp nfp = 0,
		newiteration_constraints_fp ncfp = 0, void *user_pointer = 0, bool HLBGS_verbose = true);
}