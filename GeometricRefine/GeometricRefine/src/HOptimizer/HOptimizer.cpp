#include "HOptimizer.h"

namespace IGOpt
{
	/////////////////////////////////////////////////////////////////////////
	bool HOPT_ITEM Gradient_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, newiteration_fp nfp, void *user_pointer, bool verbose)
	{
		return IGOpt_impl::Unconstrained_HOptimizer(IGOpt_impl::METHOD_GRADIENT, sol, max_iter, fp, 0, nfp, user_pointer, verbose);
	}
	bool HOPT_ITEM CG_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, newiteration_fp nfp, void *user_pointer, bool verbose)
	{
		return IGOpt_impl::Unconstrained_HOptimizer(IGOpt_impl::METHOD_CG, sol, max_iter, fp, 0, nfp, user_pointer, verbose);
	}
	bool HOPT_ITEM PCG_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, eval_hessian_fp hfp, newiteration_fp nfp, void *user_pointer, bool verbose)
	{
		return IGOpt_impl::Unconstrained_HOptimizer(IGOpt_impl::METHOD_PCG, sol, max_iter, fp, hfp, nfp, user_pointer, verbose);
	}
	bool HOPT_ITEM LBFGS_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, newiteration_fp nfp, void *user_pointer, bool verbose)
	{
		return IGOpt_impl::Unconstrained_HOptimizer(IGOpt_impl::METHOD_LBFGS, sol, max_iter, fp, 0, nfp, user_pointer, verbose);
	}
	bool HOPT_ITEM Newton_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, eval_hessian_fp hfp, newiteration_fp nfp, void *user_pointer, bool verbose)
	{
		return IGOpt_impl::Unconstrained_HOptimizer(IGOpt_impl::METHOD_HESSIAN, sol, max_iter, fp, hfp, nfp, user_pointer, verbose);
	}
	bool HOPT_ITEM PLBFGS_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, eval_hessian_fp hfp, newiteration_fp nfp, void *user_pointer, bool verbose)
	{
		return IGOpt_impl::Unconstrained_HOptimizer(IGOpt_impl::METHOD_PLBFGS, sol, max_iter, fp, hfp, nfp, user_pointer, verbose);
	}
	bool HOPT_ITEM RNA_Optimizer(std::vector<double> &sol, int max_iter, eval_funcgrad_fp fp, newiteration_fp nfp, void *user_pointer)
	{
		return IGOpt_impl::Unconstrained_HOptimizer(IGOpt_impl::METHOD_RNA, sol, max_iter, fp, 0, nfp, user_pointer, false);
	}
	/////////////////////////////////////////////////////////////////////////
	bool HOPT_ITEM AL_LBFGS_Optimizer(std::vector<double> &sol, size_t num_eq, size_t num_ieq, int max_iter,
		eval_funcgrad_fp fp, eval_constraints_fp cfp, newiteration_fp nfp,
		newiteration_constraints_fp ncfp, void *user_pointer, bool verbose)
	{
		return IGOpt_impl::Constrained_HOptimizer(IGOpt_impl::METHOD_LBFGS_AL, sol, num_eq, num_ieq, max_iter,
			fp, 0, nfp, cfp, 0, ncfp, user_pointer, verbose);
	}
	bool HOPT_ITEM AL_Optimizer(std::vector<double> &sol, size_t num_eq, size_t num_ieq, int max_iter,
		eval_funcgrad_fp fp, eval_constraints_fp cfp, eval_hessian_fp hfp, eval_constraints_hessian_fp chfp,
		newiteration_fp nfp, newiteration_constraints_fp ncfp, void *user_pointer, bool verbose)
	{
		return IGOpt_impl::Constrained_HOptimizer(IGOpt_impl::METHOD_AL, sol, num_eq, num_ieq, max_iter,
			fp, hfp, nfp, cfp, chfp, ncfp, user_pointer, verbose);
	}
	bool HOPT_ITEM PAL_Optimizer(std::vector<double> &sol, size_t num_eq, size_t num_ieq, int max_iter,
		eval_funcgrad_fp fp, eval_constraints_fp cfp, eval_hessian_fp hfp, eval_constraints_hessian_fp chfp,
		newiteration_fp nfp, newiteration_constraints_fp ncfp, void *user_pointer, bool verbose)
	{
		return IGOpt_impl::Constrained_HOptimizer(IGOpt_impl::METHOD_PAL, sol, num_eq, num_ieq, max_iter,
			fp, hfp, nfp, cfp, chfp, ncfp, user_pointer, verbose);
	}
}