#include "LinearSolver.h"
#include "SimpleAlgorithm.h"
#include <stdlib.h>
#include <cuda_runtime_api.h>
namespace volcano
{



	DenseSolver::DenseSolver()
	{
		this->A = 0;
		this->B = 0;
		this->X = 0;
	}
	int DenseSolver::setData(DenseMatrix *a, double *b, double *x)
	{
		A = a; B = b; X = x;
		return 0;
	}

	DenseLUSolver::DenseLUSolver()
	{
		A = new DenseMatrix();
		L = new DenseMatrix();
		U = new DenseMatrix();
		B = 0; X = 0;
	}
	int DenseLUSolver::DenseLU(DenseMatrix *&A, DenseMatrix *&L, DenseMatrix *&U)
	{
		if (A->m != A->n)
		{
			cout << "fail" << endl;
			return 1;
		}
		int m = A->m;
		L->m = m;
		L->n = m;
		U->m = m;
		U->n = m;
		L->val = (double **)malloc(sizeof(double *)*m);
		U->val = (double **)malloc(sizeof(double *)*m);
		for (int i = 0; i < m; i++)
		{
			L->val[i] = (double *)malloc(sizeof(double)*m);
			U->val[i] = (double *)malloc(sizeof(double)*m);
		}
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++)
			{
				L->val[i][j] = 0;
				U->val[i][j] = 0;
			}
		}
		for (int i = 0; i < m; i++)
		{
			L->val[i][i] = 1;
		}
		for (int i = 0; i < m; i++)
		{
			for (int j = i; j < m; j++)
			{
				U->val[i][j] = A->val[i][j];
				for (int k = 0; k < i; k++)
				{
					U->val[i][j] -= L->val[i][k] * U->val[k][j];
				}
			}
			for (int j = i + 1; j < m; j++)
			{
				L->val[j][i] = A->val[j][i];
				for (int k = 0; k < i; k++)
				{
					L->val[j][i] -= L->val[j][k] * U->val[k][i];
				}
				L->val[j][i] /= U->val[i][i];
			}
		}

		return 0;
	}
	int DenseLUSolver::preconditioning()
	{
		return 0;
	}
	int DenseLUSolver::execute()
	{
		DenseLU(A, L, U);
		double *y = (double *)malloc(sizeof(double)*A->m);
		for (int i = 0; i < A->m; i++)
		{
			y[i] = B[i];
			for (int j = 0; j < i; j++)
			{
				y[i] -= L->val[i][j] * y[j];
			}
			//cout << y[i] << endl;
		}
		for (int i = 0; i < A->m; i++)
		{
			X[A->m - i - 1] = y[A->m - i - 1] ;
			for (int j = 0; j < i; j++)
			{
				X[A->m - i - 1] -= U->val[A->m - i - 1][A->m - j - 1] * X[A->m - j - 1];
			}
			X[A->m - i - 1] /= U->val[A->m - i - 1][A->m - i - 1];
		}
		free(y);
		y = 0;
		return 0;
	}

	CULASolver::CULASolver()
	{
		A = 0; B = 0; X = 0;
	}

	int CULASolver::preconditioning()
	{
		//culaSparseCreate(&handle);
		//if (culaSparseCreate(&handle) != culaSparseNoError)
		//{
		//	// this should only fail under extreme conditions
		//	std::cout << "fatal error: failed to create library handle!" << std::endl;
		//	exit(EXIT_FAILURE);
		//}
		//culaSparseCreatePlan(handle, &plan);
		//culaSparseConfigInit(handle, &config);
		//config.relativeTolerance = this->tol;
		//config.maxIterations = 1000;
		//culaSparseGetConfigString(handle, &config, buffer, 512);
		//std::cout << buffer << std::endl;
		//culaSparseSetCudaPlatform(handle, plan, 0);

		///*s
		//culaSparseGmresOptions solverOpts;
		//culaSparseAinvOptions precondOpts;
		//tatus = culaSparseGmresOptionsInit(handle, &solverOpts);
		//solverOpts.restart = 20;
		//status = culaSparseSetGmresSolver(handle, plan, &solverOpts);*/
		////cout << solver << endl;
		//switch (this->solver)
		//{
		//case SolverType::NoSolver:
		//{
		//	culaSparseEmptyOptions solverOpts;
		//	culaSparseStatus status = culaSparseEmptyOptionsInit(handle, &solverOpts);
		//	status = culaSparseSetNoSolver(handle, plan, &solverOpts);
		//}
		//case SolverType::CG:
		//{
		//	culaSparseCgOptions solverOpts;
		//	culaSparseStatus status = culaSparseCgOptionsInit(handle, &solverOpts);
		//	status = culaSparseSetCgSolver(handle, plan, &solverOpts); break;
		//}
		//case SolverType::BiCG:
		//{
		//	culaSparseBicgOptions solverOpts;
		//	culaSparseStatus status = culaSparseBicgOptionsInit(handle, &solverOpts);
		//	//solverOpts.avoidTranspose = 1;
		//	status = culaSparseSetBicgSolver(handle, plan, &solverOpts); break;
		//}
		//case SolverType::BiCGSTAB:
		//{
		//	culaSparseBicgstabOptions solverOpts;
		//	culaSparseStatus status = culaSparseBicgstabOptionsInit(handle, &solverOpts);
		//	status = culaSparseSetBicgstabSolver(handle, plan, &solverOpts); break;//"Bicgstab"
		//}

		//case SolverType::BiCGSTABl:
		//{
		//	culaSparseBicgstablOptions solverOpts;
		//	culaSparseStatus status = culaSparseBicgstablOptionsInit(handle, &solverOpts);
		//	//solverOpts.l = 10;
		//	status = culaSparseSetBicgstablSolver(handle, plan, &solverOpts); break;
		//}
		//case SolverType::GMRESm:
		//{
		//	culaSparseGmresOptions solverOpts;
		//	culaSparseStatus status = culaSparseGmresOptionsInit(handle, &solverOpts);
		//	//solverOpts.restart = 5;
		//	status = culaSparseSetGmresSolver(handle, plan, &solverOpts); break;
		//}
		//case SolverType::MINRES:
		//{
		//	culaSparseMinresOptions solverOpts;
		//	culaSparseStatus status = culaSparseMinresOptionsInit(handle, &solverOpts);
		//	status = culaSparseSetMinresSolver(handle, plan, &solverOpts); break;
		//}
		//default:cout << "error solver type" << endl;
		//}



		//switch (this->preconditioner)
		//{
		//case PreconditionerType::NoPreconditioner:
		//{
		//	culaSparseEmptyOptions precondOpts;
		//	culaSparseStatus status = culaSparseEmptyOptionsInit(handle, &precondOpts);
		//	status = culaSparseSetNoPreconditioner(handle, plan, &precondOpts); break;
		//}
		//case PreconditionerType::Jacobi:
		//{
		//	culaSparseJacobiOptions precondOpts;
		//	culaSparseStatus status = culaSparseJacobiOptionsInit(handle, &precondOpts);
		//	status = culaSparseSetJacobiPreconditioner(handle, plan, &precondOpts); break;
		//}
		//case PreconditionerType::BlockJacobi:
		//{
		//	culaSparseBlockjacobiOptions precondOpts;
		//	culaSparseStatus status = culaSparseBlockjacobiOptionsInit(handle, &precondOpts);
		//	//precondOpts.blockSize = 10;
		//	status = culaSparseSetBlockJacobiPreconditioner(handle, plan, &precondOpts); break;
		//}
		//case PreconditionerType::ILUO:
		//{
		//	culaSparseIlu0Options precondOpts;
		//	culaSparseStatus status = culaSparseIlu0OptionsInit(handle, &precondOpts);
		//	status = culaSparseSetIlu0Preconditioner(handle, plan, &precondOpts); break;
		//}
		//case PreconditionerType::ApproximateInverse:
		//{
		//	culaSparseAinvOptions precondOpts;
		//	culaSparseStatus status = culaSparseAinvOptionsInit(handle, &precondOpts);
		//	//precondOpts.dropTolerance = 0.001;
		//	//precondOpts.pattern = culaSparseAPattern;
		//	/*
		//	culaSparseAPattern,         ///< Uses the same sparsity pattern as A
		//	culaSparseA2Pattern,        ///< Uses a sparsity pattern of A^2
		//	culaSparseA3Pattern,        ///< Uses a sparsity pattern of A^3
		//	culaSparseA4Pattern         ///< Uses a sparsity pattern of A^4
		//	*/
		//	status = culaSparseSetAinvPreconditioner(handle, plan, &precondOpts); break;
		//}
		//case PreconditionerType::FactorizedApproximateInverse:
		//{
		//	culaSparseFainvOptions precondOpts;
		//	culaSparseStatus status = culaSparseFainvOptionsInit(handle, &precondOpts);
		//	status = culaSparseSetFainvPreconditioner(handle, plan, &precondOpts); break;
		//}
		//}
		return 0;
	}

		
	int CULASolver::execute()
	{
		//culaSparseSetDcsrData(handle, plan, 0, A->m, A->nnz, A->val, A->rowptr, A->colind, X, B);
		//culaSparseExecutePlan(handle, plan, &config, &result);
		//
		//culaSparseGetResultString(handle, &result, buffer, 512);
		//std::cout << buffer << std::endl;
		//return 0;
	//}
	//int CULASolver::clean()
	//{
		//culaSparseDestroyPlan(plan);
		//culaSparseDestroy(handle);
		return 0;
	}

	CUSolver::CUSolver()
	{

	}
	int CUSolver::preconditioning()
	{
		cusolver_status = cusolverSpCreate(&cusolverH);
		//assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
		cusparse_status = cusparseCreateMatDescr(&descrA);
		//assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);




		return 0;
	}


	int CUSolver::execute()
	{
		cudaStat1 = cudaMalloc((void**)&d_csrValA, sizeof(double) * A->nnz);
		cudaStat2 = cudaMalloc((void**)&d_csrColIndA, sizeof(int) * A->nnz);
		cudaStat3 = cudaMalloc((void**)&d_csrRowPtrA, sizeof(int) * (A->m + 1));
		cudaStat4 = cudaMalloc((void**)&d_b, sizeof(double) *A->m);
		cudaStat5 = cudaMalloc((void**)&d_x, sizeof(double) * A->m);
		cudaStat1 = cudaMemcpy(d_csrValA, A->val, sizeof(double) *A->nnz, cudaMemcpyHostToDevice);
		cudaStat2 = cudaMemcpy(d_csrColIndA, A->colind, sizeof(int) *A->nnz, cudaMemcpyHostToDevice);
		cudaStat3 = cudaMemcpy(d_csrRowPtrA, A->rowptr, sizeof(int) * (A->m + 1), cudaMemcpyHostToDevice);
		cudaStat4 = cudaMemcpy(d_b, B, sizeof(double) * A->m, cudaMemcpyHostToDevice);
		int singularity;
		cusolverSpDcsrlsvqr(cusolverH, A->m, A->nnz, descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_b, 0.0001, 0, d_x, &singularity);
		cudaStat5 = cudaMemcpy(X, d_x, sizeof(double) * A->m, cudaMemcpyDeviceToHost);
		cudaFree(d_csrValA);
		cudaFree(d_csrColIndA);
		cudaFree(d_csrRowPtrA);
		cudaFree(d_b);
		cudaFree(d_x);
		return 0;
	}
	SparseSolverSelector::SparseSolverSelector()
	{
		this->solver = 0;
	}
	int SparseSolverSelector::setPlatform(PlatFrom p)
	{
		/*
		switch (p)
		{
		case PlatFrom::CULA:
			this->solver = new CULASolver();  break;
		case PlatFrom::CUSOLVER:
			this->solver = new CUSolver(); break;
		case PlatFrom::CUSTOMSOLVER:
			this->solver = new CustomSolver(); break;
		case PlatFrom::CUDASOLVER:
			this->solver = new CUDASolver(); break;
		default:
			break;
		}*/
		this->solver = new CUDASolver();
		return 0;
	}
	SparseSolverSelector::SparseSolverSelector(PlatFrom p)
	{
		setPlatform(p);
	}


	DenseSolverSelector::DenseSolverSelector()
	{
		this->solver = 0;
	}
	int DenseSolverSelector::setSolver(SolverType s)
	{
		/*switch (s)
		{
		case SolverType::LU:
			this->solver = new DenseLUSolver(); break;

		default:
			break;
		}*/
		return 0;
	}
	DenseSolverSelector::DenseSolverSelector(SolverType s)
	{
		setSolver(s);
	}



}