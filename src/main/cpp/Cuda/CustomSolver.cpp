#include "LinearSolver.h"
#include "SimpleAlgorithm.h"
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>


//#include "Norm.h"
namespace volcano
{
	double maxinArray(double *data, int length)
	{
		double m = data[0];
		for (int i = 0; i < length; i++)
		{
			if (data[i]>m)
			{
				m = data[i];
			}
		}
		return m;
	}
	CustomSolver::CustomSolver()
	{}
	int CustomSolver::preconditioning()
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
	//void CustomSolver::setSolver(char *s)
	//{
	//	this->solver = s;
	//}
	//void CustomSolver::setPreconditioner(char *p)
	//{
	//	this->preconditioner = p;
	//}
	//void CustomSolver::setTol(double t)
	//{
	//	this->tol = t;
	//}
	int CustomSolver::execute()
	{

		//switch (this->preconditioner)
		//{
		//case(PreconditionerType::GeneralCG) :
		//	GeneralCG(A, M1); break;
		//default:
		//	GeneralCG(A, M1);
		//	//cout << "default" << endl;
		//}
		//if (M2 == 0)
		//{
		//	switch (this->solver)
		//	{
		//	case(SolverType::CG) ://CG ConjugateGradient
		//		ConjugateGradient(CSRMul(M1, A), CSRMul(M1, B), this->X,this->tol); break;
		//	default:
		//		ConjugateGradient(CSRMul(M1, A), CSRMul(M1, B), this->X, this->tol);
		//		//cout << "default" << endl;
		//	}
		//}
		//else
		//{
		//	switch (this->solver)
		//	{
		//	case(SolverType::CG) ://CG ConjugateGradient
		//		ConjugateGradient(CSRMul(CSRMul(M1, A), M2), CSRMul(M1, B), this->y, this->tol); break;
		//	default:
		//		ConjugateGradient(CSRMul(CSRMul(M1, A), M2), CSRMul(M1, B), this->y, this->tol);
		//		//cout << "default" << endl;
		//	}
		//	this->X = CSRMul(M2, y);
		//}
		//M1->showDense();
		//NormInitialize();

		//
		//mwArray Val(1, M1->nnz, mxDOUBLE_CLASS); Val.SetData(M1->val, M1->nnz);
		//mwArray Rowptr(1, M1->m + 1, mxDOUBLE_CLASS); Rowptr.SetData(M1->rowptr, M1->m + 1);
		//mwArray Colind(1, M1->nnz, mxDOUBLE_CLASS); Colind.SetData(M1->colind, M1->nnz);
		//mwArray M(1, 1, mxDOUBLE_CLASS); M.SetData(&M1->m, 1);
		//mwArray NNZ(1, 1, mxDOUBLE_CLASS); NNZ.SetData(&M1->nnz, 1);
		//
		//mwArray ValI(1, M1->nnz, mxDOUBLE_CLASS);
		//mwArray RowptrI(1, M1->m + 1, mxDOUBLE_CLASS); 
		//mwArray ColindI(1, M1->nnz, mxDOUBLE_CLASS);
		//mwArray NNZI(1, 1, mxDOUBLE_CLASS);
		//Inverse(4, ValI, RowptrI, ColindI, NNZI, Val, Rowptr, Colind, M, NNZ);
		//M2 = new CSRMatrix(M1->m,M1->nnz);
		////M2->m = M1->m;
		//NNZI.GetData(&M2->nnz, 1);
		//ValI.GetData(M2->val, M2->nnz);
		//RowptrI.GetData(M2->rowptr, M2->m + 1);
		//ColindI.GetData(M2->colind, M2->nnz);

		////Val.GetData(M2->val,)
		//NormTerminate();
		////M2->showDense();
		////system("pause");


		//CSRMatrix *MA = CSRMul(M2, A);
		//double *MB = CSRMul(M2, B);








		////culaSparseSetDcsrData(handle, plan, 0, MA->m, MA->nnz, MA->val, MA->rowptr, MA->colind, X, MB);
		//culaSparseSetDcsrData(handle, plan, 0, A->m, A->nnz, A->val, A->rowptr, A->colind, X, B);
		//culaSparseExecutePlan(handle, plan, &config, &result);

		//culaSparseGetResultString(handle, &result, buffer, 512);
		//std::cout << buffer << std::endl;




		return 0;
	}

	int CustomSolver::clean()
	{
		/*culaSparseDestroyPlan(plan);
		culaSparseDestroy(handle);*/
		return 0;
	}
	void CustomSolver::GeneralCG(CSRMatrix *A, CSRMatrix *&M1)
	{
		M1 = CSRTran(A); 
	}


	void CustomSolver::ConjugateGradient(CSRMatrix *A, double *b, double *&x, double tolerance)
	{
		if (x != 0)
		{
			try
			{
				free(x);
				x = 0;
			}
			catch (...)
			{
				delete[] x;
				x = 0;
			}
		}
		x = (double *)malloc(sizeof(double)*A->m);
		for (int i = 0; i < A->m; i++)
		{
			x[i] = 0;
		}
		double *r = CSRMul(A,x);
		double *p = (double *)malloc(sizeof(double)*A->m);
		for (int i = 0; i < A->m; i++)
		{
			r[i] = b[i] - r[i];
			p[i] = r[i];
		}
		double alpha = 0, beta = 0;
		int k = 0;
		double res = 0;
		do
		{
			double *ap = CSRMul(A, p);
			double alpha1 = 0, alpha2 = 0;
			for (int i = 0; i < A->m; i++)
			{
				alpha1 += r[i] * r[i];
				alpha2 += p[i] * ap[i];
			}
			alpha = 0 - alpha1 / alpha2;
			for (int i = 0; i < A->m; i++)
			{
				x[i] -= alpha*p[i];
				r[i] =r[i]+ alpha*ap[i];
				//cout << r[i] << "\t";
			}
			//cout << endl;
			beta = 0;
			for (int i = 0; i < A->m; i++)
			{
				beta += r[i] * r[i];
			}
			beta /= alpha1;
			for (int i = 0; i < A->m; i++)
			{
				p[i] = r[i] + beta*p[i];
			}
			free(ap);
			//ap = 0;
			res = maxinArray(r, A->m);
			//cout << "\nRes["<<k<<"] = " << res ;
			k++;
		} while ( res> tolerance);
		free(r);
		//r = 0;
		free(p);
		//p = 0;
		
		
	}

	void  CustomSolver::GMRES(CSRMatrix *A, double *b, double *&x, double tolerance)
	{
		if (x != 0)
		{
			try
			{
				free(x);
				x = 0;
			}
			catch (...)
			{
				delete[] x;
				x = 0;
			}
		}
		x = (double *)malloc(sizeof(double)*A->m);
		for (int i = 0; i < A->m; i++)
		{
			x[i] = 0;
		}
		double *r = CSRMul(A, x);
		double *v = (double *)malloc(sizeof(double)*A->m);
		double beta = 0;
		for (int i = 0; i < A->m; i++)
		{
			r[i] = b[i] - r[i];
			beta += r[i] * r[i];
		}
		beta = sqrt(beta);
		for (int i = 0; i < A->m; i++)
		{
			v[i] = r[i] / beta;
		}
		int k = 0;
		double res = 0;
		do
		{
		

		} while (res> tolerance);
	}
	void  CustomSolver::GMRESm(CSRMatrix *A, double *b, double *&x, int m, double tolerance)
	{
		if (x != 0)
		{
			try
			{
				free(x);
				x = 0;
			}
			catch (...)
			{
				delete[] x;
				x = 0;
			}
		}
		x = (double *)malloc(sizeof(double)*A->m);
		for (int i = 0; i < A->m; i++)
		{
			x[i] = 0;
		}
		double *r = CSRMul(A, x);
		double **v = (double **)malloc(sizeof(double*)*(m + 1));
		for (int i = 0; i < m + 1; i++)
		{
			v[i] = (double*)malloc(sizeof(double)*A->m);
		}
		
		double beta = 0;
		for (int i = 0; i < A->m; i++)
		{
			r[i] = b[i] - r[i];
			beta += r[i] * r[i];
		}
		beta = sqrt(beta);
		for (int i = 0; i < A->m; i++)
		{
			v[0][i] = r[i] / beta;
		}
		int k = 0;
		double res = 0;
		double **H = (double **)malloc(sizeof(double*)*(m + 1));
		for (int i = 0; i < m + 1; i++)
		{
			H[i] = (double *)malloc(sizeof(double)*m);
			for (int j = 0; j < m; j++)
			{
				H[i][j] = 0;
			}
		}
		
		do
		{
			for (int j = 0; j < m; j++)
			{
				double *av = CSRMul(A, v[j]);
				for (int k = 0; k < A->m; k++)
				{
					v[j + 1][k] = av[k];
				}
				for (int i = 0; i <= j; i++)
				{
					H[i][j] = 0;
					for (int k = 0; k < A->m; k++)
					{
						H[i][j] += av[k] * v[i][k];
					}
					for (int k = 0; k < A->m; k++)
					{
						v[j+1][k] -=  H[i][j] * v[i][k];
					}
				}
				H[j + 1][j] = 0;
				for (int i = 0; i < A->m; i++)
				{
					H[j + 1][j] += v[j + 1][i] * v[j + 1][i];
				}
				H[j + 1][j] = sqrt(H[j + 1][j]);
				for (int i = 0; i < A->m; i++)
				{
					v[j + 1][i] = v[j + 1][i] / H[j + 1][j];
				}

			}

		} while (res> tolerance);
	}








	int CUDASolver::preconditioning()
	{


		cublasHandle = 0;
		cublasStatus = cublasCreate(&cublasHandle);
		

		cusparseHandle = 0;
		cusparseStatus = cusparseCreate(&cusparseHandle);
		
		descrA = 0;
		descrL = 0;
		descrU = 0;
		cusparseStatus = cusparseCreateMatDescr(&descrA);

		cusparseStatus = cusparseCreateMatDescr(&descrL);
		
		cusparseStatus = cusparseCreateMatDescr(&descrU);


		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);


		cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
		cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);

		cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
		cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);



		
		return 0;
	}
	
	int CUDASolver::execute()
	{
		const double one = 1.0;
		const double zero = 0;
		const double negativeone=-1.0;


		for (int i = 0; i < A->m; i++)
		{
			X[i] = 0;
		}
		double *d_val,  *d_x,*d_r,*d_f;
		int *d_col, *d_row;
		cudaMalloc((void **)&d_val, A->nnz*sizeof(double));
		cudaMalloc((void **)&d_row, (A->m+1)*sizeof(int));
		cudaMalloc((void **)&d_col, (A->nnz)*sizeof(int));
		cudaMalloc((void **)&d_x, A->m*sizeof(double));
		cudaMalloc((void **)&d_r, A->m*sizeof(double));
		cudaMalloc((void **)&d_f, A->m*sizeof(double));


		cudaMemcpy(d_col, A->colind, A->nnz*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_row, A->rowptr, (A->m + 1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_val, A->val, A->nnz*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_x, X, A->m*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_f,B, A->m*sizeof(double), cudaMemcpyHostToDevice);
		int N = A->m;
		int nz = A->nnz;
		int nzILU0 = 2 * N - 1;
		double *valsILU0 = (double *)malloc(nz*sizeof(double));
		double *d_valsILU0;
		double *d_zm1;
		double *d_zm2;
		double *d_rm2;
		
		//CUDADebug::printDeviceArray(d_f, A->m);
		//CUDADebug::printHostArray(B, A->m);

		cudaMalloc((void **)&d_valsILU0, nz*sizeof(double));
		cudaMalloc((void **)&d_zm1, (N)*sizeof(double));
		cudaMalloc((void **)&d_zm2, (N)*sizeof(double));
		cudaMalloc((void **)&d_rm2, (N)*sizeof(double));


		



		cusparseSolveAnalysisInfo_t infoA = 0;
		cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);
		

		/* Perform the analysis for the Non-Transpose case */
		cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			N, nz, descrA, d_val, d_row, d_col, infoA);

	

		/* Copy A data to ILU0 vals as input*/
		cudaMemcpy(d_valsILU0, d_val, nz*sizeof(double), cudaMemcpyDeviceToDevice);
		cusparseStatus = cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descrA, d_valsILU0, d_row, d_col, infoA);
		//checkCudaErrors(cusparseStatus);
		//cusparseSolveAnalysisInfo_t info_u;
		//cusparseCreateSolveAnalysisInfo(&info_u);

		int n =N ;


		cusparseSolveAnalysisInfo_t infoL,infoU;
		cusparseCreateSolveAnalysisInfo(&infoL);
		cusparseCreateSolveAnalysisInfo(&infoU);

		cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, d_val, d_row, d_col, infoU);
		cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrL, d_val, d_row, d_col, infoL);




		double nrmr0,nrmr;


		double *d_p,*d_q, *d_rw;
		double *d_t,*d_s, *d_ph;
		cudaMalloc((void **)&d_p, n*sizeof(double));
		cudaMalloc((void **)&d_q, n*sizeof(double));
		cudaMalloc((void **)&d_rw,n*sizeof(double));
		cudaMalloc((void **)&d_t, n*sizeof(double));
		cudaMalloc((void **)&d_s, n*sizeof(double));
		cudaMalloc((void **)&d_ph, n*sizeof(double));
		cudaMemcpy(d_q, X, A->m*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_t, X, A->m*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_s, X, A->m*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ph, X, A->m*sizeof(double), cudaMemcpyHostToDevice);
		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nz,&one,descrA,d_val, d_row, d_col, d_x,&zero, d_r);
		cublasDscal(cublasHandle, n, &negativeone, d_r, 1);
		cublasDaxpy(cublasHandle, n, &one, d_f, 1, d_r, 1);
		cublasDcopy(cublasHandle, n, d_r, 1, d_p, 1);
		cublasDcopy(cublasHandle, n, d_r, 1, d_rw, 1);
		cublasDnrm2(cublasHandle, n, d_r, 1, &nrmr0);
		int maxit = 1000;
		double alpha = 0, alpha_i = 0, beta = 0, omega = 1, omega_i;
		double rho = 0, rhop=0;
		for (int i = 0; i<maxit; i++)
		{
			rhop = rho;
			cublasDdot(cublasHandle,n, d_rw, 1, d_r, 1, &rho);
			if (i > 0)
			{
				beta = (rho / rhop)*(alpha / omega);
				omega_i = 0 - omega;
				cublasDaxpy(cublasHandle,n, &omega_i, d_q, 1, d_p, 1);
				cublasDscal(cublasHandle,n, &beta, d_p, 1);
				cublasDaxpy(cublasHandle,n, &one, d_r, 1, d_p, 1);
			}
			cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one, descrL, d_valsILU0, d_row, d_col,infoL, d_p, d_t);
			cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one, descrU, d_valsILU0, d_row, d_col,infoU, d_t,d_ph);

			cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nz,&one,descrA, d_val, d_row,d_col, d_ph, &zero, d_q);
			__device__ double temp,temp2;
			 cublasDdot(cublasHandle,n, d_rw, 1, d_q, 1,&temp);
			alpha = rho / temp;
			alpha_i = 0 - alpha;
			cublasDaxpy(cublasHandle, n, &alpha_i, d_q, 1, d_r, 1);
			cublasDaxpy(cublasHandle, n, &alpha, d_ph, 1, d_x, 1);
			cublasDnrm2(cublasHandle, n, d_r, 1,&nrmr);
			if (nrmr / nrmr0 < tol)
			{
				break;
			}
			cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one, descrL, d_valsILU0, d_row, d_col,infoL, d_r, d_t);
			cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one, descrU, d_valsILU0, d_row, d_col,infoU, d_t, d_s);
			cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n,nz, &one,descrA, d_val, d_row, d_col, d_s, &zero, d_t);
			cublasDdot(cublasHandle,n, d_t, 1, d_r, 1,&temp);
			cublasDdot(cublasHandle, n, d_t, 1, d_t, 1,&temp2);
			omega = temp / temp2;
			cublasDaxpy(cublasHandle,n, &omega, d_s, 1, d_x, 1);
			omega_i = 0 - omega;
			cublasDaxpy(cublasHandle,n, &omega_i, d_t, 1, d_r, 1);  
			cublasDnrm2(cublasHandle,n, d_r, 1,&nrmr);
			if (nrmr / nrmr0 < tol)
			{
				break;
			}
		}

		cudaMemcpy(X, d_x, A->m*sizeof(double), cudaMemcpyDeviceToHost);
		cusparseDestroySolveAnalysisInfo(infoA);
		cusparseDestroySolveAnalysisInfo(infoL);
		cusparseDestroySolveAnalysisInfo(infoU);
		cudaFree(d_val); cudaFree(d_row); cudaFree(d_col); cudaFree(d_x); cudaFree(d_r); cudaFree(d_f);cudaFree(d_valsILU0);cudaFree(d_zm1);cudaFree(d_zm2);cudaFree(d_rm2);
		cudaFree(d_p);cudaFree(d_q);cudaFree(d_rw);cudaFree(d_t);cudaFree(d_s);cudaFree(d_ph);
		return 0;
	}
	int CUDASolver::clean()
	{
		cusparseDestroy(cusparseHandle);
		cublasDestroy(cublasHandle);
		return 0;
	}
}