#ifndef _LINEAR_SOLVER_H_
#define _LINEAR_SOLVER_H_
#include "Matrix.h"
//#include "cula_sparse.h"
#include <cusparse.h>
#include <cublas_v2.h>
#include "cusolverSp.h"
//#define CULA 0
//#define CUSOLVER 1
//#define DENSELU 3
//#define CG 2
//#define CUSTOMSOLVER 4
//#define CS 4
enum PlatFrom
{
	CULA,
	CUSOLVER,
	CUSTOMSOLVER,
	CUDASOLVER
};
enum SolverType
{
	CG,
	BiCG,
	BiCGSTAB,
	BiCGSTABl,
	GMRESm,
	MINRES,
	LU,
	NoSolver
};
enum PreconditionerType
{
	NoPreconditioner,
	Jacobi,
	BlockJacobi,
	ILUO,
	ApproximateInverse,
	FactorizedApproximateInverse,
	UserDefinedPreconditioner,
	GeneralCG
};
using namespace volcano;
namespace volcano
{
	int SuperLUSolve(CSRMatrix *A, double *x, double *b, int n, int nnz);
	int jacobianIteration(CSRMatrix*A, double *b, double maxTol, int maxIterationTimes, double *&x);
	



	class SparseSolver
	{
	protected:
		CSRMatrix *A=0;
		double *B=0;
		double *X=0;
		SolverType solver;
		PreconditionerType preconditioner;
		double tol = 0.000001;
		CSRMatrix *M1 = 0, *M2 = 0;
		double *y = 0;
	public:
		SparseSolver(){A = 0;B = 0;X = 0;};
		void setA(CSRMatrix *a){ A = a; };
		void setB(double *b){B = b;};
		void setX(double *&x){X = x;};
		void setData(CSRMatrix *a, double *b, double *&x){A = a; B = b; X = x;};
		void setM(CSRMatrix *m1, CSRMatrix *m2){ M1 = m1; M2 = m2; };
		void setM(CSRMatrix *m){ M1 = m; M2 = NULL; };
		void setSolver(SolverType s){ solver = s; };
		void setPreconditioner(PreconditionerType p)  { preconditioner = p; };
		void setTol(double t){ tol = t; };
		virtual int execute()=0;
		virtual int preconditioning() = 0;
		double* getX();
	};

	class CUDASolver :public SparseSolver
	{
	private:
		cublasHandle_t cublasHandle;
		cublasStatus_t cublasStatus;
		cusparseHandle_t cusparseHandle;
		cusparseStatus_t cusparseStatus;
		cusparseMatDescr_t descrA;
		cusparseMatDescr_t descrL;
		cusparseMatDescr_t descrU;

	public:
		CUDASolver(){};
		int preconditioning();

		int execute();
		int clean();
	};

	class CULASolver :public SparseSolver
	{
	//private:
	//	culaSparseHandle handle;
	//	culaSparsePlan plan;
	//	culaSparseConfig config;
	//	char buffer[512];

	//	culaSparseStatus status;

	//	culaSparseResult result;
	public:
		CULASolver();
		int preconditioning();
		int execute();
	//	int clean();
	};



	class CustomSolver: public SparseSolver
	{
	private:
		//culaSparseHandle handle;
		//culaSparsePlan plan;
		//culaSparseConfig config;
		//char buffer[512];

		//culaSparseStatus status;

		//culaSparseResult result;
	public:
		static void GeneralCG(CSRMatrix *A, CSRMatrix *&M1);
		static void ConjugateGradient(CSRMatrix *A, double *b, double *&x, double tolerance);
		static void GMRES(CSRMatrix *A, double *b, double *&x, double tolerance);
		static void GMRESm(CSRMatrix *A, double *b, double *&x, int m, double tolerance);


		CustomSolver();
		int preconditioning();
		int execute();
		int clean();
		//int execute(char *solver);
		//int execute(char *solver, char *preconditioner);
	};



	class CUSolver :public SparseSolver
	{
	private:
		cusolverSpHandle_t cusolverH = NULL;
		
		cusparseMatDescr_t descrA = NULL;
		cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
		cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
		cudaError_t cudaStat1 = cudaSuccess;
		cudaError_t cudaStat2 = cudaSuccess;
		cudaError_t cudaStat3 = cudaSuccess;
		cudaError_t cudaStat4 = cudaSuccess;
		cudaError_t cudaStat5 = cudaSuccess;
		int *d_csrRowPtrA = NULL;
		int *d_csrColIndA = NULL;
		double *d_csrValA = NULL;
		double *d_b = NULL;
		double *d_x = NULL;
	public:
		CUSolver();
		int preconditioning();
		int execute();
		int clean();
	};

	class DenseSolver
	{
	protected:
		DenseMatrix *A;
		double *B;
		double *X;
	public:
		DenseSolver();
		int setData(DenseMatrix *a, double *b, double *x);
		virtual int execute() = 0;
		virtual int preconditioning() = 0;
		double* getX();
	};

	class DenseLUSolver:public DenseSolver
	{
	private:
		DenseMatrix *L;
		DenseMatrix *U;
		int DenseLU(DenseMatrix *&A, DenseMatrix *&L, DenseMatrix *&U);
	public:
		DenseLUSolver();
		int execute();
		int preconditioning();
	};



	class SparseSolverSelector
	{
		PlatFrom platfrom;
	public:
		SparseSolver *solver = 0;
		SparseSolverSelector(PlatFrom p);
		SparseSolverSelector();
		int setPlatform(PlatFrom p);

	};


	class DenseSolverSelector
	{
	public:
		DenseSolver *solver = 0;
		DenseSolverSelector(SolverType s);
		DenseSolverSelector();
		int setSolver(SolverType s);
	};
}



#endif // !_LINEAR_SOLVER_H_
