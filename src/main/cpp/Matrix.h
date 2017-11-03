#ifndef _MATRIX_H_
#define _MATRIX_H_
#include <iostream>
#include <fstream>
#include <stdlib.h>
using namespace std;
namespace volcano
{
	class DenseMatrix
	{
	public:
		double **val;
		int n;
		int m;
		DenseMatrix()
		{
			val = NULL;
			n = 0;
			m = 0;
		}
		DenseMatrix(int m, int n)
		{
			this->m = m;
			this->n = n;
			val = (double **)malloc(sizeof(double *)*this->m);
			for (int i = 0; i < this->m; i++)
			{
				val[i] = (double *)malloc(sizeof(double)*this->n);
			}
		}
		~DenseMatrix()
		{
			if (val != NULL)
			{
				for (int i = 0; i < this->m; i++)
				{
					if (val[i] == 0)
					{
						try
						{
							free(val[i]);
							val = 0;
						}
						catch (...)
						{
							delete[] val[i];
							val = 0;
						}
					}
				}
			}
		}
		void show()
		{
			for (int i = 0; i < m; i++)
			{
				//cout << endl;
				for (int j = 0; j < n; j++)
				{
					//cout << val[i][j] << "\t";
					if (val[i][j] >= 1e-6)
						cout << "A(" << i + 1 << "," << j + 1 << ")=" << val[i][j] << ";" << endl;
				}
				//cout << ";";
			}
		}
	};
	class CSRMatrix
	{
	public:
		double *val;
		int *rowptr;
		int *colind;
		int m;
		int n;
		int nnz;
		CSRMatrix();
		CSRMatrix(int m, int n, int nnz);
		CSRMatrix(int n, int nnz);
		~CSRMatrix();
		void show();
		DenseMatrix *toDenseMatrix();
		void showDense();
		int compressLine();
		int compressColumn();
		int compress();
		friend CSRMatrix operator*(CSRMatrix &A, CSRMatrix &B);
		friend double* operator*(CSRMatrix &A, double *B);
		//friend CSRMatrix* operator*(CSRMatrix*&A, CSRMatrix*&B);
		double getNorm(int p, double normTol);
		double getCond(int p, double normTol);
		void distributionStatistics();
	};


	CSRMatrix* CSRTran(CSRMatrix *);
	CSRMatrix* CSRTran_noVal(CSRMatrix *);
	CSRMatrix* CSRMul(CSRMatrix *, CSRMatrix *);
	double *CSRMul(CSRMatrix *, double *);
	CSRMatrix* randomCSRMatrix(int m, int n, int nnz_per_line);
	CSRMatrix* randomSymmetricCSRMatrix(int m, int nnz_per_line);
	CSRMatrix* randomCSRMatrix(int m, int n, int nnz_per_line,long seed);
	CSRMatrix* randomSymmetricCSRMatrix(int m, int nnz_per_line,long seed);
	CSRMatrix* randomLapacianMatrix(int grid_size);
	
	CSRMatrix* simpleInv(CSRMatrix*,int);
	
}
#endif
