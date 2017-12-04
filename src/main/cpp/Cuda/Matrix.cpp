#include <vector>
#include <algorithm>
#include <iostream>  
#include <iomanip>  
#include <fstream>
#include <cmath>
#include <time.h>
#include "Matrix.h"
#include "SimpleAlgorithm.h"
#include <stdlib.h>

using namespace volcano;
using namespace std;
namespace volcano
{
	std::vector<int> get_square_grid(int n)
	{
		// allocate square grid with -1 
		std::vector<int> grid(n*n, -1);

		// fill interior index map
		int idx = 0;
		for (int i = 1; i < n - 1; i++)
			for (int j = 1; j < n - 1; j++)
				grid[i*n + j] = idx++;

		return grid;
	}
	int build_lapacian_double(int n, std::vector<double>& values, std::vector<int>& rowInd, std::vector<int>& colInd)
	{
		// get square grid
		std::vector<int> grid = get_square_grid(n);

		// build index set list
		std::vector<int> index_set;
		for (size_t i = 0; i < grid.size(); i++)
			if (grid[i] != -1)
				index_set.push_back(int(i));

		// fill sparse vectors
		for (size_t i = 0; i < index_set.size(); i++)
		{
			// add self
			int self_index = grid[index_set[i]];
			values.push_back(4.0);
			rowInd.push_back(self_index);
			colInd.push_back(self_index);

			// add adjacent sides
			int degree = 1;
			int adj_index[4];
			adj_index[0] = grid[index_set[i] - 1];
			adj_index[1] = grid[index_set[i] + 1];
			adj_index[2] = grid[index_set[i] - n];
			adj_index[3] = grid[index_set[i] + n];
			for (int j = 0; j < 4; j++)
			{
				if (adj_index[j] != -1)
				{
					// values.push_back(-1.0);
					values.push_back(-1.0);
					rowInd.push_back(self_index);
					colInd.push_back(adj_index[j]);
					degree++;
				}
			}

		}

		return int(index_set.size());
	}
	double getNorm1(CSRMatrix* A)
	{
		double *sumColumn = (double*)malloc(sizeof(double)*A->n);
		for (int i = 0; i < A->n; i++)
		{
			sumColumn[i] = 0;
		}
		for (int i = 0; i < A->m; i++)
		{
			for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
			{
				sumColumn[A->colind[j]] += abs(A->val[j]);
			}
		}
		double norm1 = 0;
		for (int i = 0; i < A->n; i++)
		{
			if (sumColumn[i]>norm1)
			{
				norm1 = sumColumn[i];
			}
		}
		return norm1;
	}
	//double getNorm2(CSRMatrix* A)
	//{
	//	DenseMatrix *dA = A->toDenseMatrix();

	//}
	double getNormInf(CSRMatrix *A)
	{
		double *sumLine = (double*)malloc(sizeof(double)*A->m);
		double normInf = 0;
		for (int i = 0; i < A->m; i++)
		{
			sumLine[i] = 0;
			for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
			{
				sumLine[i] += abs(A->val[j]);
			}
			if (sumLine[i]>normInf)
			{
				normInf = sumLine[i];
			}
		}
		return normInf;
	}


	CSRMatrix::CSRMatrix()
	{
		val = NULL;
		rowptr = NULL;
		colind = NULL;
		m = 0;
		n = 0;
		nnz = 0;
	}
	CSRMatrix::CSRMatrix(int m, int n, int nnz)
	{
		this->m = m;
		this->n = n;
		this->nnz = nnz;
		val = (double*)malloc(sizeof(double)*nnz);
		colind = (int *)malloc(sizeof(int)*nnz);
		rowptr = (int *)malloc(sizeof(int)*(m + 1));
	}
	CSRMatrix::CSRMatrix(int n, int nnz)
	{
		this->m = n;
		this->n = n;
		this->nnz = nnz;
		val = (double*)malloc(sizeof(double)*nnz);
		colind = (int *)malloc(sizeof(int)*nnz);
		rowptr = (int *)malloc(sizeof(int)*(m + 1));
	}
	CSRMatrix::~CSRMatrix()
	{
		if (val != NULL)
		{
			try
			{
				free(val);
				val = 0;
			}
			catch (...)
			{
				delete[] val;
				val = 0;
			}
		}
		if (colind != NULL)
		{
			try
			{
				free(colind);
				colind = 0;
			}
			catch (...)
			{
				delete[] colind;
				colind = 0;
			}
		}
		if (rowptr != NULL)
		{
			try
			{
				free(rowptr);
				rowptr = 0;
			}
			catch (...)
			{
				delete[] rowptr;
				rowptr = 0;
			}
		}
	}
	void CSRMatrix::show()
	{
		cout << endl << "m=" << m << "\t" << "n=" << n << "\t" << "nnz=" << nnz;
		cout << endl << "val= ";
		for (int i = 0; i < m; i++)
		{
			cout << "\n";
			for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
			{
				cout << val[j] << "\t";
			}
		}
		cout << endl << endl << "rowptr= ";
		for (int i = 0; i < m + 1; i++)
		{
			cout << rowptr[i] << "\t";
		}
		cout << endl << endl << "colind= ";
		for (int i = 0; i < nnz; i++)
		{
			cout << colind[i] << "\t";
		}

	}
	DenseMatrix* CSRMatrix::toDenseMatrix()
	{
		DenseMatrix *densematrix;
		densematrix = new DenseMatrix(m, n);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				densematrix->val[i][j] = 0;
			}
		}
		for (int i = 0; i < m; i++)
		{
			for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
			{
				densematrix->val[i][colind[j]] = val[j];
			}
		}
		return densematrix;
	}
	void CSRMatrix::showDense()
	{
		DenseMatrix *densematrix = this->toDenseMatrix();
		densematrix->show();
		delete densematrix;
	}
	int CSRMatrix::compressLine()
	{
		int cpLineNum = 0;
		for (int i = 0; i< m; i++)
		{
			if (rowptr[i] == rowptr[i+1])
			{

				cpLineNum++;
			}

		}
		int M = n - cpLineNum;

		int *rowptrCompressed = (int *)malloc(sizeof(int)*M);
		rowptrCompressed[0] = 0;
		for (int i = 0,j=1; i< m; i++)
		{
			if (rowptr[i] != rowptr[i + 1])
			{
				rowptrCompressed[j++] = rowptr[i+1];
			}

		}
		m = M;
		free(rowptr);
		
		rowptr = rowptrCompressed;
		return 0;
	}
	int CSRMatrix::compressColumn()
	{
		bool *numIsExit = (bool *)malloc(sizeof(bool)*n);
		for (int i = 0; i < n; i++)
		{
			numIsExit[i] = false;
		}
		for (int i = 0; i < nnz; i++)
		{
			numIsExit[colind[i]] = true;
		}
		int *perm = (int *)malloc(sizeof(int)*n);
		int N = 0;
		for (int i = 0,j=0; i < n; i++)
		{
			if (numIsExit[i])
			{
				perm[i] = j++;
				N++;
			}
		}
		for (int i = 0; i < nnz; i++)
		{
			colind[i] = perm[colind[i]];
		}
		n = N;
		free(numIsExit); numIsExit = 0;
		free(perm); perm = 0;
		return 0;
	}
	int CSRMatrix::compress()
	{
		return compressLine() + compressColumn();
	}


	//double CSRMatrix::getNorm(int p, double normTol)
	//{
	//	//switch (p)
	//	//{
	//	//case 1:getNorm1(this); break;
	//	//case 2:getNorm1(this); break;
	//	//case INT_MAX:getNormInf(this); break;
	//	//default:cout << "Error Norm Type" << endl;
	//	//}


	//	NormInitialize();

	//	mwArray norm(1, 1, mxDOUBLE_CLASS);
	//	mwArray Val(1, this->nnz, mxDOUBLE_CLASS); Val.SetData(this->val, this->nnz);
	//	mwArray Rowptr(1, this->m + 1, mxDOUBLE_CLASS); Rowptr.SetData(this->rowptr, this->m + 1);
	//	mwArray Colind(1, this->nnz, mxDOUBLE_CLASS); Colind.SetData(this->colind, this->nnz);
	//	mwArray M(1, 1, mxDOUBLE_CLASS); M.SetData(&this->m, 1);
	//	mwArray NNZ(1, 1, mxDOUBLE_CLASS); NNZ.SetData(&this->nnz, 1);
	//	mwArray NormDim(1, 1, mxDOUBLE_CLASS); NormDim.SetData(&p, 1);
	//	mwArray Tol(1, 1, mxDOUBLE_CLASS); Tol.SetData(&normTol, 1);
	//	Norm(1, norm, Val, Rowptr, Colind, M, NNZ, NormDim, Tol);
	//	double Ans;
	//	norm.GetData(&Ans, 1);
	//	NormTerminate();
	//	return Ans;

	//}
	//double CSRMatrix::getCond(int p, double normTol)
	//{
	//	NormInitialize();
	//	
	//	mwArray cond(1, 1, mxDOUBLE_CLASS);
	//	mwArray Val(1, this->nnz, mxDOUBLE_CLASS); Val.SetData(this->val, this->nnz);
	//	mwArray Rowptr(1, this->m + 1, mxDOUBLE_CLASS); Rowptr.SetData(this->rowptr, this->m + 1);
	//	mwArray Colind(1, this->nnz, mxDOUBLE_CLASS); Colind.SetData(this->colind, this->nnz);
	//	mwArray M(1, 1, mxDOUBLE_CLASS); M.SetData(&this->m, 1);
	//	mwArray NNZ(1, 1, mxDOUBLE_CLASS); NNZ.SetData(&this->nnz, 1);
	//	mwArray CondDim(1, 1, mxDOUBLE_CLASS); CondDim.SetData(&p, 1);
	//	mwArray Tol(1, 1, mxDOUBLE_CLASS); Tol.SetData(&normTol, 1);
	//	Cond(1, cond, Val, Rowptr, Colind, M, NNZ,CondDim, Tol);
	//	double Ans;
	//	cond.GetData(&Ans, 1);
	//	NormTerminate();
	//	return Ans;
	//}

	void CSRMatrix::distributionStatistics()
	{

		fstream distributionStatisticsFile("Data//DistributionStatistics//DistributionStatistics.txt");
		if (!distributionStatisticsFile)
		{
			cout << "Unable to open distributionStatisticsFile";
			system("pause");
			exit(1); 
		}
		double min = val[0], max = val[0];
		for (int i = 0; i < this->nnz; i++)
		{
			if (val[i]>max)
			{
				max = val[i];
			}
			if (val[i] < min)
			{
				min = val[i];
			}
		}
		const int duanshu= 100;
		int *di = (int *)malloc(sizeof(int) * duanshu);
		double *mi = (double *)malloc(sizeof(double) * duanshu);
		for (int i = 0; i < duanshu; i++)
		{
			di[i] = 0;
	
		}
		for (int i = 0; i < nnz; i++)
		{
			di[(int)((val[i] - min) * duanshu / (max - min))]++;;
		}
		distributionStatisticsFile.seekp(0, ios::end);
		distributionStatisticsFile << "\n*****************************************\n";
		for (int i = 0; i < duanshu; i++)
		{
			double left = min + i*(max - min) / duanshu;
			double right = min + (i + 1)*(max - min) / duanshu;
			distributionStatisticsFile << "(" << left << " , " << right << ") =" << di[i] << endl;
			mi[i] = (left + right) / 2;
		}
		distributionStatisticsFile << "\n*****************************************\n\n";
		distributionStatisticsFile.close();

		//NormInitialize();
		//mwArray Val(1, 100, mxDOUBLE_CLASS); Val.SetData(di, duanshu);
		//mwArray ax(1, 100, mxDOUBLE_CLASS); ax.SetData(mi, duanshu);
		//cppPlot(ax, Val);
		//NormTerminate();
		//system("pause");
	}
	CSRMatrix* CSRTran(CSRMatrix *A)
	{
		int *nnz_line;
		nnz_line = (int *)malloc(sizeof(int)*A->m);
		for (int i = 0; i < A->m; i++)
		{
			nnz_line[i] = 0;
		}
		for (int i = 0; i < A->nnz; i++)
		{
			nnz_line[A->colind[i]]++;
		}
		CSRMatrix *AT;
		AT = new CSRMatrix(A->n, A->m, A->nnz);
		AT->rowptr[0] = 0;
		for (int i = 0; i < AT->n; i++)
		{
			AT->rowptr[i + 1] = AT->rowptr[i] + nnz_line[i];
		}
		free(nnz_line); nnz_line = 0;
		int *AT_line_index;
		AT_line_index = (int *)malloc(sizeof(int)*AT->n);
		for (int i = 0; i < AT->n; i++)
		{
			AT_line_index[i] = 0;
		}
		for (int i = 0; i < A->n; i++)
		{
			for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
			{
				AT->val[AT->rowptr[A->colind[j]] + AT_line_index[A->colind[j]]] = A->val[j];
				AT->colind[AT->rowptr[A->colind[j]] + AT_line_index[A->colind[j]]] = i;
				AT_line_index[A->colind[j]]++;
			}
		}
		free(AT_line_index); AT_line_index = 0;
		return AT;
	}
	CSRMatrix* CSRTran_noVal(CSRMatrix *A)
	{
		int *nnz_line;
		nnz_line = (int *)malloc(sizeof(int)*A->m);
		for (int i = 0; i < A->m; i++)
		{
			nnz_line[i] = 0;
		}
		for (int i = 0; i < A->nnz; i++)
		{
			nnz_line[A->colind[i]]++;
		}
		CSRMatrix *AT;
		AT = new CSRMatrix(A->n, A->m, A->nnz);
		AT->rowptr[0] = 0;
		for (int i = 0; i < AT->n; i++)
		{
			AT->rowptr[i + 1] = AT->rowptr[i] + nnz_line[i];
		}
		free(nnz_line);
		int *AT_line_index;
		AT_line_index = (int *)malloc(sizeof(int)*AT->n);
		for (int i = 0; i < AT->n; i++)
		{
			AT_line_index[i] = 0;
		}
		for (int i = 0; i < A->n; i++)
		{
			for (int j = A->rowptr[i]; j <A->rowptr[i + 1]; j++)
			{
				//AT->val[AT->rowptr[A->colind[j]] + AT_line_index[A->colind[j]]] = A->val[j];
				AT->colind[AT->rowptr[A->colind[j]] + AT_line_index[A->colind[j]]] = i;
				AT_line_index[A->colind[j]]++;
			}
		}
		free(AT_line_index); AT_line_index = 0;
		return AT;
	}
	CSRMatrix* CSRMul(CSRMatrix *A, CSRMatrix *B)
	{
		if (A->n != B->m)
		{
			cout << "Matrix Mul Error" << endl;
			return 0;
		}
		CSRMatrix *C;
		C = new CSRMatrix();
		C->n = A->n;
		C->m = B->m;
		C->rowptr = (int *)malloc(sizeof(int)*(C->n + 1));
		C->rowptr[0] = 0;
		CSRMatrix *BT = CSRTran(B);
		//BT->show();
		int *C_nnz_line;
		C_nnz_line = (int *)malloc(sizeof(int)*C->n);
		for (int i = 0; i < C->n; i++)
		{
			C_nnz_line[i] = 0;
		}
		for (int i = 0; i < A->n; i++)
		{
			int A_nnz_line = A->rowptr[i + 1] - A->rowptr[i];
			for (int j = 0; j < BT->n; j++)
			{
				int BT_nnz_line = BT->rowptr[j + 1] - BT->rowptr[j];
				//cout << A_nnz_line + BT_nnz_line << endl;
				////////////////////�˴����Զ��߳�
				int *AwithB_colind_line;
				AwithB_colind_line = (int *)malloc(sizeof(int)*(A_nnz_line + BT_nnz_line));
				for (int k = A->rowptr[i]; k < A->rowptr[i + 1]; k++)
				{
					AwithB_colind_line[k - A->rowptr[i]] = A->colind[k];
				}
				for (int k = BT->rowptr[j]; k < BT->rowptr[j + 1]; k++)
				{
					AwithB_colind_line[k - BT->rowptr[j] + A_nnz_line] = BT->colind[k];
				}
				//cout << endl;
				//for (int k = 0; k < A_nnz_line + BT_nnz_line; k++)
				//{
				//	cout << AwithB_colind_line[k] << "\t";
				//}

				qsort(AwithB_colind_line, A_nnz_line + BT_nnz_line, sizeof(int), comp);

				//cout << endl;
				//for (int k = 0; k < A_nnz_line + BT_nnz_line; k++)
				//{
				//	cout << AwithB_colind_line[k] << "\t";
				//}
				//free(AwithB_colind_line);
				bool nnz_flag = false;
				for (int k = 0; k < A_nnz_line + BT_nnz_line - 1; k++)
				{
					if (AwithB_colind_line[k] == AwithB_colind_line[k + 1])
					{
						nnz_flag = true;
						break;
					}
				}
				if (nnz_flag)
				{
					C_nnz_line[i]++;
				}
				free(AwithB_colind_line);
				AwithB_colind_line = 0;
			}
		}
		for (int i = 0; i < C->n; i++)
		{
			C->rowptr[i + 1] = C->rowptr[i] + C_nnz_line[i];
		}
		C->nnz = C->rowptr[C->n];
		C->colind = (int *)malloc(sizeof(int)*C->nnz);
		C->val = (double *)malloc(sizeof(double)*C->nnz);
		for (int i = 0; i < C->nnz; i++)
		{
			C->val[i] = 0;
		}

		int nnz_index = 0;
		for (int i = 0; i < A->n; i++)
		{
			int A_nnz_line = A->rowptr[i + 1] - A->rowptr[i];
			for (int j = 0; j < BT->n; j++)
			{
				int BT_nnz_line = BT->rowptr[j + 1] - BT->rowptr[j];
				int *AwithB_colind_line;
				double *AwithB_nnz_line;
				AwithB_colind_line = (int *)malloc(sizeof(int)*(A_nnz_line + BT_nnz_line));
				AwithB_nnz_line = (double *)malloc(sizeof(double)*(A_nnz_line + BT_nnz_line));
				for (int k = A->rowptr[i]; k < A->rowptr[i + 1]; k++)
				{
					AwithB_colind_line[k - A->rowptr[i]] = A->colind[k];
					AwithB_nnz_line[k - A->rowptr[i]] = A->val[k];
				}
				for (int k = BT->rowptr[j]; k < BT->rowptr[j + 1]; k++)
				{
					AwithB_colind_line[k - BT->rowptr[j] + A_nnz_line] = BT->colind[k];
					AwithB_nnz_line[k - BT->rowptr[j] + A_nnz_line] = BT->val[k];
				}
				buckets_sort_v2(AwithB_colind_line, AwithB_nnz_line, A_nnz_line + BT_nnz_line);
				//qsort(AwithB_colind_line, A_nnz_line + BT_nnz_line, sizeof(int), comp);
				bool nnz_flag = false;
				for (int k = 0; k < A_nnz_line + BT_nnz_line - 1; k++)
				{
					bool fisttime = true;
					if (AwithB_colind_line[k] == AwithB_colind_line[k + 1])
					{
						if (fisttime)
						{
							C->colind[nnz_index] = j;
							nnz_flag = true;//�ĳ�ִֻ��һ��
							fisttime = false;
						}
						C->val[nnz_index] += AwithB_nnz_line[k] * AwithB_nnz_line[k + 1];
					}
				}
				if (nnz_flag)
				{
					nnz_index++;
				}
				free(AwithB_colind_line); AwithB_colind_line = 0;
				free(AwithB_nnz_line); AwithB_nnz_line = 0;

			}
		}

		//for (int i = 0; i < A->n; i++)
		//{
		//	for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
		//	{
		//		for (int k = B->rowptr[A->colind[j]];k< B->rowptr[A->colind[j]+1]; k++)
		//		{
		//			
		//		}
		//	}
		//}
		free(C_nnz_line); C_nnz_line = 0;
		return C;

	}

	double *CSRMul(CSRMatrix *A, double *B)
	{
		double *C = (double *)malloc(sizeof(double)*A->m);
		for (int i = 0; i < A->m; i++)
		{
			C[i] = 0;
			for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
			{
				C[i] += A->val[j] * B[A->colind[j]];
			}
		}
		return C;
	}
	CSRMatrix operator*(CSRMatrix &A, CSRMatrix &B)
	{
		return *(CSRMul(&A, &B));
	}

	//CSRMatrix* operator*(CSRMatrix*&A, CSRMatrix*&B)
	//{

	//}

	double* operator*(CSRMatrix &A, double *B)
	{
		return CSRMul(&A, B);
	}


	CSRMatrix* randomCSRMatrix(int m, int n, int nnz_per_line)
	{
		CSRMatrix *csrmatrix = new CSRMatrix();
		csrmatrix->m = m;
		csrmatrix->n = n;
		csrmatrix->rowptr = (int *)malloc(sizeof(int)*(m + 1));
		csrmatrix->rowptr[0] = 0;
		for (int i = 0; i < m; i++)
		{
			csrmatrix->rowptr[i + 1] = csrmatrix->rowptr[i] + nnz_per_line;
		}
		csrmatrix->nnz = csrmatrix->rowptr[m];
		csrmatrix->colind = (int *)malloc(sizeof(int)*csrmatrix->nnz);
		csrmatrix->val = (double *)malloc(sizeof(double)*csrmatrix->nnz);

		for (int i = 0; i < m; i++)
		{
//			srand(clock());
			int *temp = randomN(csrmatrix->rowptr[i + 1] - 1 - csrmatrix->rowptr[i], 0, n - 2);
			for (int j = 0; j < csrmatrix->rowptr[i + 1] - 1 - csrmatrix->rowptr[i]; j++)
			{
				if (temp[j] >= i)
				{
					temp[j]++;
				}
			}
			for (int j = csrmatrix->rowptr[i]; j < csrmatrix->rowptr[i + 1] - 1; j++)
			{
				csrmatrix->colind[j] = temp[j - csrmatrix->rowptr[i]];
			}
			csrmatrix->colind[csrmatrix->rowptr[i + 1] - 1] = i;
			qsort(&(csrmatrix->colind[csrmatrix->rowptr[i]]), csrmatrix->rowptr[i + 1] - csrmatrix->rowptr[i], sizeof(int), comp);
			
			for (int j = csrmatrix->rowptr[i]; j < csrmatrix->rowptr[i + 1]; j++)
			{
				csrmatrix->val[j] = (double)(rand() % 2000) / 1000;
			}
			free(temp); temp = 0;
		}
		return csrmatrix;
	}
	CSRMatrix* randomCSRMatrix(int m, int n, int nnz_per_line,long seed)
	{
		srand(seed);
		return randomCSRMatrix(m, n, nnz_per_line);
	}
	CSRMatrix* randomSymmetricCSRMatrix(int m, int nnz_per_line)
	{
		CSRMatrix *csrmatrix0 = randomCSRMatrix(m, m, nnz_per_line);
		CSRMatrix *csrmatrix1 = CSRTran(csrmatrix0);
		int *axis0, *axis1;
		axis0 = (int *)malloc(sizeof(int)*m);
		axis1 = (int *)malloc(sizeof(int)*m);
		for (int i = 0; i < m; i++)
		{
			for (int j = csrmatrix0->rowptr[i]; j < csrmatrix0->rowptr[i + 1]; j++)
			{
				if (csrmatrix0->colind[j] == i)
				{
					axis0[i] = j;
					break;
				}
			}
			for (int j = csrmatrix1->rowptr[i]; j < csrmatrix1->rowptr[i + 1]; j++)
			{
				if (csrmatrix1->colind[j] == i)
				{
					axis1[i] = j;
					break;
				}
			}
		}
		CSRMatrix *csrmatrix = new CSRMatrix();
		csrmatrix->m = m;
		csrmatrix->n = m;
		csrmatrix->rowptr = (int *)malloc(sizeof(int)*(csrmatrix->m + 1));
		csrmatrix->rowptr[0] = 0;

		for (int i = 0; i < csrmatrix->m; i++)
		{
			csrmatrix->rowptr[i + 1] = csrmatrix->rowptr[i] + axis0[i] - csrmatrix0->rowptr[i] + csrmatrix1->rowptr[i + 1] - axis1[i];
			//cout << axis0[i] - csrmatrix0->rowptr[i] <<"\t"<< csrmatrix1->rowptr[i + 1] - axis1[i] <<"\t"<<csrmatrix->rowptr[i + 1] << "\n";
		}

		csrmatrix->nnz = csrmatrix->rowptr[csrmatrix->m];
		csrmatrix->colind = (int *)malloc(sizeof(int)*csrmatrix->nnz);
		csrmatrix->val = (double *)malloc(sizeof(double)*csrmatrix->nnz);
		for (int i = 0, k = 0; i < csrmatrix->m; i++)
		{
			for (int j = csrmatrix0->rowptr[i]; j < axis0[i]; j++)
			{
				csrmatrix->colind[k] = csrmatrix0->colind[j];
				csrmatrix->val[k] = csrmatrix0->val[j];
				k++;
			}
			for (int j = axis1[i]; j < csrmatrix1->rowptr[i + 1]; j++)
			{
				csrmatrix->colind[k] = csrmatrix1->colind[j];
				csrmatrix->val[k] = csrmatrix1->val[j];
				k++;
			}
		}

		free(axis0); axis0 = 0;
		free(axis1); axis1 = 0;
		return csrmatrix;
	}
	CSRMatrix* randomSymmetricCSRMatrix(int m, int nnz_per_line, long seed)
	{
		srand(seed);
		return randomSymmetricCSRMatrix(m, nnz_per_line);
	}
	CSRMatrix* randomLapacianMatrix(int grid_size)
	{
		std::vector<double> values;
		std::vector<int> rowInd;
		std::vector<int> colInd;
		int n = build_lapacian_double(grid_size, values, rowInd, colInd);
		int nnz = int(values.size());
		CSRMatrix *A = new CSRMatrix(n, nnz);
		int *nnz_row = (int *)malloc(sizeof(int)*n);
		for (int i = 0; i < n; i++)
		{
			nnz_row[i] = 0;
		}
		for (int i = 0; i < nnz; i++)
		{
			A->val[i] = values[i];
			A->colind[i] = colInd[i];
			nnz_row[rowInd[i]]++;
		}
		A->rowptr[0] = 0;
		for (int i = 0; i < n; i++)
		{
			A->rowptr[i + 1] = A->rowptr[i] + nnz_row[i];
		}
		free(nnz_row); nnz_row = 0;

		return A;
	}
	CSRMatrix *digAdd(CSRMatrix *A, CSRMatrix *B)
	{
	
		CSRMatrix *AA = new CSRMatrix(A->m, A->n, A->nnz);
		for (int i = 0; i < A->nnz; i++)
		{
			AA->colind[i] = A->colind[i];
			AA->val[i] =  A->val[i]+B->val[i];
		}
		for (int i = 0; i <A->m + 1; i++)
		{
			AA->rowptr[i] = A->rowptr[i];
		}
		return AA;
	}
	CSRMatrix *digDec(CSRMatrix *A, CSRMatrix *B)
	{

		CSRMatrix *AA = new CSRMatrix(A->m, A->n, A->nnz);
		for (int i = 0; i < A->nnz; i++)
		{
			AA->colind[i] = A->colind[i];
			AA->val[i] = A->val[i] - B->val[i];
		}
		for (int i = 0; i <A->m + 1; i++)
		{
			AA->rowptr[i] = A->rowptr[i];
		}
		return AA;
	}
	CSRMatrix *digInv(CSRMatrix *A)
	{
		CSRMatrix *AA = new CSRMatrix(A->m, A->n, A->nnz);
		for (int i = 0; i < A->nnz; i++)
		{
			AA->colind[i] =A->colind[i];
			AA->val[i] = 1 / A->val[i];
		}
		for (int i = 0; i <A->m + 1; i++)
		{
			AA->rowptr[i] = A->rowptr[i];
		}
		return AA;
	}
	CSRMatrix *digMulNor(CSRMatrix *A, CSRMatrix *B)
	{
		CSRMatrix *AA = new CSRMatrix(B->m,B->n, B->nnz);
		for (int i = 0; i < B->nnz; i++)
		{
			AA->colind[i] = B->colind[i];
			AA->val[i] =  B->val[i]*A->val[B->colind[i]];
		}
		for (int i = 0; i <B->m + 1; i++)
		{
			AA->rowptr[i] = B->rowptr[i];
		}
		return AA;
	}
	CSRMatrix *norMulNor(CSRMatrix *A, CSRMatrix *B)
	{
		CSRMatrix *AA = new CSRMatrix(A->m, B->n,A->m);
		for (int i = 0; i < AA->nnz; i++)
		{
			AA->colind[i] = i;
			AA->val[i] = 0;
		}
		for (int i = 0; i <AA->m + 1; i++)
		{
			AA->rowptr[i] =i;
		}
		//CSRMatrix *tranB = CSRTran(B);
		for (int i = 0; i < A->nnz; i++)
		{
			AA->val[B->colind[i]] = A->val[i] * B->val[i];
		}
		return AA;

	}
	CSRMatrix *norMulDig(CSRMatrix *A, CSRMatrix *B)
	{
		CSRMatrix *AA = new CSRMatrix(A->m, A->n, A->nnz);
		for (int i = 0; i < A->nnz; i++)
		{
			AA->colind[i] = A->colind[i];
			AA->val[i] = A->val[i] * B->val[A->colind[i]];
		}
		for (int i = 0; i <A->m + 1; i++)
		{
			AA->rowptr[i] = A->rowptr[i];
		}
		return AA;
	}
	CSRMatrix* simpleInv(CSRMatrix* A,int n)
	{
		for (int i = 0; i < A->m; i++)
		{
			if (A->rowptr[i + 1] - A->rowptr[i] > 2)
			{
				cout << "err";
				return 0;
			}
		}
		CSRMatrix *X1 = new CSRMatrix(n - 1, n - 1, n - 1);
		CSRMatrix *X2 = new CSRMatrix(n - 1, A->n - n + 1, A->n - n + 1);
		CSRMatrix *X3 = new CSRMatrix(A->m - n + 1, n - 1, A->m - n + 1);
		CSRMatrix *X4 = new CSRMatrix(A->m - n + 1, A->n - n + 1, A->m - n + 1);
		int x1val_i = 0, x2val_i = 0, x3val_i = 0, x4val_i = 0;
		X1->rowptr[0] = 0; X2->rowptr[0] = 0; X3->rowptr[0] = 0; X4->rowptr[0] = 0;
		for (int i = 0; i < n-1; i++)
		{
			X1->colind[i] = i;
			X1->rowptr[i + 1] = i+1;
			X2->rowptr[i + 1] = X2->rowptr[i];
			for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
			{
				if (A->colind[j] < n - 1)
				{
					X1->val[x1val_i++] = A->val[j];
				}
				else
				{
					X2->val[x2val_i] = A->val[j];
					X2->colind[x2val_i] = A->colind[j] - n + 1;
					X2->rowptr[i + 1] = X2->rowptr[i] + 1;
					x2val_i++;
				}
			}
		}

		for (int i = n - 1; i < A->m; i++)
		{
			
			for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
			{
				if (A->colind[j] < n - 1)
				{
					X3->colind[x3val_i] = A->colind[j];
					X3->rowptr[x3val_i + 1] = i + 2-n;
					X3->val[x3val_i] = A->val[j];
					x3val_i++;
				}
				else
				{
					X4->colind[x4val_i] = A->colind[j] - n + 1;
					X4->rowptr[x4val_i + 1] = i + 2-n;
					X4->val[x4val_i] = A->val[j];
					x4val_i++;
				}
				
			}
		}
		//A->showDense();
		//X1->showDense();
		//X2->showDense();
		//X3->showDense();
		//X4->showDense();
		CSRMatrix *Y1 = new CSRMatrix(n - 1, n - 1, n - 1);
		CSRMatrix *Y2 = new CSRMatrix(n - 1, A->n - n + 1, A->n - n + 1);
		CSRMatrix *Y3 = new CSRMatrix(A->m - n + 1, n - 1, A->m - n + 1);
		CSRMatrix *Y4 = new CSRMatrix(A->m - n + 1, A->n - n + 1, A->m - n + 1);


		for (int i = 0; i < Y1->nnz; i++)
		{
			Y1->colind[i] = X1->colind[i];
		}
		for (int i = 0; i < Y1->m+1; i++)
		{
			Y1->rowptr[i] = X1->rowptr[i];
		}

		for (int i = 0; i < Y2->nnz; i++)
		{
			Y2->colind[i] = X2->colind[i];
		}
		for (int i = 0; i < Y2->m + 1; i++)
		{
			Y2->rowptr[i] = X2->rowptr[i];
		}

		for (int i = 0; i < Y3->nnz; i++)
		{
			Y3->colind[i] = X3->colind[i];
		}
		for (int i = 0; i < Y3->m + 1; i++)
		{
			Y3->rowptr[i] = X3->rowptr[i];
		}

		for (int i = 0; i < Y4->nnz; i++)
		{
			Y4->colind[i] = X4->colind[i];
		}
		for (int i = 0; i < Y4->m + 1; i++)
		{
			Y4->rowptr[i] = X4->rowptr[i];
		}

		Y1->val = digInv(digDec(X1, digMulNor(norMulDig(X2, digInv(X4)), X3)))->val;
		Y2->val = norMulDig(digMulNor(digInv(X1), X2), digInv(digDec(norMulNor(norMulDig(X3, digInv(X1)), X2), X4)))->val;
		Y3->val = norMulDig(digMulNor(digInv(X4), X3), digInv(digDec(norMulNor(norMulDig(X2, digInv(X4)), X3), X1)))->val;
		Y4->val = digInv(digDec(X4, norMulNor(norMulDig(X3, digInv(X1)), X2)))->val;

		CSRMatrix *invA = new CSRMatrix(A->m, A->n, A->nnz);
		for (int i = 0; i < A->nnz; i++)
		{
			invA->colind[i] = A->colind[i];
		}
		for (int i = 0; i < A->m + 1; i++)
		{
			invA->rowptr[i] = A->rowptr[i];
		}
		int invAval_i = 0;
		for (int i = 0; i < Y1->m; i++)
		{
			for (int j = Y1->rowptr[i]; j< Y1->rowptr[i + 1]; j++)
			{
				invA->val[invAval_i++] = Y1->val[j];
			}
			for (int j = Y2->rowptr[i]; j< Y2->rowptr[i + 1]; j++)
			{
				invA->val[invAval_i++] = Y2->val[j];
			}
		}
		for (int i = 0; i < Y3->m; i++)
		{
			for (int j = Y3->rowptr[i]; j< Y3->rowptr[i + 1]; j++)
			{
				invA->val[invAval_i++] = Y3->val[j];
			}
			for (int j = Y4->rowptr[i]; j< Y4->rowptr[i + 1]; j++)
			{
				invA->val[invAval_i++] = Y4->val[j];
			}
		}
		//invA->showDense();
		return invA;
	}

	

}