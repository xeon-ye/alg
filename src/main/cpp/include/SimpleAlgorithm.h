#ifndef __volcano_SimpleAlgorithm_H__
#define	__volcano_SimpleAlgorithm_H__
#include<stdlib.h>

namespace volcano
{
	void buckets_sort_v2(int *data, int n);
	void buckets_sort_v2(int *data, double *data1, int n);
	int comp(const void*a, const void*b);
	void Qsort(int *a, int low, int high);
	void Qsort(int *data1, double *data2, int low, int high);
	//template<typename T> void Qsort(int *data1, T *data2, int low, int high);
	void Qsort(int *data1, void *data2, size_t size, int low, int high);
	int* randomN(int n, int min, int max);
	template<class T>size_t BKDRHash(const T *str);
}

namespace volcano//Hashing
{
	template<class T>
	size_t BKDRHash(const T *str)
	{
		register size_t hash = 0;
		while (size_t ch = (size_t)*str++)
		{
			hash = hash * 131 + ch;   // Ҳ���Գ���31��131��1313��13131��131313..  
			// ����˵���˷��ֽ�Ϊλ���㼰�Ӽ����������Ч�ʣ��罫��ʽ���Ϊ��hash = hash << 7 + hash << 1 + hash + ch;  
			// ����ʵ��Intelƽ̨�ϣ�CPU�ڲ��Զ��ߵĴ���Ч�ʶ��ǲ��ģ�  
			// �ҷֱ������100�ڴε������������㣬���ֶ���ʱ�������Ϊ0�������Debug�棬�ֽ��λ�����ĺ�ʱ��Ҫ��1/3����  
			// ��ARM����RISCϵͳ��û�в��Թ�������ARM�ڲ�ʹ��Booth's Algorithm��ģ��32λ�����˷����㣬����Ч��������йأ�  
			// ������8-31λ��Ϊ1��0ʱ����Ҫ1��ʱ������  
			// ������16-31λ��Ϊ1��0ʱ����Ҫ2��ʱ������  
			// ������24-31λ��Ϊ1��0ʱ����Ҫ3��ʱ������  
			// ������Ҫ4��ʱ������  
			// ��ˣ���Ȼ��û��ʵ�ʲ��ԣ���������Ȼ��Ϊ����Ч���ϲ�𲻴�          
		}
		return hash;
	}
}

#endif