#include <iostream>
#include <time.h>
#include <cstring>
#include "SimpleAlgorithm.h"
using namespace volcano;
namespace volcano
{
	void buckets_sort_v2(int *data, int n)
	{
		int biggest_num = data[0];
		for (int i = 1; i < n; i++)
		{
			if (data[i] > biggest_num)
			{
				biggest_num = data[i];
			}
		}
		int sort_times = 0;
		while (biggest_num > 0)
		{
			biggest_num = biggest_num / 10;
			sort_times++;
		}
		int *buckets;
		buckets = (int *)malloc(sizeof(int)*n);
		int exptimes = 1;
		int *buckets_index;
		buckets_index = (int *)malloc(sizeof(int) * 10);
		int *n_number;
		n_number = (int *)malloc(sizeof(int) * 10);
		int *yushu;
		yushu = (int *)malloc(sizeof(int)*n);
		for (int i = 0; i < sort_times; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				n_number[j] = 0;
			}
			for (int j = 0; j < n; j++)
			{
				yushu[j] = (data[j] / exptimes) % 10;
				n_number[yushu[j]]++;
			}

			buckets_index[0] = 0;
			for (int j = 1; j < 10; j++)
			{
				buckets_index[j] = n_number[j - 1] + buckets_index[j - 1];
			}
			for (int j = 0; j < n; j++)
			{
				buckets[buckets_index[yushu[j]]] = data[j];
				buckets_index[yushu[j]]++;
			}
			exptimes = exptimes * 10;
			for (int j = 0; j < n; j++)
			{
				data[j] = buckets[j];
			}
		}
		free(buckets);
		buckets = 0;
		free(buckets_index);
		buckets_index = 0;
		free(n_number);
		n_number = 0;
		free(yushu);
		yushu = 0;
	}

	void buckets_sort_v2(int *data, double *data1, int n)
	{
		int biggest_num = data[0];
		for (int i = 1; i < n; i++)
		{
			if (data[i] > biggest_num)
			{
				biggest_num = data[i];
			}
		}
		int sort_times = 0;
		while (biggest_num > 0)
		{
			biggest_num = biggest_num / 10;
			sort_times++;
		}
		int *buckets;
		buckets = (int *)malloc(sizeof(int)*n);
		double *buckets1;
		buckets1 = (double *)malloc(sizeof(double)*n);
		int exptimes = 1;
		int *buckets_index;
		buckets_index = (int *)malloc(sizeof(int) * 10);
		int *n_number;
		n_number = (int *)malloc(sizeof(int) * 10);
		int *yushu;
		yushu = (int *)malloc(sizeof(int)*n);

		for (int i = 0; i < sort_times; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				n_number[j] = 0;
			}
			for (int j = 0; j < n; j++)
			{
				yushu[j] = (data[j] / exptimes) % 10;
				n_number[yushu[j]]++;
			}

			buckets_index[0] = 0;
			for (int j = 1; j < 10; j++)
			{
				buckets_index[j] = n_number[j - 1] + buckets_index[j - 1];
			}
			for (int j = 0; j < n; j++)
			{
				buckets[buckets_index[yushu[j]]] = data[j];
				buckets1[buckets_index[yushu[j]]] = data1[j];
				buckets_index[yushu[j]]++;
			}
			exptimes = exptimes * 10;
			for (int j = 0; j < n; j++)
			{
				data[j] = buckets[j];
				data1[j] = buckets1[j];
			}
		}
		free(buckets);
		buckets = 0;
		free(buckets1);
		buckets1 = 0;
		free(buckets_index);
		buckets_index = 0;
		free(n_number);
		n_number = 0;
		free(yushu);
		yushu = 0;
	}

	int comp(const void*a, const void*b)
	{
		return *(int*)a - *(int*)b;
	}

	void Qsort(int *a, int low, int high)
	{
		if (low >= high)
		{
			return;
		}
		int first = low;
		int last = high;
		int key = a[first];/*���ֱ�ĵ�һ����¼��Ϊ����*/

		while (first < last)
		{
			while (first < last && a[last] >= key)
			{
				--last;
			}

			a[first] = a[last];/*���ȵ�һ��С���Ƶ��Ͷ�*/

			while (first < last && a[first] <= key)
			{
				++first;
			}

			a[last] = a[first];
			/*���ȵ�һ������Ƶ��߶�*/
		}
		a[first] = key;/*�����¼��λ*/
		Qsort(a, low, first - 1);
		Qsort(a, first + 1, high);
	}

	void Qsort(int *data1, double *data2, int low, int high)
	{
		if (low >= high)
		{
			return;
		}
		int first = low;
		int last = high;
		int key1 = data1[first];
		double key2 = data2[first];
		while (first < last)
		{
			while (first < last&&data1[last] >= key1)
			{
				--last;
			}
			data1[first] = data1[last];
			data2[first] = data2[last];
			while (first < last && data1[first] <= key1)
			{
				++first;
			}
			data1[last] = data1[first];
			data2[last] = data2[first];
			data1[first] = key1;/*�����¼��λ*/
			data2[first] = key2;
			Qsort(data1, data2, low, first - 1);
			Qsort(data1, data2, first + 1, high);
		}
	}
	//template<typename T>
	//void Qsort(int *data1, T *data2, int low, int high)
	//{
	//	if (low >= high)
	//	{
	//		return;
	//	}
	//	int first = low;
	//	int last = high;
	//	int key1 = data1[first];
	//	T key2 = data2[first];
	//	while (first < last)
	//	{
	//		while (first < last&&data1[last] >= key1)
	//		{
	//			--last;
	//		}
	//		data1[first] = data1[last];
	//		data2[first] = data2[last];
	//		while (first < last && data1[first] <= key1)
	//		{
	//			++first;
	//		}
	//		data1[last] = data1[first];
	//		data2[last] = data2[first];
	//		data1[first] = key1;/*�����¼��λ*/
	//		data2[first] = key2;
	//		Qsort(data1, data2, low, first - 1);
	//		Qsort(data1, data2, first + 1, high);
	//	}
	//}
	void Qsort(int *data1, void *data2, size_t size, int low, int high)
	{
		if (low >= high)
		{
			return;
		}
		int first = low;
		int last = high;
		int key1 = data1[first];
		void *key2 = malloc(size);
		//size_t a = data1;
		//memcpy(key2, (void *)((unsigned long)data2 + size*first), size);
		memcpy(key2, (char*)data2 + size*first, size);
		while (first < last)
		{
			while (first < last&&data1[last] >= key1)
			{
				--last;
			}
			data1[first] = data1[last];
			memcpy((char*)data2 + size*first, (char*)data2 + size*last, size);
			while (first < last && data1[first] <= key1)
			{
				++first;
			}
			data1[last] = data1[first];
			memcpy((char*)data2 + size*last, (char*)data2 + size*first, size);
			data1[first] = key1;/*�����¼��λ*/
			memcpy((char*)data2 + size*first, key2, size);
			Qsort(data1, data2, size,low, first - 1);
			Qsort(data1, data2, size,first + 1, high);
		}
	}
	
	int* randomN(int n, int min, int max)
	{
		//srand(clock());
		if (max < min || n<0 || n>(max - min + 1))
		{
			std::cout << "err" << std::endl;
			return 0;
		}
		int *a = (int *)malloc(sizeof(int)*(max - min + 1));
		for (int i = 0; i < max - min + 1; i++)
		{
			a[i] = i;
		}
		for (int i = 0; i < n; i++)
		{
			int Rand = rand() % (max - min + 1);
			int abc = a[Rand];
			a[Rand] = a[i];
			a[i] = abc;
		}
		for (int i = 0; i < n; i++)
		{
			a[i] += min;
		}
		a = (int *)realloc(a, sizeof(int)*n);
		return a;
	}
}
