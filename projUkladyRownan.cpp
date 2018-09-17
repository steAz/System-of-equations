#include <iostream>
#include <vector>
#include <math.h>
#include <ctime>
#include <fstream>

using namespace std;

void packVector(vector<double> &vec, int N, int f) {
	for (int i = 0; i < N; i++) {
		vec.push_back(sin(i * (f + 1) / 50));
	}
}


void allocMatrices(double **(&matrix), double **(&jacIndMat), int N) {
	matrix = new double*[N];
	jacIndMat = new double*[N];
	for (int i = 0; i < N; i++) {
		matrix[i] = new double[N];
		jacIndMat[i] = new double[N];
	}
}

void allocMatrixGauss(double **(&matrix), int N) {
	matrix = new double*[N];
	for (int i = 0; i < N; i++) {
		matrix[i] = new double[N + 1];
	}
}


void initDvecInvAndvecX(double *DvecInv, double **(&matrix), double* vecX, int N) {
	for (int i = 0; i < N; i++) { // initialize D^-1
		DvecInv[i] = 1 / matrix[i][i];
		vecX[i] = 1 / N; 
	}
}


void initJacIndMat(double **(&matrix), double **(&jacIndMat), double *jacDvecInv, int N) {
	for (int i = 0; i < N; i++) { // initialize jacIndirectMat  = -D^-1 (L + U)
		for (int j = 0; j < N; j++) {
			if (i == j)
				jacIndMat[i][j] = 0;
			else
				jacIndMat[i][j] = -(matrix[i][j] * jacDvecInv[i]);
		}
	}
}


void jacobi(double **(&matrix), vector<double> &vecB, double **(&jacIndMat), double* vecX, int N) {
	double *jacDvecInv = new double[N];
	initDvecInvAndvecX(jacDvecInv, matrix, vecX, N);
	initJacIndMat(matrix, jacIndMat, jacDvecInv, N);

	double *vecIndX = new double[N];
	double *residuum = new double[N];
	double normRes;
	bool ifResiduum = false;
	int iteracja = 0;
	while (!ifResiduum) {
		for (int i = 0; i < N; i++) {
			vecIndX[i] = jacDvecInv[i] * vecB[i];
			for (int j = 0; j < N; j++) {
				vecIndX[i] += jacIndMat[i][j] * vecX[j];
			}
		}
		for (int i = 0; i < N; i++) {
			vecX[i] = vecIndX[i];
		}

		for (int i = 0; i < N; i++) {
			residuum[i] = -vecB[i];
			for (int j = 0; j < N; j++) {
				residuum[i] += matrix[i][j] * vecX[j];
			}
		}

		double indirectNorm = 0;
		for (int i = 0; i < N; i++) {
			indirectNorm += pow(residuum[i], 2);
		}
		normRes = sqrt(indirectNorm);
		cout << normRes << endl;

		/*cout << normRes << endl;
		cout << "Iteracja " << iteracja << endl;*/
		iteracja++;

		if (normRes < 1e-9) ifResiduum = true;
	}

	cout << "Liczba iteracji: " << iteracja << endl;
}


double sumGaussSeidel(double **(&matrix), double* vecIndX, int numI, int numOfSum, int beg) {
	double sum = 0;
	for (int i = beg; i < numOfSum; i++) {
		sum += matrix[numI][i] * vecIndX[i];
	}
	return sum;
}


void gaussSeidel(double **(&matrix), vector<double> &vecB, double* vecX, int N) {
	for (int i = 0; i < N; i++) {
		vecX[i] = 1 / N; // w rozwiazania wpisujemy 1/N
	}

	double *vecIndX = new double[N];
	double *residuum = new double[N];
	double normRes;
	bool ifResiduum = false;
	int iteracja = 0;
	while (!ifResiduum) {
		for (int i = 0; i < N; i++) {
			vecIndX[i] = ((-1)* (sumGaussSeidel(matrix, vecIndX, i, i, 0) + sumGaussSeidel(matrix, vecX, i, N, (i + 1))) + vecB[i]) / matrix[i][i];
		}
		for (int i = 0; i < N; i++) {
			vecX[i] = vecIndX[i];
		}

		for (int i = 0; i < N; i++) {
			residuum[i] = -vecB[i];
			for (int j = 0; j < N; j++) {
				residuum[i] += matrix[i][j] * vecX[j];
			}
		}

		double indNorm = 0;
		for (int i = 0; i < N; i++) {
			indNorm += pow(residuum[i], 2);
		}
		normRes = sqrt(indNorm);
		cout << normRes << endl;
		iteracja++;

		if (normRes < 1e-9) ifResiduum = true;
	}


	cout << "Liczba iteracji: " << iteracja << endl;
}


void createMatrix(double **(&matrix), int N, int a1, int a2, int a3) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j)
				matrix[i][j] = a1;
			else if (i - j == 1 || j - i == 1)
				matrix[i][j] = a2;
			else if (i - j == 2 || j - i == 2)
				matrix[i][j] = a3;
			else
				matrix[i][j] = 0;
		}
	}
}


void createMatrixGauss(double **(&matrix), vector<double> &vecB, int N) {
	for (int i = 0; i < N; i++) {
		matrix[i][N] = vecB[i];
	}
}


void deAlloc(double **(&matrix), double **(&jacIndMat), vector<double> &vecB, double *vecXgauss, double *vecXgaussSeidel, double *vecXjac, int N) {
	if (matrix != NULL)
	{
		for (int i = 0; i < N; i++)
		{
			delete[] matrix[i];
		}

		delete[] matrix;
		matrix = NULL;
	}

	if (jacIndMat != NULL)
	{
		for (int i = 0; i < N; i++)
		{
			delete[] jacIndMat[i];
		}

		delete[] jacIndMat;
		jacIndMat = NULL;
	}

	if (vecXgauss != NULL) {
		delete[] vecXgauss;
		vecXgauss = NULL;
	}
	if (vecXjac != NULL) {
		delete[] vecXjac;
		vecXjac = NULL;
	}
	if (vecXgaussSeidel != NULL) {
		delete[] vecXgaussSeidel;
		vecXgaussSeidel = NULL;
	}

	vecB.clear();
	vecB.shrink_to_fit();
}

void deAllocMatrixGauss(double **(&matrix), int N) {
	if (matrix != NULL)
	{
		for (int i = 0; i < N; i++)
		{
			delete[] matrix[i];
		}

		delete[] matrix;
		matrix = NULL;
	}
}


void gauss(double** (&matrix), double* vecX, int N, vector<double> &vecB, int a1, int a2, int a3) {
	double** A;
	allocMatrixGauss(A, N);
	createMatrix(A, N, a1, a2, a3);
	createMatrixGauss(A, vecB, N);

	for (int i = 0; i < N; i++) {
		double max = abs(A[i][i]);  // szukamy maximum w kolumnie
		int maxRow = i;
		for (int j = i + 1; j < N; j++) {
			if (abs(A[j][i]) > max) {
				max = abs(A[j][i]);
				maxRow = j;
			}
		}

		for (int k = i; k < N + 1; k++) swap(A[i][k], A[maxRow][k]); //zamieniamy maxy z odpowiednimi

		for (int j = i + 1; j < N; j++) {
			double indResult = -A[j][i] / A[i][i];
			for (int k = i; k < N + 1; k++) {
				if (i == k)
					A[j][k] = 0; // zmien wszystkie wiersze pod elementami w zera - macierz trojkatna
				else
					A[j][k] += indResult * A[i][k];
			}
		}
	}

	for (int i = N - 1; i >= 0; i--) {  // znajdujemy rozwiazanie z diagonali
		vecX[i] = A[i][N] / A[i][i];
		for (int j = i - 1; j >= 0; j--) {
			A[j][N] -= A[j][i] * vecX[i];
		}
	}

	double *residuum = new double[N];
	double normRes; // oblicz norme na glownej macierzy
	for (int i = 0; i < N; i++) {
		residuum[i] = -vecB[i];
		for (int j = 0; j < N; j++) {
			residuum[i] += matrix[i][j] * vecX[j];
		}
	}

	double indNorm = 0;
	for (int i = 0; i < N; i++) {
		indNorm += pow(residuum[i], 2);
	}
	normRes = sqrt(indNorm);
	cout << "Gauss norma: " << normRes << endl;

	deAllocMatrixGauss(A, N);
}


int main() {
	int N = 912;  //160512 
	int f = 0;
	int e = 5;
	int a1 = e+5;
	int a3;
	int a2 = a3 = -1;
	double **matrix, **jacIndMat;
	vector<double> vecB;
	double *vecXgauss, *vecXjac, *vecXgaussSeidel;
	
	ofstream myF;
	myF.open("dane.txt");  // tGAUSS  tJACOBI, tSEIDEL, N
	//for (int i = 0; i < 10; i++) {
		vecXgauss = new double[N];
		vecXjac = new double[N];
		vecXgaussSeidel = new double[N];
		packVector(vecB, N, f);
		allocMatrices(matrix, jacIndMat, N);
		createMatrix(matrix, N, a1, a2, a3);
		gaussSeidel(matrix, vecB, vecXgaussSeidel, N);

		////myF << "N teraz wynosi: " << N << endl;
		//clock_t cStart = clock();
		//gauss(matrix, vecXgauss, N, vecB, a1, a2, a3);
		//clock_t cEnd = clock();
		//float time1 = (float)(cEnd - cStart) / CLOCKS_PER_SEC;
		////myF << time1 << " " << N << endl;

		//cStart = clock();
		//jacobi(matrix, vecB, jacIndMat, vecXjac, N);
		//cEnd = clock();
		//float time2 = (float)(cEnd - cStart) / CLOCKS_PER_SEC;
		////myF << time1 << " " << N << endl;

		//cStart = clock();
		//gaussSeidel(matrix, vecB, vecXgaussSeidel, N);
		//cEnd = clock();
		//float time3 = (float)(cEnd - cStart) / CLOCKS_PER_SEC;

		//myF << N << " " << time1 << " " << time2 << " " << time3 << endl;
		//deAlloc(matrix, jacIndMat, vecB, vecXgauss, vecXgaussSeidel, vecXjac, N);

		//N += 50;
	//}
	myF.close();

	system("pause");
	return 0;
}
