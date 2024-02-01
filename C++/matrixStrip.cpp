#include <mpi.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>

using namespace std;

int ProcNum = 0;  // Number of available processes (size)
int ProcRank = 0; // Rank of current process (rank)


/* Stop program conditions */
void ExitCondition(int Size) {
    if (Size % ProcNum != 0) {
        if (ProcRank == 0)
            cerr << "Size of matrices must be divisible by the ProcNum size!\n";
        MPI_Finalize();
        exit(1);
    }
}


/* Function for initialization of unit matrix */
void DummyDataInitialization(double* Matrix_A, double* Matrix_B, int Size) {
    for (int i = 0; i < Size; i++)
        for (int j = 0; j < Size; j++) {
            Matrix_A[i * Size + j] = 1;
            Matrix_B[i * Size + j] = 1;
        }
}
/* Function for initialization of random matrix */
void RandomDataInitialization(double* Matrix_A, double* Matrix_B, int Size) {
    srand(unsigned(clock()));
    for (int i = 0; i < Size; i++)
        for (int j = 0; j < Size; j++) {
            Matrix_A[i * Size + j] = rand() / double(1000);
            Matrix_B[i * Size + j] = rand() / double(1000);
        }
}
void DataInitialization(double* Matrix_A, double* Matrix_B, int Size, const string& mode) {
    if (mode == "1" || mode == "ones")
        DummyDataInitialization(Matrix_A, Matrix_B, Size);
    else
        RandomDataInitialization(Matrix_A, Matrix_B, Size);
}


/* First data distribute */
void DataDistribution(double* Matrix_A, double* Matrix_B, int MatrixSize, int LineSize) {
    for (int i = 1; i < ProcNum; i++)
        MPI_Send(Matrix_B,MatrixSize*MatrixSize,MPI_DOUBLE,i,0,MPI_COMM_WORLD);

    for (int j = 1; j < ProcNum; j++)
        MPI_Send(Matrix_A + j*LineSize*MatrixSize,MatrixSize*LineSize,MPI_DOUBLE,j,1,MPI_COMM_WORLD);
}


/* Function for vector×matrix and matrix×matrix multiplication*/
void MatrixMultiplication(const double* Matrix_A, const double* Matrix_B, double* Matrix_C, int MatrixSize, int StripSize) {
    for (int i = 0; i < StripSize; i++)
        for (int j = 0; j < MatrixSize; j++)
            for (int k = 0; k < MatrixSize; k++)
                Matrix_C[i * MatrixSize + j] += Matrix_A[i * MatrixSize + k] * Matrix_B[k * MatrixSize + j];
}


/* Print Matrix */
void PrintMatrix(double* Matrix, int RowCount, int ColCount) {
    int i, j; // Loop variables
    for (i = 0; i < RowCount; i++) {
        for (j = 0; j < ColCount; j++)
            printf("%7.4f ", Matrix[i * ColCount + j]);
        printf("\n");
    }
}
/* Test with Sequential algorithm */
void TestResult(double* Matrix_A, double* Matrix_B, double* Matrix_C, int MatrixSize) {
    if (ProcRank == 0) {
        auto* SerialResult = new double[MatrixSize * MatrixSize]();
        double e = 1.e-6;
        bool equal = true;

        MatrixMultiplication(Matrix_A, Matrix_B, SerialResult, MatrixSize, MatrixSize);

        for (int i = 0; i < MatrixSize * MatrixSize; i++)
            if (fabs(SerialResult[i] - Matrix_C[i]) >= e) {
                equal = false;
                break;
            }

        if (equal)
            cout << "The results of serial and parallel algorithms are identical." << endl;
        else
            cout << "The results of serial and parallel algorithms are NOT identical. Check your code." << endl;

        PrintMatrix(Matrix_C, MatrixSize, MatrixSize);
        delete[] SerialResult;
    }
}


int main(int argc,char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    setvbuf(stdout, nullptr, _IONBF, 0);

    if (argc != 3) {
        if (ProcRank == 0)
            cerr << "Usage: argc != 3";
        MPI_Finalize();
        exit(1);
    }

    int MatrixSize = stoi(argv[1]);    // Size of matrices
    int StripSize = MatrixSize / ProcNum;  // Sizes of matrix block

    ExitCondition(MatrixSize);

    double* Matrix_A; // The first matrix
    double* Matrix_B; // The second matrix
    double* Matrix_C; // The result matrix

    double* Strip_A; // The strip of Matrix_A
    double* Strip_C; // The strip of Matrix_C

    if (ProcRank==0) {
        double start_time = MPI_Wtime();

        Matrix_A = new double[MatrixSize * MatrixSize];
        Matrix_B = new double[MatrixSize * MatrixSize];
        Matrix_C = new double[MatrixSize * MatrixSize]();
        Strip_C = new double[MatrixSize * StripSize]();

        // Создаём и рассылаем данные по процессам
        DataInitialization(Matrix_A, Matrix_B, MatrixSize, argv[2]);
        DataDistribution(Matrix_A, Matrix_B, MatrixSize, StripSize);

        // Рассчитать и записать 1 полосу
        MatrixMultiplication(Matrix_A, Matrix_B, Matrix_C, MatrixSize, StripSize);

        // Получить и записать со 2-й по поселению полосы
        for (int k = 1; k < ProcNum; k++) {
            MPI_Recv(Strip_C, StripSize*MatrixSize, MPI_DOUBLE, k, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(&Matrix_C[k * StripSize * MatrixSize], Strip_C, StripSize * MatrixSize * sizeof(double));
        }

        double end_time = MPI_Wtime();
        double duration = (end_time - start_time) * 1e6;
        cout << duration << endl;

        //TestResult(Matrix_A, Matrix_B, Matrix_C, MatrixSize);

        delete[] Matrix_A;
        delete[] Matrix_B;
        delete[] Matrix_C;
        delete[] Strip_C;
    }

    else {
        Matrix_B = new double[MatrixSize * MatrixSize];
        Strip_A = new double[MatrixSize * StripSize];
        Strip_C = new double[MatrixSize * StripSize]();

        // Получить данные с главного процесса
        MPI_Recv(Matrix_B,MatrixSize*MatrixSize,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(Strip_A, MatrixSize*StripSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Рассчитать со 2-й по последнею полосу и отправить главному процессу
        MatrixMultiplication(Strip_A, Matrix_B, Strip_C, MatrixSize, StripSize);
        MPI_Send(Strip_C, StripSize*MatrixSize, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);

        delete[] Matrix_B;
        delete[] Strip_A;
        delete[] Strip_C;
    }

    MPI_Finalize();
    return 0;
}