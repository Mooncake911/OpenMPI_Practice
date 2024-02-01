#include <mpi.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>


using namespace std;

int ProcNum = 0; // Number of available processes
int ProcRank = 0; // Rank of current process

int GridSize; // Size of virtual processor grid
int GridCoords[2]; // Coordinates of current processor in grid

MPI_Comm GridComm; // Grid communicator
MPI_Comm ColComm; // Column communicator
MPI_Comm RowComm; // Row communicator


/* Stop program conditions */
void ExitCondition(int Size) {
    if (ProcNum != GridSize * GridSize) {
        if (ProcRank == 0)
            cerr << "Number of processes must be a perfect square \n";
        MPI_Finalize();
        exit(1);
    }

    if (Size % GridSize != 0) {
        if (ProcRank == 0)
            cerr << "Size of matrices must be divisible by the grid size!\n";
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


/* Function for matrix×matrix multiplication*/
void MatrixMultiplication(const double* Matrix_A, const double* Matrix_B, double* Matrix_C, int MatrixSize) {
    for (int i = 0; i < MatrixSize; i++)
        for (int j = 0; j < MatrixSize; j++)
            for (int k = 0; k < MatrixSize; k++)
                Matrix_C[i * MatrixSize + j] += Matrix_A[i * MatrixSize + k] * Matrix_B[k * MatrixSize + j];
}


/* Creation of two-dimensional grid communicator and communicators for each row and each column of the grid */
void CreateGridCommunicators() {
    int DimSize[2] = {GridSize, GridSize}; // Number of processes in each dimension of the grid
    int Periodic[2] = {0, 0}; // =1, if the grid dimension should be periodic
    int Subdims[2]; // =1, if the grid dimension should be fixed

    // Creation of the Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);
    // Determination of the cartesian coordinates for every process
    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);

    // Creating communicators for rows
    Subdims[0] = 0; // Dimensionality fixing
    Subdims[1] = 1; // The presence of the given dimension in the subgrid
    MPI_Cart_sub(GridComm, Subdims, &RowComm);

    // Creating communicators for columns
    Subdims[0] = 1;
    Subdims[1] = 0;
    MPI_Cart_sub(GridComm, Subdims, &ColComm);
}


/* Function for memory allocation and data initialization */
void ProcessInitialization(double*& Matrix_A, double*& Matrix_B, double*& Matrix_C,
                           double*& Block_A, double*& Block_B, double*& Block_C, double*& TemporaryBlock_A,
                           int& MatrixSize, int& BlockSize, const string& mode) {
    MPI_Bcast(&MatrixSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    BlockSize = MatrixSize / GridSize;
    Block_A = new double[BlockSize * BlockSize];
    Block_B = new double[BlockSize * BlockSize];
    Block_C = new double[BlockSize * BlockSize]();
    TemporaryBlock_A = new double[BlockSize * BlockSize];
    for (int i = 0; i < BlockSize * BlockSize; i++) {
        Block_C[i] = 0;
    }
    if (ProcRank == 0) {
        Matrix_A = new double[MatrixSize * MatrixSize];
        Matrix_B = new double[MatrixSize * MatrixSize];
        Matrix_C = new double[MatrixSize * MatrixSize];
        DataInitialization(Matrix_A, Matrix_B, MatrixSize, mode);
    }
}


/* Function for checkerboard matrix decomposition */
void CheckerboardMatrixScatter(double* pMatrix, double* pMatrixBlock, int Size, int BlockSize)
{
    auto* MatrixRow = new double[BlockSize * Size];
    if (GridCoords[1] == 0)
        MPI_Scatter(pMatrix, BlockSize * Size, MPI_DOUBLE,
                    MatrixRow, BlockSize * Size, MPI_DOUBLE, 0, ColComm);
    for (int i = 0; i < BlockSize; i++)
        MPI_Scatter(&MatrixRow[i * Size], BlockSize, MPI_DOUBLE,
                    &(pMatrixBlock[i * BlockSize]), BlockSize, MPI_DOUBLE, 0, RowComm);
    delete[] MatrixRow;
}
void DataDistribution(double* Matrix_A, double* Matrix_B, double* MatrixBlock_A, double* Block_B, int MatrixSize, int BlockSize) {
    // Scatter the matrix among the processes of the first grid column
    CheckerboardMatrixScatter(Matrix_A, MatrixBlock_A, MatrixSize, BlockSize);
    CheckerboardMatrixScatter(Matrix_B, Block_B, MatrixSize, BlockSize);
}


/* Broadcasting matrix A blocks to process grid rows */
void ABlockCommunication(int iter, double* Block_A, const double* MatrixBlock_A, int BlockSize) {
    // Defining the leading process of the process grid row
    int Pivot = (GridCoords[0] + iter) % GridSize;
    // Copying the transmitted block in a separate memory buffer
    if (GridCoords[1] == Pivot) {
        for (int i = 0; i < BlockSize * BlockSize; i++)
            Block_A[i] = MatrixBlock_A[i];
    }
    // Block broadcasting
    MPI_Bcast(Block_A, BlockSize * BlockSize, MPI_DOUBLE, Pivot, RowComm);
}
/* Cyclic shift of matrix B blocks in the process grid columns */
void BBlockCommunication(double* Block_B, int BlockSize) {
    MPI_Status Status;
    int NextProc = GridCoords[0] + 1;
    if (GridCoords[0] == GridSize - 1) NextProc = 0;
    int PrevProc = GridCoords[0] - 1;
    if (GridCoords[0] == 0) PrevProc = GridSize - 1;
    MPI_Sendrecv_replace(Block_B, BlockSize * BlockSize, MPI_DOUBLE,
                         NextProc, 0, PrevProc, 0, ColComm, &Status);
}
void ParallelResultCalculation(double* Block_A, double* Block_B, double* Block_C, double* MatrixBlock_A, int BlockSize) {
    for (int iter = 0; iter < GridSize; iter++) {
        // Sending blocks of matrix A to the process grid rows
        ABlockCommunication(iter, Block_A, MatrixBlock_A, BlockSize);
        // Block multiplication
        MatrixMultiplication(Block_A, Block_B, Block_C, BlockSize);
        // Cyclic shift of blocks of matrix B in process grid columns
        BBlockCommunication(Block_B, BlockSize);
    }
}


/* Function for gathering the result matrix */
void ResultCollection(double* Matrix_C, double* Block_C, int MatrixSize, int BlockSize) {
    auto* pResultRow = new double[MatrixSize * BlockSize];
    for (int i = 0; i < BlockSize; i++)
        MPI_Gather(&Block_C[i * BlockSize], BlockSize, MPI_DOUBLE,
                   &pResultRow[i * MatrixSize], BlockSize, MPI_DOUBLE, 0, RowComm);
    if (GridCoords[1] == 0)
        MPI_Gather(pResultRow, BlockSize * MatrixSize, MPI_DOUBLE,
                   Matrix_C, BlockSize * MatrixSize, MPI_DOUBLE, 0, ColComm);
    delete[] pResultRow;
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
/* Test printing of the matrix block */
void TestBlocks(double* pBlock, int BlockSize, char str[])
{
    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0) {
        printf("%s \n", str);
    }
    for (int i = 0; i < ProcNum; i++) {
        if (ProcRank == i) {
            printf("ProcRank = %d \n", ProcRank);
            PrintMatrix(pBlock, BlockSize, BlockSize);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
/* Test with Sequential algorithm */
void TestResult(double* Matrix_A, double* Matrix_B, double* Matrix_C, int MatrixSize) {
    if (ProcRank == 0) {
        auto* SerialResult = new double[MatrixSize * MatrixSize]();
        double e = 1.e-6;
        bool equal = true;

        MatrixMultiplication(Matrix_A, Matrix_B, SerialResult, MatrixSize);

        for (int i = 0; i < MatrixSize * MatrixSize; i++)
            if (fabs(SerialResult[i] - Matrix_C[i]) >= e) {
                equal = false;
                break;
            }

        if (equal)
            cout << "The results of serial and parallel algorithms are identical." << endl;
        else
            cout << "The results of serial and parallel algorithms are NOT identical. Check your code." << endl;

        //PrintMatrix(Matrix_C, MatrixSize, MatrixSize);
        //PrintMatrix(SerialResult, MatrixSize, MatrixSize);
        delete[] SerialResult;
    }
}


/* Free memory */
void ProcessTermination(const double* Matrix_A, const double* Matrix_B, const double* Matrix_C,
                        const double* Block_A, const double* Block_B, const double* Block_C, const double* MatrixBlock_A) {
    if (ProcRank == 0) {
        delete[] Matrix_A;
        delete[] Matrix_B;
        delete[] Matrix_C;
    }
    delete[] Block_A;
    delete[] Block_B;
    delete[] Block_C;
    delete[] MatrixBlock_A;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    setvbuf(stdout, nullptr, _IONBF, 0);

    if (argc != 3) {
        if (ProcRank == 0)
            cerr << "Usage: argc != 3";
        MPI_Finalize();
        exit(1);
    }

    int MatrixSize = stoi(argv[1]); // Size of matrices
    int BlockSize; // Sizes of matrix blocks on current process
    GridSize = static_cast<int>(sqrt((double)ProcNum));

    double* Matrix_A; // The first matrix
    double* Matrix_B; // The second matrix
    double* Matrix_C; // The result matrix

    double* Block_A; // Block of matrix A
    double* Block_B; // Block of matrix B
    double* Block_C; // Block of matrix C
    double* MatrixBlock_A;

    ExitCondition(MatrixSize);

    double start_time = MPI_Wtime();

    CreateGridCommunicators();

    ProcessInitialization(Matrix_A, Matrix_B, Matrix_C, Block_A, Block_B, Block_C, MatrixBlock_A,
                          MatrixSize, BlockSize, argv[2]);

    DataDistribution(Matrix_A, Matrix_B, MatrixBlock_A, Block_B, MatrixSize, BlockSize);

    ParallelResultCalculation(Block_A, Block_B, Block_C, MatrixBlock_A, BlockSize);

    ResultCollection(Matrix_C, Block_C, MatrixSize, BlockSize);

    double end_time = MPI_Wtime();
    double duration = (end_time - start_time) * 1e6;
    if (ProcRank == 0)
        cout << duration << endl;

    //TestResult(Matrix_A, Matrix_B, Matrix_C, MatrixSize);

    ProcessTermination(Matrix_A, Matrix_B, Matrix_C, Block_A, Block_B, Block_C, MatrixBlock_A);

    MPI_Finalize();
    return 0;
}