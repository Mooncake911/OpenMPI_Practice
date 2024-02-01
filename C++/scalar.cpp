#include <mpi.h>

#include <iostream>
#include <vector>
#include <string>
#include <random>


using namespace std;

// 0.
double scalar(vector<double>& v1, vector<double>& v2) {
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i)
        result += v1[i] * v2[i];
    return result;
}

/* For DEBUG */
void TestResult(double r1, double r2) {
    double e = 1.e-3;
    if (fabs(r1 - r2) < e)
        cout << "Results are identical!" << endl;
    else
        cerr << "Results are NOT identical!" << endl;
}

/* Generate vector with random values in (a, b) */
void fill_vector(vector<double>& data, double a, double b) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(a, b);

    for (double& i : data)
        i = dist(gen);
}

/* Calculate scalar multiplication */
double FindScalar(vector<double>& v1, vector<double>& v2) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int global_size = static_cast<int>(v1.size());
    int local_size = global_size / size;
    int remainder = global_size % size;
    int offset = rank * local_size;
    if (rank == size - 1)
        local_size += remainder; // add remains to the last process

    MPI_Bcast(v1.data(), global_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2.data(), global_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> localVec1(v1.begin() + offset, v1.begin() + offset + local_size);
    vector<double> localVec2(v2.begin() + offset, v2.begin() + offset + local_size);

    double localResult = 0.0;
    for (size_t i = 0; i < local_size; ++i)
        localResult += localVec1[i] * localVec2[i];

    double globalResult;
    MPI_Reduce(&localResult, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return globalResult;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0)
            cerr << "Usage: argc != 4";
        MPI_Finalize();
        exit(1);
    }

    unsigned int N = stoll(argv[1]);
    double a = stod(argv[2]);
    double b = stod(argv[3]);

    vector<double> vector1(N);
    vector<double> vector2(N);

    if (rank == 0) {
        fill_vector(vector1, a, b);
        fill_vector(vector2, a, b);
    }

    double start_time = MPI_Wtime();
    double result = FindScalar(vector1, vector2);
    double end_time = MPI_Wtime();
    double duration = (end_time - start_time) * 1e6;

    if (rank == 0) {
        cout << duration << endl;
        // cout << result << endl;
        //TestResult(result, scalar(vector1, vector2));
    }

    MPI_Finalize();
    return 0;
}
