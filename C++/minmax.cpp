#include <mpi.h>

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <limits>


using namespace std;

struct Result {
    double min = numeric_limits<double>::max();
    double max = numeric_limits<double>::min();

    void update(double value) {
        if (value < min)
            min = value;
        if (value > max)
            max = value;
    }
};

/* Generate vector with random values in range (a, b) */
void fill_vector(vector<double>& data, double a, double b) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(a, b);

    for (double& i : data)
        i = dist(gen);
}


Result FindMinMax(vector<double>& vec) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_size = static_cast<int>(vec.size() / size);
    vector<double> local_vec(local_size);

    MPI_Scatter(vec.data(), local_size, MPI_DOUBLE,
                local_vec.data(), local_size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    Result local;
    for (int i = 0; i < local_size; ++i)
        local.update(local_vec[i]);

    Result global;
    MPI_Reduce(&local.min, &global.min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local.max, &global.max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return global;
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

    vector<double> vec(N);

    if (rank == 0)
        fill_vector(vec, a, b);

    double start_time = MPI_Wtime();
    Result result = FindMinMax(vec);
    double end_time = MPI_Wtime();
    double duration = (end_time - start_time) * 1e6;

    if (rank == 0) {
        cout << duration << endl;
//        cout << result.min << endl;
//        cout << result.max << endl;
    }

    MPI_Finalize();
    return 0;
}
