#include <mpi.h>

#include <iostream>
#include <string>


using namespace std;


void SynchroWithFeedback(char* m, const int& m_len, const unsigned int& r) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (unsigned int i = 0; i < r; ++i) {
        if (rank == 0) {
            MPI_Ssend(m, m_len, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        } else if (rank == 1) {
            MPI_Recv(m, m_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}


void SynchroNoFeedback(char* m, const int& m_len, const unsigned int& r) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (unsigned int i = 0; i < r; ++i) {
        if (rank == 0) {
            MPI_Send(m, m_len, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        } else if (rank == 1) {
            MPI_Recv(m, m_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}


void ASynchro(char* m, const int& m_len, const unsigned int& r) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Request send_request, recv_request;
    MPI_Status status;

    for (unsigned int i = 0; i < r; ++i) {
        if (rank == 0) {
            MPI_Isend(m, m_len, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &send_request);
            // Wait for the completion of the send and receive operations
            MPI_Wait(&send_request, &status);
        } else if (rank == 1) {
            MPI_Irecv(m, m_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &recv_request);
            // Wait for the completion of the send and receive operations
            MPI_Wait(&recv_request, &status);
        }
    }
}


void CommonMemory(char* m, const int& m_len, const unsigned int& r) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Win win;
    MPI_Win_create(m, m_len, sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    for (unsigned int i = 0; i < r; ++i) {
        if (rank == 0) {
            MPI_Win_fence(0, win);
            MPI_Put(m, m_len, MPI_CHAR, 1, 0, m_len, MPI_CHAR, win);
            MPI_Win_fence(0, win);
        } else if (rank == 1) {
            MPI_Win_fence(0, win);
            MPI_Get(m, m_len, MPI_CHAR, 0, 0, m_len, MPI_CHAR, win);
            MPI_Win_fence(0, win);
        }
    }

    MPI_Win_free(&win);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((size != 2) || (argc != 4)){
        if (rank == 0)
            cerr << "Usage: n != 2 or argc != 4." << endl;
        MPI_Finalize();
        exit(1);
    }

    unsigned int repeat = stoll(argv[1]);
    const int n = stoi(argv[2]);
    const int mode = stoi(argv[3]);

    char* message = new char[n];

    double start_time = MPI_Wtime();
    if (mode == 0)
        SynchroWithFeedback(message, n, repeat);
    else if (mode == 1)
        SynchroNoFeedback(message, n, repeat);
    else if (mode == 2)
        ASynchro(message, n, repeat);
    else if (mode == 3)
        CommonMemory(message, n, repeat);
    double end_time = MPI_Wtime();
    double duration = (end_time - start_time) * 1e6;

    delete[] message;

    if (rank == 0) {
        cout << duration << endl;
    }

    MPI_Finalize();
    return 0;
}
