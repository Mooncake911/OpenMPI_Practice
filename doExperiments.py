import csv
import subprocess


# # Пока не используется
# def file_exists_decorator(func):
#     def wrapper(filename, *args, **kwargs):
#         if os.path.exists(filename):
#             print(f"File '{filename}' already exists. Skipping function execution. \n")
#         else:
#             func(filename, *args, **kwargs)
#
#     return wrapper


class DoExperiments:
    def __init__(self, exe_path: str,
                 from_n: int = 1, to_n: int = 16,
                 from_iters: int = 90, to_iters: int = 9000000, iters_step: float = 10,
                 args_in_right_order=tuple()):

        self.path = exe_path
        self.from_n = from_n
        self.to_n = to_n

        self.from_iters = from_iters
        self.to_iters = to_iters
        self.iters_step = iters_step

        self.my_args = args_in_right_order
        self.data_list = []

    def start_open_mpi(self, n, *args):
        """ C++ code to start in commandline interface """
        command = f"mpiexec -n {n} {self.path} {' '.join(map(str, args))}"
        # print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if stdout:
            output = [float(x) for x in stdout.decode().split()]
            return output
        if stderr:
            raise Exception(stderr.decode())

    def run(self, filename):
        """ Run the experiments with OpenMPI """
        if self.iters_step <= 1:
            raise ValueError(f"iters_step: {self.iters_step} <= 1")

        iters = self.from_iters
        while iters <= self.to_iters:
            iters = int(iters)
            for n in range(self.from_n, self.to_n + 1):
                time = self.start_open_mpi(n, iters, *self.my_args)[0]
                self.data_list.append([n, iters, time])
            iters *= self.iters_step

        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Num_Threads', 'Iter', 'Time'])
            csv_writer.writerows(self.data_list)
        print(f"Results saved to {filename}")
