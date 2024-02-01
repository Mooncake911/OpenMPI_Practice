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
                 from_iters: int = 10, to_iters: int = 1000000,
                 iter_step: float = 10, step_mode: str = "*",
                 args_in_right_order=tuple(),
                 columns_names: list = None):

        self.path = exe_path
        self.from_n = min(int(from_n), int(to_n))
        self.to_n = max(int(from_n), int(to_n))

        self.from_iters = min(int(from_iters), int(to_iters))
        self.to_iters = max(int(from_iters), int(to_iters))
        self.iter_step = iter_step
        self.step_mode = step_mode

        self.my_args = args_in_right_order
        self.data_list = []
        self.columns = columns_names

        self.rise_error()

    def rise_error(self):
        if self.from_n == 0:
            raise ValueError(f"from_n: {self.from_n} != 0")
        if self.iter_step <= 1:
            raise ValueError(f"iters_step: {self.iter_step} <= 1")

    def make_step(self, s):
        if self.step_mode == "+":
            s += self.iter_step
        if self.step_mode == "*":
            s *= self.iter_step
        return s

    def add_columns(self, n):
        while len(self.columns) < n:
            self.columns.append(str(len(self.columns)))

    def start_open_mpi(self, n, *args):
        """ C++ code to start in commandline interface """
        command = f"mpiexec -n {n} {self.path} {' '.join(map(str, args))}"
        # print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if stdout:
            output = [x for x in stdout.decode().split()]
            return output
        elif stderr:
            print(f"[::COMMAND RISE ERROR]\n {command} =>", Exception(stderr.decode()))
            return [None]

    def run(self, filename):
        """ Run the experiments with OpenMPI """
        iters = self.from_iters
        while iters <= self.to_iters:
            for n in range(self.from_n, self.to_n + 1):
                output = self.start_open_mpi(n, iters, *self.my_args)
                self.data_list.append([n, iters, *output])
            iters = int(self.make_step(iters))

        self.add_columns(len(self.data_list[0]))
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(self.columns)
            csv_writer.writerows(self.data_list)
        print(f"Results saved to {filename}")
