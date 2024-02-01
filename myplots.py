import matplotlib.pyplot as plt
import numpy as np


class Plots:
    def __init__(self, tables, num_threads=str, iters=str, time=str):
        self.num_threads = str(num_threads)
        self.iters = str(iters)
        self.time = str(time)

        self.tables = tables
        self.thread_groups = []
        self.iter_groups = []

        for t in self.tables.values():
            self.thread_groups.append(t.groupby(self.num_threads, as_index=False))
            self.iter_groups.append(t.groupby(self.iters, as_index=False))

    def time_iter_plot(self, size=(15, 15)):
        title = f'Dependency of {self.iters} on {self.time} for Different {self.num_threads}'
        plt.figure(figsize=size)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.suptitle(title, fontsize=30)

        for group, label in zip(self.thread_groups, self.tables.keys()):
            s = np.ceil(np.sqrt(len(group))).astype(int)
            for index, (thread_count, group_data) in enumerate(group, start=1):
                plt.subplot(s, s, index)
                plt.plot(group_data[self.iters], group_data[self.time],
                         linestyle='--', marker='o',
                         label=f'{label} - {self.num_threads} = {thread_count}')

                plt.xlabel(self.iters)
                plt.ylabel(self.time)
                plt.legend()

        plt.show()

    def time_thread_plot(self, size=(15, 15)):
        title = f'Dependency of {self.num_threads} on {self.time} for Different {self.iters}'
        plt.figure(figsize=size)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.suptitle(title, fontsize=30)

        for group, label in zip(self.iter_groups, self.tables.keys()):
            for index, (iter_value, group_data) in enumerate(group, start=1):
                group_data = group_data.reset_index()

                plt.subplot(len(group), 1, index)
                plt.plot(group_data[self.num_threads], group_data[self.time],
                         marker='o', linestyle='-',
                         label=f'{label} - {self.iters} = {iter_value}')

                # Выделение точки с наименьшим значением красным цветом
                min_time_idx = group_data[self.time].idxmin()
                plt.scatter(group_data[self.num_threads].iloc[min_time_idx], group_data[self.time].iloc[min_time_idx],
                            color='black', marker='X', zorder=10)

                plt.xlabel(self.num_threads)
                plt.ylabel(self.time)
                plt.legend()

        plt.show()

    def speedup_plot(self, base_num_threads=1, size=(15, 15)):
        title = f'Dependency of {self.num_threads} on Speedup for Different {self.iters}'
        plt.figure(figsize=size)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.suptitle(title, fontsize=30)

        for group, label in zip(self.iter_groups, self.tables.keys()):
            for index, (iter_value, group_data) in enumerate(group, start=1):
                group_data = group_data.reset_index()

                base_time = group_data[group_data[self.num_threads] == base_num_threads][self.time].values[0]
                speedup = base_time / group_data[self.time]

                plt.subplot(len(group), 1, index)
                plt.plot(group_data[self.num_threads], speedup,
                         marker='o', linestyle='-',
                         label=f'{label} - {self.iters} = {iter_value}')

                # Выделение точки с наибольшим значением ускорения красным цветом
                max_speedup_idx = np.argmax(speedup)
                plt.scatter(group_data[self.num_threads].iloc[max_speedup_idx], speedup.iloc[max_speedup_idx],
                            color='black',
                            marker='X', zorder=10)

                plt.xlabel(self.num_threads)
                plt.ylabel('Speedup')
                plt.legend()

        plt.show()
