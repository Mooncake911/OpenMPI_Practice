import matplotlib.pyplot as plt
import numpy as np


def time_iter_plot(thread_groups, labels, size=(15, 15),
                   title: str = 'Dependency of Iter on Time for Different Num_Threads'):
    plt.figure(figsize=size)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle(title, fontsize=30)

    for group, label in zip(thread_groups, labels):
        s = np.ceil(np.sqrt(len(group))).astype(int)
        for index, (thread_count, group_data) in enumerate(group, start=1):
            plt.subplot(s, s, index)
            plt.plot(group_data['Iter'], group_data['Time'],
                     linestyle='--', marker='o',
                     label=f'{label} - Num_Threads = {thread_count}')

            plt.xlabel('Iter')
            plt.ylabel('Time')
            plt.legend()

    plt.show()


def time_thread_plot(iter_groups, labels, size=(15, 15),
                     title: str = 'Dependency of Num_Threads on Time for Different Iter'):
    plt.figure(figsize=size)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle(title, fontsize=30)

    for group, label in zip(iter_groups, labels):
        for index, (iter_value, group_data) in enumerate(group, start=1):
            group_data = group_data.reset_index()

            plt.subplot(len(group), 1, index)
            plt.plot(group_data['Num_Threads'], group_data['Time'],
                     marker='o', linestyle='-',
                     label=f'{label} - Iter = {iter_value}')

            # Выделение точки с наименьшим значением красным цветом
            min_time_idx = group_data['Time'].idxmin()
            plt.scatter(group_data['Num_Threads'].iloc[min_time_idx], group_data['Time'].iloc[min_time_idx],
                        color='black', marker='X', zorder=10)

            plt.xlabel('Num_Threads')
            plt.ylabel('Time')
            plt.legend()

    plt.show()


def speedup_plot(iter_groups, labels, base_num_threads=1, size=(15, 15),
                 title: str = 'Dependency of Num_Threads on Speedup for Different Iter'):
    plt.figure(figsize=size)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle(title, fontsize=30)

    for group, label in zip(iter_groups, labels):
        for index, (iter_value, group_data) in enumerate(group, start=1):
            group_data = group_data.reset_index()

            base_time = group_data[group_data['Num_Threads'] == base_num_threads]['Time'].values[0]
            speedup = base_time / group_data['Time']

            plt.subplot(len(group), 1, index)
            plt.plot(group_data['Num_Threads'], speedup,
                     marker='o', linestyle='-',
                     label=f'{label} - Iter = {iter_value}')

            # Выделение точки с наибольшим значением ускорения красным цветом
            max_speedup_idx = np.argmax(speedup)
            plt.scatter(group_data['Num_Threads'].iloc[max_speedup_idx], speedup.iloc[max_speedup_idx], color='black',
                        marker='X', zorder=10)

            plt.xlabel('Num_Threads')
            plt.ylabel('Speedup')
            plt.legend()

    plt.show()
