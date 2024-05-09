# 每天进步一点点
# 5/3/2024 5:11 PM
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from dfr_simulation_template import dfr_simulation

if __name__ == "__main__":
    task_list = [1, 2, 3, 4]  # select task we want to run
    for task in task_list:
        if not isinstance(task, int):
            raise TypeError('Only support number type\n')
        elif task == 1:
            # task 1: dfr_simulation without fec and interleaving tech, using pl = 1e-4 and 1e-3
            pl_list = [1e-4, 1e-3]
            lenx = len(pl_list)
            decoded_rate_t1 = np.zeros(lenx)
            if_fec = False
            if_ci = False
            for pl, i in zip(pl_list, range(lenx)):
                decoded_rate_t1[i] = dfr_simulation(random_seed=777, video_trace='silenceOfTheLambs_verbose',
                                                    num_frames=10000, loss_probability=pl, fec=if_fec, ci=if_ci)
                print(f'Task1: Symbol loss rate: {pl:.4f} -> decoded rate: {decoded_rate_t1[i]:.2%} \n')
        elif task == 2:
            # task 2: dfr_simulation with fec but without interleaving tech, using pl = 1e-4 and 1e-3
            pl_list = [1e-4, 1e-3]
            lenx = len(pl_list)
            decoded_rate_t2 = np.zeros(lenx)
            if_fec = True
            if_ci = False
            for pl, i in zip(pl_list, range(lenx)):
                decoded_rate_t2[i] = dfr_simulation(random_seed=777, video_trace='silenceOfTheLambs_verbose',
                                                    num_frames=10000, loss_probability=pl, fec=if_fec, ci=if_ci)
                print(f'Task2: Symbol loss rate: {pl:.4f} -> decoded rate: {decoded_rate_t2[i]:.2%} \n')
        elif task == 3:
            # task 3: dfr_simulation with fec interleaving tech, using pl = 1e-4 and 1e-3
            pl_list = [1e-4, 1e-3]
            lenx = len(pl_list)
            decoded_rate_t3 = np.zeros(lenx)
            if_fec = True
            if_ci = True
            for pl, i in zip(pl_list, range(lenx)):
                decoded_rate_t3[i] = dfr_simulation(random_seed=777, video_trace='silenceOfTheLambs_verbose',
                                                    num_frames=10000, loss_probability=pl, fec=if_fec, ci=if_ci)
                print(f'Task3: Symbol loss rate: {pl:.4f} -> decoded rate: {decoded_rate_t3[i]:.2%} \n')
        # task 4: Compare three different experiments setting results and plot the dfr curve
        elif task == 4:
            # task4:
            print('this simulation cost 6254 seconds on my weak computer, i5-10210U for git holder\n'
                  'if you do not need a precise curve, you can simply change the [resolution] variable to 2 or more')
            unit = 1e-4
            resolution = 1
            up_limit = 40
            down_limit = 1
            pl_list = [x * unit for x in range(down_limit, up_limit + resolution, resolution)]
            lenx = len(pl_list)
            decoded_rate_1 = np.zeros(lenx)
            decoded_rate_2 = np.zeros(lenx)
            decoded_rate_3 = np.zeros(lenx)
            print('simulation starts!\n')
            result_directory = 'simulation_result/'
            folder_name = result_directory + 'task4_pl1_40'
            # 使用os.makedirs()创建文件夹，如果文件夹已经存在，则会抛出异常
            try:
                os.makedirs(folder_name)
                print(f"folder '{folder_name}' created successfully")
            except FileExistsError:
                print(f"folder'{folder_name}' already exist")
            except Exception as e:
                print(f"folder'{folder_name}' created error：{e}")
            time_start = time.time()
            for pl, i in zip(pl_list, range(lenx)):
                decoded_rate_1[i] = dfr_simulation(random_seed=777, video_trace='silenceOfTheLambs_verbose',
                                                   num_frames=10000, loss_probability=pl, fec=False, ci=False)
                decoded_rate_2[i] = dfr_simulation(random_seed=777, video_trace='silenceOfTheLambs_verbose',
                                                   num_frames=10000, loss_probability=pl, fec=True, ci=False)
                decoded_rate_3[i] = dfr_simulation(random_seed=777, video_trace='silenceOfTheLambs_verbose',
                                                   num_frames=10000, loss_probability=pl, fec=True, ci=True)
                print(f'Symbol loss rate: {pl:.4f} -> decoded rate: '
                      f'1. {decoded_rate_1[i]:.2%} 2. {decoded_rate_2[i]:.2%} 3.{decoded_rate_3[i]:.2%}\n')
            time_end = time.time()
            time_consumption = time_end - time_start
            print(f'Simulation running time is {time_consumption:}')
            decoded_rate_1_str = [f'Symbol loss rate:{pl} -> decoded rate:{rate:.2%}\n'
                                  for pl, rate in zip(pl_list, decoded_rate_1)]
            decoded_rate_2_str = [f'Symbol loss rate:{pl} -> decoded rate:{rate:.2%}\n'
                                  for pl, rate in zip(pl_list, decoded_rate_2)]
            decoded_rate_3_str = [f'Symbol loss rate:{pl} -> decoded rate:{rate:.2%}\n'
                                  for pl, rate in zip(pl_list, decoded_rate_3)]

            # 将列表保存为.npy文件
            #
            np.save(f'{folder_name}/decoded_rate_1_.npy', decoded_rate_1)
            with open(f'{folder_name}/decoded_rate_1_str.txt', 'w') as f:
                for item in decoded_rate_1_str:
                    f.write("%s\n" % item)
            #
            np.save(f'{folder_name}/decoded_rate_2.npy', decoded_rate_2)
            with open(f'{folder_name}/decoded_rate_2_str.txt', 'w') as f:
                for item in decoded_rate_2_str:
                    f.write("%s\n" % item)

            np.save(f'{folder_name}/decoded_rate_3.npy', decoded_rate_3)
            with open(f'{folder_name}/decoded_rate_3_str.txt', 'w') as f:
                for item in decoded_rate_3_str:
                    f.write("%s\n" % item)
            fig, ax = plt.subplots()
            ax.plot(pl_list, decoded_rate_1, label='no fec and ci')
            ax.plot(pl_list, decoded_rate_2, label='fec only')
            ax.plot(pl_list, decoded_rate_3, label='fec and ci')
            plt.legend()
            plt.xlabel('Symbol loss rate')
            plt.ylabel('Frame decode rate')
            plt.savefig(f'{folder_name}compare_3_setting.png')
            plt.show()

        else:
            raise ValueError('Only 4 tasks are supported now, please choose from 1 to 4!\n')



