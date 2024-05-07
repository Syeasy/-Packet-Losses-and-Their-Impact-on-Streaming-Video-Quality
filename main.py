# 每天进步一点点
# 5/3/2024 5:11 PM
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from dfr_simulation_template import dfr_simulation
if __name__ == "__main__":
    unit = 1e-4
    resolution = 1
    up_limit = 40
    down_limit = 1
    lenx = int(up_limit/resolution)
    decoded_rate_1 = np.zeros(lenx)
    decoded_rate_2 = np.zeros(lenx)
    decoded_rate_3 = np.zeros(lenx)
    pl_list = [x * unit for x in range(down_limit, up_limit + resolution, resolution)]
    print('simulation starts!\n')

    folder_name = 'simulation_result'
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
        print(f'Symbol loss rate: {pl} -> decoded rate: '
              f'1.{decoded_rate_1[i]:.2%} 2.{decoded_rate_2[i]:.2%} 3.{decoded_rate_3[i]:.2%}\n')
    decoded_rate_1_str = [f'Symbol loss rate:{pl} -> decoded rate:{rate:.2%}\n'
                          for pl, rate in zip(pl_list, decoded_rate_1)]
    decoded_rate_2_str = [f'Symbol loss rate:{pl} -> decoded rate:{rate:.2%}\n'
                          for pl, rate in zip(pl_list, decoded_rate_2)]
    decoded_rate_3_str = [f'Symbol loss rate:{pl} -> decoded rate:{rate:.2%}\n'
                          for pl, rate in zip(pl_list, decoded_rate_3)]



    # 将列表保存为.npy文件
    # np.save(f'{folder_name}/decoded_rate_3_str_{time_start}.npy', decoded_rate_1_str)
    np.save(f'{folder_name}/decoded_rate_3_{time_start}.npy', decoded_rate_1)

    # np.save(f'{folder_name}/decoded_rate_3_str_{time_start}.npy', decoded_rate_3_str)
    np.save(f'{folder_name}/decoded_rate_3_{time_start}.npy', decoded_rate_3)

    # np.save(f'{folder_name}/decoded_rate_3_str_{time_start}.npy', decoded_rate_3_str)
    np.save(f'{folder_name}/decoded_rate_3_{time_start}.npy', decoded_rate_3)
    fig, ax = plt.subplots()
    ax.plot(pl_list, decoded_rate_1)
    ax.plot(pl_list, decoded_rate_2)
    ax.plot(pl_list, decoded_rate_3)













