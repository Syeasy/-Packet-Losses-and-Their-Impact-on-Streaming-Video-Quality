# 每天进步一点点
# 5/3/2024 5:11 PM

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from dfr_simulation_template import dfr_simulation
if __name__ == "__main__":
    resolution = 1e-4
    up_limit = 4e-3
    down_limit = 2e-3
    lenx = int(up_limit/resolution)
    decoded_rate_1 = np.zeros(lenx)
    decoded_rate_2 = np.zeros(lenx)
    decoded_rate_3 = np.zeros(lenx)
    for pl, i in zip(np.arange(down_limit, up_limit + resolution, resolution), range(lenx)):
        # decoded_rate_1[i] = dfr_simulation(num_frames=10000, loss_probability=pl, fec=False, ci=False)
        # decoded_rate_2[i] = dfr_simulation(num_frames=10000, loss_probability=pl, fec=True, ci=False)
        decoded_rate_3[i] = dfr_simulation(random_seed=777, video_trace='silenceOfTheLambs_verbose',
                                           num_frames=10000, loss_probability=pl, fec=True, ci=True)
        print(f'Symbol loss rate: {pl} -> decoded rate:{decoded_rate_3[i]:.2%}\n')
    decoded_rate_3_str = [f'Symbol loss rate:{pl} -> decoded rate:{rate:.2%}\n'
                          for pl, rate in zip(range(down_limit, up_limit, resolution), decoded_rate_3)]
    # 将列表保存为.npy文件
    np.save('decoded_rate_3_str.npy', decoded_rate_3_str)
    np.save('decoded_rate_3.npy', decoded_rate_3)
    fig, ax = plt.subplots()
    ax.plot(range(down_limit, up_limit, resolution), decoded_rate_3)












