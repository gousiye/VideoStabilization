import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

data_path = 'Data/total_data.xlsx'
data = pd.read_excel(data_path, 'Sheet1', index_col=0)
reverse_optim_t = data['dn_t'] - data['dn_unstable_t']
reverse_optim_p = data['dn_p'] - data['dn_unstable_p']

# 有多少个负面防抖的
lent = len(reverse_optim_t[reverse_optim_t > 0])
lenp = len(reverse_optim_p[reverse_optim_p > 0])
fig1 = plt.figure()
fig1.canvas.set_window_title('figure1')
plt.bar(x=['tradition', 'deep'], height=[lent, lenp],
        alpha=0.2, label='柱状图', color='black')
plt.title("防抖效果产生负面效果", fontsize=20)
plt.ylabel('视频个数', fontsize=15)
plt.tick_params(labelsize=15)
plt.show()


# 查看防抖后的晃动系数
fig2, axes = plt.subplots(1, 1)
x_list = range(1, 62)
y1 = data['dn_t']
y2 = data['dn_p']
y3 = data['dn_unstable_t']
y4 = data['dn_unstable_p']
y5 = data['dn_stable']

print(y1)
fig2.canvas.set_window_title('figure2')
axes.plot(x_list, y1,
          label='tradition', color='red')
axes.plot(x_list, y2,
          label='deep', color='blue')
axes.plot(x_list, y3,
          label='unstable_t', color='yellow')
axes.plot(x_list, y4,
          label='unstable_p', color='green')
axes.plot(x_list, y5,
          label='stable', color='black')
axes.set_title("防抖后的晃动系数", fontsize=20)
axes.set_ylabel('防抖后的晃动系数', fontsize=15)
axes.tick_params(labelsize=15)
axes.legend(loc='best', fontsize=17)
plt.show()


# 查看防抖的绝对值变化
fig3, axes = plt.subplots(1, 1)
x_list = range(1, 62)
y1 = data['dn_unstable_t'] - data['dn_t']
y2 = data['dn_unstable_p'] - data['dn_p']
print(y1)
fig3.canvas.set_window_title('figure3')
axes.plot(x_list, y1,
          label='tradition', color='red')
axes.plot(x_list, y2,
          label='deep', color='blue')
axes.set_title("晃动系数的绝对下降值", fontsize=20)
axes.set_ylabel('晃动系数绝对下降值', fontsize=15)
axes.tick_params(labelsize=15)
axes.legend(loc='best', fontsize=17)
plt.show()


# 查看防抖的相对变化
fig4, axes = plt.subplots(1, 1)
x_list = range(1, 62)
y1 = data['dn_unstable_t'] - data['dn_t']
y2 = data['dn_unstable_p'] - data['dn_p']
y1 = y1 / data['dn_unstable_t']
y2 = y2 / data['dn_unstable_p']
print(y1)
fig4.canvas.set_window_title('figure4')
axes.plot(x_list, y1,
          label='tradition', color='red')
axes.plot(x_list, y2,
          label='deep', color='blue')
axes.set_title("晃动系数的相对下降值", fontsize=20)
axes.set_ylabel('晃动系数的相对下降值', fontsize=15)
axes.tick_params(labelsize=15)
axes.legend(loc='best', fontsize=17)
plt.show()
