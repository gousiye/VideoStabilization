import tradition
import deep
import os
import pandas as pd
import util
unstable_path = './Data/unstable/'
stable_path = './Data/stable/'

files = os.listdir(unstable_path)
files.sort(key=lambda x: int(x[:-4]))  # 将'.jpg'左边的字符转换成整数型进行排序


# 创建存放结果的文件夹
if not os.path.exists('Data/result'):
    os.mkdir('Data/result')

if not os.path.exists('Data/result/tradition'):
    os.mkdir('Data/result/tradition')

if not os.path.exists("Data/result/deep"):
    os.mkdir("Data/result/deep")

# if not os.path.exists('Data/result/compare'):
#     os.mkdir("Data/result/compare")

# 用于记录各视频的晃动指标
column_name = ['dn_unstable_t', 'dn_unstable_p', 'dn_t', 'dn_p', 'dn_stable']
dn_table = pd.DataFrame(None, columns=column_name)


# files = files[:]
# 数据集可能有点大，需要分批处理
files = files[0:3]
for file in files:
    input = unstable_path + file
    reference = stable_path + file
    out_t = './Data/result/tradition/'
    out_p = './Data/result/deep/'

    [dn_unstable_t, dn_t] = tradition.tradition_stabilization(input, out_t)
    [dn_unstable_p, dn_p] = deep.deep_stabilization(input, out_p)
    dn_stable = util.get_d_a_video(reference)

    temp_data = pd.DataFrame(
        [[dn_unstable_t, dn_unstable_p, dn_t, dn_p, dn_stable]], columns=column_name)
    dn_table = dn_table.append(temp_data)

print(dn_table)
dn_table.to_csv('./Data/data.csv')
