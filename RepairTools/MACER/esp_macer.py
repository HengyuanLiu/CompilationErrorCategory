import os
import sys
import time

import numpy as np
import pandas as pd

from testRepair import repairProgram

# 程序启动时间
print("-"*15 + "| start log at " + time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())) + " |" + "-"*15)

# 读取 目标代码库
with open(r"data/for_research/Code_deepfix.txt") as fp:
    codes = eval(fp.read())
# with open(r"data/for_research/Code_tegcer.txt") as fp:
#     codes.update(eval(fp.read()))

# 结果记录库
result_data = r'log/macer_result.txt'

log_name = r'log/log4eps.txt'  # 运行日志，每次修复目标代码之前追加其对应id
error_log_name = r'log/log4epsErr.txt'  # 错误代码日志，每次重启时，首先将上次修复中断代码的id提取出来，并放入记录

error_flag = True  # 程序自然运行中断为True，如果是人为调试请设为False

# 根据运行日志，对当前需要修复的目标代码库进行筛选，去除已经运行的
if not os.path.isfile(log_name):
    with open(log_name, mode='w') as fp_init:
        fp_init.write('')

with open(log_name, mode='r') as fp_start:
    code_done_ids = fp_start.read().strip().split('\n')
    if code_done_ids[0] == '':
        code_done_ids = []
    while code_done_ids:
        del_id = code_done_ids.pop()

        if not code_done_ids and error_flag:
            with open(error_log_name, mode='w') as fp_error:
                fp_error.write(del_id)

        del codes[del_id]


# 修复工具设置
fname = r'data/for_research/single.c'
predAtk = 5

# 修复进度辅助计数器
time_counter = 0

# 辅助观察修复计时
time_list = []

for prog_id, code in codes.items():
    # 记录当前所修复的目标代码代号
    with open(log_name, mode='a') as fp_id:
        fp_id.write(prog_id+'\n')

    start_time = time.time()  # 当前代码修复起始时间

    # 将当前代码写入目标文件
    with open(fname, mode='w') as fp_code:
        fp_code.write(code)

    result_dict = repairProgram(fname, 5)

    if result_dict["C"]:
        fix_state = "CompletelyFixed"
    elif result_dict["P"]:
        fix_state = "PartiallyFixed"
    else:
        fix_state = "Unfixed"

    with open(result_data, mode='a') as fp_result:
        fp_result.write("{},{}\n".format(prog_id, fix_state))

    time_counter += 1  # 计数器增加
    end_time = time.time()  # 当前代码修复终止时间

    time_list.append(float('{:.3f}'.format(end_time-start_time)))
    print("代码代号:{}   本次耗时:{:.3f}s".format(prog_id, end_time-start_time))
    if time_counter % 10 == 0:
        print("进度:{:.2f}%    本轮平均耗时:{:.3f}s".format(time_counter/len(codes)*100, np.mean(time_list)))
        time_list = []


print("-"*15 + "| end log at " + time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())) + " |" + "-"*15)
