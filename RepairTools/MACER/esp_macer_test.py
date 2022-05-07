import os
import sys
import time

import pandas as pd

from testRepair import repairProgram

print("-"*15 + "| start log at " + time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())) + " |" + "-"*15)

with open(r"data/for_research/Code_deepfix.txt") as fp:
    codes = eval(fp.read())
with open(r"data/for_research/Code_tegcer.txt") as fp:
    codes.update(eval(fp.read()))

result_data = pd.DataFrame(columns=['prog_id', 'fix_state'])

fname = r'data/for_research/test.c'
predAtk = 5
prog_id = "prog43976"  # 正常
prog_id = "prog01857"  # 有问题 但可以跳过

# prog_id = "tegcer02111"  # 有问题 目前无法跳过
code = codes[prog_id]
fp = open(fname, mode='w')
fp.write(code)
fp.close()
result_dict = repairProgram(fname, 5)

if result_dict["C"]:
    fix_state = "CompletelyFixed"
elif result_dict["P"]:
    fix_state = "PartiallyFixed"
else:
    fix_state = "Unfixed"

result_data = result_data.append({'prog_id':prog_id , 'fix_state':fix_state}, ignore_index=True)


