import os
import sys

fp = open(r'Code.txt', mode='r')
codes = eval(fp.read())
fp.close()

os.system('mkdir esp_results')
for prog_id, code in codes.items():
    fp = open(r'./fix_test/test.c', mode='w')
    fp.write(code)
    fp.close()
    os.system('python -O neural_net/agent.py data/network_inputs/RLAssist-seed-1189/bin_0/ data/checkpoints/RLAssist-seed-1189/bin_0/ -eval 64040758 -esp fix_test/test.c')
    os.system('mv logs/RLAssist-seed-1189-ge_01_bin_0-single-eval_at-64040758-single.txt esp_results/{}.txt'.format(prog_id))

fp = open(r'over.txt', mode='w')
fp.write('All program have been eval!')
fp.close()
