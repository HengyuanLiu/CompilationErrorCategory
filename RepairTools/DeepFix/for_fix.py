import numpy as np
import sqlite3
import json
from util.c_tokenizer import C_Tokenizer

tokenize = C_Tokenizer().tokenize

prob_id = 'prob0'
user_id = 'user0'

with open(r"data/for_research/Code_buctoj.txt") as fp:
    codes = eval(fp.read())

test_data = {}
test_data[prob_id] = []
time_counter = 0
for prog_id, code in codes.items():

    tokenized_code, name_dict, name_seq = tokenize(code)

    if time_counter == 0:
        print(tokenized_code)

    test_data[prob_id].append(
        (tokenized_code, name_dict, name_seq, user_id, prog_id)
    )
    time_counter += 1
    print 'prog_id:{}   processed:{:.2f}%'.format(prog_id, time_counter * 1.0 / len(codes) * 100)

test_data = np.array(test_data)
np.save('test_data', test_data)
