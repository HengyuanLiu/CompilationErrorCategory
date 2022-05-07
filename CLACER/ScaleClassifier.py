import os
import sys
import time
import numpy as np
import pandas as pd
import paddle
import paddle.fluid as fluid
from DataProcessor import *

paddle.enable_static()


def SyntacticErrorClassifier(codes, error_flag):
    # 程序启动时间
    print("-" * 15 + "| start log at " + time.strftime('%Y.%m.%d %H:%M:%S',
                                                       time.localtime(time.time())) + " |" + "-" * 15)
    # 修改工作路径
    path_ori = os.getcwd()
    path_target = os.path.dirname(sys.argv[0])

    # 结果记录库
    result_data = r'log_id/deepfix_classify_result.txt'

    log_name = r'log_id/log4done.txt'  # 运行日志，每次修复目标代码之前追加其对应id
    error_log_name = r'log_id/log4doneErr.txt'  # 错误代码日志，每次重启时，首先将上次修复中断代码的id提取出来，并放入记录

    # error_flag = True  # 程序自然运行中断为True，如果是人为调试请设为False

    # 根据运行日志，对当前需要修复的目标代码库进行筛选，去除已经运行的
    if not os.path.isfile(log_name):
        with open(log_name, mode='w') as fp_init:
            fp_init.write('')

    with open(log_name, mode='r') as fp_start:
        code_done_ids = fp_start.read().strip().split('\n')
        if code_done_ids[0] == '':
            code_done_ids = []
        while code_done_ids:
            del_id = str(code_done_ids.pop())

            if not code_done_ids and error_flag:
                with open(error_log_name, mode='w') as fp_error:
                    fp_error.write(del_id)

            del codes[int(del_id)]

    if path_ori != path_target:
        os.chdir(path_target)

    def get_top_N(np_array, N):
        """获取np数组的前几大的位置"""
        np_array = np_array.copy()
        imax_N = []
        for i in range(N):
            imax_N.append(np_array.argmax())
            np_array[0, np_array.argmax()] = np_array.min()
        return imax_N

    start_time = time.time()  # 当前代码修复起始时间
    # 读取 语料库 以及 向量化库
    corpus = Corpus()
    corpus.get()
    corpus.get_dict_map()
    end_time = time.time()  # 当前代码修复终止时间
    print("读取语料库耗时:{:.3f}s".format(end_time - start_time))

    start_time = time.time()  # 当前代码修复起始时间
    vec_repository = VecRepository(error_line_len=80, error_message_len=20)
    vec_repository.get()
    end_time = time.time()  # 当前代码修复终止时间
    print("读取向量化库耗时:{:.3f}s".format(end_time - start_time))


    # 读取模型
    start_time = time.time()  # 当前代码修复起始时间
    exe = fluid.Executor(fluid.CPUPlace())
    path = 'E:\PYprojects\SyntacticErrorHelper\param_repository\param_repository\param_all_TextCNN_512_128_5_0.001_None_None'
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
    end_time = time.time()  # 当前代码修复终止时间
    print("加载模型耗时:{:.3f}s".format(end_time - start_time))

    # 辅助观察修复计时
    time_list = []

    for code_id, code_str in codes_dict.items():
        # 记录当前所修复的目标代码代号
        with open(log_name, mode='a') as fp_id:
            fp_id.write(str(code_id) + '\n')

        start_time = time.time()  # 当前代码修复起始时间

        try:
            # 处理目标代码，获取网络输入
            code = CodeData(code_str=code_str)
            code.code_annotation_strip()
            code.get_error_message()
            code.get_first_error_message()
            code.token_frame_genr()
            code.code_abstraction()
            code.error_message_process()
            # 需要的数据
            code_str = code.code_str  # "code"
            error_message = code.error_message  # "error_message"
            error_loc_true = code.error_loc_true  # "error_loc_true"
            error_line = code.code_lines_pickup(index=code.error_loc_true, code_str=code.code_str)  # "error_line"

            error_line_abs = code.code_lines_pickup(index=code.error_loc_true,
                                                    code_str=code.code_str_abs)  # "error_line_abs"
            error_message_useful_abs = code.error_message_useful_abs  # "error_message_useful_abs"

            vecs = [0, 0]
            vecs[0] = corpus.text2vec(error_line_abs)
            vecs[1] = corpus.text2vec(error_message_useful_abs)

            vecs[0] = vec_repository.vec_process(vecs[0], len_max=vec_repository.error_line_len)
            vecs[1] = vec_repository.vec_process(vecs[1], len_max=vec_repository.error_message_len)

            code_v = vecs[0] + vecs[1]

            code_v = np.array(code_v).astype(dtype='int64').reshape([100, 1])
            code_v = fluid.create_lod_tensor(code_v, [[100]], place=fluid.CPUPlace())

            top_N = 3  # 预测概率前三
            results = exe.run(inference_program,
                              feed={feed_target_names[0]: code_v},
                              fetch_list=fetch_targets)

            top_N_pred = get_top_N(results[0], top_N)

            result_dict = {
                "code": code_str,
                "error_message": error_message,
                "error_loc_true": error_loc_true,
                "error_line": error_line,
                "error_class_id": top_N_pred[0]
            }
        except Exception as e:
            top_N = 3  # 预测概率前三
            top_N_pred = [-1]*top_N
            result_dict = {
                "code": code_str,
                "error_message": '',
                "error_loc_true": -1,
                "error_line": '',
                "error_class_id": -1
            }

        with open(result_data, mode='a') as fp_result:
            fp_result.write(
                "{},{},{},{}\n".format(code_id, top_N_pred[0], top_N_pred[1], top_N_pred[2])
            )

        end_time = time.time()  # 当前代码修复终止时间
        time_list.append(float('{:.3f}'.format(end_time-start_time)))
        print("代码代号:{} 代码错误类型:{} 本次耗时:{:.3f}s".format(code_id, top_N_pred[0], end_time-start_time))

        if code_id % 100 == 0:
            print("进度:{:.2f}%    本轮平均耗时:{:.3f}s".format(code_id / max(codes.keys()) * 100, np.mean(time_list)))
            time_list = []

    os.chdir(path_ori)
    print(
        "-" * 15 + "| end log at " + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())) + " |" + "-" * 15)


if __name__ == "__main__":
    # 目标代码库
    with open("Codes_mutate_id.txt", mode='r') as fp:
        codes_dict = eval(fp.read())
    # # 目标代码库
    # with open("Codes_mutate.txt", mode='r') as fp:
    #     codes_dict = eval(fp.read())

    # 分类结果记录
    SyntacticErrorClassifier(codes_dict, error_flag=True)
