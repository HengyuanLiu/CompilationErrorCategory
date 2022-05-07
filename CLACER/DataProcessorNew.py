import os
import sys
import re
import copy
import numpy as np
import pandas as pd

import lex_analysis

class CodeData:
    """
    定义了一些用于处理c代码作为字符串的函数
    """

    def __init__(self, code_str):
        self.code_str = code_str  # 用来存储代码原本的字符串

        self.code_list = None  # 当调用self.code_str_split方法后生成

        self.code_str_abs = None  # 当调用self.code_abstraction方法后生成

        self.token_frame = None  # 当调用self.token_frame_genr方法后生成
        self.token_frame_abs = None  # 当调用self.tokens_abstraction或self.tokens_abstraction_distinction方法后生成
        self.abs_dict = None  # 当调用self.tokens_abstraction_distinction方法后生成
        self.abs_dict_count = None  # 当调用self.tokens_abstraction_distinction方法后生成

        self.concrete_map = None  # 当调用self.concrete_map_genr方法后生成

        self.error_message = None  # 当调用self.get_error_message生成
        self.error_message_useful = None  # 当调用self.error_message_process生成
        self.error_message_useful_abs = None  # 当调用self.error_message_process生成
        self.errors = None
        self.warnings = None

        self.error_loc_expc = None  # 当调用self.get_error_message生成
        self.error_loc_true = None  # 当调用self.get_error_message生成
        self.error_loc_dis = None  # 当调用self.get_error_message生成

        self.compile_path = 'lexer_file/test.c'
        self.error_message_path = 'lexer_file/error_message.txt'
        self.path = os.path.dirname(os.path.abspath(__file__))

    def code_annotation_strip(self):
        """给代码去除注释行"""

        def annotation_interval(code_str):
            """给出注释的区间,annotation_strip的辅助函数"""

            def annotation_dict(code_str):
                """annotation_interval的辅助函数"""
                # 给出注释符号的所有出现位置
                annotation_labels = ['//', '/*', '*/']
                anno_dict = {}
                for annotation_label in annotation_labels:
                    anno_dict[annotation_label] = []
                    flag = -2
                    while flag != -1:
                        if flag == -2:
                            flag = code_str.find(annotation_label)
                        else:
                            flag = code_str.find(annotation_label, flag + 2)
                        anno_dict[annotation_label].append(flag)
                    del anno_dict[annotation_label][-1]
                return anno_dict

            anno_dict = annotation_dict(code_str)
            # 确定多行注释
            anno_intervals = []
            for i in range(len(anno_dict['/*'])):
                for j in range(len(anno_dict['*/'])):
                    if anno_dict['/*'][i] <= anno_dict['*/'][j]:
                        anno_intervals.append([anno_dict['/*'][i], anno_dict['*/'][j] + 2])
                        break
            # 确定单行注释
            for i in range(len(anno_dict['//'])):
                find_flag = 0
                for j in range(anno_dict['//'][i], len(code_str)):
                    if code_str[j] == '\n':
                        anno_intervals.append([anno_dict['//'][i], j])
                        find_flag = 1
                        break
                if find_flag == 0:
                    anno_intervals.append([anno_dict['//'][i], len(code_str)])

            # 对注释区间进行排序
            anno_intervals = sorted(anno_intervals, key=lambda x: x[0])
            # 对注释区间去冗余
            index = 0
            while index < len(anno_intervals) - 1:
                if anno_intervals[index][0] <= anno_intervals[index + 1][0] \
                        and anno_intervals[index][1] >= anno_intervals[index + 1][1]:
                    del anno_intervals[index + 1]
                else:
                    index += 1
            return anno_intervals

        anno_interval = annotation_interval(code_str=self.code_str)
        code_str = list(self.code_str)
        for interval in anno_interval:
            for i in range(interval[0], interval[1]):
                if code_str[i] != '\n':
                    code_str[i] = ' '
        code_str = ''.join(code_str)
        self.code_str = code_str

    # check over
    def get_error_message(self, strip=True):
        """获取代码的编译错误信息"""
        if strip:
            self.code_annotation_strip()

        c_file = self.compile_path
        error_message_file = self.error_message_path
        command = 'gcc -c ' + c_file + ' 2> ' + error_message_file

        with open(c_file, 'w', encoding='utf-8') as f:
        # with open(c_file, 'w') as f:
            temp = self.code_str
            temp = list(temp)
            while '\r' in temp:
                temp.remove('\r')
            f.write(''.join(temp))
        os.system(command)
        with open(error_message_file, 'r', encoding='UTF-8') as f:
        # with open(error_message_file, 'r') as f:
            self.error_message = f.read()

        error_message_list = self.error_message.split('\n')
        path = self.compile_path
        remove_len = len(path)
        error_message_list = [error_message_line[remove_len + 1:] for error_message_line in error_message_list]
        self.error_message = '\n'.join(error_message_list)

    def get_first_error_message(self):
        """
        将数据集中错误信息分为各个单错误
        :param error_message: str,错误信息字符串
        """
        # 错误信息转化为列表
        error_message = self.error_message.split('\n')
        errors = []  # 存储分割后的错误
        # 函数索引与错误信息索引(用于定位)
        func_index_list = []
        error_index_list = []
        warning_index_list = []

        # 函数匹配与错误信息匹配标准
        func_str = 'In function'
        error_str = 'error:'
        warning_str = 'warning:'

        # 求的函数索引与错误信息索引
        for i in range(len(error_message)):
            if func_str in error_message[i]:
                func_index_list.append(i)
            if error_str in error_message[i] or warning_str in error_message[i]:
                error_index_list.append(i)

        error_index_list.append(len(error_message))  # 使得分割完整
        func_index_list.append(len(error_message))  # 便于判断该段错误信息所在函数

        # 生成单条错误信息列表
        for i in range(len(error_index_list) - 1):
            message_start_index = error_index_list[i]
            message_end_index = error_index_list[i + 1]
            # 摘出错误信息并组合
            error = '\n'.join(error_message[message_start_index:message_end_index])
            # 加入所在函数
            for j in range(len(func_index_list) - 1):
                if message_start_index >= func_index_list[j] and message_end_index <= func_index_list[j + 1]:
                    error = error_message[func_index_list[j]] + '\n' + error
                    break
            # 添加错误
            errors.append(error)
        # 对errors进一步格式化，只保留函数信息和错误信息
        errors = [error.strip() for error in errors]
        errors_new = []
        warnings = []
        patt_error = r'[0-9]*:[0-9]*: error:'
        patt_warning = r'[0-9]*:[0-9]*: warning:'
        patt_message = r'[0-9]*:[0-9]*: [A-Za-z]*:'
        for error_inx in range(len(errors)):
            error = errors[error_inx]
            error_prefix = re.findall(patt_message, error)
            error_text = re.split(patt_message, error)
            error = error_text.pop(0)
            for text_inx in range(len(error_text)):
                if re.findall(patt_error, error_prefix[text_inx]):
                    error = error.strip() + ' \n ' + error_prefix[text_inx].strip() + ' ' + error_text[text_inx].strip()
                    errors_new.append(error)
                elif re.findall(patt_warning, error_prefix[text_inx]):
                    warning = error.strip() + ' \n ' + error_prefix[text_inx].strip() + ' ' + error_text[
                        text_inx].strip()
                    warnings.append(warning)

        self.errors = errors_new
        self.warnings = warnings
        if errors_new:
            self.error_message = errors_new[0]
        elif warnings:
            self.error_message = warnings[0]
        else:
            print('没有错误信息')

    # check over
    def code_str_split(self, seq='\n', code_str=None, drop=False):
        """将代码字符串根据分隔符分解为代码列表"""
        if code_str is None:
            code_str = self.code_str
        code_list = code_str.split(seq)
        if drop:
            self.code_list = code_list
        else:
            return code_list

    def error_message_process(self):
        """输入error_message，返回编译报错行以及调整错误行"""

        def get_error_loc_expc(error_message):
            """获取编译器提示的错误位置信息error_loc_expc"""
            # 错误行标匹配模式
            mode_loc = re.compile(r"(.*)error:", re.S)
            mode_lineno = re.compile(r"[0-9]+", re.S)

            try:
                error_message = error_message.split('\n')
                find_flag = False
                while not find_flag:
                    if 'error:' not in error_message[0]:
                        del error_message[0]
                    else:
                        find_flag = True
                error_message = '\n'.join(error_message)

                loc = re.findall(mode_loc, error_message)
                error_loc_expc = re.findall(mode_lineno, loc[0])
                error_loc_expc = int(error_loc_expc[0])
                return error_loc_expc
            except (TypeError, IndexError) as reason:
                print("-" * 50)
                print('错误原因为', reason)
                return -1

        def get_error_loc_true(error_message, code_list):
            """若报错位置在报错行首个非空分词且报错信息含有before时,给出上个非注释非空行的行号"""

            # 若报错位置在报错行首个非空分词且报错信息含有before时,给出上个非注释非空行的行号
            def get_complie_loc(error_message):
                """
                获取编译信息的错误位置和错误列（单条错误信息）
                :param error_message:
                :return:
                """
                patt_error = r'[0-9]*:[0-9]*: error:'

                lineno_colno = re.findall(patt_error, error_message)[0].split(':')
                lineno_error = int(lineno_colno[0])
                colno_error = int(lineno_colno[1])
                return lineno_error, colno_error

            def error_loc_isfirst(code_str, lineno_error, colno_error):
                # 辅助判断报错位置是否为该行的首个非空字符
                try:
                    return code_str[lineno_error - 1][0:colno_error - 1].strip() == ''
                except IndexError:
                    return False

            def get_the_not_empty_lineno_before(code_str, lineno_error):
                """
                在判断错误应该出现在报错行的上面时,给出其向后搜索时的第一个非注释非空行的行号
                :param code_str:代码列表(按行分开)
                :param lineno: 当前报错行
                :return: lineno_before上一个非空的代码行标
                """
                find_flag = False
                lineno_before = lineno_error - 1
                while not find_flag:
                    lineno_before -= 1
                    if code_str[lineno_before].strip() == '':
                        pass
                    else:
                        find_flag = True
                return lineno_before + 1

            def error_loc_isbefore(error_message):
                if 'before' in error_message:
                    return True
                else:
                    return False

            # 去除非error信息
            # error_message = error_message.split('\n')
            # find_flag = False
            # while not find_flag:
            #     if 'error:' not in error_message[0]:
            #         del error_message[0]
            #     else:
            #         find_flag = True
            # error_message = '\n'.join(error_message)

            lineno_error, colno_error = get_complie_loc(error_message)
            lineno_before = lineno_error
            if error_loc_isfirst(code_list, lineno_error, colno_error) and error_loc_isbefore(error_message):
                lineno_before = get_the_not_empty_lineno_before(code_list, lineno_error)

            return lineno_before


        self.code_str_split(drop=True)
        self.error_loc_expc = get_error_loc_expc(error_message=self.error_message)
        self.error_loc_true = get_error_loc_true(error_message=self.error_message, code_list=self.code_list)

        return self.error_loc_expc, self.error_loc_true


