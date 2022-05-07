"""
Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import sqlite3
import numpy as np
from util.helpers import get_lines, extract_line_number, get_rev_dict, FailedToGetLineNumberException, _truncate_fix
import regex as re


class VectorizationFailedException(Exception):
    pass


class EmptyFixException(Exception):
    pass


def filter_minus_one(vector):
    result = []
    for each in vector:
        if each != '-1':
            result.append(each)

    return result


def devectorize(vector, dictionary, reverse=False):
    result = []
    reversed_dictionary = get_rev_dict(dictionary)

    for each in vector:
        if reversed_dictionary[each] != '_pad_' and (not reverse or reversed_dictionary[each] != '_eos_'):
            result.append(reversed_dictionary[each])

    if reverse:
        result = result[::-1]

    if len(result) == 0:
        raise EmptyFixException('Empty vector: {} = {} passed in devectorize'.format(
            vector, [reversed_dictionary[v] for v in vector]))

    return ' '.join(filter_minus_one(result))


def _is_stop_signal(fix):
    if _truncate_fix(fix) == '':
        return True


def meets_criterion(incorrect_program_tokens, fix, type_, silent=True):
    lines = get_lines(incorrect_program_tokens)
    fix = _truncate_fix(fix)

    if _is_stop_signal(fix):
        return False

    try:
        fix_line_number = extract_line_number(fix)
    except FailedToGetLineNumberException:
        return False

    if fix_line_number >= len(lines):
        return False

    fix_line = lines[fix_line_number]

    # Make sure number of IDs is the same
    if len(re.findall('_<id>_\w*', fix_line)) != len(re.findall('_<id>_\w*', fix)):
        if not silent:
            print 'number of ids is not the same'
        return False

    keywords_regex = '_<keyword>_\w+|_<type>_\w+|_<APIcall>_\w+|_<include>_\w+'

    if type_ == 'replace' and re.findall(keywords_regex, fix_line) != re.findall(keywords_regex, fix):
        if not silent:
            print 'important words (keywords, etc.) change drastically'
        return False

    return True


def get_final_results(database):
    with sqlite3.connect(database) as conn:
        c = conn.cursor()

        error_counts = []

        for row in c.execute("SELECT iteration, COUNT(*) FROM error_messages GROUP BY iteration ORDER BY iteration;"):
            error_counts.append(row[1])

        query1 = """SELECT COUNT(*)
        FROM error_messages
        WHERE iteration = 0 AND prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        for row in c.execute(query1):
            initial_errors = row[0]

        query2 = """SELECT COUNT(*)
        FROM error_messages
        WHERE iteration = 10 AND prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        for row in c.execute(query2):
            final_errors = row[0]

        query3 = """SELECT COUNT(DISTINCT prog_id)
        FROM error_message_strings
        WHERE iteration = 10 AND error_message_count = 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        for row in c.execute(query3):
            fully_fixed = row[0]

        #####################################################################################
        query3_ = """SELECT DISTINCT prog_id
        FROM error_message_strings
        WHERE iteration = 10 AND error_message_count = 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        completely_fixed = {}
        for row in c.execute(query3_):
            completely_fixed[row[0]] = 0

        with open(r"MYTEST.txt", mode='a') as fp:
            for p_i, e_c in completely_fixed.items():
                fp.write("{},{}\n".format(p_i, 'CompletelyFixed'))
        #####################################################################################

        query4 = """SELECT DISTINCT prog_id, error_message_count FROM error_message_strings
        WHERE iteration = 0 AND error_message_count > 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        query5 = """SELECT DISTINCT prog_id, error_message_count FROM error_message_strings
        WHERE iteration = 10 AND error_message_count > 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        original_errors = {}
        for row in c.execute(query4):
            original_errors[row[0]] = int(row[1])

        partially_fixed = {}
        unfixed = {}
        for row in c.execute(query5):
            if int(row[1]) < original_errors[row[0]]:
                partially_fixed[row[0]] = int(row[1])
            elif int(row[1]) == original_errors[row[0]]:
                unfixed[row[0]] = int(row[1])
            else:
                print row[0], row[1], original_errors[row[0]]

        #####################################################################################
        with open(r"MYTEST.txt", mode='a') as fp:
            for p_i, e_c in partially_fixed.items():
                fp.write("{},{}\n".format(p_i, 'PartiallyFixed'))

        with open(r"MYTEST.txt", mode='a') as fp:
            for p_i, e_c in unfixed.items():
                fp.write("{},{}\n".format(p_i, 'Unfixed'))
        #####################################################################################

        token_counts = []
        assignments = None

        for row in c.execute("SELECT COUNT(DISTINCT prob_id) FROM programs p WHERE prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"):
            assignments = int(row[0])

        for row in c.execute("SELECT code FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count <> 0;"):
            token_counts += [len(row[0].split())]

        avg_token_count = np.mean(token_counts)

        print "-------"
        print "Assignments:", assignments
        print "Program count:", len(token_counts)
        print "Average token count:", avg_token_count
        print "Error messages:", initial_errors
        print "-------"

        print "Errors remaining:", final_errors
        print "Reduction in errors:", (initial_errors - final_errors)
        print "Completely fixed programs:", fully_fixed
        print "partially fixed programs:", len(partially_fixed)
        print "unfixed programs:", len(unfixed)
        print "-------"
