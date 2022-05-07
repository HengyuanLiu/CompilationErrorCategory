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

import sys
import argparse
import sqlite3
import json
import time
from util.helpers import tokens_to_source, compilation_errors, apply_fix, logger, InvalidFixLocationException
from post_processing.postprocessing_helpers import meets_criterion, get_final_results


parser = argparse.ArgumentParser(description="Apply generated fixes")
parser.add_argument("database", help="sqlite3 database to use",)
parser.add_argument('--is_timing_experiment', action="store_true",
                    help="This is a timing experiment, do not store results")
args = parser.parse_args()

log = logger(args.database.split('.db')[
             0] + '_results', move_to_logs_dir=False)
print 'logging into {}'.format(log.log_file)
sys.stdout = log

print 'applying combined fixes ...'

conn = sqlite3.connect(args.database)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS error_message_strings (
                prog_id text NOT NULL,
                iteration text NOT NULL,
                network text NOT NULL,
                error_message_string text NOT NULL,
                error_message_count integer NOT NULL,
                FOREIGN KEY(prog_id, iteration, network) REFERENCES iterations(prog_id, iteration, network)
             )''')

problem_ids = []

for row in c.execute('SELECT DISTINCT prob_id FROM programs'):
    problem_ids.append(row[0])

c.close()

reconstruction = {}
errors = {}
errors_full = {}
errors_test = {}

fixes_per_stage = [0] * 10

total_count = 0


def do_problem(problem_id):
    global reconstruction, errors, errors_full, total_count, errors_test

    c = conn.cursor()

    reconstruction[problem_id] = {}
    errors[problem_id] = {}
    errors_full[problem_id] = {}
    errors_test[problem_id] = []
    candidate_programs = []

    for row in c.execute('SELECT user_id, prog_id, code, name_dict, name_seq FROM programs WHERE prob_id = ?', (problem_id,)):
        user_id, prog_id, initial = row[0], row[1], row[2]
        name_dict = json.loads(row[3])
        name_seq = json.loads(row[4])

	##########################################################################
	# print prog_id, type(initial), type(name_dict), type(name_seq)
	if initial is None or type(name_dict) != dict or type(name_seq) != list:
	    # time.sleep(2)
	    pass
	else:
	##########################################################################
            candidate_programs.append(
                (user_id, prog_id, initial, name_dict, name_seq,))

    for _, prog_id, initial, name_dict, name_seq in candidate_programs:
	##########################################################################
	print "processed:", prog_id
	##########################################################################
        fixes_suggested_by_typo_network = []
        fixes_suggested_by_undeclared_network = []

        for row in c.execute('SELECT fix FROM iterations WHERE prog_id=? AND network = \'typo\' ORDER BY iteration', (prog_id,)):
            fixes_suggested_by_typo_network.append(row[0])

        for row in c.execute('SELECT fix FROM iterations WHERE prog_id=? AND network = \'ids\' ORDER BY iteration', (prog_id,)):
            fixes_suggested_by_undeclared_network.append(row[0])

        reconstruction[problem_id][prog_id] = [initial]
        temp_errors, temp_errors_full = compilation_errors(
            tokens_to_source(initial, name_dict, False))
        errors[problem_id][prog_id] = [temp_errors]
        errors_full[problem_id][prog_id] = [temp_errors_full]

        try:
            for fix in fixes_suggested_by_typo_network:
                if meets_criterion(reconstruction[problem_id][prog_id][-1], fix, 'replace'):
                    temp_prog = apply_fix(
                        reconstruction[problem_id][prog_id][-1], fix, 'replace')
                    temp_errors, temp_errors_full = compilation_errors(
                        tokens_to_source(temp_prog, name_dict, False))

                    if len(temp_errors) > len(errors[problem_id][prog_id][-1]):
                        break
                    else:
                        reconstruction[problem_id][prog_id].append(temp_prog)
                        errors[problem_id][prog_id].append(temp_errors)
                        errors_full[problem_id][prog_id].append(
                            temp_errors_full)
                else:
                    break

        except InvalidFixLocationException:
            print 'Localization failed'

        while len(reconstruction[problem_id][prog_id]) <= 5:
            reconstruction[problem_id][prog_id].append(
                reconstruction[problem_id][prog_id][-1])
            errors[problem_id][prog_id].append(errors[problem_id][prog_id][-1])
            errors_full[problem_id][prog_id].append(
                errors_full[problem_id][prog_id][-1])

        already_fixed = []

        try:
            for fix in fixes_suggested_by_undeclared_network:
                if fix not in already_fixed:
                    temp_prog = apply_fix(
                        reconstruction[problem_id][prog_id][-1], fix, 'insert')
                    already_fixed.append(fix)
                    temp_errors, temp_errors_full = compilation_errors(
                        tokens_to_source(temp_prog, name_dict, False))

                    if len(temp_errors) > len(errors[problem_id][prog_id][-1]):
                        break
                    else:
                        reconstruction[problem_id][prog_id].append(temp_prog)
                        errors[problem_id][prog_id].append(temp_errors)
                        errors_full[problem_id][prog_id].append(
                            temp_errors_full)
                else:
                    pass

        except InvalidFixLocationException:
            print 'Localization failed'

        while len(reconstruction[problem_id][prog_id]) <= 10:
            reconstruction[problem_id][prog_id].append(
                reconstruction[problem_id][prog_id][-1])
            errors[problem_id][prog_id].append(errors[problem_id][prog_id][-1])
            errors_full[problem_id][prog_id].append(
                errors_full[problem_id][prog_id][-1])

        errors_test[problem_id].append(errors[problem_id][prog_id])

        if not args.is_timing_experiment:
            for k, errors_t, errors_full_t in zip(range(len(errors[problem_id][prog_id])), errors[problem_id][prog_id], errors_full[problem_id][prog_id]):
                c.execute("INSERT INTO error_message_strings VALUES(?, ?, ?, ?, ?)", (
                    prog_id, k, 'typo', errors_full_t.decode('utf-8', 'ignore'), len(errors_t)))

                for error_ in errors_t:
                    c.execute("INSERT INTO error_messages VALUES(?, ?, ?, ?)",
                              (prog_id, k, 'typo', error_.decode('utf-8', 'ignore'),))

    count_t = len(candidate_programs)
    total_count += count_t

    if not args.is_timing_experiment:
        print 'Committing changes to database...'
        conn.commit()
        print 'Done!'
    else:
        print 'Done problem with', count_t, 'programs'

    c.close()


start = time.time()

for problem_id in problem_ids:
    do_problem(problem_id)

time_t = time.time() - start

conn.commit()
conn.close()

print 'Total time:', time_t, 'seconds'
print 'Total programs processed:', total_count
print 'Average time per program:', int(float(time_t) / float(total_count) * 1000), 'ms'


def subset(arr1, arr2):
    for x in arr1:
        if x not in arr2:
            return False

    return True


total_fixes_num = {}
errors_before = {}

for problem_id in errors_test:
    total_fixes_num[problem_id] = 0

    for j, seq in enumerate(errors_test[problem_id]):
        error_numbers = [len(x) for x in seq]
        skip = False

        for i in range(len(error_numbers) - 1):
            assert (not error_numbers[i + 1] > error_numbers[i])
            total_fixes_num[problem_id] += error_numbers[i] - \
                error_numbers[i + 1]

            if error_numbers[i] != error_numbers[i + 1]:
                fixes_per_stage[i] += error_numbers[i] - error_numbers[i + 1]

total_numerator = 0
total_denominator = 0

for problem_id in errors_test:
    print problem_id,
    print '%d/%d' % (total_fixes_num[problem_id], sum([len(x[0]) for x in errors_test[problem_id]])),
    print '(%g%%)' % (float(total_fixes_num[problem_id]) / sum([len(x[0]) for x in errors_test[problem_id]]) * 100)

    total_numerator += total_fixes_num[problem_id]
    total_denominator += sum([len(x[0]) for x in errors_test[problem_id]])

print
print '~',
print int(float(total_numerator) * 100.0 / float(total_denominator)),
print '%'
print

for stage in range(len(fixes_per_stage)):
    print 'Stage', stage, ':', fixes_per_stage[stage]

# print final result for this dataset!
get_final_results(args.database)
