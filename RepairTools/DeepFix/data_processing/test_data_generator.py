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

import os
import sqlite3
import json
from functools import partial
import numpy as np


def generate_raw_test_data(db_path, bins, min_program_length, max_program_length):

    raw_test_data = {}
    program_lengths = []

    problem_list = []
    for bin_ in bins:
        for problem_id in bin_:
            problem_list.append(problem_id)

    skipped = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT user_id, code_id, tokenized_code, name_dict, name_seq FROM Code " +\
            "WHERE problem_id=? and codelength>? and codelength<? and errorcount>0;"
        for problem_id in problem_list:
            for row in cursor.execute(query, (problem_id, min_program_length, max_program_length)):
                user_id, code_id, tokenized_code = map(str, row[:-2])
                name_dict, name_sequence = json.loads(
                    row[3]), json.loads(row[4])

                program_length = len(tokenized_code.split())
                program_lengths.append(program_length)
                if program_length >= min_program_length and program_length <= max_program_length:
                    try:
                        raw_test_data[problem_id].append(
                            (tokenized_code, name_dict, name_sequence, user_id, code_id))
                    except KeyError:
                        raw_test_data[problem_id] = [
                            (tokenized_code, name_dict, name_sequence, user_id, code_id)]

                else:
                    skipped += 1
                    print 'out of length range:', problem_id, user_id, code_id

    print 'problem_count:', len(raw_test_data),
    print 'program_count:', sum([len(raw_test_data[problem_id]) for problem_id in raw_test_data]),
    print 'discared_problems:', skipped

    program_lengths = np.sort(program_lengths)

    print 'Statistics'
    print '----------'
    print 'Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', \
        program_lengths[int(0.95 * len(program_lengths))]

    return raw_test_data


def generate_seeded_test_data(db_path, bins, min_program_length, max_program_length, max_fix_length,
                              kind_mutations, max_mutations, programs_per_problem, seed):
    rng = np.random.RandomState(seed)

    if kind_mutations == 'typo':
        from data_processing.typo_mutator import LoopCountThresholdExceededException, FailedToMutateException, Typo_Mutate, typo_mutate
        mutator_obj = Typo_Mutate(rng)
        mutate = partial(typo_mutate, mutator_obj, just_one=True)
    else:
        from data_processing.undeclared_mutator import LoopCountThresholdExceededException, FailedToMutateException, id_mutate
        mutate = partial(id_mutate, rng)

    result = {}

    exceptions_in_mutate_call = 0
    total_mutate_calls = 0
    program_lengths, fix_lengths = [], []

    problem_list = []
    for bin_ in bins:
        for problem_id in bin_:
            problem_list.append(problem_id)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT user_id, code_id, tokenized_code, name_dict, name_seq FROM Code " +\
            "WHERE problem_id=? and codelength>? and codelength<? and errorcount=0;"
        for problem_id in problem_list:
            for row in cursor.execute(query, (problem_id, min_program_length, max_program_length)):
                user_id, code_id, tokenized_code = map(str, row[:-2])
                name_dict, name_sequence = json.loads(
                    row[3]), json.loads(row[4])

                program_length = len(tokenized_code.split())
                program_lengths.append(program_length)
                if program_length >= min_program_length and program_length <= max_program_length:
                    # Mutate
                    total_mutate_calls += 1
                    try:
                        iterator = mutate(
                            tokenized_code, max_mutations, num_mutated_progs=1)

                    except FailedToMutateException:
                        exceptions_in_mutate_call += 1
                    except LoopCountThresholdExceededException:
                        exceptions_in_mutate_call += 1
                    except ValueError:
                        exceptions_in_mutate_call += 1
                        if kind_mutations == 'typo':
                            raise
                    except AssertionError:
                        exceptions_in_mutate_call += 1
                        if kind_mutations == 'typo':
                            raise
                    except Exception:
                        exceptions_in_mutate_call += 1
                        if kind_mutations == 'typo':
                            raise
                    else:
                        for corrupt_program, fix in iterator:
                            corrupt_program_length = len(
                                corrupt_program.split())
                            fix_length = len(fix.split())
                            fix_lengths.append(fix_length)

                            if corrupt_program_length >= min_program_length and \
                               corrupt_program_length <= max_program_length and fix_length <= max_fix_length:

                                try:
                                    result[problem_id].append(
                                        (corrupt_program, name_dict, name_sequence, user_id, code_id))
                                except KeyError:
                                    result[problem_id] = [
                                        (corrupt_program, name_dict, name_sequence, user_id, code_id)]

                if problem_id in result and len(result[problem_id]) >= programs_per_problem:
                    break

    seeded_test_data = {problem_id: result[problem_id] for problem_id in result if
                        len(result[problem_id]) == programs_per_problem}

    program_lengths = np.sort(program_lengths)
    fix_lengths = np.sort(fix_lengths)

    print 'Statistics'
    print '----------'
    print 'Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', \
        program_lengths[int(0.95 * len(program_lengths))]
    try:
        print 'Mean fix length: Mean =', np.mean(fix_lengths), '\t95th %ile = ', \
            fix_lengths[int(0.95 * len(fix_lengths))]
    except Exception as e:
        print e
        print 'fix_lengths'
        print fix_lengths
    print 'Total mutate calls:', total_mutate_calls
    print 'Exceptions in mutate() call:', exceptions_in_mutate_call, '\n'

    return seeded_test_data, mutator_obj.get_mutation_distribution() if kind_mutations == 'typo' else {}


def save_bins(destination, test_data, bins, which='raw'):
    print which
    full_list = []
    for bin_ in bins:
        for problem_id in bin_:
            full_list.append(problem_id)

    for i, bin_ in enumerate(bins):
        test_problems = bin_

        this_fold = {}
        count = 0

        for problem_id in sorted(test_problems):
            if problem_id in test_data:
                this_fold[problem_id] = test_data[problem_id]
                count += len(test_data[problem_id])

        assert os.path.isdir(os.path.join(destination, 'bin_%d' % i))

        print "Fold %d: %d Test programs in %d problems" % (i, count, len(this_fold))
        np.save(os.path.join(destination, 'bin_%d' %
                             i, 'test_%s_bin_%d.npy' % (which, i)), this_fold)


if __name__ == "__main__":
    max_program_length = 450
    min_program_length = 75
    max_fix_length = 25
    max_mutations = 5
    programs_per_problem = 100
    seed = 1189

    db_path = os.path.join('data', 'iitk-dataset', 'dataset.db')
    bins = np.load(os.path.join('data', 'iitk-dataset', 'bins.npy'))

    kind_mutations = 'typo'
    iitk_typo_output_directory = os.path.join('data/network_inputs', 'iitk-%s-%d' % (kind_mutations, seed))
    kind_mutations = 'ids'
    iitk_ids_output_directory = os.path.join('data/network_inputs', 'iitk-%s-%d' % (kind_mutations, seed))

    assert os.path.exists(iitk_typo_output_directory)
    assert os.path.exists(iitk_ids_output_directory)

    raw_test_data = generate_raw_test_data(
        db_path, bins, min_program_length, max_program_length)

    seeded_typo_test_data, _ = generate_seeded_test_data(db_path, bins, min_program_length,
                                                         max_program_length, max_fix_length, 'typo', max_mutations,
                                                         programs_per_problem, seed)

    seeded_ids_test_data, _ = generate_seeded_test_data(db_path, bins, min_program_length,
                                                        max_program_length, max_fix_length, 'ids', max_mutations,
                                                        programs_per_problem, seed)

    # Save raw test dataset
    save_bins(iitk_typo_output_directory, raw_test_data, bins, 'raw')
    save_bins(iitk_ids_output_directory, raw_test_data, bins, 'raw')

    # Save seeded test datasets
    save_bins(iitk_typo_output_directory, seeded_typo_test_data, bins, 'seeded-typo')
    save_bins(iitk_ids_output_directory, seeded_ids_test_data, bins, 'seeded-ids')

    print 'test dataset generated!'
