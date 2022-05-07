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
import argparse
import time
import math
import numpy as np
import tensorflow as tf
from data_processing.training_data_generator import vectorize
from neural_net.train import load_data, seq2seq_model as model
from post_processing.postprocessing_helpers import devectorize, \
    VectorizationFailedException
from util.helpers import apply_fix, vstack_with_right_padding, make_dir_if_not_exists
from util.helpers import InvalidFixLocationException, SubstitutionFailedException


parser = argparse.ArgumentParser(
    description="Predict and store fixes in a database.")

parser.add_argument("checkpoint_directory", help="Checkpoint directory")
parser.add_argument('-dd', "--data_directory", help="Data directory")
parser.add_argument('-d', "--database", help="sqlite3 database to use")
parser.add_argument('-w', '--which', help='Which test data.',
                    choices=['raw', 'seeded', 'MYTEST'], default="raw")
parser.add_argument('-b', '--batch_size', type=int,
                    help="batch_size", default=100)
parser.add_argument("--embedding_dim", type=int,
                    help="embedding_dim", default=50)
parser.add_argument('-m', "--memory_dim", type=int,
                    help="memory_dim", default=300)
parser.add_argument('-n', "--num_layers", type=int,
                    help="num_layers", default=4)
parser.add_argument('-c', '--cell_type',
                    help='One of RNN, LSTM or GRU.', default="LSTM")
parser.add_argument(
    '-v', '--vram', help='Fraction of GPU memory to use', type=float, default=0.9)
parser.add_argument('-a', '--max_attempts',
                    help='How many iterations to limit the repair to', type=int, default=5)
parser.add_argument("-r", "--resume_at", type=int,
                    help="Checkpoint to resume from (leave blank to resume from the best one)", default=None)
parser.add_argument("-t", "--task", help="Specify the task for which the network has been trained",
                    choices=['typo', 'ids'], default='typo')
parser.add_argument("--max_prog_length", type=int,
                    help="maximum length of the programs in tokens", default=450)
parser.add_argument('-o', '--max_output_seq_len',
                    help='max_output_seq_len', type=int, default=28)
parser.add_argument('--is_timing_experiment', action="store_true",
                    help="This is a timing experiment, do not store results")

args = parser.parse_args()

database_path = args.checkpoint_directory.replace(
    '''data/checkpoints/''', '''data/results/''')
args.checkpoint_directory = args.checkpoint_directory + \
    ('' if args.checkpoint_directory.endswith('/') else '/')
bin_id = None
try:
    if args.checkpoint_directory.find('bin_') == -1:
        raise ValueError('ERROR: failed to find the bin id')
    bin_id = int(args.checkpoint_directory[-2])
    print 'bin_id:', bin_id
except:
    raise

if args.database:
    database = args.database
else:
    make_dir_if_not_exists(database_path)
    database_name = args.which + '_' + args.task + '.db'
    database = os.path.join(database_path, database_name)

print 'using database:', database

if not args.data_directory:
    training_args = np.load(os.path.join(
        args.checkpoint_directory, 'experiment-configuration.npy')).item()['args']
    args.data_directory = training_args.data_directory

print 'data directory:', args.data_directory

conn = sqlite3.connect(database)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS programs (
                prog_id text NOT NULL,
                user_id text NOT NULL,
                prob_id text NOT NULL,
                code text NOT NULL,
                name_dict text NOT NULL,
                name_seq text NOT NULL,
                PRIMARY KEY(prog_id)
             )''')

c.execute('''CREATE TABLE IF NOT EXISTS iterations (
                prog_id text NOT NULL,
                iteration text NOT NULL,
                network text NOT NULL,
                fix text NOT NULL,
                PRIMARY KEY(prog_id, iteration)
             )''')

c.execute('''CREATE TABLE IF NOT EXISTS error_messages (
                prog_id text NOT NULL,
                iteration text NOT NULL,
                network text NOT NULL,
                error_message text NOT NULL,
                FOREIGN KEY(prog_id, iteration, network) REFERENCES iterations(prog_id, iteration, network)
             )''')


# Checkpoint information
if args.resume_at is None:
    best_checkpoint = None

    for checkpoint_name in os.listdir(os.path.join(args.checkpoint_directory, 'best')):
        if 'meta' in checkpoint_name:
            this_checkpoint = int(checkpoint_name[17:].split('.')[0])

            if best_checkpoint is None or this_checkpoint > best_checkpoint:
                best_checkpoint = this_checkpoint

    print "Resuming at", best_checkpoint, "..."
else:
    best_checkpoint = args.resume_at

# Load data
dataset = load_data(args.data_directory, shuffle=False, load_only_dicts=True)
dictionary = dataset.get_tl_dictionary()

# Build the network
scope = 'typo' if 'typo' in args.data_directory else 'ids'

with tf.variable_scope(scope):
    seq2seq = model(dataset.vocabulary_size, args.embedding_dim,
                    args.max_output_seq_len,
                    cell_type=args.cell_type,
                    memory_dim=args.memory_dim,
                    num_layers=args.num_layers,
                    dropout=0
                    )


def get_fix(sess, program):
    X, X_len = tuple(dataset.prepare_batch(program))
    return seq2seq.sample(sess, X, X_len)[0]


def get_fixes_in_batch(sess, programs):
    X, X_len = tuple(dataset.prepare_batch(programs))
    fixes = seq2seq.sample(sess, X, X_len)
    assert len(programs) == np.shape(fixes)[0]
    return fixes


def get_fixes(sess, programs):
    num_programs = len(programs)
    all_fixes = []

    for i in range(int(math.ceil(num_programs * 1.0 / args.batch_size))):
        start = i * args.batch_size
        end = (i + 1) * args.batch_size
        fixes = get_fixes_in_batch(sess, programs[start:end])
        all_fixes.append(fixes)

    fixes = vstack_with_right_padding(all_fixes)
    assert num_programs == np.shape(fixes)[
        0], 'num_programs: {}, fixes-shape: {}'.format(num_programs, np.shape(fixes))

    return fixes


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.vram)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if args.resume_at is None:
    seq2seq.load_parameters(sess, os.path.join(
        args.checkpoint_directory, 'best', 'saved-model-attn-' + str(best_checkpoint)))
else:
    seq2seq.load_parameters(sess, os.path.join(
        args.checkpoint_directory, 'saved-model-attn-' + str(best_checkpoint)))

if args.which == 'raw':
    test_dataset = np.load(os.path.join(
        args.data_directory, 'test_%s_bin_%d.npy' % (args.which, bin_id))).item()
###################################################################################################################
#                                               MY
###################################################################################################################
elif args.which == 'MYTEST':
    test_dataset = np.load(os.path.join(
        args.data_directory, 'test_data.npy')).item()
else:
    test_dataset = np.load(os.path.join(
        args.data_directory, 'test_%s-%s_bin_%d.npy' % (args.which, args.task, bin_id))).item()

# Attempt to repair
sequences_of_programs = {}
fixes_suggested_by_network = {}

if args.task == 'typo':
    normalize_names = True
    fix_kind = 'replace'
else:
    assert args.task == 'ids'
    normalize_names = False
    fix_kind = 'insert'

times = []
counts = []

print 'test data length:', sum([len(test_dataset[pid]) for pid in test_dataset])

for problem_id, test_programs in test_dataset.iteritems():
    sequences_of_programs[problem_id] = {}
    fixes_suggested_by_network[problem_id] = {}
    start = time.time()

    entries = []

    for program, name_dict, name_sequence, user_id, program_id in test_programs:
        sequences_of_programs[problem_id][program_id] = [program]
        fixes_suggested_by_network[problem_id][program_id] = []
        entries.append(
            (program, name_dict, name_sequence, user_id, program_id,))

        if not args.is_timing_experiment:
            c.execute("INSERT OR IGNORE INTO programs VALUES (?,?,?,?,?,?)", (program_id,
                                                                              user_id, problem_id, program, json.dumps(name_dict), json.dumps(name_sequence)))

    for round_ in range(args.max_attempts):
        to_delete = []
        input_ = []

        for i, entry in enumerate(entries):
            program, name_dict, name_sequence, user_id, program_id = entry

            try:
                program_vector = vectorize(sequences_of_programs[problem_id][program_id][-1], dictionary, args.max_prog_length,
                                           normalize_names, reverse=True)
            except:
                program_vector = None

            if program_vector is not None:
                input_.append(program_vector)
            else:
                to_delete.append(i)

        to_delete = sorted(to_delete)[::-1]

        for i in to_delete:
            del entries[i]

        assert len(input_) == len(entries)

        if len(input_) == 0:
            print 'Stopping before iteration %d (no programs remain)' % (round_ + 1)
            break

        # Get fix predictions
        fix_vectors = get_fixes(sess, input_)
        fixes = []

        # devectorize fixes
        for i, fix_vector in enumerate(fix_vectors):
            _, _, _, _, program_id = entries[i]

            fix = devectorize(fix_vector, dictionary)
            fixes_suggested_by_network[problem_id][program_id].append(fix)
            fixes.append(fix)

        to_delete = []

        # Apply fixes
        for i, entry, fix in zip(range(len(fixes)), entries, fixes):
            program, name_dict, name_sequence, user_id, program_id = entry

            try:
                program = sequences_of_programs[problem_id][program_id][-1]

                program = apply_fix(program, fix, fix_kind,
                                    flag_replace_ids=False)
                sequences_of_programs[problem_id][program_id].append(program)
            except ValueError as e:
                to_delete.append(i)
                sequences_of_programs[problem_id][program_id].append(
                    '{{localization_failed}}')
            except VectorizationFailedException as e:
                to_delete.append(i)
                sequences_of_programs[problem_id][program_id].append(
                    '{{vectorization_failed %s}}' % e.args[0])
            except InvalidFixLocationException as e:
                to_delete.append(i)
                sequences_of_programs[problem_id][program_id].append(
                    '{{localization_failed}}')
            except SubstitutionFailedException as e:
                to_delete.append(i)
                sequences_of_programs[problem_id][program_id].append(
                    '{{back_substitution_failed}}')
            except Exception as e:
                raise e
            else:
                if (not args.is_timing_experiment):
                    c.execute("INSERT OR IGNORE INTO iterations VALUES (?,?,?,?)",
                              (program_id, round_ + 1, args.task, fix))

        to_delete = sorted(to_delete)[::-1]

        for i in to_delete:
            del entries[i]

    times.append(time.time() - start)
    counts.append(len(test_programs))

    if not args.is_timing_experiment:
        print 'Committing changes to database...'
        conn.commit()
        print 'Done!'
    else:
        print 'Took', times[-1], 's to run for', counts[-1], 'programs'

if not args.is_timing_experiment:
    conn.commit()

print 'Total time:', np.sum(times), 'seconds'
print 'Total programs processed:', np.sum(counts)
print 'Average time per program:', int(float(np.sum(times)) / float(np.sum(counts)) * 1000), 'ms'
print
print 'results for {} dataset saved in {}'.format(args.which, database)

conn.close()
