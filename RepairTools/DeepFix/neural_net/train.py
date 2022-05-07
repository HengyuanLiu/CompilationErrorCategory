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

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell, MultiRNNCell, DropoutWrapper
import math
import os
import sys
import time
import glob
import argparse
from shutil import copy

from util.helpers import make_dir_if_not_exists, logger, get_rev_dict, Accuracy_calculator_for_deepfix, get_accuracy
from data_processing.training_data_generator import load_dictionaries


class load_data:
    def _deserialize(self, data_folder):
        train_ex = np.load(os.path.join(data_folder, 'examples-train.npy'))
        valid_ex = np.load(os.path.join(
            data_folder, 'examples-validation.npy'))
        test_ex = np.load(os.path.join(data_folder, 'examples-test.npy'))
        assert train_ex is not None and valid_ex is not None and test_ex is not None
        return train_ex, valid_ex, test_ex

    def __init__(self, data_folder, shuffle=True, load_only_dicts=False):
        self.rng = np.random.RandomState(1189)
        self.tl_dict, self.rev_tl_dict = load_dictionaries(data_folder)
        assert self.tl_dict is not None and self.rev_tl_dict is not None

        if load_only_dicts:
            return

        if not shuffle:
            self.train_ex, self.valid_ex, self.test_ex = self._deserialize(
                data_folder)

        else:
            try:
                self.train_ex, self.valid_ex, self.test_ex = self._deserialize(
                    os.path.join(data_folder, 'shuffled'))

                print "Successfully loaded shuffled data."
                sys.stdout.flush()

            except IOError:
                print "Generating shuffled data..."
                sys.stdout.flush()

                self.train_ex, self.valid_ex, self.test_ex = self._deserialize(
                    data_folder)

                self.rng.shuffle(self.train_ex)
                self.rng.shuffle(self.valid_ex)
                self.rng.shuffle(self.test_ex)

                make_dir_if_not_exists(os.path.join(data_folder, 'shuffled'))

                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-train.npy'), self.train_ex)
                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-validation.npy'), self.valid_ex)
                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-test.npy'), self.test_ex)

    def get_raw_data(self):
        return self.train_ex, self.valid_ex, self.test_ex

    @classmethod
    def prepare_batch(self, sequences, msg=False):
        sequence_lengths = [len(seq) for seq in sequences]
        batch_size = len(sequences)
        max_sequence_length = max(sequence_lengths)

        if msg:
            print 'max_sequence_length', max_sequence_length

        # initialize with _pad_ = 0
        inputs_time_major = np.zeros(
            shape=[max_sequence_length, batch_size], dtype=np.int32)
        for i, seq in enumerate(sequences):
            for j, element in enumerate(seq):
                inputs_time_major[j, i] = element
        return [inputs_time_major, np.array(sequence_lengths)]

    def get_batch(self, start, end, which='train'):
        if which == 'train':
            X, Y = zip(*self.train_ex[start:end])
        elif which == 'valid':
            X, Y = zip(*self.valid_ex[start:end])
        elif which == 'test':
            X, Y = zip(*self.test_ex[start:end])
        else:
            raise ValueError('choose one of train/valid/test for which')
        return tuple(self.prepare_batch(X) + self.prepare_batch(Y))

    def get_tl_dictionary(self):
        return self.tl_dict

    def get_rev_tl_dictionary(self):
        return self.rev_tl_dict

    @property
    def data_size(self):
        return len(self.train_ex), len(self.valid_ex), len(self.test_ex)

    @property
    def vocabulary_size(self):
        return len(self.tl_dict)


def _new_RNN_cell(memory_dim, num_layers, cell_type, dropout, keep_prob):

    assert memory_dim is not None and num_layers is not None and cell_type is not None and dropout is not None, 'At least one of the arguments is passed as None'

    if cell_type == 'LSTM':
        constituent_cell = LSTMCell(memory_dim)
    elif cell_type == 'GRU':
        constituent_cell = GRUCell(memory_dim)
    else:
        raise Exception('unsupported rnn cell type: %s' % cell_type)

    if dropout != 0:
        constituent_cell = DropoutWrapper(
            constituent_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)

    if num_layers > 1:
        return MultiRNNCell([constituent_cell for _ in range(num_layers)])

    return constituent_cell

# inspired from https://github.com/ematvey/tensorflow-seq2seq-tutorials
class seq2seq_model():
    """Seq2Seq model using blocks from new `tf.contrib.seq2seq`.
    Requires TF-1.0.1"""

    PAD = 0
    EOS = 1

    def __init__(self, vocab_size, embedding_size, max_output_seq_len,
                 cell_type='LSTM', memory_dim=300, num_layers=4, dropout=0.2,
                 attention=True,
                 scope=None,
                 verbose=False):

        assert 0 <= dropout and dropout <= 1, '0 <= dropout <= 1, you passed dropout={}'.format(
            dropout)

        tf.set_random_seed(1189)

        self.attention = attention
        self.max_output_seq_len = max_output_seq_len

        self.memory_dim = memory_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.scope = scope

        if dropout != 0:
            self.keep_prob = tf.placeholder(tf.float32)
        else:
            self.keep_prob = None

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.encoder_cell = _new_RNN_cell(
            memory_dim, num_layers, cell_type, dropout, self.keep_prob)
        self.decoder_cell = _new_RNN_cell(
            memory_dim, num_layers, cell_type, dropout, self.keep_prob)

        self._make_graph()

        if self.scope is not None:
            saver_vars = [var for var in tf.global_variables(
            ) if var.name.startswith(self.scope)]
        else:
            saver_vars = tf.global_variables()

        if verbose:
            print 'root-scope:', self.scope
            print "\n\nDiscovered %d saver variables." % len(saver_vars)
            for each in saver_vars:
                print each.name

        self.saver = tf.train.Saver(saver_vars, max_to_keep=5)

    @property
    def decoder_hidden_units(self):
        return self.memory_dim

    def _make_graph(self):
        self._init_placeholders()

        self._init_decoder_train_connectors()

        self._init_embeddings()

        self._init_simple_encoder()

        self._init_decoder()

        self._init_optimizer()

    def _init_placeholders(self):
        """ Everything is time-major """
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

    def _init_decoder_train_connectors(self):

        with tf.name_scope('decoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(
                tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat(
                [EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat(
                [self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(
                tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(
                decoder_train_targets_eos_mask, [1, 0])

            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs)

            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=True,
                                  dtype=tf.float32)
            )

    def _init_decoder(self):
        with tf.variable_scope("decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            if not self.attention:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(
                    encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=self.max_output_seq_len,
                    num_decoder_symbols=self.vocab_size,
                )
            else:
                attention_states = tf.transpose(
                    self.encoder_outputs, [1, 0, 2])

                (attention_keys,
                 attention_values,
                 attention_score_fn,
                 attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )

                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=self.max_output_seq_len,
                    num_decoder_symbols=self.vocab_size,
                )

            (self.decoder_outputs_train,
             self.decoder_state_train,
             self.decoder_context_state_train) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=True,
                    scope=scope,
                )
            )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(
                self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=True,
                    scope=scope,
                )
            )

            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1,
                                                          name='decoder_prediction_inference')

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)

        self.optimizer = tf.train.AdamOptimizer()
        gvs = self.optimizer.compute_gradients(self.loss)

        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1., 1)

        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]

        self.train_op = self.optimizer.apply_gradients(capped_gvs)

    def make_feed_dict(self, x, x_len, y, y_len):
        feed_dict = {
            self.encoder_inputs: x,
            self.encoder_inputs_length: x_len,

            self.decoder_targets: y,
            self.decoder_targets_length: y_len,
        }

        if self.dropout != 0:
            feed_dict.update({self.keep_prob: 1.0 - self.dropout})

        return feed_dict

    def load_parameters(self, sess, filename):
        self.saver.restore(sess, filename)

    def save_parameters(self, sess, filename, global_step=None):
        self.saver.save(sess, filename, global_step=global_step)

    def train_step(self, session, x, x_len, y, y_len):
        feed_dict = self.make_feed_dict(x, x_len, y, y_len)
        _, loss = session.run([self.train_op, self.loss], feed_dict)
        return loss

    def validate_step(self, session, x, x_len, y, y_len):
        feed_dict = self.make_feed_dict(x, x_len, y, y_len)
        loss, decoder_prediction, decoder_train_targets = session.run([self.loss,
                                                                       self.decoder_prediction_inference,
                                                                       self.decoder_train_targets], feed_dict)
        return loss, np.array(decoder_prediction).T, np.array(decoder_train_targets).T

    def sample(self, session, X, X_len):
        feed_dict = {self.encoder_inputs: X,
                     self.encoder_inputs_length: X_len}

        if self.dropout != 0:
            feed_dict.update({self.keep_prob: 1.0})

        decoder_prediction = session.run(
            self.decoder_prediction_inference, feed_dict)
        return np.array(decoder_prediction).T

######################################################################
######################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a fault-localization and repair seq2seq RNN.')
    parser.add_argument('data_directory', help='Data directory')
    parser.add_argument('checkpoints_directory', help='Checkpoints directory')
    parser.add_argument('-b', "--batch_size", type=int,
                        help="batch size", default=128)
    parser.add_argument("--embedding_dim", type=int,
                        help="embedding_dim", default=50)
    parser.add_argument("--memory_dim", type=int,
                        help="memory_dim", default=300)
    parser.add_argument('-n', "--num_layers", type=int,
                        help="num_layers", default=4)
    parser.add_argument('-e', "--epochs", type=int, help="epochs", default=20)
    parser.add_argument('-r', "--resume_at", type=int,
                        help="resume_at", default=0)
    parser.add_argument('-re', "--resume_epoch", type=int,
                        help="resume_epoch", default=0)
    parser.add_argument('-rmb', "--resume_minibatch",
                        type=int, help="resume_minibatch", default=0)
    parser.add_argument(
        '--cell_type', help='One of LSTM or GRU.', default='LSTM')
    parser.add_argument(
        '-c', '--ckpt_every', help='How often to save checkpoints', type=int, default=500)
    parser.add_argument('-o', '--max_output_seq_len',
                        help='max_output_seq_len', type=int, default=28)
    parser.add_argument(
        '-d', '--dropout', help='Probability to use for dropout', type=float, default=0.2)
    parser.add_argument(
        '-v', '--vram', help='Fraction of GPU memory to use', type=float, default=0.85)

    args = parser.parse_args()

    dataset_name = '_'.join(args.checkpoints_directory.split(
        '/')[2:]) if 'bin_' in args.checkpoints_directory else '_'.join(args.checkpoints_directory.split('/')[1:])
    dataset_name = dataset_name[:-1]

    log = logger('log-' + dataset_name + '.txt')
    print '\nlogging into {}'.format(log.log_file)
    sys.stdout = log

    assert args.data_directory != args.checkpoints_directory, 'data and checkpoints directories should be different!'
    assert 'typo' in args.data_directory or 'ids' in args.data_directory, 'data_directory argument has neither *typo* nor *ids* keyword!'

    which_network = 'typo' if 'typo' in args.data_directory else 'ids'

    # Make checkpoint directories
    make_dir_if_not_exists(args.checkpoints_directory)
    make_dir_if_not_exists(os.path.join(args.checkpoints_directory, 'best'))

    configuration = {}

    configuration["args"] = args
    configuration["log"] = 'log-' + dataset_name + '.txt'
    configuration['which_network'] = which_network

    np.save(os.path.join(args.checkpoints_directory,
                         'experiment-configuration.npy'), configuration)

    def update_best_ckpt(**kwargs):
        with open(os.path.join(args.checkpoints_directory, 'best', 'best.txt'), 'a+') as f:
            for each in kwargs:
                f.write('{}: {} '.format(each, kwargs[each]))
            f.write('\n')

    print 'data_directory            :', args.data_directory
    print 'checkpoints_directory     :', args.checkpoints_directory
    print 'Checkpoint every          :', args.ckpt_every
    print 'Batch size                :', args.batch_size
    print 'Embedding dim             :', args.embedding_dim
    print 'Memory dim                :', args.memory_dim
    print 'Layers                    :', args.num_layers
    print 'max_output_seq_len        :', args.max_output_seq_len
    print 'Epochs                    :', args.epochs
    print 'Resume at                 :', args.resume_at
    print 'Resume epoch              :', args.resume_epoch
    print 'Resume minibatch          :', args.resume_minibatch
    print 'cell type                 :', args.cell_type
    print 'dropout                   :', args.dropout
    print 'vram                      :', args.vram
    print 'network                   :', which_network

    dataset = load_data(args.data_directory)
    tl_dict = dataset.get_tl_dictionary()
    rev_tl_dict = dataset.get_rev_tl_dictionary()
    batch_size = args.batch_size
    acc_calc = Accuracy_calculator_for_deepfix(tl_dict['~'])
    get_all_accuracies = acc_calc.get_all_accuracies

    num_train, num_validation, num_test = dataset.data_size
    print 'Training:', num_train, 'examples', '\nValidation:', num_validation, 'examples', '\nTest:', num_test, 'examples'
    print 'vocabulary size:', dataset.vocabulary_size

    print '\n\n===================== initializing model =====================\n\n'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.vram)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    scope = which_network

    with tf.variable_scope(scope):
        seq2seq = seq2seq_model(dataset.vocabulary_size, args.embedding_dim,
                                args.max_output_seq_len,
                                cell_type=args.cell_type,
                                memory_dim=args.memory_dim,
                                num_layers=args.num_layers,
                                dropout=args.dropout,
                                scope=scope
                                )

    if args.resume_at == 0:
        sess.run(tf.global_variables_initializer())
    else:
        seq2seq.load_parameters(sess, os.path.join(
            args.checkpoints_directory, 'saved-model-attn-' + str(args.resume_at)))

    step = args.resume_at
    resume_minibatch = args.resume_minibatch
    best_overall_accuracy = 0

    def test_or_validate(which='valid'):
        epoch = t + 1
        loss, token_acc, repair_acc, localization_acc = [], [], [], []
        num_examples = num_validation if which == 'valid' else num_test

        for i in range(num_examples / batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size
            minibatch = i

            X, X_len, Y, Y_len = dataset.get_batch(start, end, which=which)

            loss_t, Y_hat, Y_targets = seq2seq.validate_step(
                sess, X, X_len, Y, Y_len)
            loss.append(loss_t)
            token_acc.append(get_accuracy(Y_targets, Y_hat, which='token'))

            repair_accuracy = 0
            localization_accuracy = 0

            for y_idx, (y, y_hat) in enumerate(zip(Y_targets, Y_hat)):
                localization_equality, fix_equality = get_all_accuracies(
                    y, y_hat)

                if localization_equality:
                    localization_accuracy += 1
                if fix_equality:
                    repair_accuracy += 1

            repair_acc.append(float(repair_accuracy) / float(batch_size))
            localization_acc.append(
                float(localization_accuracy) / float(batch_size))

        loss, token_acc, repair_accuracy, \
            localization_accuracy = np.mean(loss), np.mean(token_acc), np.mean(repair_acc), \
            np.mean(localization_acc)

        # Print epoch step and validation information
        print "[{}] Epoch: {}, Loss: {}, Token-level-acc: {}, loc-acc: {}, repair-acc: {}" .format(which,
                                                                                                   epoch, loss, token_acc, localization_accuracy, repair_accuracy)

        if (num_test > 0 and which == 'test') or (num_test == 0 and which == 'valid'):
            global best_overall_accuracy
            if repair_accuracy > best_overall_accuracy:
                best_overall_accuracy = repair_accuracy
                for each_file in glob.glob(os.path.join(args.checkpoints_directory, 'saved-model-attn-%d*' % step)):
                    copy(each_file, os.path.join(
                        args.checkpoints_directory, 'best'))

                update_best_ckpt(epoch=epoch, step=step, loss=loss, token_acc=token_acc, localization_accuracy=localization_accuracy,
                                 repair_accuracy=repair_accuracy)

                print "[Best Checkpoint] Checkpointed at Epoch %d, Minibatch %d." % (t + 1, 0)

    ##############################################################################

    # Training
    for t in range(args.resume_epoch, args.epochs):
        # Training
        start_time = time.time()
        train_loss = []

        for i in range(resume_minibatch, num_train / args.batch_size):
            start = i * args.batch_size
            end = (i + 1) * args.batch_size
            x, x_len, y, y_len = dataset.get_batch(start, end, which='train')

            loss = seq2seq.train_step(sess, x, x_len, y, y_len)

            train_loss.append(loss)

            # Print progress
            step += 1
            print "Step: {},\tMinibatch: {},\tEpoch: {},\tLoss: {}".format(step, i, t + float(i + 1) / (num_train / args.batch_size), train_loss[-1])

            # Checkpoint
            if step % args.ckpt_every == 0:
                seq2seq.save_parameters(sess, os.path.join(
                    args.checkpoints_directory, 'saved-model-attn'), global_step=step)
                print "[Checkpoint] Checkpointed at Epoch %d, Minibatch %d." % (t, i)

        train_loss = np.mean(train_loss, 0)
        resume_minibatch = 0

        # Checkpoint before going into validation/testing
        if step % args.ckpt_every != 0:
            seq2seq.save_parameters(sess, os.path.join(
                args.checkpoints_directory, 'saved-model-attn'), global_step=step)
            print "[Checkpoint] Checkpointed at Epoch {}, Minibatch {}.".format(t + 1, 0)

        print "End of Epoch: {}".format(t + 1)
        print "[Training] Loss: {}".format(train_loss)

        ##############################################################################

        test_or_validate('valid')
        test_or_validate('test')

        ##############################################################################

        print "[Time] Took {} minutes to run." .format((time.time() - start_time) / 60)

    sess.close()
    log.close()
