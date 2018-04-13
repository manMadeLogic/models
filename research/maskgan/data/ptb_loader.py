# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""PTB data loader and helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
# Dependency imports
import numpy as np

import tensorflow as tf

EOS_INDEX = 0


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", " <eos>").split()


def build_vocab(filename):
  data = _read_words(filename)
  # print(data)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  print("<eos>:", word_to_id["<eos>"])
  global EOS_INDEX
  EOS_INDEX = word_to_id["<eos>"]

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_iterator(raw_data, batch_size, sequence_length, epoch_size_override=None):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    sequence_length: int, the number of unrolls. sequence_size

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  # data_len = len(raw_data)
  # batch_len = data_len // batch_size
  # data = np.full([batch_size, batch_len], EOS_INDEX, dtype=np.int32)
  # for i in range(batch_size):
  #   data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  #
  # if epoch_size_override:
  #   epoch_size = epoch_size_override
  # else:
  #   epoch_size = (batch_len - 1) // sequence_length
  #
  # if epoch_size == 0:
  #   raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  # print("Number of batches per epoch: %d" % epoch_size)
  # for i in range(epoch_size):
  #   x = data[:, i * sequence_length:(i + 1) * sequence_length]
  #   y = data[:, i * sequence_length + 1:(i + 1) * sequence_length + 1]
  #   w = np.ones_like(x)
  #   yield (x, y, w)

  # print('raw_data', len(raw_data))
  sentences = np.split(raw_data, np.where(raw_data == EOS_INDEX)[0] + 1)
  # print(raw_data)
  # print(EOS_INDEX)
  # print(sentences)
  sentence_len = len(sentences)
  print('sentence_len', sentence_len)
  data = np.full([sentence_len, sequence_length+1], EOS_INDEX, dtype=np.int32)
  for i in range(sentence_len):
    sent = sentences[i][:sequence_length+1]
    data[i][:len(sent)] = sent

  if epoch_size_override:
    raise NotImplementedError

  epoch_size = sentence_len//batch_size
  print('epoch_size', epoch_size)
  for i in range(epoch_size):
    x = data[i*batch_size:(i+1)*batch_size, :-1]
    y = data[i*batch_size:(i+1)*batch_size, 1:]
    w = np.ones_like(x)
    print("for loop")
    print(x, y, w)
    yield (x, y, w)



if __name__ == '__main__':
#   data_path = '/home/chenxi410402/tmp/ptb'
  data_path = '/Users/xi/Downloads/ptb/'
  train_path = os.path.join(data_path, "ptb.train.txt")
  word_to_id = build_vocab(train_path)

  # path = '/home/chenxi410402/tmp/ptb'
  # train_data, valid_data, test_data, vocabulary = ptb_raw_data(path)
  # word_to_id = build_vocab()
  # iterator = ptb_iterator(train_data, 10, 20)
  # for x, y, _ in iterator:
  #   print(x)
  #   print(y)
  #   break
