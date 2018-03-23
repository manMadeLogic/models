import tensorflow as tf
from collections import Counter
import os
import re
import sys
import random

train_pos = '/Users/Boya/Desktop/Courses/DL/aclImdb/train/pos'
train_neg = '/Users/Boya/Desktop/Courses/DL/aclImdb/train/neg'
# train_unsup = '/Users/Boya/Desktop/Courses/DL/aclImdb/train/unsup'

filenames = [os.path.join(train_pos, f) for f in os.listdir(train_pos)]
filenames += [os.path.join(train_neg, f) for f in os.listdir(train_neg)]
# filenames += [os.path.join(train_unsup, f) for f in os.listdir(train_unsup)]

train_filename = '/Users/Boya/Desktop/Courses/DL/aclImdb/train_lm.tfrecords'
test_filename = '/Users/Boya/Desktop/Courses/DL/aclImdb/test_lm.tfrecords'
vocab_file = '/Users/Boya/Desktop/Courses/DL/aclImdb/vocab.txt'


def write_to_tfrecords(train, label, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    train_label = zip(train, label)
    for i, (review, label) in enumerate(train_label):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(train)))
            sys.stdout.flush()

        token_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])) for idx in review]
        feature_list = {
            'token_id': tf.train.FeatureList(feature=token_features)
        }
        context = {
            'feature': {
                'class': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
        }
        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        example = tf.train.SequenceExample(feature_lists=feature_lists, context=context)

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def load_data(fnames):
    all_words = []
    reviews = []
    train = []
    classes = []
    for f in fnames:
        with open(f, 'r') as file:
            review = file.read().strip()
        words = re.sub( '\s+', ' ', review).strip().split(' ')# convert multiple spaces to one
        words = words[:40]
        all_words += words
        reviews.append(words)
        label = int(f.split('/')[-2] == 'pos')
        classes.append(label)

    idx2word = [element[0] for element in Counter(all_words).most_common()]
    dir_words = dict((word, idx) for idx, word in enumerate(idx2word))

    with open(vocab_file, 'w') as f:
        f.write('\n'.join(idx2word))

    for words in reviews:
        train.append([dir_words[word] for word in words])

    return train, classes

train, classes = load_data(filenames)
training = list(zip(train, classes))
random.shuffle(training)
train, classes = zip(*training)
train = list(train)
classes = list(classes)
train_size = int(len(train) * 0.9)
write_to_tfrecords(train[:train_size], classes[:train_size], train_filename)
write_to_tfrecords(train[train_size:], classes[:train_size], test_filename)
