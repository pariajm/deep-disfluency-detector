"""Utilities for reading Switchboard files."""
import numpy as np
from tensorflow.contrib import learn
import os


def _read_data(filename):
    with open(filename, 'r') as fp:
        lines = fp.read().split('\n')
        return lines


def _length(sentences):
    sent_length = [len(sen.split()) for sen in sentences]
    return sent_length


def _mask(sentences, max_length):
    """
    Build a mask array to ignore padded integers for calculating precision, recall and fscore.

    Args:
         sentences: a list of input sentences.
         max_length: maximum length used for padding sentences.

    Returns:
        mask_array
        an array of actual length of sentences.

    """
    sent_length = _length(sentences)
    mask_array = np.zeros((len(sent_length)*max_length, 1), dtype=np.float64)
    row_num = 0
    for length in sent_length:
        mask_array[row_num:length+row_num] = 1
        row_num += length + (max_length - length)
    return mask_array


def swi_data(data_path=None):
    """
    Load Switchboard input and output files from data directory "data_path".

    Read Switchboard files, convert strings to integer ids.

    Create mask arrays for input files.

    Args:
        data_path: string path to the directory where train, dev and test input and output files are stored.

    Returns:
        tuple (train_input_ids, dev_input_ids, test_input_ids,
               train_output_ids, dev_output_ids, test_output_ids,
               train_mask, dev_mask, test_mask,
               max_length, input_vocab_processor)
        where each of the data objects can be passed to swi_minibathes.
    """
    train_input_data = _read_data(os.path.join(data_path, "swi.train.txt"))
    dev_input_data = _read_data(os.path.join(data_path, "swi.dev.txt"))
    test_input_data = _read_data(os.path.join(data_path, "swi.test.txt"))

    max_length = max(_length(train_input_data))

    input_vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=1)
    train_input_ids = np.array(list(input_vocab_processor.fit_transform(train_input_data)))
    dev_input_ids = np.array(list(input_vocab_processor.transform(dev_input_data)))
    test_input_ids = np.array(list(input_vocab_processor.transform(test_input_data)))


    train_output_data = _read_data(os.path.join(data_path, "swi.train.label.txt"))
    dev_output_data = _read_data(os.path.join(data_path, "swi.dev.label.txt"))
    test_output_data = _read_data(os.path.join(data_path, "swi.test.label.txt"))

    # label_vocab = {'_': 0, 'E': 1}
    label_vocab = {'F': 0, 'E': 1}
    output_vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, vocabulary=label_vocab)
    train_output_ids = np.array(list(output_vocab_processor.transform(train_output_data)))
    dev_output_ids = np.array(list(output_vocab_processor.transform(dev_output_data)))
    test_output_ids = np.array(list(output_vocab_processor.transform(test_output_data)))

    train_mask = _mask(train_input_data, max_length)
    dev_mask = _mask(dev_input_data, max_length)
    test_mask = _mask(test_input_data, max_length)

    return train_input_ids, dev_input_ids, test_input_ids, \
           train_output_ids, dev_output_ids, test_output_ids, \
           train_mask, dev_mask, test_mask, \
           max_length, input_vocab_processor


def swi_minibatches(input_ids, output_ids, mask_data, batch_size, num_epochs, max_length, shuffle=True):
    """
    Iterate on the Switchboard input and output files.

    Args:
        input_ids: one of the input id files from swi_data.
        output_ids: one of the output id files from swi_data.
        mask_data: one of the mask files from swi_data.
        batch_size: int, the batch size.
        num_epochs: int, the number of training epochs.
        max_length: int, the maximum length used for padding.
        shuffle: Boolean, whether to shuffle training data or not.

    Returns:
        tuple (x, y, z)
        which are minibathes of (input, output, mask)
    """

    output_ids = np.reshape(np.array(output_ids), (-1, max_length))
    mask_data = np.reshape(np.array(mask_data), (-1, max_length))

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(len(input_ids)))
        input_ids = input_ids[shuffle_indices]
        output_ids = output_ids[shuffle_indices]
        mask_data = mask_data[shuffle_indices]

    input_ids = np.array([np.concatenate(input_ids, 0)]).T
    output_ids = np.array([np.concatenate(output_ids, 0)]).T
    mask_data = mask_data.reshape(-1, 1)

    data_size = len(input_ids)/max_length
    num_batches_per_epoch = data_size / batch_size

    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = (batch_num * batch_size) * max_length
            end_index = (min((batch_num + 1) * batch_size, data_size)) * max_length
            x = np.reshape(input_ids[start_index:end_index], (batch_size, max_length))
            y = output_ids[start_index:end_index]
            z = mask_data[start_index:end_index]

            yield (x, y, z)


