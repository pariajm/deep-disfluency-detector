"""

Utilities for reading Switchboard files

"""

import numpy as np
from tensorflow.contrib import learn
import os


def _read_data(filename):
    with open(filename) as fp:
        lines = fp.readlines()
        return lines


def _length(sentences):
    sn_length = [len(sn.split()) for sn in sentences]
    return sn_length


def _mask(sentences, max_length):
    """
    - Builds a mask array to ignore padded integers for calculating precision, recall and fscore

    Args:
         sentences: a list of input sentences
         max_length: maximum length used for padding sentences

    Returns:
        mask_array: an array of actual length of sentences

    """
    sn_length = _length(sentences)
    mask_array = np.zeros((len(sn_length) * max_length, 1), dtype=np.float64)
    row_num = 0
    for length in sn_length:
        mask_array[row_num:length+row_num] = 1
        row_num += length + (max_length - length)
    return mask_array


def swbd_data(data_path=None):
    """
    - Loads Switchboard input and output files from data dir "./data_path",

    - Then, reads Switchboard files and converts strings to integer ids,

    - Finally, creates mask arrays for input files.

    Args:
        data_path: string path to the dir where train, dev and test input and output files are stored
        (check out ./sample_data for the input format)

    Returns:
        tuple (
               train_input_ids, 
               dev_input_ids, 
               test_input_ids,
               train_output_ids, 
               dev_output_ids, 
               test_output_ids,
               train_mask, 
               dev_mask, 
               test_mask,
               max_length, 
               input_vocab_processor
        ): where each of the data objects can be passed to swbd_minibathes
    """
    train_input_data = _read_data(os.path.join(data_path, "swbd.train.txt"))
    dev_input_data = _read_data(os.path.join(data_path, "swbd.dev.txt"))
    test_input_data = _read_data(os.path.join(data_path, "swbd.test.txt"))

    max_length = max(_length(train_input_data))

    input_vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=1)
    train_input_ids = np.array(list(input_vocab_processor.fit_transform(train_input_data)))
    dev_input_ids = np.array(list(input_vocab_processor.transform(dev_input_data)))
    test_input_ids = np.array(list(input_vocab_processor.transform(test_input_data)))

    train_output_data = _read_data(os.path.join(data_path, "swbd.train.label.txt"))
    dev_output_data = _read_data(os.path.join(data_path, "swbd.dev.label.txt"))
    test_output_data = _read_data(os.path.join(data_path, "swbd.test.label.txt"))

    label_vocab = {'F': 0, 'E': 1}
    output_vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, vocabulary=label_vocab)
    train_output_ids = np.array(list(output_vocab_processor.transform(train_output_data)))
    dev_output_ids = np.array(list(output_vocab_processor.transform(dev_output_data)))
    test_output_ids = np.array(list(output_vocab_processor.transform(test_output_data)))

    train_mask = _mask(train_input_data, max_length)
    dev_mask = _mask(dev_input_data, max_length)
    test_mask = _mask(test_input_data, max_length)

    return train_input_ids, \
        dev_input_ids, \
        test_input_ids, \
        train_output_ids, \
        dev_output_ids, \
        test_output_ids, \
        train_mask, \
        dev_mask, \
        test_mask, \
        max_length, \
        input_vocab_processor


def swbd_minibatches(input_ids, output_ids, mask_data, batch_size, num_epochs, max_length, shuffle=True):
    """
    - Iterates on the Switchboard input and output files

    Args:
        input_ids: one of the input id files from swbd_data
        output_ids: one of the output id files from swbd_data
        mask_data: one of the mask files from swbd_data
        batch_size: int, the batch size
        num_epochs: int, the number of training epochs
        max_length: int, the maximum length used for padding
        shuffle: Boolean, whether to shuffle training data or not

    Returns:
        tuple (x, y, z): which are minibathes of (input, output, mask)
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

    data_size = len(input_ids) // max_length
    num_batches_per_epoch = data_size // batch_size
     for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = (batch_num * batch_size) * max_length
            end_index = (min((batch_num + 1) * batch_size, data_size)) * max_length
            x = np.reshape(input_ids[start_index:end_index], (batch_size, max_length))
            y = output_ids[start_index:end_index]
            z = mask_data[start_index:end_index]
            yield (x, y, z)
            
            
def batch_iter(input_id, max_length, mask):
    """
    - Iterates on input data (usef for prediction)

    Args:
        input_id: list of input sentences mapped to integers
        max_length: maximum length of sentences
        mask: list of actual length of sentences
     
    Returns:
        tuple (x_input, z_mask): which are minibathes of (input, mask)
    """

    x = np.array(input_id)
    for sn in range(len(input_id)):
        start = sn * max_length
        end = (1 + sn) * max_length
        x_input = x[sn : sn + 1]
        z_mask = mask[start:end]
        yield (x_input, z_mask)
 
