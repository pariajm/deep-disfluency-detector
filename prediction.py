#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import reader
from tensorflow.contrib import learn

tf.flags.DEFINE_string("input_path", '', "Data source for input file")
tf.flags.DEFINE_string("checkpoint_dir", '', "Checkpoint dir from training run")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("output_path", '', "Directory to save results")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

# Load input data:
input_data = reader._read_data(FLAGS.input_path)

# Map data into vocabulary:
vocab_path = os.path.join(FLAGS.checkpoint_dir, '', "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
input_id = np.array(list(vocab_processor.transform(input_data)))

# Create result file:
output_file = open(os.path.join(FLAGS.output_path, "results.txt"),'w')

# Evaluation:
print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(os.path.join(
    FLAGS.checkpoint_dir, "checkpoints")
    )
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables:
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name:
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
       
        # Tensors we want to evaluate:
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches:
        sn_length = reader._length(input_data)
        max_length = max(sn_length)
        mask = reader._mask(input_data, max_length)
        batches = reader.batch_iter(input_id, max_length, mask)
        batch_size = graph.get_operation_by_name("batch_size").outputs[0]
            
        # Collect the predictions:  
        indx = 0
        for batch in batches:
            x_batch = batch[0]
            z_batch = batch[1]
            batch_predictions = sess.run(
                predictions, {input_x: x_batch, batch_size: 1, dropout_keep_prob: 1.0}
                )    
            words = input_data[indx].split()
            # "E" stands for disfluent words and "F" for fluent words:
            for i in range(sn_length[indx]):
                label = 'E' if batch_predictions[i] == 1 else 'F'
                output_file.write(words[i]+' '+label+' ')
            output_file.write('\n')
            indx += 1
     
print('\nEvaluation Done!')
