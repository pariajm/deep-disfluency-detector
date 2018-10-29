from __future__ import division
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import reader
from acnn import ACNN


# paths to input, output and result files
tf.flags.DEFINE_string("data_path", '', "Data source for the input and output files.")
tf.flags.DEFINE_string("checkpoint_dir", '', "Directory to save the checkpoints and training summaries")

# model hyper-parameters
tf.flags.DEFINE_integer("embed_dim", 290, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("num_filters", 120, "Number of filters per filter size")
tf.flags.DEFINE_string("conv1_filter_sizes", "12,7", "Comma-separated conv1 filter sizes")
tf.flags.DEFINE_string("conv2_filter_sizes", "10,6", "Comma-separated conv2 filter sizes")
tf.flags.DEFINE_string("conv3_filter_sizes", "8,5", "Comma-separated conv3 filter sizes")
tf.flags.DEFINE_float("dropout_keep_prob", 0.53, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.23, "L2 regularization lambda")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("embed_initial", 0.09, "The standard deviation of word embedding initializer")
tf.flags.DEFINE_float("weight_initial", 0.09, "The standard deviation of weight initializers")
tf.flags.DEFINE_integer("batch_size", 25, "Training batch size")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_integer("dev_batch_size", 143, "Dev batch size")

# other parameters
tf.flags.DEFINE_string("device_name", '/cpu:0', "Device name to be used in ACNN layer")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        Data Preparation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Loading training, dev and test data
print("Loading data...")

x_train, x_dev, x_test, y_train, y_dev, y_test, z_train, z_dev, z_test, max_length, vocab = reader.swi_data(FLAGS.data_path)

# evaluate model on dev data and save the model at the end of each epoch
evaluate_every = (len(x_train) - 1) // FLAGS.batch_size

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                           Training
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
graph = tf.Graph()
with graph.as_default():
    assert FLAGS.data_path, '`data_path` is missing.'
    assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = ACNN(
            max_length=max_length,
            num_classes=1,
            vocab_size=len(vocab.vocabulary_),
            embedding_size=FLAGS.embed_dim,
            conv1_filter_sizes=list(map(int, FLAGS.conv1_filter_sizes.split(","))),
            conv2_filter_sizes=list(map(int, FLAGS.conv2_filter_sizes.split(","))),
            conv3_filter_sizes=list(map(int, FLAGS.conv3_filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            embed_initial= FLAGS.embed_initial,
            weight_initial= FLAGS.weight_initial,
            device_name=FLAGS.device_name
        )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss
        loss_summary = tf.summary.scalar("loss", cnn.loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # ------------------------------------------------------------------------------
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print "                         EPOCH %s" % 1
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        cost = [] # saving loss of each training step
        e = [] # saving num of edited predictions in training set
        c = [] # saving num of correct edited predictions in training set
        g = [] # saving num of ground truth edited labels in training set
        def train_step(x_batch, y_batch, z_batch):
            """
            A single training step

            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.input_z: z_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              cnn.l2_reg_lambda: FLAGS.l2_reg_lambda,
              cnn.batch_size: FLAGS.batch_size

            }
            _, step, summaries, loss, nprediction, ncorrect, ntarget = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.nprediction, cnn.ncorrect, cnn.ntarget],
                 feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            train_summary_writer.add_summary(summaries, step)

            cost.append(loss)
            e.append(nprediction)
            c.append(ncorrect)
            g.append(ntarget)

        # ------------------------------------------------------------------------------
        cost_dev = [] # saving loss of each dev step
        e_dev = [] # saving num of edited predictions in dev set
        c_dev = [] # saving num of correct edited predictions in dev set
        g_dev = [] # saving num of ground truth edited labels in dev set
        def dev_step(x_batch, y_batch, z_dev, writer=None):
            """
            Evaluates model on a dev set

            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_z: z_dev,
                cnn.dropout_keep_prob: 1.0,
                cnn.l2_reg_lambda: 0.0,
                cnn.batch_size: FLAGS.dev_batch_size
            }
            step, summaries, loss, dev_nprediction, dev_ncorrect, dev_ntarget = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.nprediction, cnn.ncorrect, cnn.ntarget],
                feed_dict)

            cost_dev.append(loss)
            e_dev.append(dev_nprediction)
            c_dev.append(dev_ncorrect)
            g_dev.append(dev_ntarget)
            if writer:
                writer.add_summary(summaries, step)

        # ------------------------------------------------------------------------------
        cost_test = [] # saving loss of each test step
        e_test = [] # saving num of edited predictions in test set
        c_test = [] # saving num of correct edited predictions in test set
        g_test = [] # saving num of ground truth edited labels in test set
        def test_step(x_batch, y_batch, z_dev):
            """
            Evaluates model on a test set

            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_z: z_dev,
                cnn.dropout_keep_prob: 1.0,
                cnn.l2_reg_lambda: 0.0,
                cnn.batch_size: len(x_test)
                }
            loss, test_nprediction, test_ncorrect, test_ntarget = sess.run(
                [cnn.loss, cnn.nprediction, cnn.ncorrect, cnn.ntarget],
                feed_dict)
            cost_test.append(loss)
            e_test.append(test_nprediction)
            c_test.append(test_ncorrect)
            g_test.append(test_ntarget)

        #------------------------------------------------------------------------------
        # Generate batches
        batches = reader.swi_minibatches(
            x_train, y_train, z_train, FLAGS.batch_size, FLAGS.num_epochs, max_length, shuffle=True)

        # Training loop. For each batch...
        epoch_counter = 1
        for batch in batches:
            x_batch = batch[0]
            y_batch = batch[1]
            z_batch = batch[2]

            train_step(x_batch, y_batch, z_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print ">>> EPOCH %s: Train Loss %f, Precision %f, Recall %f, F-score %f" % \
                      (epoch_counter, np.mean(cost), sum(c) / (sum(e) + (1e-100)),
                       sum(c) / (sum(g) + (1e-100)), (2 * sum(c)) / (sum(g) + sum(e) + 1e-100))
                print len(cost), 'len'
                cost = []
                e = []
                c = []
                g = []
                # ------------------------------------------------------------------------------
                # Evaluating the model on the dev set at the end of each training epoch:
                print("\nEvaluation:")
                dev_batches = reader.swi_minibatches(
                    x_dev, y_dev, z_dev, FLAGS.dev_batch_size, num_epochs=1, max_length=max_length, shuffle=False)

                for batch_num in dev_batches:
                    x_dev_batch = batch_num[0]
                    y_dev_batch = batch_num[1]
                    z_dev_batch = batch_num[2]

                    dev_step(x_dev_batch, y_dev_batch, z_dev_batch, writer=dev_summary_writer)

                print "\n >>> EPOCH %s: Evaluation Loss %f,  Precision %f, Recall %f, F-score %f" %\
                      (epoch_counter, np.mean(cost_dev),
                       sum(c_dev) / (sum(e_dev) + 1e-100), sum(c_dev) / (sum(g_dev) + 1e-100),
                       (2 * sum(c_dev)) / (sum(g_dev) + sum(e_dev) + 1e-100))

                cost_dev = []
                e_dev = []
                c_dev = []
                g_dev = []
                epoch_counter += 1

                print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                print "                         EPOCH %s" % (epoch_counter)
                print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"


            if current_step % evaluate_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


        # Evaluating the model on the test set at the end of training epochs:
        print("\nTest:")
        test_batches = reader.swi_minibatches(x_test, y_test, z_test, batch_size=1, num_epochs=1, max_length=max_length, shuffle=False)

        for batch_num in test_batches:
            x_test = batch_num[0]
            y_test = batch_num[1]
            z_test = batch_num[2]

            test_step(x_test, y_test, z_test)

        print ">>>Test Loss %f, Precision %f, Recall %f, F-score %f" % \
                (np.mean(cost_test),
                sum(c_test) / (sum(e_test) + 1e-100), sum(c_test) / (sum(g_test) + 1e-100),
                 (2 * sum(c_test)) / (sum(g_test) + sum(e_test) + 1e-100))
