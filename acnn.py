from __future__ import division
import tensorflow as tf

class ACNN(object):
    """

    An ACNN for disfluency detection:
    includes an embedding layer, followed by a drop-out layer, an auto-correlational layer, two convolutional layers and a sigmoid layer.

    """
    def __init__(self, max_length, num_classes, vocab_size, embedding_size, conv1_filter_sizes,
                 conv2_filter_sizes, conv3_filter_sizes, num_filters, embed_initial, weight_initial, device_name):

        # Placeholders for input, output, mask, dropout, batch_size and l2_regularization
        self.input_x = tf.placeholder(tf.int32, [None, max_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_z = tf.placeholder(tf.float32, [None, 1], name="input_z")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_reg_lambda = tf.placeholder(tf.float32, name="l2_reg_lambda")
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                        Embedding Layer
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embed_weights = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -embed_initial, embed_initial), name="embed_weights")
            embedded_words = tf.nn.embedding_lookup(embed_weights, self.input_x)
            embedded_words_expanded = tf.expand_dims(embedded_words, -1)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                        Drop-out Layer
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(embedded_words_expanded, self.dropout_keep_prob)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                        Auto-Correlation Layer
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The auto-correlation layer includes two parts:
            1. An auto-correlated tensor is constructed by comparing each input vector u with the input vector v using a
            binary function f. The auto-correlated tensor is then convolved with 3D or 4D kernels B of different sizes.

            2. A vanilla CNN layer which convolves the input tensor with kernels A of different sizes.

            Each kernel group A and B outputs a tensor of the same size which are added element-wise to produce the feature
            representation that is passed to further convolutional layers. For more details, read https://www.aclweb.org/anthology/D18-1490.pdf.
        """

        combine_acnn_features = []
        for i, filter_size in enumerate(conv1_filter_sizes):
            # First part of the ACNN layer:
            with tf.device(device_name), tf.name_scope("auto-correlation"):
                # kernel_B_shape = [conv1_filter_sizes[i], conv1_filter_sizes[i], embedding_size, num_filters] # 4D kernels --> give better results
                kernel_B_shape = [conv1_filter_sizes[i], conv1_filter_sizes[i], num_filters] # 3D kernels
                kernel_B = tf.Variable(tf.truncated_normal(kernel_B_shape, stddev=weight_initial), name="w")
                flat_kernel_B = tf.reshape(kernel_B, (-1, num_filters))
                patches = tf.extract_image_patches(self.h_drop,
                                                     ksizes=[1, conv1_filter_sizes[i], embedding_size, 1],
                                                     strides=[1, 1, embedding_size, 1],
                                                     rates=[1, 1, 1, 1],
                                                     padding="SAME",
                                                     name="patches")
                reshaped_pathes = tf.reshape(patches, [-1, conv1_filter_sizes[i], embedding_size])
                # function_f = tf.einsum('ijl,ikl->ikjl', reshaped_pathes, reshaped_pathes) # function to be used for 4D kernels
                function_f = tf.einsum('ijl,ikl->ijk', reshaped_pathes, reshaped_pathes) # function to be used for 3D kernels
                reshaped_function_f = tf.reshape(function_f, [self.batch_size * max_length, -1])
                auto_correlated_input = tf.reshape(tf.matmul(reshaped_function_f, flat_kernel_B, name='auto_cor'),
                                                   (self.batch_size, max_length, 1, -1))

            # Second part of the ACNN layer:
            with tf.name_scope("conv1"):
                kernel_A_shape = [conv1_filter_sizes[i], embedding_size, 1, num_filters]
                kernel_A = tf.Variable(tf.truncated_normal(kernel_A_shape, stddev=weight_initial), name="w")
                b_conv1 = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
                convolved_input = tf.nn.conv2d(
                    self.h_drop,
                    kernel_A,
                    strides=[1, 1, embedding_size, 1],
                    padding="SAME",
                    name="conv1")

            # Here the outputs of first and second parts are added element-wise:
            added_outputs = tf.add(auto_correlated_input, convolved_input)
            auto_correlation = tf.nn.relu(tf.nn.bias_add(added_outputs, b_conv1), name="relu1")
            combine_acnn_features.append(auto_correlation)

        # Combine all the acnn features
        num_filters_total = num_filters * len(conv1_filter_sizes)
        all_acnn_features = tf.concat(combine_acnn_features, 3)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                        Convolutional Layer
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        combine_conv2_features = []
        conv2_input = tf.reshape(all_acnn_features, [-1, max_length, num_filters_total, 1])
        for j, filter_size in enumerate(conv2_filter_sizes):
            with tf.name_scope("conv2"):
                kernel_conv2_shape = [conv2_filter_sizes[j], num_filters_total, 1, num_filters]
                kernel_conv2 = tf.Variable(tf.truncated_normal(kernel_conv2_shape, stddev=weight_initial), name="w")
                b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv_2_output = tf.nn.conv2d(
                    conv2_input,
                    kernel_conv2,
                    strides=[1, 1, num_filters_total, 1],
                    padding="SAME",
                    name="conv2")
            conv2 = tf.nn.relu(tf.nn.bias_add(conv_2_output, b_conv2), name="relu2")
            combine_conv2_features.append(conv2)

        # Combine all the conv2 features
        num_filters_total = num_filters * len(conv2_filter_sizes)
        all_conv2_features = tf.concat(combine_conv2_features, 3)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                        Convolutional Layer
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        combine_conv3_features = []
        conv3_input = tf.reshape(all_conv2_features, [-1, max_length, num_filters_total, 1])
        for k, filter_size in enumerate(conv3_filter_sizes):
            with tf.name_scope("conv3"):
                kernel_conv3_shape = [conv3_filter_sizes[k], num_filters_total, 1, num_filters]
                kernel_conv3 = tf.Variable(tf.truncated_normal(kernel_conv3_shape, stddev=weight_initial), name="w")
                b_conv3 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv3_output = tf.nn.conv2d(
                    conv3_input,
                    kernel_conv3,
                    strides=[1, 1, num_filters_total, 1],
                    padding="SAME",
                    name="conv3")
            conv3 = tf.nn.relu(tf.nn.bias_add(conv3_output, b_conv3), name="relu3")
            combine_conv3_features.append(conv3)

        # Combine all the conv3 features
        num_filters_total = num_filters * len(conv3_filter_sizes)
        all_conv3_features = tf.concat(combine_conv3_features, 3)
        reshaped_conv3_features = tf.reshape(all_conv3_features, [-1, num_filters_total])

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                         1-Width Convolution
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        l2_loss = tf.constant(0.0) # keeping track of l2 regularization loss
        with tf.name_scope("local4"):
            W = tf.get_variable("ww", shape=[num_filters_total, 128], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[128]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            local4 = tf.nn.relu(tf.matmul(reshaped_conv3_features, W) + b, name="local4")

        # Final scores and predictions
        with tf.name_scope("output"):
            w = tf.get_variable(
                "w",
                shape=[128, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(w)
            scores = tf.nn.xw_plus_b(local4, w, b, name="scores")
            probs = tf.sigmoid(scores, name="sigmoid") # using sigmoid to convert scores to probabilities
            condition = tf.less(probs, tf.fill(tf.shape(probs), 0.5))
            self.predictions = tf.where(condition, tf.zeros(tf.shape(probs)), tf.ones(tf.shape(probs))) # if prob >= 0.5: 1 (i.e. disfluent);
                                                                                                        # else : 0 (i.e. fluent)

        # Calculate sigmoid cross entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            masked_losses = tf.transpose(self.input_z) * losses
            self.loss = (tf.reduce_sum(masked_losses) / tf.cast(self.batch_size, "float32")) + (self.l2_reg_lambda * l2_loss)

        # Calculate f-score
        with tf.name_scope("fscore"):
            fscore_mask = tf.cast(self.input_z, "int64")
            predictions = tf.cast(self.predictions, "int64")
            input_y = tf.cast(self.input_y, "int64")

            # e: #edited predictions
            masked_prediction = fscore_mask * predictions
            e = tf.reduce_sum(masked_prediction)
            self.nprediction = tf.to_int32(e, name="nprediction")

            # g: #edited words in gold set
            masked_input_y = fscore_mask * input_y
            g = tf.reduce_sum(masked_input_y)
            self.ntarget = tf.to_int32(g, name="ntarget")

            # c: #correct proposed edited words
            c = tf.count_nonzero(masked_input_y * masked_prediction)
            self.ncorrect = tf.to_int32(c, name="ncorrect")







