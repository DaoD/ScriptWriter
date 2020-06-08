import tensorflow as tf
import numpy as np


def normalize(inputs, epsilon=1e-8, scope="normalize", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable(name='beta', shape=params_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        gamma = tf.get_variable(name='scale', shape=params_shape, dtype=tf.float32, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta
    return outputs


def embedding(inputs, vocab_size=None, embedding_size=None, zero_pad=False, scale=False, scope="embedding", reuse=None, initializer=None):
    with tf.variable_scope(scope, reuse=reuse):
        if initializer:
            lookup_table = initializer
        else:
            lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, embedding_size]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (embedding_size ** 0.5)
    return outputs


def positional_encoding(inputs, num_units, max_len, zero_pad=True, scale=True, scope="positional_encoding", reuse=None):
    # N, T = inputs.get_shape().as_list()
    # N, T = tf.shape(inputs)
    inputs_shape = tf.shape(inputs)
    N = inputs_shape[0]
    T_real = inputs_shape[1]
    T = max_len
    with tf.variable_scope(scope, reuse=reuse, dtype=tf.float32):
        position_ind = tf.tile(tf.expand_dims(tf.range(T_real), 0), [N, 1])
        position_enc = np.array([[pos / np.power(10000, 2 * (i // 2) /num_units) for i in range(num_units)] for pos in range(T)], dtype=np.float32)
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        lookup_table = tf.convert_to_tensor(position_enc)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        if scale:
            outputs = outputs * (num_units ** 0.5)
    return outputs


def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0, is_training=True, causality=False, scope="multihead_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        attn_weights = outputs
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        outputs += queries
        outputs = normalize(outputs, scope=scope)  # (N, T_q, C)

    return outputs, attn_weights


def feedforward(inputs, num_units=[2048, 512], scope="feed_forward", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs += inputs
        outputs = normalize(outputs, scope=scope)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)

