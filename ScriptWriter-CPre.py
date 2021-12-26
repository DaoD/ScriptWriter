from modules import *
from keras.preprocessing.sequence import pad_sequences
import Utils
import Evaluate
import pickle
import os
from tqdm import tqdm
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# for train
embedding_file = "./data/embeddings.pkl"
train_file = "./data/train.gr.pkl"
val_file = "./data/dev.gr.pkl"
evaluate_file = "./data/test.gr.pkl"

save_path = "./model/cpre/"
result_path = "./output/cpre/"
log_path = "./model/cpre/"

max_sentence_len = 50
max_num_utterance = 11
batch_size = 50
eval_batch_size = 100

class ScriptWriter_cpre():
    def __init__(self, eta=0.5):
        self.max_num_utterance = max_num_utterance
        self.negative_samples = 1
        self.max_sentence_len = max_sentence_len
        self.word_embedding_size = 200
        self.hidden_units = 200
        self.total_words = 43514
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.dropout_rate = 0
        self.num_heads = 1
        self.num_blocks = 3
        self.eta = eta
        self.gamma = tf.get_variable('gamma', shape=1, dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(0.5))

        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, max_num_utterance, max_sentence_len))
        self.response_ph = tf.placeholder(tf.int32, shape=(None, max_sentence_len))
        self.gt_response_ph = tf.placeholder(tf.int32, shape=(None, max_sentence_len))
        self.y_true_ph = tf.placeholder(tf.int32, shape=(None,))
        self.narrative_ph = tf.placeholder(tf.int32, shape=(None, max_sentence_len))

        self.word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.word_embedding_size), dtype=tf.float32, trainable=False)
        self.embedding_init = self.word_embeddings.assign(self.embedding_ph)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.is_training = True
        print("current eta: ", self.eta)

    def load(self, previous_modelpath):
        sess = tf.Session()
        latest_ckpt = tf.train.latest_checkpoint(previous_modelpath)
        # print("recover from checkpoint: " + latest_ckpt)
        variables = tf.contrib.framework.get_variables_to_restore()
        saver = tf.train.Saver(variables)
        saver.restore(sess, latest_ckpt)
        return sess

    def build(self):
        all_utterances = tf.unstack(self.utterance_ph, num=self.max_num_utterance, axis=1)
        reuse = None
        alpha_1, alpha_2 = None, None

        response_embeddings = embedding(self.response_ph, initializer=self.word_embeddings)
        Hr_stack = [response_embeddings]
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                response_embeddings, _ = multihead_attention(queries=response_embeddings, keys=response_embeddings, num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                response_embeddings = feedforward(response_embeddings, num_units=[self.hidden_units, self.hidden_units])
                Hr_stack.append(response_embeddings)

        gt_response_embeddings = embedding(self.gt_response_ph, initializer=self.word_embeddings)
        Hgtr_stack = [gt_response_embeddings]
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=True):
                gt_response_embeddings, _ = multihead_attention(queries=gt_response_embeddings, keys=gt_response_embeddings, num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                gt_response_embeddings = feedforward(gt_response_embeddings, num_units=[self.hidden_units, self.hidden_units])
                Hgtr_stack.append(gt_response_embeddings)

        narrative_embeddings = embedding(self.narrative_ph, initializer=self.word_embeddings)
        Hn_stack = [narrative_embeddings]
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=True):
                narrative_embeddings, _ = multihead_attention(queries=narrative_embeddings, keys=narrative_embeddings, num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                narrative_embeddings = feedforward(narrative_embeddings, num_units=[self.hidden_units, self.hidden_units])
                Hn_stack.append(narrative_embeddings)

        Mur, Mun = [], []
        self.decay_factor = []
        last_u_reps = []
        turn_id = 0
        for utterance in all_utterances:
            utterance_embeddings = embedding(utterance, initializer=self.word_embeddings)
            Hu_stack = [utterance_embeddings]
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=True):
                    utterance_embeddings, _ = multihead_attention(queries=utterance_embeddings, keys=utterance_embeddings, num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                    utterance_embeddings = feedforward(utterance_embeddings, num_units=[self.hidden_units, self.hidden_units])
                    Hu_stack.append(utterance_embeddings)

            if turn_id == self.max_num_utterance - 1:
                last_u_reps = Hu_stack

            r_a_u_stack = []
            u_a_r_stack = []

            for i in range(self.num_blocks + 1):
                with tf.variable_scope("utterance_attention_response_{}".format(i), reuse=reuse):
                    u_a_r, _ = multihead_attention(queries=Hu_stack[i], keys=Hr_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                    u_a_r = feedforward(u_a_r, num_units=[self.hidden_units, self.hidden_units])
                    u_a_r_stack.append(u_a_r)
                with tf.variable_scope("response_attention_utterance_{}".format(i), reuse=reuse):
                    r_a_u, _ = multihead_attention(queries=Hr_stack[i], keys=Hu_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                    r_a_u = feedforward(r_a_u, num_units=[self.hidden_units, self.hidden_units])
                    r_a_u_stack.append(r_a_u)
            u_a_r_stack.extend(Hu_stack)
            r_a_u_stack.extend(Hr_stack)

            n_a_u_stack = []
            u_a_n_stack = []
            for i in range(self.num_blocks + 1):
                with tf.variable_scope("narrative_attention_response_{}".format(i), reuse=reuse):
                    n_a_u, _ = multihead_attention(queries=Hn_stack[i], keys=Hu_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                    n_a_u = feedforward(n_a_u, num_units=[self.hidden_units, self.hidden_units])
                    n_a_u_stack.append(n_a_u)
                with tf.variable_scope("response_attention_narrative_{}".format(i), reuse=reuse):
                    u_a_n, alpha_1 = multihead_attention(queries=Hu_stack[i], keys=Hn_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                    u_a_n = feedforward(u_a_n, num_units=[self.hidden_units, self.hidden_units])
                    u_a_n_stack.append(u_a_n)
            n_a_u_stack.extend(Hn_stack)
            u_a_n_stack.extend(Hu_stack)

            u_a_r = tf.stack(u_a_r_stack, axis=-1)
            r_a_u = tf.stack(r_a_u_stack, axis=-1)
            u_a_n = tf.stack(u_a_n_stack, axis=-1)
            n_a_u = tf.stack(n_a_u_stack, axis=-1)

            with tf.variable_scope('similarity'):
                # sim shape [batch, max_sent_len, max_sent_len, 2 * (stack_num + 1)]
                sim_ur = tf.einsum('biks,bjks->bijs', u_a_r, r_a_u) / tf.sqrt(200.0)  # for no rp and normal
                sim_un = tf.einsum('biks,bjks->bijs', u_a_n, n_a_u) / tf.sqrt(200.0)  # for no rp and normal

            self_n = tf.nn.l2_normalize(tf.stack(Hn_stack, axis=-1))  # #for no rp
            self_u = tf.nn.l2_normalize(tf.stack(Hu_stack, axis=-1))  # #for no rp
            Hn_stack_tensor = tf.stack(Hn_stack, axis=-1)  # [batch, o_len, embedding_size, stack]
            with tf.variable_scope('similarity'):
                self_sim = tf.einsum('biks,bjks->bijs', self_u, self_n)  # [batch, u_len, o_len, stack]
                self_sim = 1 - self.gamma * tf.reduce_sum(self_sim, axis=1)  # [batch, (1), o_len, stack]
                Hn_stack = tf.einsum('bjkl,bjl->bjkl', Hn_stack_tensor, self_sim)
                Hn_stack = tf.unstack(Hn_stack, axis=-1, num=self.num_blocks + 1)

            Mur.append(sim_ur)
            Mun.append(sim_un)
            turn_id += 1
            if not reuse:
                reuse = True

        Hn_stack_for_tracking = tf.layers.dense(tf.stack(Hn_stack, axis=2), self.hidden_units)  # [batch, o_len, stack, embedding_size]
        Hn_stack_for_tracking = tf.transpose(Hn_stack_for_tracking, perm=[0, 1, 3, 2])  # [batch, o_len, embedding_size, stack]
        Hlastu_stack_for_tracking = tf.stack(last_u_reps, axis=-1)  # [batch, u_len, embedding_size, stack]
        Hr_stack_for_tracking = tf.stack(Hgtr_stack, axis=-1)  # [batch, r_len, embedding_size, stack]
        Hlastu = tf.transpose(Hlastu_stack_for_tracking, perm=[0, 2, 3, 1])
        Hlastu = tf.squeeze(tf.layers.dense(Hlastu, 1), axis=-1)  # [batch, embedding_size, stack]
        p1_tensor = tf.nn.softmax(tf.einsum('bnds,bds->bns', Hn_stack_for_tracking, Hlastu), axis=1)  # [batch, o_len, stack]
        Hlastur = tf.transpose(Hr_stack_for_tracking, perm=[0, 2, 3, 1])
        Hlastur = tf.squeeze(tf.layers.dense(Hlastur, 1), axis=-1)  # [batch, embedding_size, stack]
        p2_tensor = tf.nn.softmax(tf.einsum('bnds,bds->bns', Hn_stack_for_tracking, Hlastur), axis=1)  # [batch, o_len, stack]
        p1 = tf.unstack(p1_tensor, num=self.num_blocks + 1, axis=-1)
        p2 = tf.unstack(p2_tensor, num=self.num_blocks + 1, axis=-1)
        KL_loss = 0.0
        for i in range(self.num_blocks + 1):
            KL_loss += tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(p1[i], p2[i]))
        KL_loss /= (self.num_blocks + 1)

        r_a_n_stack = []
        n_a_r_stack = []
        for i in range(self.num_blocks + 1):
            with tf.variable_scope("narrative_attention_response_{}".format(i), reuse=True):
                n_a_r, _ = multihead_attention(queries=Hn_stack[i], keys=Hr_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                n_a_r = feedforward(n_a_r, num_units=[self.hidden_units, self.hidden_units])
                n_a_r_stack.append(n_a_r)
            with tf.variable_scope("response_attention_narrative_{}".format(i), reuse=True):
                r_a_n, _ = multihead_attention(queries=Hr_stack[i], keys=Hn_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False, dropout_rate=self.dropout_rate)
                r_a_n = feedforward(r_a_n, num_units=[self.hidden_units, self.hidden_units])
                r_a_n_stack.append(r_a_n)

        n_a_r_stack.extend(Hn_stack)
        r_a_n_stack.extend(Hr_stack)
        n_a_r = tf.stack(n_a_r_stack, axis=-1)
        r_a_n = tf.stack(r_a_n_stack, axis=-1)

        with tf.variable_scope('similarity'):
            Mrn = tf.einsum('biks,bjks->bijs', n_a_r, r_a_n) / tf.sqrt(200.0)
        self.rosim = Mrn
        Mur = tf.stack(Mur, axis=1)
        Mun = tf.stack(Mun, axis=1) 
        with tf.variable_scope('cnn_aggregation'):
            conv3d = tf.layers.conv3d(Mur, filters=32, kernel_size=[3, 3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv1")
            pool3d = tf.layers.max_pooling3d(conv3d, pool_size=[3, 3, 3], strides=[3, 3, 3], padding="SAME")
            conv3d2 = tf.layers.conv3d(pool3d, filters=32, kernel_size=[3, 3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv2")
            pool3d2 = tf.layers.max_pooling3d(conv3d2, pool_size=[3, 3, 3], strides=[3, 3, 3], padding="SAME")
            mur = tf.contrib.layers.flatten(pool3d2)
        with tf.variable_scope('cnn_aggregation', reuse=True):
            conv3d = tf.layers.conv3d(Mun, filters=32, kernel_size=[3, 3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv1")
            pool3d = tf.layers.max_pooling3d(conv3d, pool_size=[3, 3, 3], strides=[3, 3, 3], padding="SAME")
            conv3d2 = tf.layers.conv3d(pool3d, filters=32, kernel_size=[3, 3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv2")
            pool3d2 = tf.layers.max_pooling3d(conv3d2, pool_size=[3, 3, 3], strides=[3, 3, 3], padding="SAME")
            mun = tf.contrib.layers.flatten(pool3d2)
        with tf.variable_scope('cnn_aggregation'):
            conv2d = tf.layers.conv2d(Mrn, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv2d")
            pool2d = tf.layers.max_pooling2d(conv2d, pool_size=[3, 3], strides=[3, 3], padding="SAME")
            conv2d2 = tf.layers.conv2d(pool2d, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv2d2")
            pool2d2 = tf.layers.max_pooling2d(conv2d2, pool_size=[3, 3], strides=[3, 3], padding="SAME")
            mrn = tf.contrib.layers.flatten(pool2d2)

        all_vector = tf.concat([mur, mun, mrn], axis=-1)
        logits = tf.reshape(tf.layers.dense(all_vector, 1, kernel_initializer=tf.orthogonal_initializer()), [-1])

        self.y_pred = tf.sigmoid(logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, beta1=0.9, beta2=0.98, epsilon=1e-8)
        RS_loss = tf.reduce_mean(tf.clip_by_value(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y_true_ph, tf.float32), logits=logits), -10, 10))
        self.loss = self.eta * RS_loss + (1 - self.eta) * KL_loss
        self.all_variables = tf.global_variables()
        self.grads_and_vars = optimizer.compute_gradients(self.loss)

        for grad, var in self.grads_and_vars:
            if grad is None:
                print(var)

        self.capped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in self.grads_and_vars]
        self.train_op = optimizer.apply_gradients(self.capped_gvs, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.alpha_1 = alpha_1
        # self.alpha_2 = alpha_2
        # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


def evaluate(model_path, eval_file, output_path, eta):
    with open(eval_file, 'rb') as f:
        utterance, response, narrative, gt_response, y_true = pickle.load(f)

    current_lr = 1e-3
    all_candidate_scores = []
    dataset = tf.data.Dataset.from_tensor_slices((utterance, narrative, response, gt_response, y_true)).batch(eval_batch_size)
    iterator = dataset.make_initializable_iterator()
    data_iterator = iterator.get_next()

    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)

    model = ScriptWriter_cpre(eta)
    model.build()
    sess = model.load(model_path)
    sess.run(iterator.initializer)
    sess.run(model.embedding_init, feed_dict={model.embedding_ph: embeddings})

    test_loss = 0.0
    step = 0
    try:
        with tqdm(total=len(y_true), ncols=100) as pbar:
            while True:
                bu, bn, br, bgtr, by = data_iterator
                bu, bn, br, bgtr, by = sess.run([bu, bn, br, bgtr, by])
                candidate_scores, loss = sess.run([model.y_pred, model.loss], feed_dict={
                    model.utterance_ph: bu, 
                    model.narrative_ph: bn,
                    model.response_ph: br,
                    model.gt_response_ph: bgtr,
                    model.y_true_ph: by,
                    model.learning_rate_ph: current_lr
                })
                all_candidate_scores.append(candidate_scores)
                test_loss += loss
                pbar.update(model.eval_batch_size)
                step += 1
    except tf.errors.OutOfRangeError:
        pass

    sess.close()
    tf.reset_default_graph()

    all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
    with open(output_path + "test.result.micro_session.txt", "w") as fw:
        for sc in all_candidate_scores.tolist():
            fw.write(str(sc) + "\n")
    return Evaluate.evaluate_all(all_candidate_scores, y_true), test_loss / step, all_candidate_scores.tolist()


def simple_evaluate(sess, model, eval_file):
    with open(eval_file, 'rb') as f:
        utterance, response, narrative, y_true = pickle.load(f)
    utterance, utterance_len = Utils.multi_sequences_padding(utterance, max_sentence_len, max_num_utterance=max_num_utterance)
    utterance = np.array(utterance)
    narrative = np.array(pad_sequences(narrative, padding='post', maxlen=max_sentence_len))
    response = np.array(pad_sequences(response, padding='post', maxlen=max_sentence_len))
    y_true = np.array(y_true)
    all_candidate_scores = []
    dataset = tf.data.Dataset.from_tensor_slices((utterance, narrative, response, y_true)).batch(eval_batch_size)
    iterator = dataset.make_initializable_iterator()
    data_iterator = iterator.get_next()
    sess.run(iterator.initializer)
    current_lr = 1e-3
    test_loss = 0.0
    step = 0
    try:
        with tqdm(total=len(y_true), ncols=100) as pbar:
            while True:
                bu, bn, br, by = data_iterator
                bu, bn, br, by = sess.run([bu, bn, br, by])
                candidate_scores, loss = sess.run([model.y_pred, model.loss], feed_dict={
                    model.utterance_ph: bu,
                    model.narrative_ph: bn,
                    model.response_ph: br,
                    model.y_true_ph: by,
                    model.gt_response_ph: br,
                    model.learning_rate_ph: current_lr
                })
                all_candidate_scores.append(candidate_scores)
                test_loss += loss
                pbar.update(eval_batch_size)
                step += 1
    except tf.errors.OutOfRangeError:
        pass
    all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
    return Evaluate.evaluate_all(all_candidate_scores, y_true), test_loss / step, all_candidate_scores.tolist()


def evaluate_multi_turns(test_file, model_path, output_path):
    vocab = {}
    vocab_id2word = {}

    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)

    model = ScriptWriter_cpre()
    model.build()
    sess = model.load(model_path)
    sess.run(model.embedding_init, feed_dict={model.embedding_ph: embeddings})

    with open("./data/vocab.txt", "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr):
            line = line.strip().split("\t")
            vocab[line[0]] = idx + 1
            vocab_id2word[idx + 1] = line[0]
    vocab["_PAD_"] = 0
    vocab_id2word[0] = "_PAD_"

    def initialize(test_file):
        initial_file = output_path + "test.multi.0.pkl"
        max_turn = 0
        narrative_dict = {}
        narrative_dict_score = {}

        with open(test_file, 'rb') as f:
            utterance, response, outline, labels = pickle.load(f)
        new_utterance, new_response, new_narrative, new_labels = [], [], [], []
        for i in range(len(response)):
            ut = utterance[i]
            if len(ut) == 1:
                o = outline[i]
                r = response[i]
                l = labels[i]
                new_utterance.append(ut)
                new_response.append(r)
                new_narrative.append(o)
                new_labels.append(l)
            if len(ut) > max_turn:
                max_turn = len(ut)
            o = "".join([vocab_id2word[x] for x in outline[i]])
            if o not in narrative_dict:
                narrative_dict[o] = {0: outline[i]}
                narrative_dict_score[o] = {0: [-1]}
            r = response[i]
            l = labels[i]
            if len(ut) in narrative_dict[o]:
                narrative_dict[o][len(ut)].append(r)
                narrative_dict_score[o][len(ut)].append(l)
            else:
                narrative_dict[o][len(ut)] = [r]
                narrative_dict_score[o][len(ut)] = [l]

        pickle.dump(narrative_dict, open(output_path + "response_candidate.pkl", "wb"))

        new_data = [new_utterance, new_response, new_narrative, new_labels]
        pickle.dump(new_data, open(initial_file, "wb"))

        (r2_1, r10_1, r10_2, r10_5, mrr), eva_loss, result = simple_evaluate(sess, model, initial_file)
        with open(output_path + "test.result.multi.0.txt", "w") as fw:
            fw.write("R2@1: %f, R10@1: %f, R10@2: %f, R10@5: %f, MRR: %f\n" % (r2_1, r10_1, r10_2, r10_5, mrr))
            for r in result:
                fw.write(str(r) + "\n")

        return max_turn, narrative_dict, narrative_dict_score

    max_turn, narrative_dict, narrative_dict_score = initialize(test_file)
    for turn in range(1, max_turn):
        score = []
        with open(output_path + "test.result.multi." + str(turn - 1) + ".txt", "r") as fr:
            for idx, line in enumerate(fr):
                if idx == 0:
                    continue
                score.append(float(line.strip()))
        with open(output_path + "test.multi." + str(turn - 1) + ".pkl", "rb") as fr:
            utterance, response, narrative, y_true = pickle.load(fr)

        new_utterance = []
        new_response = []
        new_narrative = []
        new_labels = []

        for i, o in enumerate(narrative):
            if i % 10 == 0:
                sent_o = "".join([vocab_id2word[x] for x in o])
                if turn + 1 in narrative_dict[sent_o]:
                    new_response.extend(narrative_dict[sent_o][turn + 1])
                    score_sub_list = score[i:i + 10]
                    response_sub_list = response[i:i + 10]
                    max_idx = score_sub_list.index(max(score_sub_list))
                    selected_response = response_sub_list[max_idx]
                    for ut in utterance[i:i + 10]:
                        tmp = ut + [selected_response]
                        new_utterance.append(tmp)
                    new_narrative.extend([o] * 10)
                    new_labels.extend(narrative_dict_score[sent_o][turn + 1])

        new_data = [new_utterance, new_response, new_narrative, new_labels]
        new_file = output_path + "test.multi." + str(turn) + ".pkl"
        pickle.dump(new_data, open(new_file, "wb"))

        (r2_1, r10_1, r10_2, r10_5, mrr), eva_loss, result = simple_evaluate(sess, model, new_file)
        with open(output_path + "test.result.multi." + str(turn) + ".txt", "w") as fw:
            fw.write("R2@1: %f, R10@1: %f, R10@2: %f, R10@5: %f, MRR: %f\n" % (r2_1, r10_1, r10_2, r10_5, mrr))
            for r in result:
                fw.write(str(r) + "\n")


def train(eta=0.5, load=False, model_path=None, logger=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    epoch = 0
    best_result = [0.0, 0.0, 0.0, 0.0, 0.0]
    with tf.Session(config=config) as sess:
        with open(embedding_file, 'rb') as f:
            embeddings = pickle.load(f, encoding="bytes")
        with open(train_file, 'rb') as f:
            utterance_train, response_train, narrative_train, gt_response_train, y_true_train = pickle.load(f)
        with open(val_file, "rb") as f:
            utterance_val, response_val, narrative_val, gt_response_val, y_true_val = pickle.load(f)

        train_dataset = tf.data.Dataset.from_tensor_slices((utterance_train, narrative_train, response_train, gt_response_train, y_true_train)).shuffle(1024).batch(batch_size)
        train_iterator = train_dataset.make_initializable_iterator()
        train_data_iterator = train_iterator.get_next()

        val_dataset = tf.data.Dataset.from_tensor_slices((utterance_val, narrative_val, response_val, gt_response_val, y_true_val)).batch(batch_size)
        val_iterator = val_dataset.make_initializable_iterator()
        val_data_iterator = val_iterator.get_next()

        model = ScriptWriter_cpre(eta=eta)
        model.build()

        if load:
            sess = model.load(model_path)

        sess.run(tf.global_variables_initializer())
        sess.run(model.embedding_init, feed_dict={model.embedding_ph: embeddings})
        current_lr = 1e-3

        while epoch < 4:
            print("\nEpoch ", epoch + 1, "/ 4")
            train_loss = 0.0
            sess.run(train_iterator.initializer)
            step = 0
            try:
                with tqdm(total=len(y_true_train), ncols=100) as pbar:
                    while True:
                        bu, bn, br, bgtr, by = train_data_iterator
                        bu, bn, br, bgtr, by = sess.run([bu, bn, br, bgtr, by])
                        _, loss = sess.run([model.train_op, model.loss], feed_dict={
                            model.utterance_ph: bu, 
                            model.narrative_ph: bn,
                            model.response_ph: br,
                            model.gt_response_ph: bgtr,
                            model.y_true_ph: by,
                            model.learning_rate_ph: current_lr
                        })
                        train_loss += loss
                        pbar.set_postfix(learning_rate=current_lr, loss=loss)
                        pbar.update(model.batch_size)
                        step += 1
                        if step % 500 == 0:
                            val_loss = 0.0
                            val_step = 0
                            sess.run(val_iterator.initializer)
                            all_candidate_scores = []
                            try:
                                while True:
                                    bu, bn, br, bgtr, by = val_data_iterator
                                    bu, bn, br, bgtr, by = sess.run([bu, bn, br, bgtr, by])
                                    candidate_scores, loss = sess.run([model.y_pred, model.loss], feed_dict={
                                        model.utterance_ph: bu, 
                                        model.narrative_ph: bn,
                                        model.response_ph: br,
                                        model.gt_response_ph: bgtr,
                                        model.y_true_ph: by,
                                    })
                                    all_candidate_scores.append(candidate_scores)
                                    val_loss += loss
                                    val_step += 1
                            except tf.errors.OutOfRangeError:
                                pass
                            all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
                            result = Evaluate.evaluate_all(all_candidate_scores, y_true_val)
                            if result[0] + result[1] + result[2] + result[3] + result[4] > best_result[0] + best_result[1] + best_result[2] + best_result[3] + best_result[4]:
                                best_result = result
                                tqdm.write("Current best result on validation set: r2@1 %.3f, r10@1 %.3f, r10@2 %.3f, r10@5 %.3f, mrr %.3f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4]))
                                logger.info("Current best result on validation set: r2@1 %.3f, r10@1 %.3f, r10@2 %.3f, r10@5 %.3f, mrr %.3f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4]))
                                model.saver.save(sess, save_path + "model")
                                patience = 0
                            else:
                                patience += 1
                                if patience >= 3:
                                    current_lr *= 0.5
            except tf.errors.OutOfRangeError:
                pass

            val_loss = 0.0
            val_step = 0
            sess.run(val_iterator.initializer)
            all_candidate_scores = []
            try:
                while True:
                    bu, bn, br, bgtr, by = val_data_iterator
                    bu, bn, br, bgtr, by = sess.run([bu, bn, br, bgtr, by])
                    candidate_scores, loss = sess.run([model.y_pred, model.loss], feed_dict={
                        model.utterance_ph: bu, 
                        model.narrative_ph: bn,
                        model.response_ph: br,
                        model.gt_response_ph: bgtr,
                        model.y_true_ph: by
                    })
                    all_candidate_scores.append(candidate_scores)
                    val_loss += loss
                    val_step += 1
            except tf.errors.OutOfRangeError:
                pass
            all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
            result = Evaluate.evaluate_all(all_candidate_scores, y_true_val)
            if result[0] + result[1] + result[2] + result[3] + result[4] > best_result[0] + best_result[1] + best_result[2] + best_result[3] + best_result[4]:
                best_result = result
                tqdm.write("Current best result on validation set: r2@1 %.3f, r10@1 %.3f, r10@2 %.3f, r10@5 %.3f, mrr %.3f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4]))
                logger.info("Current best result on validation set: r2@1 %.3f, r10@1 %.3f, r10@2 %.3f, r10@5 %.3f, mrr %.3f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4]))
                model.saver.save(sess, save_path + "model")
            tqdm.write('Epoch No: %d, the train loss is %f, the dev loss is %f' % (epoch + 1, train_loss / step, val_loss / val_step))
            epoch += 1
        sess.close()
    tf.reset_default_graph()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    log_path = "./model/cpre/all_log"
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logger = logging.getLogger(__name__)
    eta = 0.7
    save_path = "./model/cpre/"
    result_path = "./output/cpre/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    logger.info("Current Eta: %.2f" % eta)
    train(eta=eta, logger=logger)

    (r2_1, r10_1, r10_2, r10_5, mrr), eva_loss, _ = evaluate(save_path, evaluate_file, output_path=result_path, eta=eta)
    print("Loss on test set: %f, R2@1: %f, R10@1: %f, R10@2: %f, R10@5: %f, MRR: %f" % (eva_loss, r2_1, r10_1, r10_2, r10_5, mrr))

    # to evaluate multi-turn results, the vocab file is needed
    # evaluate_multi_turns(test_file=evaluate_file, model_path=save_path, output_path=result_path)
    # Evaluate.recover_and_show(result_path)
    # test_file = result_path + "test.result.multi.txt"
    # gold_file = "../data/ground_truth.result.mul.txt"
    # Evaluate.evaluate_multi_turn_result(test_file, gold_file)

