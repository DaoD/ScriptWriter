from modules import *
from keras.preprocessing.sequence import pad_sequences
import Utils
import Evaluate
import pickle
import os
from tqdm import tqdm

embedding_file = "./data/embeddings.pkl"
train_file = "./data/train.pkl"
val_file = "./data/dev.pkl"
evaluate_file = "./data/test.pkl"
evaluate_embedding_file = "./data/embeddings.pkl"

max_sentence_len = 50
max_num_utterance = 11
batch_size = 64
eval_batch_size = 64


class ScripteWriter():
    def __init__(self, data_iterator):
        self.max_num_utterance = max_num_utterance
        self.negative_samples = 1
        self.max_sentence_len = max_sentence_len
        self.word_embedding_size = 200
        self.hidden_units = 200
        self.total_words = 43514
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.initial_learning_rate = 1e-3
        self.dropout_rate = 0
        self.num_heads = 1
        self.num_blocks = 2
        # self.gamma = 0.1
        self.gamma = tf.get_variable('gamma', shape=1, dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(0.5))

        self.utterance_ph = data_iterator[0]
        self.response_ph = data_iterator[4]
        self.y_true = data_iterator[6]
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        self.response_len = data_iterator[5]
        self.all_utterance_len_ph = data_iterator[1]
        self.narrative_ph = data_iterator[2]
        self.narrative_len = data_iterator[3]
        self.word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.word_embedding_size), dtype=tf.float32, trainable=False)
        self.embedding_init = self.word_embeddings.assign(self.embedding_ph)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.is_training = True

    def load(self, previous_modelpath):
        sess = tf.Session()
        latest_ckpt = tf.train.latest_checkpoint(previous_modelpath)
        # latest_ckpt = previous_modelpath + "model.4"
        print("recover from checkpoint: " + latest_ckpt)
        variables = tf.contrib.framework.get_variables_to_restore()
        saver = tf.train.Saver(variables)
        saver.restore(sess, latest_ckpt)
        return sess

    def build(self):
        all_utterances = tf.unstack(self.utterance_ph, num=self.max_num_utterance, axis=1)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)
        reuse = None
        alpha_1 = None

        response_embeddings = embedding(self.response_ph, initializer=self.word_embeddings)
        response_embeddings = tf.layers.dropout(response_embeddings, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
        Hr_stack = [response_embeddings]
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                response_embeddings, _ = multihead_attention(queries=response_embeddings, keys=response_embeddings, num_units=self.hidden_units, num_heads=self.num_heads, dropout_rate=self.dropout_rate, is_training=self.is_training, causality=False)
                response_embeddings = feedforward(response_embeddings, num_units=[self.hidden_units, self.hidden_units])
                Hr_stack.append(response_embeddings)

        narrative_embeddings = embedding(self.narrative_ph, initializer=self.word_embeddings)
        narrative_embeddings = tf.layers.dropout(narrative_embeddings, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
        Hn_stack = [narrative_embeddings]
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=True):
                narrative_embeddings, _ = multihead_attention(queries=narrative_embeddings, keys=narrative_embeddings, num_units=self.hidden_units, num_heads=self.num_heads, dropout_rate=self.dropout_rate, is_training=self.is_training, causality=False)
                narrative_embeddings = feedforward(narrative_embeddings, num_units=[self.hidden_units, self.hidden_units])
                Hn_stack.append(narrative_embeddings)

        Mur, Mun = [], []
        self.decay_factor = []

        for utterance, utterance_len in zip(all_utterances, all_utterance_len):
            utterance_embeddings = embedding(utterance, initializer=self.word_embeddings)
            Hu_stack = [utterance_embeddings]
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=True):
                    utterance_embeddings, _ = multihead_attention(queries=utterance_embeddings, keys=utterance_embeddings, num_units=self.hidden_units, num_heads=self.num_heads, dropout_rate=self.dropout_rate, is_training=self.is_training, causality=False)
                    utterance_embeddings = feedforward(utterance_embeddings, num_units=[self.hidden_units, self.hidden_units])
                    Hu_stack.append(utterance_embeddings)

            r_a_u_stack = []
            u_a_r_stack = []

            for i in range(self.num_blocks + 1):
                with tf.variable_scope("utterance_attention_response_{}".format(i), reuse=reuse):
                    u_a_r, _ = multihead_attention(queries=Hu_stack[i], keys=Hr_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False)
                    u_a_r = feedforward(u_a_r, num_units=[self.hidden_units, self.hidden_units])
                    u_a_r_stack.append(u_a_r)
                with tf.variable_scope("response_attention_utterance_{}".format(i), reuse=reuse):
                    r_a_u, _ = multihead_attention(queries=Hr_stack[i], keys=Hu_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False)
                    r_a_u = feedforward(r_a_u, num_units=[self.hidden_units, self.hidden_units])
                    r_a_u_stack.append(r_a_u)
            u_a_r_stack.extend(Hu_stack)
            r_a_u_stack.extend(Hr_stack)

            n_a_u_stack = []
            u_a_n_stack = []
            for i in range(self.num_blocks + 1):
                with tf.variable_scope("narrative_attention_utterance_{}".format(i), reuse=reuse):
                    n_a_u, _ = multihead_attention(queries=Hn_stack[i], keys=Hu_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False)
                    n_a_u = feedforward(n_a_u, num_units=[self.hidden_units, self.hidden_units])
                    n_a_u_stack.append(n_a_u)
                with tf.variable_scope("utterance_attention_narrative_{}".format(i), reuse=reuse):
                    u_a_n, alpha_1 = multihead_attention(queries=Hu_stack[i], keys=Hn_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False)
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
                sim_ur = tf.einsum('biks,bjks->bijs', u_a_r, r_a_u) / tf.sqrt(200.0)
                sim_un = tf.einsum('biks,bjks->bijs', u_a_n, n_a_u) / tf.sqrt(200.0)

            self_n = tf.nn.l2_normalize(tf.stack(Hn_stack, axis=-1))
            self_u = tf.nn.l2_normalize(tf.stack(Hu_stack, axis=-1))
            with tf.variable_scope('similarity'):
                self_sim = tf.einsum('biks,bjks->bijs', self_u, self_n)  # [batch * len * len * stack]
                self_sim = tf.unstack(self_sim, axis=-1, num=self.num_blocks + 1)
                reuse2 = reuse
                for i in range(self.num_blocks + 1):
                    tmp_self_sim = tf.expand_dims(self_sim[i], axis=-1)
                    tmp_self_sim = 1 - self.gamma * tf.layers.conv2d(tmp_self_sim, filters=1, kernel_size=[max_sentence_len, 1], padding="valid", kernel_initializer=tf.ones_initializer, use_bias=False, trainable=False, reuse=reuse2)  # for auto2
                    tmp_self_sim = tf.squeeze(tmp_self_sim, axis=1)
                    tmp_self_sim = tf.squeeze(tmp_self_sim, axis=-1)
                    Hn_stack[i] = tf.einsum('bik,bi->bik', Hn_stack[i], tmp_self_sim)
                    reuse2 = True

            Mur.append(sim_ur)
            Mun.append(sim_un)

            if not reuse:
                reuse = True

        r_a_n_stack = []
        n_a_r_stack = []
        reuse2 = False
        for i in range(self.num_blocks + 1):
            with tf.variable_scope("narrative_attention_response_{}".format(i), reuse=reuse2):
                n_a_r, _ = multihead_attention(queries=Hn_stack[i], keys=Hr_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False)
                n_a_r = feedforward(n_a_r, num_units=[self.hidden_units, self.hidden_units])
                n_a_r_stack.append(n_a_r)
            with tf.variable_scope("response_attention_narrative_{}".format(i), reuse=reuse2):
                r_a_n, _ = multihead_attention(queries=Hr_stack[i], keys=Hn_stack[i], num_units=self.hidden_units, num_heads=self.num_heads, is_training=self.is_training, causality=False)
                r_a_n = feedforward(r_a_n, num_units=[self.hidden_units, self.hidden_units])
                r_a_n_stack.append(r_a_n)

        n_a_r_stack.extend(Hn_stack)
        r_a_n_stack.extend(Hr_stack)
        n_a_r = tf.stack(n_a_r_stack, axis=-1)
        r_a_n = tf.stack(r_a_n_stack, axis=-1)

        with tf.variable_scope('similarity'):
            Mrn = tf.einsum('biks,bjks->bijs', n_a_r, r_a_n) / tf.sqrt(200.0)

        Mur = tf.stack(Mur, axis=1)
        Mun = tf.stack(Mun, axis=1) 
        with tf.variable_scope('cnn_aggregation'):
            conv3d = tf.layers.conv3d(Mur, filters=32, kernel_size=[3, 3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv1")
            pool3d = tf.layers.max_pooling3d(conv3d, pool_size=[3, 3, 3], strides=[3, 3, 3], padding="SAME")
            conv3d2 = tf.layers.conv3d(pool3d, filters=16, kernel_size=[3, 3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv2")
            pool3d2 = tf.layers.max_pooling3d(conv3d2, pool_size=[3, 3, 3], strides=[3, 3, 3], padding="SAME")
            mur = tf.contrib.layers.flatten(pool3d2)
        with tf.variable_scope('cnn_aggregation', reuse=True):
            conv3d = tf.layers.conv3d(Mun, filters=32, kernel_size=[3, 3, 3], padding="SAME", activation=tf.nn.elu,
                                      kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv1")
            pool3d = tf.layers.max_pooling3d(conv3d, pool_size=[3, 3, 3], strides=[3, 3, 3], padding="SAME")
            conv3d2 = tf.layers.conv3d(pool3d, filters=16, kernel_size=[3, 3, 3], padding="SAME", activation=tf.nn.elu,
                                       kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv2")
            pool3d2 = tf.layers.max_pooling3d(conv3d2, pool_size=[3, 3, 3], strides=[3, 3, 3], padding="SAME")
            mun = tf.contrib.layers.flatten(pool3d2)

        with tf.variable_scope('cnn_aggregation'):
            conv2d = tf.layers.conv2d(Mrn, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv2d")
            pool2d = tf.layers.max_pooling2d(conv2d, pool_size=[3, 3], strides=[3, 3], padding="SAME")
            conv2d2 = tf.layers.conv2d(pool2d, filters=16, kernel_size=[3, 3], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01), name="conv2d2")
            pool2d2 = tf.layers.max_pooling2d(conv2d2, pool_size=[3, 3], strides=[3, 3], padding="SAME")
            mrn = tf.contrib.layers.flatten(pool2d2)

        all_vector = tf.concat([mur, mun, mrn], axis=-1)
        logits = tf.reshape(tf.layers.dense(all_vector, 1, kernel_initializer=tf.orthogonal_initializer()), [-1])

        self.y_pred = tf.sigmoid(logits)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step=self.global_step, decay_steps=1000, decay_rate=0.9, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.loss = tf.reduce_mean(tf.clip_by_value(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y_true, tf.float32), logits=logits), -10, 10))
        self.all_variables = tf.global_variables()
        self.grads_and_vars = optimizer.compute_gradients(self.loss)

        for grad, var in self.grads_and_vars:
            if grad is None:
                print(var)

        self.capped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in self.grads_and_vars]
        self.train_op = optimizer.apply_gradients(self.capped_gvs, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.alpha_1 = alpha_1


def evaluate(model_path, eval_file, output_path):
    with open(eval_file, 'rb') as f:
        utterance, response, narrative, labels = pickle.load(f)

    all_candidate_scores = []
    utterance, utterance_len = Utils.multi_sequences_padding(utterance, max_sentence_len, max_num_utterance=max_num_utterance)
    utterance, utterance_len = np.array(utterance), np.array(utterance_len)
    narrative_len = np.array(Utils.get_sequences_length(narrative, maxlen=max_sentence_len))
    narrative = np.array(pad_sequences(narrative, padding='post', maxlen=max_sentence_len))
    response_len = np.array(Utils.get_sequences_length(response, maxlen=max_sentence_len))
    response = np.array(pad_sequences(response, padding='post', maxlen=max_sentence_len))
    y_true = np.array(labels)

    dataset = tf.data.Dataset.from_tensor_slices((utterance, utterance_len, narrative, narrative_len, response, response_len, y_true))
    dataset = dataset.batch(eval_batch_size)
    iterator = dataset.make_initializable_iterator()

    data_iterator = iterator.get_next()

    with open(evaluate_embedding_file, 'rb') as f:
        embeddings = pickle.load(f)

    model = ScripteWriter(data_iterator)
    model.build()
    sess = model.load(model_path)
    sess.run(iterator.initializer)
    sess.run(model.embedding_init, feed_dict={model.embedding_ph: embeddings})

    test_loss = 0.0
    step = 0
    try:
        with tqdm(total=len(y_true)) as pbar:
            while True:
                candidate_scores, loss = sess.run([model.y_pred, model.loss])
                all_candidate_scores.append(candidate_scores)
                test_loss += loss
                pbar.update(model.eval_batch_size)
                step += 1
    except tf.errors.OutOfRangeError:
        pass

    all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
    with open(output_path + "test.result.micro_session.txt", "w") as fw:
        for sc in all_candidate_scores.tolist():
            fw.write(str(sc) + "\n")
    return Evaluate.evaluate_all(all_candidate_scores, labels), test_loss / step, all_candidate_scores.tolist()


def simple_evaluate(sess, model, iterator, utterance_ph, utterance_len_ph, narrative_ph, narrative_len_ph, response_ph, response_len_ph, y_true_ph, eval_file):
    with open(eval_file, 'rb') as f:
        utterance, response, narrative, labels = pickle.load(f)

    all_candidate_scores = []
    utterance, utterance_len = Utils.multi_sequences_padding(utterance, max_sentence_len, max_num_utterance=max_num_utterance)
    utterance, utterance_len = np.array(utterance), np.array(utterance_len)
    narrative_len = np.array(Utils.get_sequences_length(narrative, maxlen=max_sentence_len))
    narrative = np.array(pad_sequences(narrative, padding='post', maxlen=max_sentence_len))
    response_len = np.array(Utils.get_sequences_length(response, maxlen=max_sentence_len))
    response = np.array(pad_sequences(response, padding='post', maxlen=max_sentence_len))
    y_true = np.array(labels)

    sess.run(iterator.initializer, feed_dict={utterance_ph: utterance,
                                              utterance_len_ph: utterance_len,
                                              narrative_ph: narrative,
                                              narrative_len_ph: narrative_len,
                                              response_ph: response,
                                              response_len_ph: response_len,
                                              y_true_ph: y_true})

    test_loss = 0.0
    step = 0
    try:
        with tqdm(total=len(y_true)) as pbar:
            while True:
                candidate_scores, loss = sess.run([model.y_pred, model.loss])
                all_candidate_scores.append(candidate_scores)
                test_loss += loss
                pbar.update(eval_batch_size)
                step += 1
    except tf.errors.OutOfRangeError:
        pass

    all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
    return Evaluate.evaluate_all(all_candidate_scores, labels), test_loss / step, all_candidate_scores.tolist()


def evaluate_multi_turns(test_file, model_path, output_path):
    vocab = {}
    vocab_id2word = {}

    utterance_ph = tf.placeholder(tf.int32, shape=(None, max_num_utterance, max_sentence_len))
    response_ph = tf.placeholder(tf.int32, shape=(None, max_sentence_len))
    y_true_ph = tf.placeholder(tf.int32, shape=(None,))
    response_len_ph = tf.placeholder(tf.int32, shape=(None,))
    utterance_len_ph = tf.placeholder(tf.int32, shape=(None, max_num_utterance))
    narrative_ph = tf.placeholder(tf.int32, shape=(None, max_sentence_len))
    narrative_len_ph = tf.placeholder(tf.int32, shape=(None,))

    with open(evaluate_embedding_file, 'rb') as f:
        embeddings = pickle.load(f)

    dataset = tf.data.Dataset.from_tensor_slices((utterance_ph, utterance_len_ph, narrative_ph, narrative_len_ph, response_ph, response_len_ph, y_true_ph))
    dataset = dataset.batch(eval_batch_size)
    iterator = dataset.make_initializable_iterator()

    data_iterator = iterator.get_next()

    model = ScripteWriter(data_iterator)
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
            utterance, response, narrative, labels = pickle.load(f)
        new_utterance, new_response, new_narrative, new_labels = [], [], [], []
        for i in range(len(response)):
            ut = utterance[i]
            if len(ut) == 1:
                o = narrative[i]
                r = response[i]
                l = labels[i]
                new_utterance.append(ut)
                new_response.append(r)
                new_narrative.append(o)
                new_labels.append(l)
            if len(ut) > max_turn:
                max_turn = len(ut)
            o = "".join([vocab_id2word[x] for x in narrative[i]])
            if o not in narrative_dict:
                narrative_dict[o] = {0: narrative[i]}
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

        (acc, r2_1, r10_1, r10_2, r10_5, mrr), eva_loss, result = simple_evaluate(sess, model, iterator, utterance_ph, utterance_len_ph, narrative_ph, narrative_len_ph, response_ph, response_len_ph, y_true_ph, initial_file)
        with open(output_path + "test.result.multi.0.txt", "w") as fw:
            fw.write("R2@1: %f, R10@1: %f, R10@2: %f, R10@5: %f, MRR: %f\n" % (r2_1, r10_1, r10_2, r10_5, mrr))
            for r in result:
                fw.write(str(r) + "\n")

        return max_turn, narrative_dict, narrative_dict_score

    max_turn, narrative_dict, narrative_dict_score = initialize(test_file)

    for turn in range(1, max_turn):
        # for turn in range(1, 2):
        score = []
        with open(output_path + "test.result.multi." + str(turn - 1) + ".txt", "r") as fr:
            for idx, line in enumerate(fr):
                if idx == 0:
                    continue
                score.append(float(line.strip()))
        with open(output_path + "test.multi." + str(turn - 1) + ".pkl", "rb") as fr:
            utterance, response, narrative, labels = pickle.load(fr)

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

        (acc, r2_1, r10_1, r10_2, r10_5, mrr), eva_loss, result = simple_evaluate(sess, model, iterator, utterance_ph, utterance_len_ph, narrative_ph, narrative_len_ph, response_ph, response_len_ph, y_true_ph, new_file)
        with open(output_path + "test.result.multi." + str(turn) + ".txt", "w") as fw:
            fw.write("R2@1: %f, R10@1: %f, R10@2: %f, R10@5: %f, MRR: %f\n" % (r2_1, r10_1, r10_2, r10_5, mrr))
            for r in result:
                fw.write(str(r) + "\n")


def train(load=False, model_path=None):
    best_val_loss = 100000.0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    epoch = 0
    with tf.Session(config=config) as sess:
        with open(embedding_file, 'rb') as f:
            embeddings = pickle.load(f, encoding="bytes")
        with open(train_file, 'rb') as f:
            utterance, response, narrative, labels = pickle.load(f)
        with open(val_file, "rb") as f:
            utterance_val, response_val, narrative_val, labels_val = pickle.load(f)

        state = np.random.get_state()
        np.random.shuffle(utterance)
        np.random.set_state(state)
        np.random.shuffle(response)
        np.random.set_state(state)
        np.random.shuffle(labels)
        np.random.set_state(state)
        np.random.shuffle(narrative)

        utterance_ph = tf.placeholder(tf.int32, shape=(None, max_num_utterance, max_sentence_len))
        response_ph = tf.placeholder(tf.int32, shape=(None, max_sentence_len))
        y_true_ph = tf.placeholder(tf.int32, shape=(None,))
        response_len_ph = tf.placeholder(tf.int32, shape=(None,))
        utterance_len_ph = tf.placeholder(tf.int32, shape=(None, max_num_utterance))
        narrative_ph = tf.placeholder(tf.int32, shape=(None, max_sentence_len))
        narrative_len_ph = tf.placeholder(tf.int32, shape=(None,))

        utterance_train, utterance_len_train = Utils.multi_sequences_padding(utterance, max_sentence_len, max_num_utterance=max_num_utterance)
        utterance_train, utterance_len_train = np.array(utterance_train), np.array(utterance_len_train)
        response_len_train = np.array(Utils.get_sequences_length(response, maxlen=max_sentence_len))
        response_train = np.array(pad_sequences(response, padding='post', maxlen=max_sentence_len))
        narrative_len_train = np.array(Utils.get_sequences_length(narrative, maxlen=max_sentence_len))
        narrative_train = np.array(pad_sequences(narrative, padding='post', maxlen=max_sentence_len))
        y_true_train = np.array(labels)

        utterance_val, utterance_len_val = Utils.multi_sequences_padding(utterance_val, max_sentence_len, max_num_utterance=max_num_utterance)
        utterance_val, utterance_len_val = np.array(utterance_val), np.array(utterance_len_val)
        response_len_val = np.array(Utils.get_sequences_length(response_val, maxlen=max_sentence_len))
        response_val = np.array(pad_sequences(response_val, padding='post', maxlen=max_sentence_len))
        narrative_len_val = np.array(Utils.get_sequences_length(narrative_val, maxlen=max_sentence_len))
        narrative_val = np.array(pad_sequences(narrative_val, padding='post', maxlen=max_sentence_len))
        y_true_val = np.array(labels_val)

        dataset = tf.data.Dataset.from_tensor_slices((utterance_ph, utterance_len_ph, narrative_ph, narrative_len_ph, response_ph, response_len_ph,
                                                      y_true_ph)).shuffle(1000)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()

        data_iterator = iterator.get_next()

        model = ScripteWriter(data_iterator)
        model.build()

        if load:
            sess = model.load(model_path)

        sess.run(tf.global_variables_initializer())
        sess.run(model.embedding_init, feed_dict={model.embedding_ph: embeddings})

        while epoch < 8:
            train_loss = 0.0
            sess.run(iterator.initializer, feed_dict={utterance_ph: utterance_train,
                                                      utterance_len_ph: utterance_len_train,
                                                      narrative_ph: narrative_train,
                                                      narrative_len_ph: narrative_len_train,
                                                      response_ph: response_train,
                                                      response_len_ph: response_len_train,
                                                      y_true_ph: y_true_train})
            step = 0
            try:
                with tqdm(total=len(y_true_train)) as pbar:
                    while True:
                        _, loss, lr = sess.run([model.train_op, model.loss, model.learning_rate])
                        train_loss += loss
                        pbar.set_postfix(learning_rate=lr, loss=loss)
                        pbar.update(model.batch_size)
                        step += 1
            except tf.errors.OutOfRangeError:
                pass

            val_loss = 0.0
            val_step = 0
            sess.run(iterator.initializer, feed_dict={utterance_ph: utterance_val,
                                                      utterance_len_ph: utterance_len_val,
                                                      narrative_ph: narrative_val,
                                                      narrative_len_ph: narrative_len_val,
                                                      response_ph: response_val,
                                                      response_len_ph: response_len_val,
                                                      y_true_ph: y_true_val})
            try:
                while True:
                    loss = sess.run(model.loss)
                    val_loss += loss
                    val_step += 1
            except tf.errors.OutOfRangeError:
                pass

            print('Epoch No: %d, the train loss is %f, the dev loss is %f' % (epoch + 1, train_loss / step, val_loss / val_step))
            if val_loss / val_step < best_val_loss:
                best_val_loss = val_loss / val_step
                model.saver.save(sess, "./model/model.{0}".format(epoch + 1))
                print("Save model.{}".format(epoch + 1))
            epoch += 1

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    is_train = True
    previous_train_modelpath = "./model/"
    if is_train:
        train(False, previous_train_modelpath)
    else:
        # check the validation loss obtained in the training process and use the saved model with the smallest validation loss
        (acc, r2_1, r10_1, r10_2, r10_5, mrr), eva_loss, _ = evaluate(previous_train_modelpath, evaluate_file, output_path="./output/")
        print("Loss on test set: %f, Accuracy: %f, R2@1: %f, R10@1: %f, R10@2: %f, R10@5: %f, MRR: %f" % (eva_loss, acc, r2_1, r10_1, r10_2, r10_5, mrr))

        # to evaluate multi-turn results, the vocab file is needed
        # evaluate_multi_turns(test_file=evaluate_file, model_path=previous_train_modelpath, output_path="./output/")
        # Evaluate.recover_and_show(basic_directory="./output/")
        # test_file = basic_directory + "test.result.multi.txt"
        # gold_file = "./data/ground_truth.result.mul.txt"
        # Evaluate.evaluate_multi_turn_result(test_file, gold_file)
