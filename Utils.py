import concurrent.futures
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence


def multi_sequences_padding(all_sequences, max_sentence_len=50, max_num_utterance=10):
    PAD_SEQUENCE = [0] * max_sentence_len
    padded_sequences = []
    sequences_length = []
    for sequences in all_sequences:
        sequences_len = len(sequences)
        sequences_length.append(get_sequences_length(sequences, maxlen=max_sentence_len))
        if sequences_len < max_num_utterance:
            sequences += [PAD_SEQUENCE] * (max_num_utterance - sequences_len)
            sequences_length[-1] += [0] * (max_num_utterance - sequences_len)
        else:
            sequences = sequences[-max_num_utterance:]
            sequences_length[-1] = sequences_length[-1][-max_num_utterance:]
        sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
        padded_sequences.append(sequences)
    return padded_sequences, sequences_length


def get_sequences_length(sequences, maxlen):
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return sequences_length


def generate_data_with_random_samples():
    # generate negative samples randomly
    # In training set, for each sample, we randomly sample a response as a negative candidate
    # In development and test set, for each sample, we randomly sample 9 responses as negative candidates and we add a "EOS" response as a candidate to let model select when to stop
    import random
    import pickle
    vocab = {}
    positive_data = []
    EOS_ID = 7
    with open("./data/sample_vocab.txt", "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr):
            line = line.strip().split("\t")
            vocab[line[0]] = idx + 1
    with open("./data/sample_data.txt", "r", encoding="utf-8") as fr:
        tmp = []
        for line in fr:
            line = line.strip()
            if len(line) > 0:
                line = line.split("\t")
                if line[0] == "narrative":
                    tmp.append(line[1])
                elif line[0] == "script":
                    tmp.append(line[1])
            else:
                narrative = tmp[0]
                context = tmp[1:]
                narrative_id = [vocab.get(word, 0) for word in narrative.split()]
                context_id = [[vocab.get(word, 0) for word in sent.split()] for sent in context]
                if len(narrative_id) == 0 or len(context_id) == 0:
                    continue
                data = [context_id, narrative_id, 1]
                positive_data.append(data)
                tmp = []
        random.shuffle(positive_data)
        print("all suitable sessions: ", len(positive_data))
        train_num = int(len(positive_data) * 0.9)
        dev_test_num = int(len(positive_data) * 0.05)
        train, dev, test = positive_data[:train_num], positive_data[train_num: train_num + dev_test_num], positive_data[train_num + dev_test_num:]
        train_all, dev_all, test_all = [], [], []
        for context_id, narrative_id, _ in train:
            num_context = len(context_id)
            for i in range(1, num_context):
                context = context_id[:i]
                response = context_id[i]
                train_all.append([context, response, narrative_id, 1])
                flag = True
                while flag:
                    random_idx = random.randint(0, len(positive_data) - 1)
                    random_context = positive_data[random_idx][0]
                    random_idx_2 = random.randint(0, len(random_context) - 1)
                    random_response = random_context[random_idx_2]
                    if len(response) != len(random_response):
                        flag = False
                        train_all.append([context, random_response, narrative_id, 0])
                    else:
                        for idx, wid in enumerate(response):
                            if wid != random_response[idx]:
                                flag = False
                                train_all.append([context, random_response, narrative_id, 0])
                                break
        print(train_all[0], train_all[1])
        for context_id, narrative_id, _ in dev:
            num_context = len(context_id)
            for i in range(1, num_context):
                context = context_id[:i]
                response = context_id[i]
                dev_all.append([context, response, narrative_id, 1])
                count = 0
                negative_samples = []
                while count < 9:
                    random_idx = random.randint(0, len(positive_data) - 1)
                    random_context = positive_data[random_idx][0]
                    random_idx_2 = random.randint(0, len(random_context) - 1)
                    random_response = random_context[random_idx_2]
                    if random_response not in negative_samples and random_response != [EOS_ID]:
                        if len(response) != len(random_response):
                            dev_all.append([context, random_response, narrative_id, 0])
                            count += 1
                            negative_samples.append(random_response)
                        else:
                            for idx, wid in enumerate(response):
                                if wid != random_response[idx]:
                                    dev_all.append([context, random_response, narrative_id, 0])
                                    count += 1
                                    negative_samples.append(random_response)
                                    break
                if response == [EOS_ID]:
                    dev_all.append([context, [EOS_ID], narrative_id, 1])
                else:
                    dev_all.append([context, [EOS_ID], narrative_id, 0])
        print(dev_all[0], dev_all[1], dev_all[2])
        for context_id, narrative_id, _ in test:
            num_context = len(context_id)
            for i in range(1, num_context):
                context = context_id[:i]
                response = context_id[i]
                test_all.append([context, response, narrative_id, 1])
                count = 0
                negative_samples = []
                while count < 9:
                    random_idx = random.randint(0, len(positive_data) - 1)
                    random_context = positive_data[random_idx][0]
                    random_idx_2 = random.randint(0, len(random_context) - 1)
                    random_response = random_context[random_idx_2]
                    if random_response not in negative_samples and random_response != [EOS_ID]:
                        if len(response) != len(random_response):
                            test_all.append([context, random_response, narrative_id, 0])
                            negative_samples.append(random_response)
                            count += 1
                        else:
                            for idx, id in enumerate(response):
                                if id != random_response[idx]:
                                    test_all.append([context, random_response, narrative_id, 0])
                                    negative_samples.append(random_response)
                                    count += 1
                                    break
                if response == [EOS_ID]:
                    test_all.append([context, [EOS_ID], narrative_id, 1])
                else:
                    test_all.append([context, [EOS_ID], narrative_id, 0])
        print(test_all[0], test_all[1], test_all[2])
    context, response, narrative, label = [], [], [], []
    print("train size: ", len(train_all))
    for data in train_all:
        context.append(data[0])
        response.append(data[1])
        narrative.append(data[2])
        label.append(data[3])
    train = [context, response, narrative, label]
    pickle.dump(train, open("./data/train.multi.pkl", "wb"))
    context, response, narrative, label = [], [], [], []
    print("dev size: ", len(dev_all))
    for data in dev_all:
        context.append(data[0])
        response.append(data[1])
        narrative.append(data[2])
        label.append(data[3])
    dev = [context, response, narrative, label]
    pickle.dump(dev, open("./data/dev.multi.pkl", "wb"))
    context, response, narrative, label = [], [], [], []
    print("test size: ", len(test_all))
    for data in test_all:
        context.append(data[0])
        response.append(data[1])
        narrative.append(data[2])
        label.append(data[3])
    test = [context, response, narrative, label]
    pickle.dump(test, open("./data/test.multi.pkl", "wb"))


def generate_data_with_solr_samples():
    # generate negative samples from solr
    # this is only for development and test set since training set has only one negative sample
    import pickle
    import pysolr
    import jieba

    EOS_ID = 7

    def query_comt(post, num):
        # the format of the solr data: "ut1: xxxxx, ut2: xxxxx", where ut1 is the index
        solr = pysolr.Solr('xxxxxx', timeout=10)  # write your Solr address
        post = "ut1:(" + post + ")"
        results = solr.search(q=post, **{'rows': num})  # rows equal to the number of pairs you want to retrieve
        return results

    vocab = {}
    vocab_id2word = {}

    with open("./data/vocab.txt", "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr):
            line = line.strip().split("\t")
            vocab[line[0]] = idx + 1
            vocab_id2word[idx + 1] = line[0]

    dev = pickle.load(open("./data/dev.multi.pkl", "rb"))
    context, response, narrative, label = dev[0], dev[1], dev[2], dev[3]
    num = len(response)
    dev_all = []
    for i in range(num):
        # One positive sample, nine negative samples and a "EOS" sample. 1 + 9 + 1 = 11
        if i % 11 == 0 and int(label[i]) == 1:
            count = 0
            context_ = context[i]
            pos_response = "".join([vocab_id2word[x] for x in response[i]])
            last_ut = "".join([vocab_id2word[x] for x in context_[-1]]).replace(".", "").replace("?", "").replace("\"", "").replace(":", "")
            dev_all.append([context[i], response[i], narrative[i], 1])
            negative_samples = query_comt(last_ut, 15)
            for result in negative_samples:
                if result['ut2'] != pos_response:
                    negtive_sample = [vocab[x] for x in jieba.lcut(result['ut2']) if x != ' ' and x != '\xa0' and x != '\u3000']
                    dev_all.append([context[i], negtive_sample, narrative[i], 0])
                    count += 1
                if count == 8:
                    break
            if count != 8:
                last = 8 - count
                for j in range(last):
                    negtive_sample = response[i + 1 + j]
                    dev_all.append([context[i], negtive_sample, narrative[i], 0])
            if response[i] == [EOS_ID]:
                dev_all.append([context[i], [EOS_ID], narrative[i], 1])
            else:
                dev_all.append([context[i], [EOS_ID], narrative[i], 0])

    test = pickle.load(open("./data/test.multi.pkl", "rb"))
    context, response, narrative, label = test[0], test[1], test[2], test[3]
    num = len(response)
    test_all = []
    print("start test")
    for i in range(num):
        if i % 11 == 0 and int(label[i]) == 1:
            count = 0
            context_ = context[i]
            pos_response = "".join([vocab_id2word[x] for x in response[i]])
            last_ut = "".join([vocab_id2word[x] for x in context_[-1]]).replace(".", "").replace("?", "").replace("\"", "").replace(":", "")
            test_all.append([context[i], response[i], narrative[i], 1])
            negative_samples = query_comt(last_ut, 15)
            for result in negative_samples:
                if result['ut2'] != pos_response:
                    negtive_sample = [vocab[x] for x in jieba.lcut(result['ut2']) if x != ' ' and x != '\xa0' and x != '\u3000']
                    test_all.append([context[i], negtive_sample, narrative[i], 0])
                    count += 1
                if count == 8:
                    break
            if count != 8:
                last = 8 - count
                for j in range(last):
                    negtive_sample = response[i + 1 + j]
                    test_all.append([context[i], negtive_sample, narrative[i], 0])
            if response[i] == [EOS_ID]:
                test_all.append([context[i], [EOS_ID], narrative[i], 1])
            else:
                test_all.append([context[i], [EOS_ID], narrative[i], 0])

    context, response, narrative, label = [], [], [], []
    print("dev size: ", len(dev_all))
    for data in dev_all:
        context.append(data[0])
        response.append(data[1])
        narrative.append(data[2])
        label.append(data[3])
    dev = [context, response, narrative, label]
    pickle.dump(dev, open("./data/dev.pkl", "wb"))
    context, response, narrative, label = [], [], [], []
    print("test size: ", len(test_all))
    for data in test_all:
        context.append(data[0])
        response.append(data[1])
        narrative.append(data[2])
        label.append(data[3])
    test = [context, response, narrative, label]
    pickle.dump(test, open("./data/test.pkl", "wb"))
