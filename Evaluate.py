import numpy as np

def compute_r_n_m(scores, labels, count, at):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if i % 10 == 0:
            total = total + 1
            sublist = scores[i:i + count]
            pos_score = sublist[0]
            sublist = sorted(sublist, key=lambda x: x, reverse=True)
            if sublist[at - 1] <= pos_score:
                correct += 1
    return float(correct) / total

def compute_mrr(scores, labels, count=10):
    total = 0
    accumulate_mrr = 0
    for i in range(len(labels)):
        if i % 10 == 0:
            total = total + 1
            sublist = scores[i:i + count]
            arg_sort = list(np.argsort(sublist)).index(0)
            idx = len(sublist) - arg_sort
            accumulate_mrr += 1 / idx
    return float(accumulate_mrr) / total

def compute_acc(scores, labels):
    scores = (np.asarray(scores) > 0.5).astype(np.int32)
    accuracy = sum((scores == labels).astype(np.int32)) / len(labels)
    return accuracy

def evaluate_all(scores, labels):
    return compute_acc(scores, labels), compute_r_n_m(scores, labels, 2, 1), compute_r_n_m(scores, labels, 10, 1), compute_r_n_m(scores, labels, 10, 2), \
           compute_r_n_m(scores, labels, 10, 5), compute_mrr(scores, labels)

def evaluate_all_from_file(path):
    scores = []
    labels = []
    with open(path, "r") as f:
        for line in f:
            score, label = line.strip().split("\t")
            scores.append(float(score))
            labels.append(float(labels))
    evaluate_all(scores, labels)

def recover_and_show(basic_directory):
    import pickle

    vocab = {}
    vocab_id2word = {}

    with open("./data/vocab.txt", "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr):
            line = line.strip().split("\t")
            vocab[line[0]] = idx + 1
            vocab_id2word[idx + 1] = line[0]

    vocab["_PAD_"] = 0
    vocab_id2word[0] = "_PAD_"

    def initialize():
        all_outline = []
        outline_dict = {}
        initial_file = basic_directory + "test.multi.0.pkl"
        with open(initial_file, 'rb') as f:
            _, _, outline, _ = pickle.load(f)
        for o in outline:
            if o not in all_outline:
                all_outline.append(o)
                sent_o = "".join([vocab_id2word[x] for x in o])
                outline_dict[sent_o] = []
        return all_outline, outline_dict

    max_turn = 10
    all_outline, outline_dict = initialize()

    for turn in range(0, max_turn + 1):
        score = []
        with open(basic_directory + "test.result.multi." + str(turn) + ".txt", "r") as fr:
            for idx, line in enumerate(fr):
                if idx == 0:
                    continue
                score.append(float(line.strip()))
        with open(basic_directory + "test.multi." + str(turn) + ".pkl", "rb") as fr:
            utterance, response, outline, labels = pickle.load(fr)  # except for dl2r

        for i, o in enumerate(outline):
            if i % 10 == 0:
                score_sub_list = score[i:i + 10]
                response_sub_list = response[i:i + 10]
                max_idx = score_sub_list.index(max(score_sub_list))
                selected_response = response_sub_list[max_idx]
                sent_o = "".join([vocab_id2word[x] for x in o])
                outline_dict[sent_o] = utterance[i] + [selected_response]  # for MUSwO

    with open(basic_directory + "test.result.multi.txt", "w", encoding="utf-8") as fw:
        for o in all_outline:
            sent_o = "".join([vocab_id2word[x] for x in o])
            utterance = outline_dict[sent_o]
            fw.write("outline\t" + sent_o + "\n")
            for u in utterance:
                sent_u = "".join([vocab_id2word[x] for x in u])
                fw.write("script\t" + sent_u + "\n")
            fw.write("\n")

def evaluate_multi_turn_result(t_file, g_file):
    test_file = t_file
    gold_file = g_file
    ft = open(test_file, "r", encoding="utf-8")
    fg = open(gold_file, "r", encoding="utf-8")
    t, g = ft.readline(), fg.readline()
    c = -1
    s = -1
    all_s = 0
    r_all = 0
    tmp_r_all = 0
    o = None
    flag = 1
    t_s = []
    g_s = []
    while t and g:
        t = t.strip().split("\t")
        g = g.strip().split("\t")
        if len(t) > 1:
            if t[0] == "script":
                if t[1] == g[1]:
                    s += 1
                c += 1
                t_s.append(t[1])
                g_s.append(g[1])
            if t[0] == "outline":
                o = t[1]
        else:
            tmp_c = -1
            tmp_s = -1
            for at in g_s:
                tmp_c += 1
                if at in t_s:
                    tmp_s += 1
            tmp_r = tmp_s / tmp_c
            tmp_r_all += tmp_r
            r = s / c
            r_all += r
            all_s += 1
            c = -1
            s = -1
            t_s = []
            g_s = []
        t, g = ft.readline(), fg.readline()

    tmp_c = -1
    tmp_s = -1
    for at in g_s:
        tmp_c += 1
        if at in t_s:
            tmp_s += 1
    tmp_r = tmp_s / tmp_c
    tmp_r_all += tmp_r
    r = s / c
    r_all += r
    all_s += 1

    r_all += r
    print("p_strict", r_all / all_s)
    print("p_weak", tmp_r_all / all_s)
