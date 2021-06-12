import pickle
import pdb
from File_scan import File_scan
from tqdm import tqdm
import pandas as pd
import tf_glove
import numpy as np
import time
from Vector_training import process_single_data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from pprint import pprint
import math

def fetch_embeddings(corresponding_dict, embedding_matrix, single_line):
    res = []
    for sig in single_line:
        index = corresponding_dict[sig]
        embedding = embedding_matrix[index]
        res.append(embedding)
    assert len(res) > 0
    return np.mean(res, axis=0)

def find_top_n_common(input_list, expand_value=20):
    dict = {}
    for sig in tqdm(input_list):
        if sig not in dict:
            dict[sig] = 1
        else:
            dict[sig] += 1
    sort_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    return [x for x in sort_dict[:expand_value]], [x[0] for x in sort_dict[:expand_value]]

def posterior_prob(input_danmaku, target_dict):
    log_prob = 0

    for word in input_danmaku:
        if word in target_dict:
            prob_word = math.log(target_dict[word])
        else:
            prob_word = math.log(target_dict['Lap_smooth'])

        log_prob += prob_word

    return -log_prob

def build_prob_dict(ori_dict, unjoint_split_dict, conditional_dict_list):
    res = {}
    for key in ori_dict:
        local_dict = {}

        danmaku_list = ori_dict[key]
        split_list = unjoint_split_dict[key]
        prob_dict = conditional_dict_list[key]

        for (danmaku, split_danmaku) in tqdm(zip(danmaku_list, split_list)):
            if danmaku not in local_dict:
                sentence_prob = posterior_prob(split_danmaku, prob_dict)
                local_dict[danmaku] = sentence_prob
        sorted_value = {k: v for k, v in sorted(local_dict.items(), key=lambda item: item[1])}
        sorted_keys = list(sorted_value.keys())
        res[key] = sorted_keys[:100]
    return res

def calculate_conditional_freq(split_dict):
    'get the p(word|class)'

    conditional_dict_list = {}
    for key in split_dict:
        current_dict = {}
        content = split_dict[key]
        for sig in tqdm(content):
            if sig not in current_dict:
                current_dict[sig] = 1
            else:
                current_dict[sig] += 1
        sum_val = sum(current_dict.values())
        current_dict = {k: (v + 1) / (total + 2) for total in (sum(current_dict.values()),) for k, v in current_dict.items()}
        current_dict['Lap_smooth'] = 1 / sum_val
        conditional_dict_list[key] = current_dict
    return conditional_dict_list

if __name__ == '__main__':
    quick_spl = np.load("quick_spl.npy", allow_pickle=True)

    with open('corresponding_dict.pickle', 'rb') as handle:
        corresponding_dict = pickle.load(handle)

    embedding_matrix = np.load(f"embedding/glove-{256}.npy")

    duplicate_set = set()
    index_list = []
    for index, single_line in tqdm(enumerate(quick_spl)):
        can_use_current_line = True
        for sig in single_line:
            if sig not in corresponding_dict:
                'true if it contains at least one in the dictionary'
                can_use_current_line = False

        joint_single_line = "".join(single_line)
        if can_use_current_line and joint_single_line not in duplicate_set:
            index_list.append(index)
            duplicate_set.add(joint_single_line)

    index_list = np.asarray(index_list)

    quick_spl = quick_spl[index_list]

    # embedding_list = []
    #
    # for single_line in tqdm(quick_spl):
    #     embedding_list.append(fetch_embeddings(corresponding_dict, embedding_matrix, single_line))
    #
    # grid_search_list = [42]
    # # A list holds the SSE values for each k
    # sse = []
    # for k in grid_search_list:
    #     kmeans = KMeans(n_clusters=k, n_jobs=14, verbose=2)
    #     kmeans.fit(embedding_list)
    #     sse.append(kmeans.inertia_)
    #
    # # plt.style.use("fivethirtyeight")
    # # plt.plot(grid_search_list, sse)
    # # plt.xticks(grid_search_list)
    # # plt.show()
    # #
    # # kl = KneeLocator(grid_search_list, sse, curve="convex", direction="decreasing")
    # # print(kl.elbow)
    #
    # pickle.dump(kmeans, open("kmean_model.pkl", "wb"))

    kmeans = pickle.load(open("kmean_model.pkl", "rb"))
    model_label = kmeans.labels_

    quick_ori = np.load("quick_ori.npy", allow_pickle=True)
    quick_ori = quick_ori[index_list]

    assert len(quick_ori) == len(model_label)

    global_dict = {}
    ori_dict = {}
    unsplit_dict = {}

    print("merging danmaku from different labels")
    for index in tqdm(range(len(model_label))):
        if model_label[index] not in global_dict:
            global_dict[model_label[index]] = []
            ori_dict[model_label[index]] = []
            unsplit_dict[model_label[index]] = []

        global_dict[model_label[index]].extend(quick_spl[index])
        ori_dict[model_label[index]].append(quick_ori[index])
        unsplit_dict[model_label[index]].append(quick_spl[index])

    conditional_dict_list = calculate_conditional_freq(global_dict)
    res = build_prob_dict(ori_dict, unsplit_dict, conditional_dict_list)
    pprint(res)
    pdb.set_trace()
    res_dict = {}
    real_dict = {}

    for key in global_dict:
        res_dict[key], available_list = find_top_n_common(global_dict[key])
        for (single_line, ori_line) in zip(unsplit_dict[key], ori_dict[key]):
            can_use_current_line = True
            for sig in single_line:
                if sig not in available_list:
                    'true if it contains at least one in the dictionary'
                    can_use_current_line = False
            if can_use_current_line:
                if key not in real_dict:
                    real_dict[key] = []
                real_dict[key].append(ori_line)

        pprint(res_dict[key])
        pprint(real_dict[key])
        pdb.set_trace()