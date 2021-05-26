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

def fetch_embeddings(corresponding_dict, embedding_matrix, single_line):
    res = []
    for sig in single_line:
        index = corresponding_dict[sig]
        embedding = embedding_matrix[index]
        res.append(embedding)
    assert len(res) > 0
    return np.mean(res, axis=0)

if __name__ == '__main__':
    # quick_ori = np.load("quick_ori.npy", allow_pickle=True)
    quick_spl = np.load("quick_spl.npy", allow_pickle=True)

    with open('corresponding_dict.pickle', 'rb') as handle:
        corresponding_dict = pickle.load(handle)

    embedding_matrix = np.load(f"embedding/glove-{256}.npy")

    index_list = []
    for index, single_line in tqdm(enumerate(quick_spl)):
        can_use_current_line = True
        for sig in single_line:
            if sig not in corresponding_dict:
                'true if it contains at least one in the dictionary'
                can_use_current_line = False

        if can_use_current_line:
            index_list.append(index)

    index_list = np.asarray(index_list)

    # quick_ori = quick_ori[index_list]
    quick_spl = quick_spl[index_list]

    embedding_list = []

    for single_line in tqdm(quick_spl):
        embedding_list.append(fetch_embeddings(corresponding_dict, embedding_matrix, single_line))

    kmeans = KMeans(n_clusters=50, n_jobs=14, verbose=2)
    kmeans.fit(embedding_list)

    pdb.set_trace()