import pandas as pd
import numpy as np
import pdb
import os
from File_scan import File_scan
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
from pprint import pprint

class Cluster_trainer(object):
    def __init__(self, path = "./Cleaned_database/"):
        file_scan = File_scan(path)
        self.all_file_paths = file_scan.path_gen(extension='.pkl')
        self.stride_size = 100
        self.window_size = 200
        self.time_interval_thres = 100
        self.stop_list = ['了','吧','的','啊','吗']

    def Model_trainer(self):
        print("Loading data")

        quick_ori = []
        quick_spl = []

        for single_file_path in tqdm(self.all_file_paths):
            ori, spl = self.quick_sol(single_file_path)
            quick_ori.extend(ori)
            quick_spl.extend(spl)

        np.save("quick_ori", quick_ori)
        np.save("quick_spl", quick_spl)
        pdb.set_trace()


        if not os.path.isfile("unjoint_all_danmaku.npy"):
            unjoint_all_documents = []
            for single_file_path in tqdm(self.all_file_paths):
                unjoint_all_documents.extend(self.process_single_data(single_file_path, join_strings=False))

            np.save("unjoint_all_danmaku", unjoint_all_documents)
        else:
            unjoint_all_documents = np.load("unjoint_all_danmaku.npy")

        if not os.path.isfile("all_danmaku.npy"):
            all_documents = []
            for single_file_path in tqdm(self.all_file_paths):
                all_documents.extend(self.process_single_data(single_file_path))

            np.save("all_danmaku", all_documents)
        else:
            all_documents = np.load("all_danmaku.npy")

        'Tf-idf'
        print("processing 'Tf-idf'")
        tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=6000, stop_words=self.stop_list).fit(all_documents)
        sparse_result = tfidf_model.transform(all_documents)

        np.save("vocabulary", tfidf_model.vocabulary_)
        'Kmean'
        print("processing Kmean")

        kmeans = KMeans(n_clusters=27, n_jobs=14, verbose=2)
        kmeans.fit(sparse_result)
        self.save_model(kmeans)

    def save_model(self, model):
        pickle.dump(model, open("kmean_model.pkl", "wb"))

    def load_model(self):
        return pickle.load(open("kmean_model.pkl", "rb"))

    def quick_sol(self, single_file_path):
        original_file_path = f"./database/{single_file_path.split('/')[2]}/{single_file_path.split('/')[3][:-4]}.csv"
        original_df = pd.read_csv(original_file_path)
        cleaned_df = pd.read_pickle(single_file_path)
        index = cleaned_df['message'].map(len) > 0

        original_df = original_df[index]
        cleaned_df = cleaned_df[index]

        assert len(original_df) == len(cleaned_df)
        return original_df['message'].tolist(), cleaned_df['message'].tolist()


    def process_single_data(self, single_file_path, join_strings = True):
        'prepare training data'
        df = pd.read_pickle(single_file_path)
        df = df[df['message'].map(len) > 0]
        df.index = range(len(df))

        document = []
        for i in range(0, len(df) - self.window_size, self.stride_size):
            data_slice = df.loc[i:i + self.window_size - 1]
            assert len(data_slice) == self.window_size
            time_interval = data_slice.loc[i + self.window_size - 1][0] - data_slice.loc[i][0]
            if time_interval <= self.time_interval_thres:
                message_list = data_slice['message'].tolist()
                if join_strings:
                    sum_list = sum(message_list, [])
                    document.append(" ".join(sum_list))
                else:
                    sum_list = message_list
                    document.append(sum_list)
        return document

    def load_corresponding_data(self, single_file_path):
        'prepare testing data and original data'
        original_file_path = f"./database/{single_file_path.split('/')[2]}/{single_file_path.split('/')[3][:-4]}.csv"
        original_df = pd.read_csv(original_file_path)
        cleaned_df = pd.read_pickle(single_file_path)
        index = cleaned_df['message'].map(len) > 0

        original_df = original_df[index]
        cleaned_df = cleaned_df[index]

        assert len(original_df) == len(cleaned_df)
        original_df.index = range(len(original_df))
        document = []
        for i in range(0, len(original_df) - self.window_size, self.stride_size):
            data_slice = original_df.loc[i:i + self.window_size - 1]
            assert len(data_slice) == self.window_size
            time_interval = data_slice.loc[i + self.window_size - 1][0] - data_slice.loc[i][0]
            if time_interval <= self.time_interval_thres:
                message_list = data_slice['message'].tolist()
                document.append(message_list)
        return document

    def get_danmaku_from_all_classes(self):
        print("Loading original data")
        if not os.path.isfile("original_documents.npy"):
            original_documents = []
            for single_file_path in tqdm(self.all_file_paths):
                original_documents.extend(self.load_corresponding_data(single_file_path))
            np.save("original_documents", original_documents)
        else:
            original_documents = np.load("original_documents.npy")

        assert os.path.isfile('all_danmaku.npy'), "file not exist"
        all_documents = np.load("all_danmaku.npy")
        unjoint_all_documents = np.load("unjoint_all_danmaku.npy", allow_pickle=True)
        assert len(all_documents) == len(original_documents), "Something wrong here"

        model_label = self.load_model().labels_
        assert len(model_label) == len(original_documents),"Fatal error"
        global_dict = {}
        split_dict = {}
        unjoint_split_dict = {}

        print("merging danmaku from different labels")
        for index in tqdm(range(len(model_label))):
            if model_label[index] not in global_dict:
                global_dict[model_label[index]] = []
                split_dict[model_label[index]] = []
                unjoint_split_dict[model_label[index]] = []

            global_dict[model_label[index]].extend(original_documents[index])
            split_dict[model_label[index]].extend(all_documents[index].split(' '))
            unjoint_split_dict[model_label[index]].extend(unjoint_all_documents[index])

        with open('unjoint_split_dict.pickle', 'wb') as handle:
            pickle.dump(unjoint_split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('global_dict.pickle', 'wb') as handle:
            pickle.dump(global_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('split_dict.pickle', 'wb') as handle:
            pickle.dump(split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        res_dict = {}
        'max freq'
        for sig in global_dict:
            res_dict[sig] = self.find_top_n_common(global_dict[sig])

        pprint(res_dict)

    def find_top_n_common(self, input_list):
        dict = {}
        for sig in tqdm(input_list):
            if sig not in dict:
                dict[sig] = 1
            else:
                dict[sig] += 1
        sort_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sort_dict[:10]]

if __name__ == '__main__':
    CT = Cluster_trainer()
    CT.Model_trainer()
    CT.get_danmaku_from_all_classes()