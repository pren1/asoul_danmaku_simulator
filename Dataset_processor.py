import pandas as pd
import numpy as np
import pdb
import os
from File_scan import File_scan
import jieba
from tqdm import tqdm
import concurrent.futures
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

class process_dataset(object):
    def __init__(self, path = "./database/"):
        file_scan = File_scan(path)
        self.all_file_paths = file_scan.path_gen(extension='.csv')
        self.stride_size = 50
        self.window_size = 200
        self.time_interval_thres = 100

    def process_all(self):
        executor = concurrent.futures.ProcessPoolExecutor(62)
        futures = [executor.submit(self.divide_single_file, single_file_path) for
                   single_file_path in self.all_file_paths]
        concurrent.futures.wait(futures)
        # self.divide_single_file('./database/22625025/22625025_2020-12-26.csv')

    def divide_single_file(self, single_file_path):
        roomid = single_file_path.split('/')[2]
        file_name = single_file_path.split('/')[-1][:-4]
        self.create_folder(roomid)
        df = pd.read_csv(single_file_path)
        for index, row in tqdm(df.iterrows()):
            row['message'] = self.split_string(row['message'])
            df.loc[index] = row

        df.to_pickle(f"Cleaned_database/{roomid}/{file_name}.pkl")
        return
        df = pd.read_pickle(f"Cleaned_database/{roomid}/{file_name}.pkl")
        pdb.set_trace()
        document = []
        for i in range(0, len(df)-self.window_size, self.stride_size):
            data_slice = df.loc[i:i+self.window_size-1]
            assert len(data_slice) == self.window_size
            time_interval = data_slice.loc[i+self.window_size-1][0] - data_slice.loc[i][0]
            if time_interval <= self.time_interval_thres:
                message_list = data_slice['message'].tolist()
                sum_list = sum(message_list, [])
                document.append(" ".join(sum_list))

        tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(document)
        sparse_result = tfidf_model.transform(document)
        # # 指定分成50个类
        # for i in range(1, 50):
        #     kmeans = KMeans(n_clusters=i, n_jobs=14)
        #     kmeans.fit(sparse_result)
        #     print("inertia: {}".format(kmeans.inertia_))
        # pdb.set_trace()


    def split_string(self, input):
        remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】 ［］。《》？“”‘’\[\\]^_`{|}~]+'
        distinct_string = ''.join(dict.fromkeys(input))
        res_string = re.sub(remove_chars, "", distinct_string)
        if len(res_string) == 0:
            return []
        if len(res_string) > 1:
            words = jieba.cut(res_string)
            return list(words)
        else:
            return [res_string]

    def create_folder(self, roomid):
        path = f"./Cleaned_database/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = f"./Cleaned_database/{roomid}/"
        if not os.path.exists(path):
            os.mkdir(path)

if __name__ == '__main__':
    PD = process_dataset()
    PD.process_all()