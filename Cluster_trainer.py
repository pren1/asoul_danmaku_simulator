import pandas as pd
import numpy as np
import pdb
import os
from File_scan import File_scan
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from pprint import pprint

class Cluster_trainer(object):
    def __init__(self, path = "./Cleaned_database/"):
        file_scan = File_scan(path)
        self.all_file_paths = file_scan.path_gen(extension='.pkl')
        self.stride_size = 50
        self.window_size = 200
        self.time_interval_thres = 100
        self.stop_list = ['了','吧','的','啊','吗']

    def Model_trainer(self):
        print("Loading data")

        if not os.path.isfile("all_danmaku.npy"):
            all_documents = []
            for single_file_path in tqdm(self.all_file_paths):
                all_documents.extend(self.process_single_data(single_file_path))

            np.save("all_danmaku", all_documents)
        else:
            all_documents = np.load("all_danmaku.npy")

        'Tf-idf'
        print("processing 'Tf-idf'")
        tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=3000, stop_words=self.stop_list).fit(all_documents)
        sparse_result = tfidf_model.transform(all_documents)
        'Kmean'
        print("processing Kmean")

        K = [38]
        variances = []
        # silhouette_coefficients = []
        for i in tqdm(K):
            kmeans = KMeans(n_clusters=i, n_jobs=14, verbose=2)
            kmeans.fit(sparse_result)
            variances.append(kmeans.inertia_)
            # score = silhouette_score(sparse_result, kmeans.labels_)
            # silhouette_coefficients.append(score)

        self.save_model(kmeans)
        # plt.plot(K, silhouette_coefficients)
        # plt.ylabel("Inertia ( Total Distance )")
        # plt.xlabel("K Value")
        # plt.show()
        #
        # plt.plot(K, variances)
        # plt.ylabel("Inertia ( Total Distance )")
        # plt.xlabel("K Value")
        # plt.show()

    def save_model(self, model):
        pickle.dump(model, open("kmean_model.pkl", "wb"))

    def load_model(self):
        return pickle.load(open("kmean_model.pkl", "rb"))

    def process_single_data(self, single_file_path):
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
                sum_list = sum(message_list, [])
                document.append(" ".join(sum_list))
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

        assert len(all_documents) == len(original_documents), "Something wrong here"

        model_label = self.load_model().labels_
        assert len(model_label) == len(original_documents),"Fatal error"
        global_dict = {}
        split_dict = {}
        print("merging danmaku from different labels")
        for index in tqdm(range(len(model_label))):
            if model_label[index] not in global_dict:
                global_dict[model_label[index]] = []
                split_dict[model_label[index]] = []
            global_dict[model_label[index]].extend(original_documents[index])
            split_dict[model_label[index]].extend(all_documents[index].split(' '))

        res_dict = {}
        'max freq'
        for sig in global_dict:
            res_dict[sig] = self.find_top_n_common(global_dict[sig])

        pprint(res_dict)
        pdb.set_trace()

    def find_top_n_common(self, input_list):
        dict = {}
        for sig in tqdm(input_list):
            if sig not in dict:
                dict[sig] = 1
            else:
                dict[sig] += 1
        sort_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        return sort_dict[:10]

    # def word_cloud(self, all_documents):
    #     long_string = " ".join(all_documents)
    #
    #     my_wordcloud = WordCloud(
    #         background_color='white',  # 设置背景颜色
    #         # mask=img,  # 背景图片
    #         max_words=200,  # 设置最大显示的词数
    #         # stopwords=STOPWORDS,  # 设置停用词
    #         # # 设置字体格式，字体格式 .ttf文件需自己网上下载，最好将名字改为英文，中文名路径加载会出现问题。
    #         font_path='/System/Library/AssetsV2/com_apple_MobileAsset_Font6/bf625d290b705582d5fd619878281f3325b075d0.asset/AssetData/STHEITI.ttf',
    #         relative_scaling=1.0,
    #         max_font_size=100,  # 设置字体最大值
    #         random_state=50,  # 设置随机生成状态，即多少种配色方案
    #     ).generate(long_string)
    #
    #     pdb.set_trace()
    #     # image_colors = ImageColorGenerator(img)
    #     plt.imshow(my_wordcloud)
    #     plt.axis('off')
    #     plt.show()
    #     my_wordcloud.to_file('res.jpg')

if __name__ == '__main__':
    CT = Cluster_trainer()
    CT.Model_trainer()
    CT.get_danmaku_from_all_classes()