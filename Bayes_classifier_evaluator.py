import pdb
import pickle
import numpy as np
from tqdm import tqdm
import pprint
from File_scan import File_scan
import pandas as pd
import math

class Bayes_classifier_evalutor(object):
    def __init__(self):
        with open('prior_dict.pickle', 'rb') as handle:
            self.prior_dict = pickle.load(handle)

        with open('conditional_dict_list.pickle', 'rb') as handle:
            self.conditional_dict_list = pickle.load(handle)

        with open('global_dict.pickle', 'rb') as handle:
            self.global_dict = pickle.load(handle)

        with open('unjoint_split_dict.pickle', 'rb') as handle:
            self.unjoint_split_dict = pickle.load(handle)

        # file_scan = File_scan("./Cleaned_database/")
        # self.all_file_paths = file_scan.path_gen(extension='.pkl')
        #
        # all_documents = []
        # for single_file_path in tqdm(self.all_file_paths):
        #     all_documents.extend(self.process_single_data(single_file_path))
        #
        # pdb.set_trace()

    def build_prob_dict(self):
        for key in self.global_dict:
            local_dict = {}

            danmaku_list = self.global_dict[key]
            split_list = self.unjoint_split_dict[key]
            prob_dict = self.conditional_dict_list[key]

            for (danmaku, split_danmaku) in tqdm(zip(danmaku_list, split_list)):
                if danmaku not in local_dict:
                    sentence_prob = self.posterior_prob(split_danmaku, prob_dict)
                    local_dict[danmaku] = sentence_prob

            sorted_value = {k: v for k, v in sorted(local_dict.items(), key=lambda item: item[1])}
            sorted_keys = list(sorted_value.keys())
            pdb.set_trace()

    # def process_single_data(self, single_file_path):
    #     'prepare training data'
    #     original_file_path = f"./database/{single_file_path.split('/')[2]}/{single_file_path.split('/')[3][:-4]}.csv"
    #     original_df = pd.read_csv(original_file_path)
    #     cleaned_df = pd.read_pickle(single_file_path)
    #     index = cleaned_df['message'].map(len) > 0
    #
    #     original_df = original_df[index]
    #     cleaned_df = cleaned_df[index]
    #
    #     assert len(original_df) == len(cleaned_df)
    #     cleaned_df.index = range(len(cleaned_df))
    #     original_df.index = range(len(original_df))
    #     pdb.set_trace()

    def posterior_prob(self, input_danmaku, target_dict):
        log_prob = 0

        for word in input_danmaku:
            if word in target_dict:
                prob_word = math.log(target_dict[word])
            else:
                prob_word = math.log(target_dict['Lap_smooth'])

            log_prob += prob_word

        return -log_prob




if __name__ == '__main__':
    BCE = Bayes_classifier_evalutor()
    BCE.build_prob_dict()
