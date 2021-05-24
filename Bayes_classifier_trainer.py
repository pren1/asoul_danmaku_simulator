import pdb
import pickle
import numpy as np
from tqdm import tqdm
import pprint

class Bayes_classifier_trainer(object):
    def __init__(self):
        # with open('global_dict.pickle', 'rb') as handle:
        #     self.global_dict = pickle.load(handle)

        with open('split_dict.pickle', 'rb') as handle:
            self.split_dict = pickle.load(handle)

        self.vocabulry_list = np.load("vocabulary.npy", allow_pickle=True)

        self.prior_dict = {}

        self.conditional_dict_list = []

    def calculate_prior_freq(self):
        'using the counts for each class, do the prior calculation'

        for key in self.global_dict:
            class_length = len(self.global_dict[key])
            self.prior_dict[key] = class_length

        self.prior_dict = {k: v / total for total in (sum(self.prior_dict.values()),) for k, v in self.prior_dict.items()}
        pprint.pprint(self.prior_dict)

        with open('prior_dict.pickle', 'wb') as handle:
            pickle.dump(self.prior_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_conditional_freq(self):
        'get the p(word|class)'

        for key in self.split_dict:
            current_dict = {}
            content = self.split_dict[key]
            for sig in tqdm(content):
                if sig not in current_dict:
                    current_dict[sig] = 1
                else:
                    current_dict[sig] += 1
            sum_val = sum(current_dict.values())
            current_dict = {k: (v + 1) / (total + 2) for total in (sum(current_dict.values()),) for k, v in current_dict.items()}
            current_dict['Lap_smooth'] = 1 / sum_val
            self.conditional_dict_list.append(current_dict)

        with open('conditional_dict_list.pickle', 'wb') as handle:
            pickle.dump(self.conditional_dict_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # pdb.set_trace()

if __name__ == '__main__':
    BC = Bayes_classifier_trainer()
    # BC.calculate_prior_freq()
    BC.calculate_conditional_freq()