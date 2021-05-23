import pandas as pd
import os
from File_scan import File_scan
import jieba
from tqdm import tqdm
import concurrent.futures
import re

'Cut data using jieba'
class process_dataset(object):
    def __init__(self, path = "./database/"):
        file_scan = File_scan(path)
        self.all_file_paths = file_scan.path_gen(extension='.csv')

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