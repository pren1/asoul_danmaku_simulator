import pdb
from File_scan import File_scan
from txt_processor import txt_processor
from tqdm import tqdm
import os
import pandas as pd
import pprint
import csv
import pickle

class Dataset_builder(object):
    def __init__(self, danmaku_folder):
        file_scan = File_scan(danmaku_folder)
        self.all_file_paths = file_scan.path_gen()
        self.df = pd.read_csv("/Users/renpeng/PycharmProjects/crawl/crew_list2.csv")
        self.uid_list = self.df['uid']
        self.res_dict = {}
        for sig_uid in self.uid_list:
            self.res_dict[f"{sig_uid}"] = [0, 0]

    def handle_data(self):
        for single_file_path in tqdm(self.all_file_paths):
            'first check the file size'
            # filesize = os.path.getsize(single_file_path)/(1024)
            # if filesize < f500:
            #     'cut size samller than 500 kb'
            #     continue
            processor = txt_processor(single_file_path)
            candidate_danmakus = processor.read_target_txt()
            if len(candidate_danmakus) > 0:
                room_id, _ = self.return_room_and_time(single_file_path)
                if self.within_list(room_id):
                    asoul = True
                else:
                    asoul = False
                for single_danmaku in candidate_danmakus:
                    assert len(single_danmaku) == 3, "Error, array length incorrect %d %s" % (
                    len(single_danmaku), single_danmaku)
                    UID = single_danmaku[1]
                    message = single_danmaku[2]
                    if len(message) == 0:
                        continue
                    if UID in self.res_dict:
                        if asoul:
                            self.res_dict[UID][0] += 1
                        else:
                            self.res_dict[UID][1] += 1
                    # pdb.set_trace()
        with open('target.pickle', 'wb') as handle:
            pickle.dump(self.res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pprint.pprint(self.res_dict)
        pdb.set_trace()

    def within_list(self, room_id):
        return room_id in ['22625025', '22625027', '22632157', '22632424', '22634198', '22637261']

    def return_room_and_time(self, single_file_path):
        # print(single_file_path)
        room_id = single_file_path.split(os.sep)[-2]
        YMD_time = self.add_leading_zero_to_date(single_file_path.split(os.sep)[-1][:-4])
        return room_id, YMD_time

    def add_leading_zero_to_date(self, YMD_time):
        date_list = YMD_time.split('-')
        assert len(date_list) == 3, "YMD time format error"
        date_list[1] = '{:02}'.format(int(date_list[1]))
        date_list[2] = '{:02}'.format(int(date_list[2]))
        return '-'.join(date_list)

    def create_folder(self, roomid):
        path = f"./database/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = f"./database/{roomid}/"
        if not os.path.exists(path):
            os.mkdir(path)

if __name__ == '__main__':
    # input_path = "./ASOUL_dataset/"
    dataset_builder = Dataset_builder("../bilibili-vtuber-danmaku/")
    dataset_builder.handle_data()