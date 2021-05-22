import pdb
from File_scan import File_scan
from txt_processor import txt_processor
from tqdm import tqdm
import os
import csv

class Dataset_builder(object):
    def __init__(self, danmaku_folder):
        file_scan = File_scan(danmaku_folder)
        self.all_file_paths = file_scan.path_gen()

    def handle_data(self):
        for single_file_path in tqdm(self.all_file_paths):
            'first check the file size'
            filesize = os.path.getsize(single_file_path)/(1024)
            # if filesize < 500:
            #     'cut size samller than 500 kb'
            #     continue
            processor = txt_processor(single_file_path)
            candidate_danmakus = processor.read_target_txt()
            if len(candidate_danmakus) > 0:
                room_id, YMD_time = self.return_room_and_time(single_file_path=single_file_path)
                self.create_folder(room_id)
                with open(f'./database/{room_id}/{room_id}_{YMD_time}.csv', mode='w') as csv_file:
                    fieldnames = ['Time (s)', 'message']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for single_danmaku in candidate_danmakus:
                        assert len(single_danmaku) == 3, "Error, array length incorrect %d %s" % (
                        len(single_danmaku), single_danmaku)
                        single_pass_time = single_danmaku[0]
                        # UID = single_danmaku[1]
                        message = single_danmaku[2]
                        # print(single_danmaku)
                        if len(message) == 0:
                            continue
                        writer.writerow({'Time (s)': single_pass_time, 'message': message})

    def return_room_and_time(self, single_file_path):
        print(single_file_path)
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
    input_path = "/Users/renpeng/Downloads/ASOUL_dataset/"
    dataset_builder = Dataset_builder(input_path)
    dataset_builder.handle_data()