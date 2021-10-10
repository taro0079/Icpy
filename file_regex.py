import glob
import re
import os


class FileRegex:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_path_list = glob.glob(self.folder_path + '*.dat')
        self.file_name_list = [os.path.basename(ff) for ff in self.file_path_list]

    def extract_sample_number(self):
        matched = [re.search(r'^\d{2}', ff) for ff in self.file_name_list]
        return [int(ff.group()) for ff in matched]

    def extract_magnetic_filed(self):
        matched = [re.search(r'\d{2}(?=T)', ff) for ff in self.file_name_list]
        return [int(ff.group()) for ff in matched]
