import codecs
import json

import pandas as pd

from src.constants.constants import *


def read_xlsx(path):
    df = pd.read_excel(path)
    return df


def load_notebook(file_path):
    f = codecs.open(file_path, 'r')
    source = f.read()
    y = json.loads(source)

    cells = []
    for x in y['cells']:
        if x['cell_type'] == 'code':
            cells.append(x['source'])

    f.close()
    return cells


def delete_file(path):
    os.remove(path)


class FileReader:
    base_folder_path = None
    csv_file_paths = []
    all_csv_data = pd.DataFrame()
    all_ipynb_paths = list()

    def __init__(self, path):
        self.base_folder_path = path
        self.__get_all_csv_from_dataset_folder(self.base_folder_path)
        self.__read_csv(self.csv_file_paths)
        self.__sort_and_drop_unnecessary_columns()
        self.__get_all_notebook_rows()
        self.__save_csv(combined_csv_filename)

    def __get_all_csv_from_dataset_folder(self, base_folder_path):
        self.csv_file_paths = []
        for subdir, dirs, files in os.walk(base_folder_path):
            for file in files:
                if file.startswith(combined_csv_filename):
                    continue
                if file.endswith('.csv'):
                    self.csv_file_paths.append(os.path.join(subdir, file))

    def __read_csv(self, csv_file_paths):
        self.all_csv_data = pd.DataFrame()
        for csv_file in csv_file_paths:
            df = pd.read_csv(csv_file)
            self.all_csv_data = pd.concat([self.all_csv_data, df], ignore_index=True)

    def __sort_and_drop_unnecessary_columns(self):
        self.all_csv_data = self.all_csv_data.sort_values(by=["totalVotes"], ascending=False)
        self.all_csv_data.drop(
            self.all_csv_data.index[self.all_csv_data[total_votes] < least_votes], inplace=True)
        self.all_csv_data = self.all_csv_data[selected_rows]
        self.all_csv_data.reset_index(drop=True, inplace=True)

    def __get_all_notebook_rows(self):
        is_ipynb = list()

        for index, row in self.all_csv_data.iterrows():
            competition_folder = os.path.join(self.base_folder_path, row["competitionId"])
            author = row["author"]
            file = row["ref"].split(os.sep)[-1]

            ipynb_file_path = os.path.join(competition_folder, author, file + ".ipynb")

            if os.path.exists(ipynb_file_path):
                self.all_ipynb_paths.append(ipynb_file_path)
                is_ipynb.append(index)
            # print(index)

        self.all_csv_data = self.all_csv_data.iloc[is_ipynb]

    def __save_csv(self, name):
        self.all_csv_data.to_csv(os.path.join(self.base_folder_path, name), index=False, encoding='utf-8')
