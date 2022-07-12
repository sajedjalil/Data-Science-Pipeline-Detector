import os
import pandas as pd


class FileReader:
    base_folder_path = None
    csv_file_paths = []
    all_csv_data = pd.DataFrame()

    # default constructor
    def __init__(self, path):
        self.base_folder_path = path
        self.get_all_csv_from_dataset_folder()
        self.read_csv()
        self.sort_and_drop_unnecessary_columns()
        self.save_csv('combined-data.csv')

    def get_all_csv_from_dataset_folder(self):
        for subdir, dirs, files in os.walk(self.base_folder_path):
            for file in files:
                if file.endswith('.csv'):
                    self.csv_file_paths.append( os.path.join(subdir, file))

    def read_csv(self):
        for csv_file in self.csv_file_paths:
            df = pd.read_csv(csv_file)
            self.all_csv_data = pd.concat([self.all_csv_data, df], ignore_index=True)

    def sort_and_drop_unnecessary_columns(self):
        self.all_csv_data = self.all_csv_data.sort_values(by=["totalVotes"], ascending=False)
        self.all_csv_data.drop(self.all_csv_data.index[self.all_csv_data['totalVotes'] < 15], inplace=True)
        self.all_csv_data = self.all_csv_data[['ref', 'title', 'author', 'lastRunTime', 'totalVotes', 'competitionId']]

    def save_csv(self, name):
        self.all_csv_data.to_csv(os.path.join(self.base_folder_path, name), index=False, encoding='utf-8')
        print(self.all_csv_data[-1:]['totalVotes'])
