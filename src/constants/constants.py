import os

base_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
res_folder = 'res'
dataset_folder = 'dataset'
dataset_base_path = os.path.join(base_path, dataset_folder)

combined_csv_filename = "combined-data.csv"
selected_rows = ['ref', 'title', 'author', 'lastRunTime', 'totalVotes', 'competitionId']
api_dict_file = "API-dictionary.xlsx"

total_votes = "totalVotes"
least_votes = 15
