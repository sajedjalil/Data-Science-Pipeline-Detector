import os

base_path = os.getcwd()
dataset_base_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]), 'dataset')
res_folder_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]), 'res')

combined_csv_filename = "combined-data.csv"
selected_rows = ['ref', 'title', 'author', 'lastRunTime', 'totalVotes', 'competitionId']
api_dict_file = "API-dictionary.xlsx"

total_votes = "totalVotes"
least_votes = 15
