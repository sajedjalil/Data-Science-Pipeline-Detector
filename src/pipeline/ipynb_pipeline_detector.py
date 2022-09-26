import os

import pandas as pd

from src.ast.ast_parser import Parser
from src.constants.constants import *
from src.utils.file_reader import read_xlsx, delete_file


class IpynbPipelineDetector:

    def __init__(self, ipynb_paths, path):
        self.all_note_book_paths = ipynb_paths
        self.api_dict_df = read_xlsx(os.path.join(path, res_folder, api_dict_file))

    def get_results(self):
        results = []
        for idx, path in enumerate(self.all_note_book_paths):
            parser = Parser(self.api_dict_df, path)
            nodes = parser.ast_parse()
            print("Done idx:", idx)
            for node in nodes:
                cols = [path, path.split(os.sep)[-3]]
                cols.extend(str(node).split("; "))
                results.append(cols)
        df = pd.DataFrame(results)

        return df

    def remove_non_parsable_files(self):
        for idx, path in enumerate(self.all_note_book_paths):
            parser = Parser(self.api_dict_df, path)
            # print(idx)
            if parser.ast_contains_errors():
                # print(idx, path)
                delete_file(path)
