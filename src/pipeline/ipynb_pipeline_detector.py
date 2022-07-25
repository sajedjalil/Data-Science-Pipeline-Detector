from src.utils.file_reader import read_xlsx, delete_file
from src.constants.constants import *
from src.pipeline.ast_parser import Parser
import os


class IpynbPipelineDetector:

    def __init__(self, paths):
        self.all_note_book_paths = paths
        self.api_dict_df = read_xlsx(os.path.join(res_folder_path, api_dict_file))

    def get_results(self):
        results = []
        for idx, path in enumerate(self.all_note_book_paths):
            parser = Parser(self.api_dict_df, path)
            results.extend([path, parser.ast_parse()])
            print(idx)
        print(len(results))

    def remove_non_parsable_files(self):
        for idx, path in enumerate(self.all_note_book_paths):
            parser = Parser(self.api_dict_df, path)
            print(idx)
            if parser.ast_contains_errors():
                print(idx, path)
                delete_file(path)
