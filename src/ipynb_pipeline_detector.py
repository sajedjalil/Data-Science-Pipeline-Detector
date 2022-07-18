from pprint import pprint
import numpy as np
import pandas as pd
from file_reader import read_xlsx, load_notebook
from constants import *
import os
import ast
import astpretty


class IpynbPipelineDetector:

    def __init__(self, paths):
        self.all_note_book_paths = paths
        self.api_dict_df = read_xlsx(os.path.join(res_folder_path, api_dict_file))
        # self.api_dict_df = get_dataframe_for_each_columns(self.api_dict_df)

        self.__get_ast_notebook_file(self.all_note_book_paths[0])
        # print(self.all_note_book_paths[1])
    def __get_ast_notebook_file(self, path):
        # print(self.api_dict_df['Others'].dropna())
        analyzer = Analyzer(self.api_dict_df)
        cells = load_notebook(path)

        for idx_cell, cell in enumerate(cells):
            for idx_line, line in enumerate(cell):
                try:
                    # pprint(ast.dump(ast.parse(line)))

                    analyzer.visit(ast.parse(line))
                    analyzer.set_info(idx_cell, idx_line)

                except SyntaxError as e:
                    print(line, ": Syntax Error")

        for result in analyzer.result_nodes:
            pprint(vars(result))


class Analyzer(ast.NodeVisitor):
    result_nodes = []
    line_no = None
    cell_no = None

    def __init__(self, api_dict_df):
        self.api_dict_df = api_dict_df
        self.result_nodes = []

    def set_info(self, cell_no, line_no):
        self.cell_no = cell_no
        self.line_no = line_no

    def visit_Attribute(self, node):
        response = self.make_result_node(node)
        if response is not None:
            self.result_nodes.append(response)

        self.generic_visit(node)

    def generic_visit(self, node):
        # print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def make_result_node(self, node):
        # pprint(vars(node))
        for col in self.api_dict_df.columns:
            is_found = self.api_dict_df[col].isin([node.attr])
            keyword = self.api_dict_df[col].values[is_found]

            if len(keyword) > 0:
                return Result(col, keyword, node, self.cell_no, self.line_no)

        return None


class Result:
    pipeline_step: str = None
    keyword: str = None
    cell_no: int = None
    line_no: int = None
    column_no: int = None

    def __init__(self, pipeline, keyword, node, cell_no, line_no):
        self.keyword = keyword
        self.pipeline_step = pipeline
        self.column_no = int(node.end_col_offset)
        self.cell_no = cell_no
        self.line_no = line_no
