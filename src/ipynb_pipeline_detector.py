from pprint import pprint
import numpy as np
import pandas as pd
from file_reader import read_xlsx, load_notebook
from constants import *
import os
import ast
import astpretty


def replace_non_parsable_line(cell):
    keywords = ["!", "--", "pip", "%matplotlib"]
    """
    Formats the cell split with '\n' for each line
    Replace the lines that are not parsable in AST
    :param cell: list(string)
    :return: list(string)
    """
    if cell is None:
        return ""
    if isinstance(cell, str):
        cell = cell.strip().split("\n")

    # print(cell)
    for idx, line in enumerate(cell):
        stripped = line.strip()
        if line.startswith("%"):
            cell[idx] = ""
        for key in keywords:
            if stripped.startswith(key):
                cell[idx] = ""

        if not cell[idx].endswith("\n"):
            cell[idx] += "\n"

    # print(cell)
    return "".join(cell)


def get_ast_notebook_file(api_dict_df, path):
    cells = load_notebook(path)
    analyzer = Analyzer(api_dict_df)

    for idx_cell, cell in enumerate(cells):
        # print(idx_cell)
        try:
            cell = replace_non_parsable_line(cell)
            analyzer.set_info(idx_cell + 1)
            analyzer.visit(ast.parse(cell))
        except SyntaxError as e:
            print(cell, ": Syntax Error", path)
        except RecursionError as r:
            print(cell, ": Recursion Error", path)

    # for result in analyzer.result_nodes:
    #     print("%12s%20s%5s" % (result.pipeline_step, result.keyword, result.cell_no))

    return analyzer.result_nodes


class IpynbPipelineDetector:
    error_files = set()

    def __init__(self, paths):
        self.all_note_book_paths = paths
        self.api_dict_df = read_xlsx(os.path.join(res_folder_path, api_dict_file))

        # self.all_note_book_paths = ["/home/sajed/GitHub/Data-Science-Pipeline-Detector/dataset/jigsaw-unintended-bias-in-toxicity-classification/Cristina Sierra/pretext-lstm-tuning-v3.ipynb"]
        for idx, path in enumerate(self.all_note_book_paths[:1]):
            get_ast_notebook_file(self.api_dict_df, path)
            # if idx % 10 == 0:
            #     print(idx)
            print(path)

        # print(len(self.error_files))


class Analyzer(ast.NodeVisitor):
    result_nodes = []
    line_no = None
    cell_no = None

    def __init__(self, api_dict_df):
        self.api_dict_df = api_dict_df
        self.result_nodes = []

    def set_info(self, cell_no):
        self.cell_no = cell_no

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
                return Result(col, keyword, node, self.cell_no)

        return None


class Result:
    pipeline_step: str = None
    keyword: str = None
    cell_no: int = None
    line_no: int = None
    column_no: int = None

    def __init__(self, pipeline, keyword, node, cell_no):
        self.keyword = keyword
        self.pipeline_step = pipeline
        self.column_no = int(node.end_col_offset)
        self.cell_no = cell_no
        self.line_no = node.lineno
