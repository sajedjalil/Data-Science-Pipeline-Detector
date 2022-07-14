from pprint import pprint

from file_reader import read_xlsx, load_notebook
from constants import *
import os
import ast
import astpretty


class IpynbPipelineDetector:

    def __init__(self, paths):
        self.api_dict_df = read_xlsx(os.path.join(res_folder_path, api_dict_file))
        self.all_note_book_paths = paths
        self.__get_ast_notebook_file(self.all_note_book_paths[0])

    def __get_ast_notebook_file(self, path):
        print(path)
        cells = load_notebook(path)

        for cell in cells:

            for line in cell:
                try:
                    # print(line)
                    print(ast.dump(ast.parse(line)))
                    # analyzer = Analyzer()
                    # analyzer.visit(tree)
                except SyntaxError as e:
                    print(line, ": Syntax Error")


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": [], "from": []}

    def visit_Import(self, node):
        print(node.lineno)
        for alias in node.names:
            self.stats["import"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        print(node.names, node.lineno)
        for alias in node.names:
            self.stats["from"].append(alias.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        print(node.__str__())

    def generic_visit(self, node):
        # print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    _const_node_type_names = {
        bool: 'NameConstant',  # should be before int
        type(None): 'NameConstant',
        int: 'Num',
        float: 'Num',
        complex: 'Num',
        str: 'Str',
        bytes: 'Bytes',
        type(...): 'Ellipsis',
    }

    def visit_Constant(self, node):
        print(node.value, node.col_offset)
        value = node.value
        type_name = self._const_node_type_names.get(type(value))

        print(type_name)

    def report(self):
        pprint(self.stats)
