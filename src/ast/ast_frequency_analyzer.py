import ast
from pprint import pprint

from src.ast.ast_static import make_result_node


class Frequency(ast.NodeVisitor):
    result_nodes = None
    line_no = None
    cell_no = None

    def __init__(self, api_dict_df):
        self.api_dict_df = api_dict_df
        self.result_nodes = []

    def set_info(self, cell_no):
        self.cell_no = cell_no

    def visit_Attribute(self, node):
        response = make_result_node(self.api_dict_df, node, self.cell_no)
        if response:
            self.result_nodes.append(response)
        self.generic_visit(node)

    def visit_Call(self, node):
        response = make_result_node(self.api_dict_df, node, self.cell_no)
        if response:
            self.result_nodes.append(response)
        self.generic_visit(node)

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)
