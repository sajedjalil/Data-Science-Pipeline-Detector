import ast
from src.models.result import Result
from pprint import pprint
import copy


def is_constructor(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name)


def get_tuple_name(node) -> list:
    names = []
    for name in node.elts:
        names.append(name.id)
    return names


class Analyzer(ast.NodeVisitor):
    result_nodes = []
    line_no = None
    cell_no = None
    temp_result_node: Result = None

    def __init__(self, api_dict_df):
        self.api_dict_df = api_dict_df
        self.result_nodes = []

    def set_info(self, cell_no):
        self.cell_no = cell_no

    def visit_Attribute(self, node):
        response = self.make_result_node(node)

        if response:
            self.temp_result_node = response
            self.generic_visit(node)
            self.result_nodes.append(self.temp_result_node)
        else:
            self.generic_visit(node)

    def visit_Name(self, node):
        # if self.temp_result_node:
        #     pprint(ast.dump(node))
        #     print(self.temp_result_node)
        #     # self.temp_result_node.parameters.append(node.id)
        self.generic_visit(node)

    def visit_Constant(self, node):

        if self.temp_result_node:
            self.temp_result_node.parameters.append(node.value)
        self.generic_visit(node)

    def visit_Call(self, node):
        if is_constructor(node):
            response = self.make_result_node(node)
            if response is not None:
                self.temp_result_node = response
                self.generic_visit(node)
                self.result_nodes.append(self.temp_result_node)
                self.temp_result_node = None
        else:
            self.generic_visit(node)

    def visit_keyword(self, node):
        if self.temp_result_node:
            if isinstance(node.value, ast.Constant):
                # print(node.value.value, node.arg)
                self.temp_result_node.parameters.append([node.arg, node.value.value])
            elif isinstance(node.value, ast.Tuple):
                # print(node.value.value, node.arg)
                # self.visit_Tuple(node.value)
                self.temp_result_node.parameters.append([node.arg, get_tuple_name(node.value)])
            elif isinstance(node.value, ast.Name):
                print(node.value.id)

    # def visit_arg(self, node):
    #     pprint( ast.dump(node))
    #     self.generic_visit(node)

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def make_result_node(self, node):
        value = None

        if isinstance(node, ast.Call):
            value = getattr(node.func, "id")
        elif isinstance(node, ast.Attribute):
            value = node.attr

        for col in self.api_dict_df.columns:
            is_found = self.api_dict_df[col].isin([value])
            keyword = self.api_dict_df[col].values[is_found]

            if len(keyword) > 0:
                return Result(col, keyword, node, self.cell_no)

        return None
