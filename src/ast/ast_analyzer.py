from src.ast.ast_static import *


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
        response = make_result_node(self.api_dict_df, node, self.cell_no)
        if response:
            self.temp_result_node = Result(response)
            self.generic_visit(node)
            if self.temp_result_node:
                self.result_nodes.append(self.temp_result_node)
            # self.temp_result_node = None
        else:
            self.generic_visit(node)

    def visit_Call(self, node):
        if is_constructor(node):
            response = make_result_node(self.api_dict_df, node, self.cell_no)
            if response is None:
                return
            self.temp_result_node = Result(response)
            self.visit_arguments(node.args)
            self.generic_visit(node)
            if self.temp_result_node:
                self.result_nodes.append(self.temp_result_node)
            self.temp_result_node = None
        elif is_attribute(node):
            # print(ast.dump(node))
            self.generic_visit(node)
            self.visit_arguments(node.args)
            self.temp_result_node = None

    def visit_arguments(self, nodes):
        params = []

        if not isinstance(nodes, list):
            return

        for node in nodes:
            if hasattr(node, 'value') and isinstance(node, ast.Constant):
                params.append(node.value)
            elif hasattr(node, 'value') and isinstance(node, ast.Tuple):
                params.append([node.arg, get_tuple_name(node.value)])
            elif hasattr(node, 'id') and isinstance(node, ast.Name):
                params.append(node.id)
        if self.temp_result_node and len(params):
            self.temp_result_node.parameters.extend(params)

    def visit_keyword(self, node):
        if self.temp_result_node:
            if hasattr(node, 'value') and isinstance(node.value, ast.Constant):
                self.temp_result_node.parameters.append([node.arg, node.value.value])
            elif hasattr(node, 'value') and isinstance(node.value, ast.Tuple):
                self.temp_result_node.parameters.append([node.arg, get_tuple_name(node.value)])
            elif hasattr(node.value, 'id') and isinstance(node.value, ast.Name):
                self.temp_result_node.parameters.append([node.arg, node.value.id])

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)
