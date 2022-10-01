from _ast import Import, ImportFrom

from src.ast.ast_static import *


class Analyzer(ast.NodeVisitor):
    result_nodes = []
    line_no = None
    cell_no = None
    temp_result_node: Result = None
    import_alias_dict = dict()

    def __init__(self, api_dict_df):
        self.api_dict_df = api_dict_df
        self.result_nodes = []

    def set_info(self, cell_no):
        self.cell_no = cell_no

    def visit_Attribute(self, node):
        response = make_result_node(self.api_dict_df, self.import_alias_dict, node, self.cell_no)
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
            response = make_result_node(self.api_dict_df, self.import_alias_dict, node, self.cell_no)
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
                self.temp_result_node.parameters.append([node.arg + "=" + str(node.value.value)])
            elif hasattr(node, 'value') and isinstance(node.value, ast.Tuple):
                self.temp_result_node.parameters.append([node.arg, get_tuple_name(node.value)])
            elif hasattr(node.value, 'id') and isinstance(node.value, ast.Name):
                if node.arg is not None:
                    self.temp_result_node.parameters.append([node.arg + "=" + str(node.value.id)])
                else:
                    self.temp_result_node.parameters.append(node.value.id)

    """
    The import aliases must be identified to find missed keywords
    """

    def visit_Import(self, node: Import):

        for alias in node.names:
            if alias.asname is not None:
                name = alias.name.split(".")

                if len(name) > 1:
                    self.import_alias_dict[alias.asname] = name[-1]
                else:
                    self.import_alias_dict[alias.asname] = alias.name

    def visit_ImportFrom(self, node: ImportFrom):

        for alias in node.names:
            if alias.asname is not None:
                name = alias.name.split(".")

                if len(name) > 1:
                    self.import_alias_dict[alias.asname] = name[-1]
                else:
                    self.import_alias_dict[alias.asname] = alias.name

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)
