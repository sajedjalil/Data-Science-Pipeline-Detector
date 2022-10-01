import ast

from src.ast.ast_static import make_result_node


class Frequency(ast.NodeVisitor):
    result_nodes = None
    line_no = None
    cell_no = None

    def __init__(self, api_dict_df):
        self.api_dict_df = api_dict_df
        self.result_nodes = []
        self.import_alias_dict = dict()

    def set_info(self, cell_no):
        self.cell_no = cell_no

    def visit_Attribute(self, node):
        response = make_result_node(self.api_dict_df, self.import_alias_dict, node, self.cell_no)
        if response:
            self.result_nodes.append(response)
        self.generic_visit(node)

    def visit_Call(self, node):
        response = make_result_node(self.api_dict_df, self.import_alias_dict, node, self.cell_no)
        if response:
            self.result_nodes.append(response)
        self.generic_visit(node)

    def visit_Import(self, node):

        for alias in node.names:
            if alias.asname is not None:
                name = alias.name.split(".")

                if len(name) > 1:
                    self.import_alias_dict[alias.asname] = name[-1]
                else:
                    self.import_alias_dict[alias.asname] = alias.name

    def visit_ImportFrom(self, node):

        for alias in node.names:
            if alias.asname is not None:
                name = alias.name.split(".")

                if len(name) > 1:
                    self.import_alias_dict[alias.asname] = name[-1]
                else:
                    self.import_alias_dict[alias.asname] = alias.name

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)
