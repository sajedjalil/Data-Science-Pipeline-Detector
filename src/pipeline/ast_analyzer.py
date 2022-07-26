import ast
from src.models.result import Result
from pprint import pprint

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
        print(ast.dump(node), node.lineno, node.col_offset, node.end_col_offset)
        response = self.make_result_node(node)
        if response is not None:
            self.result_nodes.append(response)
        self.generic_visit(node)

    def visit_Name(self, node):
        pprint(ast.dump(node))
        print(node.lineno, node.col_offset, node.end_col_offset)
        self.generic_visit(node)

    def visit_Constant(self, node):
        pprint(ast.dump(node))
        print(node.lineno, node.col_offset, node.end_col_offset)
        self.generic_visit(node)

    def visit_Call(self, node):
        pprint(ast.dump(node))
        # pprint(ast.dump(node))
        # pprint( ast.dump(node.func))
        # for args in node.args:
        #     self.visit_arg(args)
        # pprint( ast.dump(node.args[0]) )
        # for keyword in node.keywords:
        #     self.visit_keyword(keyword)
        self.generic_visit(node)

    # def visit_keyword(self, node):
    #     # pprint(ast.dump(node))
    #     # pprint(node.arg)
    #
    #     if isinstance(node.value, ast.Constant):
    #         print(node.value.value, node.value.lineno)
    #     if isinstance(node.value, ast.Name):
    #         print(node.value.id, node.value.lineno, node.value.col_offset, node.value.end_col_offset)

    # def visit_arg(self, node):
    #     pprint( ast.dump(node))
    #     self.generic_visit(node)

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
