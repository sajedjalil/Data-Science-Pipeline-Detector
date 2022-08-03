import ast
from src.models.result import Result


def is_constructor(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name)


def is_attribute(node):
    return isinstance(node, ast.Call) and isinstance(node.args, list)


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
            self.temp_result_node = Result(response)
            self.generic_visit(node)
            self.result_nodes.append(self.temp_result_node)
        else:
            self.generic_visit(node)

    def visit_Constant(self, node):
        if self.temp_result_node:
            self.temp_result_node.parameters.append(node.value)
        self.generic_visit(node)

    def visit_Call(self, node):
        if is_constructor(node):
            response = self.make_result_node(node)
            if response is not None:
                # pprint(ast.dump(node))
                self.temp_result_node = Result(response)
                self.generic_visit(node)
                self.get_named_arguments(node.args)
                self.result_nodes.append(self.temp_result_node)
        else:
            self.generic_visit(node)
            if is_attribute(node):
                # pprint(ast.dump(node))
                self.get_named_arguments(node.args)

    def visit_keyword(self, node):
        if self.temp_result_node:
            if hasattr(node, 'value') and isinstance(node.value, ast.Constant):
                self.temp_result_node.parameters.append([node.arg, node.value.value])
            elif hasattr(node, 'value') and isinstance(node.value, ast.Tuple):
                self.temp_result_node.parameters.append([node.arg, get_tuple_name(node.value)])
            elif hasattr(node, 'id') and isinstance(node, ast.Name):
                self.temp_result_node.parameters.append(node.id)

    def get_named_arguments(self, nodes):
        if not self.temp_result_node:
            return
        params = []
        for node in nodes:
            if hasattr(node, 'value') and isinstance(node.value, ast.Constant):
                params.append([node.arg, node.value.value])
            elif hasattr(node, 'value') and isinstance(node.value, ast.Tuple):
                params.append([node.arg, get_tuple_name(node.value)])
            elif hasattr(node, 'id') and isinstance(node, ast.Name):
                params.append(node.id)

        if len(params):
            self.temp_result_node.parameters.extend(params)

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def make_result_node(self, node):
        value = None

        if isinstance(node, ast.Call):
            value = getattr(node.func, "id")
        elif isinstance(node, ast.Attribute):
            value = node.attr

        for pipeline in self.api_dict_df.columns:
            is_found = self.api_dict_df[pipeline].isin([value])
            keyword = self.api_dict_df[pipeline].values[is_found]

            if len(keyword) > 0:
                return Result(pipeline, keyword, node, self.cell_no)

        return None
