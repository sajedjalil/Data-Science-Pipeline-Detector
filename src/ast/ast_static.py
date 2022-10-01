import ast

from src.models.result import Result


def is_constructor(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name)


def is_attribute(node):
    return isinstance(node, ast.Call) and isinstance(node.args, list)


def get_tuple_name(node) -> list:
    names = []
    for name in node.elts:
        if hasattr(name, 'value'):
            names.append(name.value)
        elif hasattr(name, 'id'):
            names.append(name.id)
    return names


def make_result_node(api_dict_df, import_alias_dict, node, cell_no):
    value = None

    if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
        value = getattr(node.func, "id")
    elif isinstance(node, ast.Attribute):
        value = node.attr

    # Check if the value is in import_alias_dict
    if value in import_alias_dict:
        value = import_alias_dict[value]

    for pipeline in api_dict_df.columns:
        is_found = api_dict_df[pipeline].isin([value])
        keyword = api_dict_df[pipeline].values[is_found]

        if len(keyword) > 0:
            return Result(pipeline, keyword, node, cell_no, ast.unparse(node))

    return None
