import ast

from src.ast.ast_analyzer import Analyzer
from src.ast.ast_frequency_analyzer import Frequency
from src.utils.file_reader import load_notebook


class Parser:

    def __init__(self, api_dict_df, path):
        self.api_dict_df = api_dict_df
        self.path = path

    @staticmethod
    def __replace_non_parsable_line(cell):
        keywords = ["!", "--", "pip ", "%matplotlib", "%time", "%%time", "ls ", "cd ", "rm "]
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

        for idx, line in enumerate(cell):
            stripped = line.strip()
            if line.startswith("%"):
                cell[idx] = ""
            for key in keywords:
                if stripped.startswith(key):
                    cell[idx] = ""

            if not cell[idx].endswith("\n"):
                cell[idx] += "\n"

        return "".join(cell)

    def ast_parse(self):
        cells = load_notebook(self.path)
        results = []
        for idx_cell, cell in enumerate(cells):
            cell = self.__replace_non_parsable_line(cell)

            analyzer = Analyzer(self.api_dict_df)
            analyzer.set_info(idx_cell + 1)
            analyzer.visit(ast.parse(cell))

            results.extend(analyzer.result_nodes)

        return results

    def ast_parse_frequency(self):
        cells = load_notebook(self.path)
        results = []
        for idx_cell, cell in enumerate(cells):
            cell = self.__replace_non_parsable_line(cell)

            analyzer = Frequency(self.api_dict_df)
            analyzer.set_info(idx_cell + 1)
            analyzer.visit(ast.parse(cell))

            results.extend(analyzer.result_nodes)

        return results

    def ast_contains_errors(self) -> bool:
        """
        Check whether notebook file contains any ast parsing issues.

        :return: bool
        """
        cells = load_notebook(self.path)
        errors: bool = False
        for idx_cell, cell in enumerate(cells):
            analyzer = Analyzer(self.api_dict_df)
            try:
                cell = self.__replace_non_parsable_line(cell)
                analyzer.set_info(idx_cell + 1)
                analyzer.visit(ast.parse(cell))
                # pprint(ast.dump(ast.parse(cell)))
            except SyntaxError as e:
                errors = True
            except RecursionError as r:
                errors = True

        return errors
