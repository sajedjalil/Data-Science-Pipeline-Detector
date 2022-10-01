import numpy as np


class Result:
    pipeline_step: str = None
    keyword: str = None
    cell_no: int = None
    line_no: int = None
    column_no: int = None
    parameters: list = None
    code: str = None

    def __init__(self, *args):
        if isinstance(args[0], str) and len(args) == 5:
            self.pipeline_step = args[0]
            self.keyword = args[1]
            self.column_no = int(args[2].end_col_offset)
            self.line_no = args[2].lineno
            self.cell_no = args[3]
            self.parameters = []
            self.code = args[4]
        elif isinstance(args[0], str):
            self.pipeline_step = args[0]
            self.keyword = args[1]
            self.column_no = args[2]
            self.line_no = args[3]
            self.cell_no = args[4]
            self.parameters = args[5]
        else:
            self.keyword = args[0].keyword
            self.pipeline_step = args[0].pipeline_step
            self.column_no = args[0].column_no
            self.cell_no = args[0].cell_no
            self.line_no = args[0].line_no
            self.parameters = args[0].parameters
            self.code = args[0].code

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Result):
            first = np.array(self.parameters, dtype=object)
            second = np.array(other.parameters, dtype=object)

            return self.pipeline_step == other.pipeline_step and self.keyword == other.keyword \
                   and self.cell_no == other.cell_no and self.line_no == other.line_no \
                   and self.column_no == other.column_no and np.array_equal(first, second)
        return False

    def __str__(self):
        data = "; ".join([self.pipeline_step, self.keyword[0], self.cell_no.__str__(),
                          self.line_no.__str__(), self.column_no.__str__(), self.parameters.__str__(),
                          self.code])
        return data
