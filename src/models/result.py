class Result:
    pipeline_step: str = None
    keyword: str = None
    cell_no: int = None
    line_no: int = None
    column_no: int = None
    parameters: list = None

    def __init__(self, *args):
        if isinstance(args[0], str):
            self.pipeline_step = args[0]
            self.keyword = args[1]
            self.column_no = int(args[2].end_col_offset)
            self.line_no = args[2].lineno
            self.cell_no = args[3]
            self.parameters = []
        else:
            self.keyword = args[0].keyword
            self.pipeline_step = args[0].pipeline_step
            self.column_no = args[0].column_no
            self.cell_no = args[0].cell_no
            self.line_no = args[0].line_no
            self.parameters = args[0].parameters
