class Result:
    pipeline_step: str = None
    keyword: str = None
    cell_no: int = None
    line_no: int = None
    column_no: int = None

    def __init__(self, pipeline, keyword, node, cell_no):
        self.keyword = keyword
        self.pipeline_step = pipeline
        self.column_no = int(node.end_col_offset)
        self.cell_no = cell_no
        self.line_no = node.lineno
