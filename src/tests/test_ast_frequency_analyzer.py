from pprint import pprint
from unittest import TestCase
from src.constants.constants import *
from src.ast.ast_parser import *
from src.utils.file_reader import read_xlsx
from src.models.result import Result


class TestAnalyzer(TestCase):

    @classmethod
    def setUpClass(cls):
        api_dict_df = read_xlsx(os.path.join(os.getcwd(), res_folder, api_dict_file))
        cls.parser = Parser(api_dict_df, os.path.join(os.getcwd(), res_folder, test_notebook_ok_file_name))
        cls.results = cls.parser.ast_parse_frequency()

    def test_match_keyword(self):
        pipeline_expected = [['Sequential'], ['LSTM'], ['Dense'], ['compile'], ['KernelRidge'], ['fit'], ['fit'],
                             ['fit'], ['fit'], ['fit'], ['predict'], ['KernelRidge'], ['train_test_split'],
                             ['Activation'], ['LSTM'], ['Sequential'], ['LSTM'], ['Sequential'], ['plot']]
        pipeline_actual = []
        for res in self.results:
            pipeline_actual.append(res.keyword)
        print(pipeline_actual)
        self.assertEqual(pipeline_expected, pipeline_actual)
