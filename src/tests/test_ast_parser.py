from src.constants.constants import *
from src.pipeline.ast_parser import *
from src.utils.file_reader import read_xlsx
import os
from unittest import TestCase
from pprint import pprint


class TestParser(TestCase):

    @classmethod
    def setUpClass(cls):
        api_dict_df = read_xlsx(os.path.join(os.getcwd(), res_folder, api_dict_file))
        cls.parser = Parser(api_dict_df, os.path.join(os.getcwd(), res_folder, test_notebook_ok_file_name))

    def test_ast_contains_errors(self):
        parser_error = Parser(self.parser.api_dict_df,
                              os.path.join(os.getcwd(), res_folder, test_notebook_error_file_name))
        self.assertNotEqual(len(parser_error.api_dict_df), 0)
        self.assertTrue(parser_error.ast_contains_errors())

    def test_ast_parse(self):
        results = self.parser.ast_parse()
        # for result in results:
        #     pprint(vars(result))
        assert True
