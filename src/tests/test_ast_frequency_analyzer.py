from collections import OrderedDict
from unittest import TestCase
from src.constants.constants import *
from src.ast.ast_parser import *
from src.utils.file_reader import read_xlsx, NotebookReader


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
        self.assertEqual(pipeline_expected, pipeline_actual)

    # def test_count_keyword_frequency_modeling_step(self):
    #     api_dict_df = read_xlsx(os.path.join(os.getcwd(), res_folder, api_dict_file))
    #     all_ipynb_paths = NotebookReader(os.path.join(os.getcwd(), dataset_folder)).all_ipynb_paths
    #
    #     threshold_map = {}
    #     for keyword in api_dict_df['Modeling']:
    #         threshold_map[keyword] = 0
    #
    #     for idx, path in enumerate(all_ipynb_paths):
    #         parser = Parser(api_dict_df, path)
    #         results = parser.ast_parse_frequency()
    #         for res in results:
    #             if res.pipeline_step == "Modeling":
    #                 threshold_map[res.keyword[0]] = threshold_map[res.keyword[0]] + 1
    #         print(idx, path)
    #
    #     for pair in threshold_map:
    #         print(pair, threshold_map[pair])
    #
    #     self.assertEqual(len(all_ipynb_paths), 14498)

    # def test_count_keyword_per_class_modeling_step(self):
    #     api_dict_df = read_xlsx(os.path.join(os.getcwd(), res_folder, api_dict_file))
    #     all_ipynb_paths = NotebookReader(os.path.join(os.getcwd(), dataset_folder)).all_ipynb_paths
    #
    #     threshold_map = {}
    #     for keyword in api_dict_df['Modeling']:
    #         threshold_map[keyword] = 0
    #
    #     for idx, path in enumerate(all_ipynb_paths):
    #         parser = Parser(api_dict_df, path)
    #         results = parser.ast_parse_frequency()
    #
    #         keyword_set = set()
    #         for res in results:
    #             if res.pipeline_step == "Modeling":
    #                 keyword_set.add(res.keyword[0])
    #         print(idx, path)
    #
    #         for item in keyword_set:
    #             threshold_map[item] += 1
    #
    #     for pair in threshold_map:
    #         print(pair, threshold_map[pair])
    #
    #     self.assertEqual(len(all_ipynb_paths), 14498)

    # def test_pipeline_step_count_per_class(self):
    #     api_dict_df = read_xlsx(os.path.join(os.getcwd(), res_folder, api_dict_file))
    #     all_ipynb_paths = NotebookReader(os.path.join(os.getcwd(), dataset_folder)).all_ipynb_paths
    #
    #     threshold_map = {}
    #
    #     for idx, path in enumerate(all_ipynb_paths):
    #         parser = Parser(api_dict_df, path)
    #         key = len(parser.ast_parse_frequency())
    #
    #         if key not in threshold_map.keys():
    #             threshold_map[key] = 0
    #         threshold_map[key] += 1
    #
    #     od = OrderedDict(sorted(threshold_map.items()))
    #
    #     for key in od:
    #         print(key, od[key])
    #
    #     self.assertEqual(len(all_ipynb_paths), 14498)
