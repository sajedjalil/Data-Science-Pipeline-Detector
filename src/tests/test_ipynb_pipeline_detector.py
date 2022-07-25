from src.utils.file_reader import *
from unittest import TestCase
from src.pipeline.ipynb_pipeline_detector import get_ast_notebook_file
from src.constants.constants import *


class TestIpynbPipelineDetector(TestCase):

    def test_get_ast_notebook_file_1(self):
        api_dict_df = read_xlsx(os.path.join(res_folder_path, api_dict_file))
        path = dataset_base_path + "/house-prices-advanced-regression-techniques/Pedro Marcelino, " \
                                   "PhD/comprehensive-data-exploration-with-python.ipynb"
        result_nodes = get_ast_notebook_file(api_dict_df, path)
        self.assertEqual(len(result_nodes), 50)
    #
    # def test_get_ast_notebook_file_2(self):
    #     api_dict_df = read_xlsx(os.path.join(res_folder_path, api_dict_file))
    #     path = dataset_base_path + "/titanic/Masum Rumi/a-statistical-analysis-ml-workflow-of-titanic.ipynb"
    #     result_nodes = get_ast_notebook_file(api_dict_df, path)
    #     self.assertEqual(len(result_nodes), 203)

    # def test_get_ast_notebook_file_3(self):
    #     api_dict_df = read_xlsx(os.path.join(res_folder_path, api_dict_file))
    #     path = dataset_base_path + "/titanic/Masum Rumi/test.ipynb" + "/home/sajed/GitHub/reference-model.ipynb"
    #     result_nodes = get_ast_notebook_file(api_dict_df, path)
    #     self.assertEqual(len(result_nodes), 2)
