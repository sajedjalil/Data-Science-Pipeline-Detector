from ipynb_pipeline_detector import IpynbPipelineDetector, replace_non_parsable_line
from file_reader import *
from unittest import TestCase
from ipynb_pipeline_detector import get_ast_notebook_file


class TestIpynbPipelineDetector(TestCase):

    def test_get_ast_notebook_file(self):
        api_dict_df = read_xlsx(os.path.join(res_folder_path, api_dict_file))
        path = "/home/sajed/GitHub/Data-Science-Pipeline-Detector/dataset/house-prices-advanced-regression-techniques" \
               "/Pedro Marcelino, PhD/comprehensive-data-exploration-with-python.ipynb"
        result_nodes = get_ast_notebook_file(api_dict_df, path)
        self.assertEqual(len(result_nodes), 50)
