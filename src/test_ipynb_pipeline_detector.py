from ipynb_pipeline_detector import IpynbPipelineDetector
from file_reader import *
from unittest import TestCase


class TestIpynbPipelineDetector(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = IpynbPipelineDetector(NotebookReader().all_ipynb_paths)

    def test_load_api_dict_column(self):
        data = self.pipeline.load_api_dict_column('Acquisition')
        self.assertEqual(len(data), 25)

        data = self.pipeline.load_api_dict_column('Preparation')
        self.assertEqual(len(data), 132)

