from ipynb_pipeline_detector import IpynbPipelineDetector
from file_reader import *
from unittest import TestCase


class TestIpynbPipelineDetector(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = IpynbPipelineDetector(NotebookReader().all_ipynb_paths)


