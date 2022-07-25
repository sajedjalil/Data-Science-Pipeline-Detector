from src.utils.file_reader import FileReader, NotebookReader
from src.constants.constants import *
from src.pipeline.ipynb_pipeline_detector import IpynbPipelineDetector

if __name__ == '__main__':
    f = FileReader(dataset_base_path)

    pipeline = IpynbPipelineDetector(NotebookReader().all_ipynb_paths)
