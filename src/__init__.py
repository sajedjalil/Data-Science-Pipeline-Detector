from src.utils.file_reader import FileReader, NotebookReader
from src.constants.constants import *
from src.pipeline.ipynb_pipeline_detector import IpynbPipelineDetector

if __name__ == '__main__':
    # FileReader(dataset_base_path)
    all_ipynb_paths = NotebookReader(dataset_base_path).all_ipynb_paths
    pipeline = IpynbPipelineDetector(all_ipynb_paths, base_path)
    # pipeline.remove_non_parsable_files()
    pipeline.get_results()
