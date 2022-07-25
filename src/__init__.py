from src.utils.file_reader import FileReader, NotebookReader
from src.constants.constants import *
from src.pipeline.ipynb_pipeline_detector import IpynbPipelineDetector

if __name__ == '__main__':
    # FileReader(dataset_base_path)
    pipeline = IpynbPipelineDetector(NotebookReader(dataset_base_path).all_ipynb_paths)
    # pipeline.remove_non_parsable_files()
    pipeline.get_results()
