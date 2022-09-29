from src.constants.constants import *
from src.pipeline.ipynb_pipeline_detector import IpynbPipelineDetector
from src.utils.file_reader import FileReader

if __name__ == '__main__':
    reader = FileReader(dataset_base_path)
    pipeline = IpynbPipelineDetector(reader.all_ipynb_paths, base_path)
    # pipeline.remove_non_parsable_files()
    df = pipeline.get_results()

    if len(df) != 0:
        df.columns = ["ref", "competitionId", "pipeline_step", "keyword", "cell_no", "line_no", "column_no",
                      "parameters"]
        df.to_csv(os.path.join(base_path, res_folder, pipeline_csv_file), index=False, encoding='utf-8')
