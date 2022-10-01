from pathlib import Path
from unittest import TestCase

from src.utils.file_reader import *


class TestFileReader(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reader = FileReader(os.path.join(os.getcwd(), dataset_folder))

    def test_last_entry_vote(self):
        # Get the last entry from csv to check whether it conforms to the minimum vote count
        last_entry_vote = self.reader.all_csv_data.tail(1)[total_votes].astype(int).item()
        self.assertGreaterEqual(last_entry_vote, least_votes)

    def test_csv_file_paths(self):
        self.assertNotEqual(len(self.reader.csv_file_paths), 0)

    def test_dataset_base_path(self):
        self.assertEqual(self.reader.base_folder_path, os.path.join(os.getcwd(), dataset_folder))

    def test_read_xlsx(self):
        df = read_xlsx(os.path.join(os.getcwd(), res_folder, api_dict_file))
        self.assertNotEqual(len(df), 0)

    def test_delete_file(self):
        file_name = Path("test.ipynb")
        fp = open(file_name, 'w')
        fp.close()
        delete_file('test.ipynb')

        self.assertFalse(file_name.exists())

    def test_all_notebook_paths(self):
        self.assertEqual(len(self.reader.all_ipynb_paths), 14498)

    # def test_load_notebook(self):
    #     cells = load_notebook(self.reader.all_ipynb_paths[0])
    #     self.assertNotEqual(len(cells), 0)
