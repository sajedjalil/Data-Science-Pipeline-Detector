from unittest import TestCase
from constants import *
from file_reader import FileReader, NotebookReader, read_xlsx, load_notebook


class TestFileReader(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reader = FileReader(dataset_base_path)

    def test_last_entry_vote(self):
        # Get the last entry from csv to check whether it conforms to the minimum vote count
        last_entry_vote = self.reader.all_csv_data.tail(1)[total_votes].astype(int).item()
        self.assertGreaterEqual(last_entry_vote, least_votes)

    def test_csv_file_paths(self):
        self.assertNotEqual(len(self.reader.csv_file_paths), 0)

    def test_dataset_base_path(self):
        self.assertEqual(self.reader.base_folder_path, dataset_base_path)

    def test_read_xlsx(self):
        df = read_xlsx(os.path.join(res_folder_path, api_dict_file))
        self.assertNotEqual(len(df), 0)


class TestNotebookReader(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb_reader = NotebookReader()

    def test_all_notebook_paths(self):
        self.assertEqual(len(self.nb_reader.all_ipynb_paths), 15079)
        self.assertEqual(len(self.nb_reader.all_py_paths), 903)

    def test_load_notebook(self):
        cells = load_notebook(self.nb_reader.all_ipynb_paths[0])
        self.assertNotEqual(len(cells), 0)
