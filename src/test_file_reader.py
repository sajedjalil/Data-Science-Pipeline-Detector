from file_reader import FileReader, NotebookReader
from unittest import TestCase
import constants


# class TestFileReader(TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.reader = FileReader(constants.dataset_base_path)
#
#     def test_last_entry_vote(self):
#         # Get the last entry from csv to check whether it conforms to the minimum vote count
#         last_entry_vote = self.reader.all_csv_data.tail(1)[constants.total_votes].astype(int).item()
#         self.assertGreaterEqual(last_entry_vote, constants.least_votes)
#
#     def test_csv_file_paths(self):
#         self.assertNotEqual(len(self.reader.csv_file_paths), 0)
#
#     def test_dataset_base_path(self):
#         self.assertEqual(self.reader.base_folder_path, constants.dataset_base_path)


class TestNotebookReader(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb_reader = NotebookReader()

    def test_all_notebook_paths(self):
        self.assertNotEqual(len(self.nb_reader.all_notebook_paths), 0)

    def test_load_notebook(self):
        cells = self.nb_reader.load_notebook(self.nb_reader.all_notebook_paths[0])
        self.assertNotEqual( len(cells), 0)
