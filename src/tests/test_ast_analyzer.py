from pprint import pprint
from unittest import TestCase

from src.ast.ast_parser import *
from src.constants.constants import *
from src.models.result import Result
from src.utils.file_reader import read_xlsx


class TestAnalyzer(TestCase):
    parser = None

    @classmethod
    def setUpClass(cls):
        cls.api_dict_df = read_xlsx(os.path.join(os.getcwd(), res_folder, api_dict_file))
        cls.parser = Parser(cls.api_dict_df, os.path.join(os.getcwd(), res_folder, test_notebook_ok_file_name))
        cls.results = cls.parser.ast_parse()

    def test_generic_visit_1(self):
        expected = Result('Modeling', 'Sequential', 20, 2, 1, [])
        self.assertEqual(expected, self.results[0])

    def test_generic_visit_2(self):
        expected = Result('Modeling', 'LSTM', 72, 5, 1, [50,
                                                         ['activation=relu'],
                                                         ['input_shape', ['n_steps', 'n_features']]])
        self.assertEqual(expected, self.results[1])

    def test_generic_visit_3(self):
        expected = Result('Modeling', 'Dense', 18, 6, 1, [1])
        self.assertEqual(expected, self.results[2])

    def test_generic_visit_4(self):
        expected = Result('Modeling', 'compile', 13, 7, 1, [['optimizer=adam'], ['loss=mse']])
        self.assertEqual(expected, self.results[3])

    def test_generic_visit_5(self):
        expected = Result('Modeling', 'KernelRidge', 70, 8, 1, [['alpha=0.1'],
                                                                ['kernel=polynomial'],
                                                                ['degree=7'],
                                                                ['coef0=2.5']])
        self.assertEqual(expected, self.results[4])

    def test_generic_visit_6(self):
        expected = Result('Training', 'fit', 7, 9, 1, ['x_train1', 'y_train1'])
        self.assertEqual(expected, self.results[5])

    def test_generic_visit_7(self):
        expected = Result('Training', 'fit', 7, 10, 1, [['y_train1=5'], 'x_train1', 'x_train1'])
        self.assertEqual(expected, self.results[6])

    def test_generic_visit_8(self):
        expected = Result('Training', 'fit', 7, 11, 1, [['y_train1=5'], 5])
        self.assertEqual(expected, self.results[7])

    def test_generic_visit_9(self):
        expected = Result('Training', 'fit', 7, 12, 1, [5, 5])
        self.assertEqual(expected, self.results[8])

    def test_generic_visit_10(self):
        expected = Result('Training', 'fit', 7, 13, 1, ['x_train1', 'y_train1'])
        self.assertEqual(expected, self.results[9])

    def test_generic_visit_11(self):
        expected = Result('Prediction', 'predict', 21, 14, 1, ['x_val1'])
        self.assertEqual(expected, self.results[10])

    def test_generic_visit_12(self):
        expected = Result('Modeling', 'KernelRidge', 70, 15, 1, [['alpha=0.1'],
                                                                 ['kernel=polynomial'],
                                                                 ['degree=7'],
                                                                 ['coef0=2.5']])
        self.assertEqual(expected, self.results[11])

    def test_generic_visit_13(self):
        expected = Result('Training', 'train_test_split', 92, 16, 1,
                          ['X_can', 'y_ca', ['random_state=42'], ['test_size=0.2']])
        self.assertEqual(expected, self.results[12])

    def test_generic_visit_14(self):
        expected = Result('Modeling', 'Activation', 34, 17, 1, [['k=9'], 'relu'])
        self.assertEqual(expected, self.results[13])

    def test_generic_visit_15(self):
        expected = Result('Others', 'plot', 9, 1, 4,
                          [['x=var'], ['y=SalePrice'], ['ylim', [0, 800000]], 123])
        self.assertEqual(expected, self.results[16])

    # def test_generic_visit_15(self):
    #     for res in self.results:
    #         pprint(vars(res))
    #     assert True
    def test_import(self):
        import_parser = Parser(self.api_dict_df, os.path.join(os.getcwd(), res_folder, test_notebook_import_file_name))
        results = import_parser.ast_parse()
        self.assertEqual("LinearRegression", results[0].keyword)
        self.assertEqual("LogisticRegression", results[1].keyword)

    def test_import_from(self):
        import_parser = Parser(self.api_dict_df, os.path.join(os.getcwd(), res_folder, test_notebook_import_file_name))
        results = import_parser.ast_parse()
        self.assertEqual("LinearRegression", results[2].keyword)
        self.assertEqual("ElasticNetCV", results[3].keyword)
