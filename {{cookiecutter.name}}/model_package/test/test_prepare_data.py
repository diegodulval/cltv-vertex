import unittest
from pathlib import Path
from trainer import prepare_data


class TesLoadData(unittest.TestCase):

    def test_output_exists(self):
        """ Test if output is there. """
        path = Path("test.joblib")
        prepare_data.load_dataset(path)
        self.assertEqual((str(path), path.is_file()), (str(path), True))
