import unittest
import pathlib


def run_all_tests(verbosity: int = 1):
    test_path = str(pathlib.Path(__file__).parent)
    all_tests = unittest.TestLoader().discover(start_dir=test_path)
    unittest.TextTestRunner(verbosity=verbosity).run(all_tests)


if __name__ == '__main__':
    run_all_tests()
