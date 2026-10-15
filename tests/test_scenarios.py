import unittest

from evals.scenarios import SCENARIOS


class TestScenarios(unittest.TestCase):
    pass


def _make(fn):
    def test(self):
        self.assertEqual(fn(), [])
    return test


for _name, _desc, _fn in SCENARIOS:
    setattr(TestScenarios, f"test_{_name}", _make(_fn))

if __name__ == "__main__":
    unittest.main()
