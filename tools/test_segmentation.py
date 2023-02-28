import unittest

import yaml
from easydict import EasyDict as edict
from src.pipeline import PipeLine


class TestTraining(unittest.TestCase):

    def test_main(self):
        config_file = 'tests/configs/mmseg.yaml'
        with open(config_file, 'r') as fp:
            cfg = edict(yaml.safe_load(fp))

        v = PipeLine(cfg)
        v.run()


if __name__ == '__main__':
    unittest.main()
