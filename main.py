"""
train or test
"""

from tester import Tester
from configs.test import cfg as cfg_test


def parse_arguments():
    pass


if __name__ == '__main__':
    tester = Tester(cfg_test)
    tester.run()
