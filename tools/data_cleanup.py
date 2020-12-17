"""
data clean tools
"""


class DataCleaner(object):
    def __init__(self, junk_dir=None):
        self.junk_dir = junk_dir
        if self.junk_dir is None:
            self.junk_dir = './junk'

    def check_duplicate(self, root):
        pass

    def check_valid(self, root, junk_dir):
        pass
    
    def rename_to_md5(self, root):
        pass
