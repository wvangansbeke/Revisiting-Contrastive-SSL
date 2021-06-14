#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os


class Path(object):
    """
    User-specific path configuration.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/path/to/datasets/'
        db_names = ['VOCSegmentation', 'cityscapes']

        if database == '':
            return db_root

        if database == 'VOCSegmentation':
            return '/path/to/datasets/VOCSegmentation'

        elif database == 'cityscapes':
            return '/path/to/datasets/cityscapes'

        else:
            raise ValueError('Invalid database {}'.format(database))    
