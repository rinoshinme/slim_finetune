"""
prepare testing images for audition (violence)
"""

import os
import sys
import shutil
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(current_dir, '../'))

from options.violence9 import demo_options
from tfvortex.demo import Demo


class_names = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest']


def main():
    options = demo_options()
    model = Demo(options)
    image_folder = r'E:\projects\ContentSupervision\Violence\data2\war'
    index = 4
    target_folder = r'D:\projects\ContentSupervision\Violence\test_for_audition\war'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for name in os.listdir(image_folder):
        print(name)
        image_path = os.path.join(image_folder, name)
        result = model.test(image_path)
        result = result[0]
        print(result)

        if result[index] > 0.5:
            # move file
            target_path = os.path.join(target_folder, name)
            if not os.path.exists(target_path):
                shutil.copy(image_path, target_path)


if __name__ == '__main__':
    main()

