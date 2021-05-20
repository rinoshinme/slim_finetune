"""
preprocess data using nsfw model trained by mao
"""
import os
import sys
import shutil
import numpy as np

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(current_dir, '../'))
from options.porn_mao5 import protobuf_options
from tfvortex.protobuf import PbTester



def main():
    pb_path = '../data/models_pb/nsfw_resnet152_0725.pb'
    options = protobuf_options()

    class_names = options.CLASS_NAMES
    model = PbTester(pb_path, options)
    image_folder = r'E:\projects\ContentSupervision\Porn\data_raw\sexy'
    threshold = 0.5
    target_root = r'E:\projects\ContentSupervision\Porn\data\v2'

    for name in os.listdir(image_folder):
        print(name)
        image_path = os.path.join(image_folder, name)
        result = model.infer(image_path)
        result = result[0]
        print(result)

        max_id = np.argmax(result)
        if result[max_id] > threshold:
            # move file
            target_folder = os.path.join(target_root, class_names[max_id])
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            target_path = os.path.join(target_folder, name)
            shutil.move(image_path, target_path)


if __name__ == '__main__':
    main()
