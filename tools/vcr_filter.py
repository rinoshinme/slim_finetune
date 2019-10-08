"""

"""
import os
import shutil
import numpy as np

result_text = r'E:\violent_reference\vcr\vcr1images.txt'
class_names = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest']
ng_indices = [1, 2, 3, 5, 7, 8, ]
ok_indices = [0, 4, 6]


def read_text(text_file):
    paths = []
    scores = []
    with open(text_file, 'r') as fp:
        for line in fp.readlines():
            line = line.strip()
            fields = line.split('\t')
            fname = fields[0]
            prob = [float(v) for v in fields[1].split(',')]
            paths.append(fname)
            scores.append(prob)
    return paths, scores


def filter_images(paths, scores, target_folder):
    file_pairs = []
    for name, score in zip(paths, scores):
        score = np.array(score)
        index = np.argmax(score)
        if index == 0:
            # skip normal
            continue
        prob = np.max(score)
        if prob < 0.99:
            continue
        dst_path = os.path.join(target_folder, class_names[index])
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        # copy file
        filename = os.path.split(name)[-1]
        dst_file = os.path.join(dst_path, filename)
        if os.path.exists(dst_file):
            continue
        # shutil.copy(name, dst_file)
        file_pairs.append((name, dst_file))
        shutil.move(name, dst_file)
    return file_pairs


def write_file_pairs(file_pairs, txt_file):
    with open(txt_file, 'wa') as f:
        # append to the end of the file
        for pair in file_pairs:
            f.write('%s\t%s\n' % (pair[0], pair[1]))


if __name__ == '__main__':
    txt_name = r'E:\violent_reference\vcr\vcr1images.txt'
    paths, scores = read_text(txt_name)
    target_folder = r'E:\violent_reference\vcr\splits'
    file_pairs = filter_images(paths, scores, target_folder)
    write_file_pairs(file_pairs, r'E:\violent_reference\vcr\filepairs.txt')
