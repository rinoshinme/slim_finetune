import os


def generate_dataset(dataset_root, class_names, output_path):
    with open(output_path, 'w') as fp:
        for idx, c in enumerate(class_names):
            sub_folder = os.path.join(dataset_root, c)
            if not os.path.exists(sub_folder):
                continue
            file_names = os.listdir(sub_folder)
            for f in file_names:
                if not f.endswith('.jpg'):
                    continue
                fpath = os.path.join(sub_folder, f)
                fp.write('{}\t{}\n'.format(fpath, idx))


if __name__ == '__main__':
    root = r'F:\\data\\violence\\bloody3_20191019\\val'
    class_names = [
        'normal', 'medium', 'bloody',
    ]
    output_path = './dataset.txt'
    generate_dataset(root, class_names, output_path)
