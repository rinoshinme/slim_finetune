import time
import os
import matplotlib.pyplot as plt
from api_test.utils import b64_data_uri
from api_test.utils import predict


def read_results(result_text):
    names = []
    values = []
    if not os.path.exists(result_text):
        return names, values

    with open(result_text, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            fields = line.split(',')
            names.append(fields[0])
            values.append(float(fields[1]))
    return names, values


def test_response_time(image_folder, scenes, result_text):
    """test response time using name set of images"""
    filenames = os.listdir(image_folder)
    num = len(filenames)
    print("{} files will be detected under folder:{}, ".format(num, image_folder))
    flag = 0

    names, values = read_results(result_text)

    for file_name in filenames:
        flag += 1
        if file_name in names:
            continue

        file_path = os.path.join(image_folder, file_name)
        b64_data = b64_data_uri(file_path)
        data = [b64_data]

        start_time = time.time()
        response_text = predict(data, scenes)
        end_time = time.time()
        time_taken = end_time - start_time

        print('detection time = {}'.format(time_taken))
        with open(result_text, 'a') as f:
            f.write('%s,%f\n' % (file_name, time_taken))

        print("detecting {}/{}:{}\nresult:{}".format(flag, num, file_name, response_text))
        time.sleep(1)


def plot_histogram(result_text):
    names, values = read_results(result_text)
    nbin = 40
    bins_edges = [0.1 * i for i in range(nbin + 1)]
    hist = [0 for _ in range(nbin)]
    for v in values:
        idx = int(v / 0.1)
        if idx >= nbin:
            hist[nbin - 1] += 1
        else:
            hist[idx] += 1

    # bin_centers = [(bins_edges[i] + bins_edges[i + 1]) / 2 for i in range(nbin)]

    plt.hist(hist, bins_edges)
    # plt.plot(bin_centers, hist)
    plt.grid()
    plt.show()


def filesize_vs_time(result_text):
    # plot the relationship of file size and recognition time.
    folder = r'D:\temp\21cn_test\mixed_size'
    names, values = read_results(result_text)

    file_sizes = []
    for name, value in zip(names, values):
        file_path = os.path.join(folder, name)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
        else:
            size = 0
        file_sizes.append(size)

    # plot curve
    plt.plot(file_sizes, values, 'r.')
    plt.title('recog time vs file size')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    test_folder = r'D:\temp\21cn_test\mixed_size'

    # test_scenes = ['politic']
    # test_text = r'./results/politic_response_time.txt'
    # test_response_time(test_folder, test_scenes, test_text)

    test_scenes = ['violence']
    test_text = r'./results/violence_response_time.txt'
    test_response_time(test_folder, test_scenes, test_text)

    test_scenes = ['ocr']
    test_text = r'./results/ocr_response_time.txt'
    test_response_time(test_folder, test_scenes, test_text)\

    test_scenes = ['porn', 'politic', 'violence', 'ocr']
    test_text = r'./results/all_response_time.txt'
    test_response_time(test_folder, test_scenes, test_text)
