import requests
import json
import time
import hashlib
import hmac
import os
from api_test.utils import b64_data_uri
from api_test.utils import predict
from multiprocessing import Process, Pool, Queue


def test_qps(image_dir, scenes, npr, text_file):
    files = os.listdir(image_dir)
    num = len(files)
    print("{} files will be detected under folder:{}, ".format(num, image_dir))

    num_batches = num // npr
    for batch in range(num_batches):
        print('testing batch {}'.format(batch))
        data = []
        for i in range(npr):
            path = os.path.join(image_dir, files[batch * npr + i])
            b64_data = b64_data_uri(path)
            data.append(b64_data)
        
        start_time = time.time()
        result = predict(data, scenes)
        end_time = time.time()
        time_taken = end_time - start_time

        with open(text_file, 'a') as f:
            f.write('%f\n' % time_taken)


def test_qps_mp(image_dir, scenes, num_processes, npr):
    # npr: num of images per request
    files = os.listdir(image_dir)
    num = len(files)
    print("{} files will be detected under folder:{}, ".format(num, image_dir))
    pool = Pool(num_processes)

    num_batches = num // npr

    time_start = time.time()
    for batch in range(num_batches):
        data = []
        for i in range(npr):
            path = os.path.join(image_dir, files[batch * npr + i])
            b64_data = b64_data_uri(path)
            data.append(b64_data)
        ruslt = pool.apply_async(predict, args=(data, scenes, True))
    
    print('等待所有添加的进程运行完毕...')
    pool.close()
    pool.join()
    time_end = time.time()
    print('time taken = %f' % (time_end - time_start))
    print("主进程结束!")


if __name__ == '__main__':
    image_folder = r'D:\temp\21cn_test\mixed_size'
    scenes = ['porn', 'politic', 'violence', 'ocr']
    result_text = r'./porn_response_time.txt'
    npr = 4
    test_qps(scenes, scenes, npr, result_text)
