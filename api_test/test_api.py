import requests
import json
import time
import hashlib
import hmac
import os
from api_test.utils import b64_data_uri


def predict(url_list):
    host = "http://101.91.169.58:8000"
    # host = "http://127.0.0.1:3100"
    # host = "http://apiserver.ai.ctyun.cn:3446"
    path = "/api/v1/image.json"
    url = host + path
    app_key = 'd5e0fdfd7fb4c6d2'
    app_secret = 'ae652745a335c06d0bd3b302bf809e4b'
    # app_key = '5b12a2269fe0b37c'
    # app_secret = '5541b1b98ad4fe6b21efd4e4e60b8420'
    # app_key = '960be7c60d61b4bf'
    # app_secret = 'aa49263b4415a400e72ffcca281261ca'

    # app_key = '5d833444ba52e3fd'
    # app_secret = '6599b3a8387311fd77f576671f40b821'

    # get the body(payload)
    payload = {
        'scenes': ['porn', 'violence', 'politic', 'ocr'],
        'data': url_list
    }

    current_timestamp = str(round(time.time() * 1000))
    # print('current_timestamp: '+current_timestamp)

    # get the payload's sha1 value, get the bytes type and change it to str
    payload_sha1 = hashlib.sha1(json.dumps(payload).encode()).hexdigest()
    # print('payload_sha1: ', type(payload_sha1), payload_sha1)

    # get the headers' authorization
    # get the request_str(构造待签名数据字符串)
    request_str = 'POST' + '\n' + path + '\n' + '' + '\n' + current_timestamp + '\n' + payload_sha1
    # 使用App Secret 对待签名串进行签名
    sign = hmac.new(app_secret.encode(), request_str.encode(), digestmod=hashlib.sha256).hexdigest()
    # 将签名添加至请求头部
    authorization = 'CTYUN ' + app_key + ':' + sign
    # print('authorization: ', type(authorization), authorization)

    # 构造header
    headers = {
               'Content-Type': "application/json",
               'Authorization': authorization,
               'X-CTYUN-DT': current_timestamp
    }
    # 访问API接口
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response_result = json.loads(response.text)
    # print(response_text)
    return response_result


# def file_name(file_dir):
#     for root, dirs, files in os.walk(file_dir):
#         return files

def parse_response(response):
    result = response['result']
    labels = []
    confs = []
    for name in ['porn', 'violence', 'politic', 'ocr']:
        if name in result.keys():
            label = result[name][0]['label']
            confidence = result[name][0]['confidence']
        else:
            label = -1
            confidence = -1
        labels.append(label)
        confs.append(confidence)
    return labels, confs


def write(image_path, labels, confs, result_file):
    with open(result_file, 'a') as f:
        parts = [image_path]
        for lbl, conf in zip(labels, confs):
            rep = '%d,%f' % (lbl, conf)
            parts.append(rep)
        line = ','.join(parts)
        f.write('%s\n' % line)


if __name__ == '__main__':
    image_dir = r'D:\data\privacy\privacy'
    result_txt = r'D:\ProjectFile\21cn\detect_result_1023.txt'
    files = os.listdir(image_dir)
    label_data = []
    confidence_data = []
    name_data = []
    num = len(files)
    print("{} files will be detected under folder:{}, ".format(num, image_dir))
    flag = 0
    for file_name in files:
        flag += 1
        file_path = os.path.join(image_dir, file_name)
        b64_data = b64_data_uri(file_path)
        data = [b64_data]
        response_text = predict(data)
        print("detecting {}/{}:{}, result:{}".format(flag, num, file_name, response_text))
        time.sleep(1)
        try:
            label_data.append(response_text["result"]["politic"][0]["label"])
            confidence_data.append(response_text["result"]["politic"][0]["confidence"])
            name_data.append(file_name)
        except ValueError:
            print("******* ValueError *************")
        except IndexError:
            print("******* IndexError *************")

    # with open(result_txt, 'a') as f:
    #     for i in range(len(name_data)):
    #         s=str(label_data[i])+"|"+str(confidence_data[i])+"|"+name_data[i]
    #         f.write(s+'\n')
