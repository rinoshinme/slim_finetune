import hashlib
import hmac
import base64
import time
import os


def curl_cmd(url, headers, payload):
    heads = []
    for k, v in headers.items():
        heads.append("-H \"" + f"{k}: {v}" + "\"")
    head = ' '.join(heads)
    import json
    data = json.dumps(payload)
    req = f'curl -X POST -H "Host: apiserver.ai.ctyun.cn" {head} -d \'{data}\' {url}'
    # req = f'curl -X POST {head} -d \'{data}\' {url}'
    # print(req)
    # with open('/home/fan/windows/share/images/req.txt', 'w') as f:
    #     f.write(req)


def sign(http_method, request_path, request_query, x_ctyun_dt, request_body):
    body_digest = hashlib.sha1(request_body).hexdigest()
    request_str = "{}\n{}\n{}\n{}\n{}".format(http_method, request_path, request_query, x_ctyun_dt, body_digest)
    signature = hmac.new(app_secret.encode(), request_str.encode(), digestmod=hashlib.sha256).hexdigest()
    return {"Authorization": "CTYUN {}:{}".format(app_key, signature)}


def unescape(s):
    return s.encode().decode('unicode-escape')


def text_b64_encode(text):
    return f"data:application/octet-stream;base64,{base64.urlsafe_b64encode(text.encode('utf-8')).decode('utf-8')}"


def b64_data_uri(image_path):
    with open(image_path, 'rb') as f:
        content = f.read()
    b64_content = base64.urlsafe_b64encode(content)
    return "data:application/octet-stream;base64,{}".format(b64_content.decode())


def unixtimestamp():
    return str(int(time.time() * 1000))
