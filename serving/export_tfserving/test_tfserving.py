import requests
import base64

SERVER_URL = 'http://101.91.169.58:9000/1/models/violence_det'


def run(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()

    data64 = base64.encodebytes(data)

    predict_request = '{"instances":%s}' % str(data64)
    response = requests.post(SERVER_URL, data=predict_request)
    prediction = response.json()['predict'][0]

    print(prediction)


if __name__ == '__main__':
    image_path = r'D:/data020_004367.jpg'
    run(image_path)
