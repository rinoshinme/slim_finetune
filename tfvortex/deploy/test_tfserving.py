import base64
import os
import requests

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
# SERVER_URL = 'http://localhost:8501/v1/models/resnet:predict'
SERVER_URL = 'http://localhost:9091/v1/models/violence:predict'

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
IMAGE_PATH = os.path.expanduser('~/Desktop/fire_002626.jpg')
IMAGE_FOLDER = os.path.expanduser('~/Desktop/violence_images')


def read_imagesb64():
    files = os.listdir(IMAGE_FOLDER)
    data = []
    for name in files:
        if not name.endswith('.jpg'):
            continue
        name = os.path.join(IMAGE_FOLDER, name)
        with open(name, 'rb') as f:
            img_data = f.read()
            data.append(base64.b64encode(img_data, b'-_').decode('utf-8'))
    return data


def test_multiple_images():
    jpeg_data = read_imagesb64()
    for data in jpeg_data:
        predict_request = '{"instances" : ["%s"]}' % data

        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        prediction = response.json()
        print(prediction)


def main():
    # Download the image
    # dl_request = requests.get(IMAGE_URL, stream=True)
    # dl_request.raise_for_status()

    # Compose a JSON Predict request (send JPEG image in base64).
    # jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
    with open(IMAGE_PATH, 'rb') as f:
        data = f.read()
        jpeg_bytes = base64.b64encode(data, b'-_').decode('utf-8')

    # input instances are a list of strings.
    predict_request = '{"instances" : ["%s"]}' % jpeg_bytes

    # Send few requests to warm-up the model.
    for _ in range(3):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()

    # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 1
    for _ in range(num_requests):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()
        print(prediction)


def run(image_path):
    SERVER_URL = 'http://101.91.169.58:9000/1/models/violence_det'
    with open(image_path, 'rb') as f:
        data = f.read()

    data64 = base64.encodebytes(data)

    predict_request = '{"instances":%s}' % str(data64)
    response = requests.post(SERVER_URL, data=predict_request)
    prediction = response.json()['predict'][0]

    print(prediction)



if __name__ == '__main__':
    # main()
    test_multiple_images()

    # 
    image_path = r'D:/data020_004367.jpg'
    run(image_path)
