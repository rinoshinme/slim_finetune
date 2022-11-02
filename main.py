import os
from options.violence9 import demo_options, train_options, test_options, protobuf_options
from tfvortex.demo import Demo
from tfvortex.test import Tester
from tfvortex.train import Trainer
from tfvortex.protobuf import PbExporter, PbTester


def run_demo():
    options = demo_options()
    demo = Demo(options)
    image_path = r'D:\liyu100.jpg'
    res = demo.test(image_path)
    print(res)

def run_demo_folder():
    options = demo_options()
    demo = Demo(options)
    # image_folder = r'D:\projects\Violence\test_for_audition\ng'
    image_folder = r'E:\projects\Violence\21cn_images\21cn_image_test\normal2'
    # save_path = r'D:\projects\Violence\test_for_audition\ng.txt'
    save_path = r'E:\projects\Violence\21cn_images\21cn_image_test\normal2.txt'

    fp = open(save_path, 'w')
    for idx, name in enumerate(os.listdir(image_folder)):
        print('processing {} {}'.format(idx, name))
        image_path = os.path.join(image_folder, name)
        try:
            res = demo.test(image_path)
            score = sum([res[i] for i in [1, 3, 4, 5, 6, 7, 8]])
            fp.write('{},{},{}\n'.format(name, 0, score))
        except:
            pass
    fp.close()


def run_train():
    options = train_options()
    trainer = Trainer(options)
    trainer.train()


def run_test():
    options = test_options()
    tester = Tester(options)
    tester.test()


def run_proto():
    options = protobuf_options()
    # pb_exporter = PbExporter(options)
    # pb_exporter.export('./data/resnetv1_101.pb')

    pb_tester = PbTester('./data/resnetv1_101.pb', options)
    image_path = r'D:\liyu100.jpg'
    output = pb_tester.infer(image_path)
    print(output)


def save_tfserving():
    """
    save tfserving models, need customization.
    """
    from tfvortex.deploy.export_savedmodel import run
    run()


if __name__ == '__main__':
    # run_test()
    # run_demo()
    run_demo_folder()
    # run_train()
    # run_proto()
    # save_tfserving()
