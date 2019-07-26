import tensorflow as tf
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard


def insepct(pb_model, log_path):
    # model = 'model.pb'
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    graph_def.ParseFromString(tf.gfile.FastGFile(pb_model, 'rb').read())
    tf.import_graph_def(graph_def, name='graph')
    summaryWriter = tf.summary.FileWriter(log_path, graph)


def inspect2(pb_model, log_path):
    import_to_tensorboard(model_dir=pb_model, log_dir=log_path)


if __name__ == '__main__':
    # pb_model = r'D:/projects/slim_finetune/export_tfserving/violence_det/1/saved_model.pb'
    pb_model = r'D:/projects/sjl-deployment/nsfw/1/saved_model.pb'
    log_path = r'D:/logs'
    inspect2(pb_model, log_path)
