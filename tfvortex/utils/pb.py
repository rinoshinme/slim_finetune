import tensorflow as tf


def inspect_protobuf(pb_path):
    with tf.Session() as sess:
        with open(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            for node in graph_def.node:
                print(node.name)
