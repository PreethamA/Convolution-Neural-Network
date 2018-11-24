"""
test script to test the new saples with the graph model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import argparse
import os.path
import re
import sys
import tarfile
import pandas as pd
import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


# pylint: enable=line-too-long

# from tensorflow.python.platform import gfile




def print_label_and_predictions(image_list, probabilities_list):
    """
    uses label-array and prediction-array + image_list
    builds a 2D-array of them

    return: table
    """
    table = [[0 for image_label in range(11)] for datensatz in range(image_list + 1)]
    table[0][0] = 'image_file'
    with open(os.path.join(FLAGS.model_dir + "output_labels.txt"), "r") as r:
        for i, label in enumerate(r):
            table[0][i + 1] = label
    print(table[0])
    for p, probabilities in enumerate(probabilities_list):
        table[p + 1][0] = image_list[p]
        for q, probability in enumerate(probabilities):
            table[p + 1][q + 1] = probability
        print(table[p + 1])

    return table
    # print(label.rstrip(), probabilities[i])


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            # FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            FLAGS.model_dir, 'output_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image, T):
    """Runs inference on an image.

    Args:
      image: Image file name.

    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    if T == 0:
        create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)


        return predictions


def main(_):
    image_dir = FLAGS.image_dir
    image_list = os.listdir(image_dir)
    image_files, prediction_list = [], []
    for i, image in enumerate(image_list):
        image_files.append(image_dir + image)
    for i, image_file in enumerate(image_files):

        prediction_for_image = run_inference_on_image(image_file, i)

        prediction_list.append(prediction_for_image)



    lines = list(tuple(open(os.path.join(FLAGS.model_dir, "output_labels.txt"), 'r')))
    # print(len(lines))
    col=['filename']
    mergedlist=col+lines
    d_1 = np.column_stack((image_list, np.round(prediction_list)*100))

    df_proba = pd.DataFrame(d_1)

    df_proba.columns = mergedlist
    writer = pd.ExcelWriter(os.path.join(FLAGS.image_dir,'cnntestresults.xlsx'), engine='xlsxwriter')
    df_proba.to_excel(writer, sheet_name='cnntestresults')
    writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.join(tempfile.gettempdir(),'savemodel'),
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Absolute path to image dir.'
    )
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
