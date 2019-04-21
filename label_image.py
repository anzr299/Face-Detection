from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import cv2
import pyttsx3

cap = cv2.VideoCapture(0)
prev=False
engine = pyttsx3.init() 

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  float_caster = tf.cast(file_name, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result



def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label
while(True):
  if __name__ == "__main__":
    ret,frame = cap.read()
    file_name = frame
    model_file = \
      "output_graph.pb"
    label_file = "output_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"


    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
      if(labels[i]=="faces" and results[i]>0.95 and prev==False):
        print(labels[i], results[i])
        engine.say("hello")
        engine.runAndWait()
        engine.say("and")
        engine.runAndWait()
        engine.say("welcome")
        engine.runAndWait()
        engine.say("to")
        engine.runAndWait()
        engine.say("DPS")
        engine.runAndWait()
        engine.say("Sharjah")
        engine.runAndWait()
        prev = True
      elif(labels[i]=="faces" and results[i]<0.95):
        prev=False