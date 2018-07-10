import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import cv2
cap = cv2.VideoCapture('videoplay.mp4')

width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file


# In[10]:

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      count += 1
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      '''
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      '''
      #width, height = image_np.size
      aov = 55
      ch = 180
      imageHeight = int(height)
      imageWidth = int(width)
      imageHeightCenter = imageHeight / 2
      imageWidthCenter = imageWidth / 2
      pixelDegree = float(aov) / imageWidth

      # Convert tensorflow data to pandas data frams
      df = pd.DataFrame(boxes.reshape(100, 4), columns=['y_min', 'x_min', 'y_max', 'x_max'])
      df1 = pd.DataFrame(classes.reshape(100, 1), columns=['classes'])
      df2 = pd.DataFrame(scores.reshape(100, 1), columns=['scores'])
      df5 = pd.concat([df, df1, df2], axis=1)

      # Transform box bound coordinates to pixel coordintate

      df5['y_min_t'] = df5['y_min'].apply(lambda x: x * imageHeight)
      df5['x_min_t'] = df5['x_min'].apply(lambda x: x * imageWidth)
      df5['y_max_t'] = df5['y_max'].apply(lambda x: x * imageHeight)
      df5['x_max_t'] = df5['x_max'].apply(lambda x: x * imageWidth)

      # Create objects pixel location x and y
      # X
      df5['ob_wid_x'] = df5['x_max_t'] - df5["x_min_t"]
      df5['ob_mid_x'] = df5['ob_wid_x'] / 2
      df5['x_loc'] = df5["x_min_t"] + df5['ob_mid_x']
      # Y
      df5['ob_hgt_y'] = df5['y_max_t'] - df5["y_min_t"]
      df5['ob_mid_y'] = df5['ob_hgt_y'] / 2
      df5['y_loc'] = df5["y_min_t"] + df5['ob_mid_y']

      # Find object degree of angle, data is sorted by score, select person with highest score
      df5['object_angle'] = df5['x_loc'].apply(lambda x: -(imageWidthCenter - x) * pixelDegree)

      df6 = df5.loc[df5['classes'] == 1]

      if (df6.empty) or (df6.iloc[0]['scores'] < 0.20) :

          continue

      df7 = df6.iloc[0]['object_angle']

      w = int(df6.iloc[0]['ob_wid_x'])
      x = int(df6.iloc[0]['x_min_t'])
      h = int(df6.iloc[0]['ob_hgt_y'])
      y = int(df6.iloc[0]["y_min_t"])

      AOB = df7 + ch
      AOB_str = str(round(AOB, 4))
      #print df6.head()
      #print imageHeight, imageWidth

      labelBuffer = int(df6.iloc[0]['y_min_t']) - int(df6.iloc[0]['ob_hgt_y'] * 0.1)

      # print
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(image_np, AOB_str, (int(df6.iloc[0]['x_min_t']), labelBuffer), font, 0.8, (0, 255, 0), 2)


      cv2.rectangle(image_np, (x,y), (x+w, y+h), (0, 255, 0), 2)
      roi = image_np[y:y+h, x:x+w]
      cv2.imwrite("save_image/frame%d.jpg" % count, roi)
      print width, height, x,y,w,h


      cv2.imshow("Presentation Tracker", cv2.resize(roi, (640,480)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
