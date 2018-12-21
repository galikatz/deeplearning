# USAGE
# python3 test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from imutils import paths
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-r", "--results", required=True,
	help="path to results dir")

args = vars(ap.parse_args())
RESULTS_DIR = args["results"]

IMAGE_DIMS = (96, 96, 3)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])


# load the image
test_image_path = sorted(list(paths.list_images(args["image"])))
image_index = 1
for image_path in test_image_path:
	image = cv2.imread(image_path)
	orig = image.copy()
	# pre-process the image for classification
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# classify the input image
	# labels list
	image_name = image_path.split(os.path.sep)[-1]
	file_name_arr = image_name.split('_')
	counting_lable = file_name_arr[4].split('.')[0]

	# build the label
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = counting_lable

	label = "{}: {:.2f}%".format(label, proba[idx] * 100)

	# draw the label on the image
	output = imutils.resize(orig, width=400)
	cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 2)
	file_name = 'Output_counting_class_'+label.replace(' ', '_').replace('%', '').replace(':', '')
	# show the output image
	cv2.imwrite(RESULTS_DIR+'/'+file_name+'.png', output)
	image_index += 1

