# Utility functions for this project
# Xiaolin Li
# Last modified on Dec 4, 2017
import random
def make_trainable(net, boolVal):
	# freeze or release the layer
	net.trainable = boolVal
	for ly in net.layers:
		ly.trainable = boolVal

def img_crop_generator(img_collections, l, m, seed=999):
	#randomly sample one image from image_collections
	#then crop a lxm sub-image from this sampled image
	random.seed(seed)
	n_imgs = len(img_collections)
	img_index = random.randint(0, len(img_collections)-1)
	img = img_collections[img_index]

	if len(img.shape) == 2:
		L1, L2 = img.shape
	elif len(img.shape) == 3:
		L1, L2, _ = img.shape
	
	assert L1 >= l
	assert L2 >= m

	x1 = random.randint(0, L1 - l)
	y1 = random.randint(0, L2 - m)

	if len(img.shape) == 2:
		cropped_img = img[x1: x1+l, y1: y1+m]
	elif len(img.shape) == 3:
		cropped_img = img[x1: x1+l, y1: y1+m, :]

	yield cropped_img
