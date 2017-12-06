# Utility functions for this project
# Xiaolin Li & Zijiang Yang

import random
import numpy as np
def sample_cropped_img(img_collections, batch_size, l, m, n_channel, seed=999):
	#randomly sample one image from image_collections
	#then crop a lxm sub-image from this sampled image
	random.seed(seed)
	n_imgs = len(img_collections)
	img_batch = np.zeros((batch_size, l, m, n_channel))

	for batch_idx in xrange(batch_size):

		img_index = random.randint(0, len(img_collections)-1) # pick an image
		img = img_collections[img_index, :, :, :]

		if n_channel == 1:
			L1, L2 = img.shape
		elif n_channel == 3:
			L1, L2, _ = img.shape
	
		assert L1 >= l
		assert L2 >= m

		x1 = random.randint(0, L1 - l)
		y1 = random.randint(0, L2 - m)

		if len(img.shape) == 2:
			cropped_img = img[x1: x1+l, y1: y1+m]
		elif len(img.shape) == 3:
			cropped_img = img[x1: x1+l, y1: y1+m, :]

		img_batch[batch_idx, :, :, :] = cropped_img

	return img_batch
