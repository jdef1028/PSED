# Utility functions for this project
# Xiaolin Li & Zijiang Yang
from __future__ import print_function
import h5py
import random
import numpy as np
from math import sqrt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
def sample_cropped_img(img_collections, batch_size, l, m, n_channel, seed=999):
	#randomly sample one image from image_collections
	#then crop a lxm sub-image from this sampled image
	random.seed(seed)
	n_imgs = img_collections.shape[0]
	img_batch = np.zeros((batch_size, l, m, n_channel))

	for batch_idx in xrange(batch_size):

		img_index = random.randint(0, len(img_collections)-1) # pick an image
		img = img_collections[img_index, :]
		img_size = int(sqrt(img.shape[0]))
		img = np.reshape(img, (img_size, img_size))

		L1, L2 = img.shape
	
		assert L1 >= l
		assert L2 >= m

		x1 = random.randint(0, L1 - l)
		y1 = random.randint(0, L2 - m)



		cropped_img = img[x1: x1+l, y1: y1+m]
		for i in xrange(n_channel):
			img_batch[batch_idx, :, :, i] = cropped_img

	return img_batch

def generate_image_snapshots(img_batch, num_img2plot, img_path):
	batch_size = img_batch.shape[0]
	img_to_pick = np.random.permutation(batch_size)[:num_img2plot]
	img_to_plot = img_batch[img_to_pick]
	img_to_plot = np.squeeze(img_to_plot, axis=3)
	fig, axs = plt.subplots(1, num_img2plot, edgecolor='r')
	axs = axs.ravel()
	for i in xrange(num_img2plot):
		axs[i].imshow(img_to_plot[i], cmap='Greys', interpolation='nearest')
		axs[i].axis('off')
	plt.savefig(img_path+'.png')







if __name__ == '__main__':

	with h5py.File('./data/data.mat', 'r') as f:
		img_collections = list(f['IMG'])
	img_collections = np.array(img_collections)
	img_collections = img_collections.T
	print(img_collections.shape)
	img_batch = sample_cropped_img(img_collections, 32, 50, 50, 3)
	print(img_batch.shape)
	temp = img_batch[1, :, :, :]
	plt.imshow(temp)
	plt.show()
