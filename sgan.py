from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Input
from keras.initializers import Zeros, RandomNormal, RandomUniform
from keras.optimizers import Adam
from keras.regularizers import l2
from utils import *
import numpy as np
import scipy.io as sio
import os
import datetime
# timestamp now
now = datetime.datetime.now()

# input tensor dimensions
Z_l = 2
Z_m = 2
Z_d = 100

# image data specification
data_path = './data/data.mat' #needs to be a .mat file
data_var_name = 'image'

# image channel num

n_channel = 3

# generator dimensions
g_filter_sizes = [(5, 5)] * 5 + [(5, 5)]
g_filter_depths = [1024, 512, 256, 128, 64] + [n_channel]

# discrimitor dimensions
d_filter_sizes = [(5, 5)] * 5 + [(5, 5)]
d_filter_depths = [64, 128, 256, 512, 1024] + [1]

assert len(g_filter_depths) == len(g_filter_sizes)
assert len(d_filter_depths) == len(d_filter_sizes)

# compute cropped image size
X_l = Z_l * (2**len(g_filter_depths))
X_m = Z_m * (2**len(g_filter_depths)) 

print("The dimension of the cropped image is: (" + str(X_l) + " x " + str(X_m) +")")

#training parameters
batch_size = 64
epoch_num = 100
ratio_btwn_D_G = 5 # train k steps for discriminator then 1 step for generator


# regularization penalty parameter
regularizers_weight = 0.02
# optimization paramters
adam_opt = Adam(lr=0.0005, beta_1=0.5, epsilon=1e-7)

class SGAN(object):
	def __init__(self):
		self._build_sgan()
		self.train()

	def _build_sgan(self):

		# build generator
		Z_dim = (Z_l, Z_m, Z_d)
		self.generator = self._build_generator(Z_dim)
		self.generator.compile(loss='binary_crossentropy',
							   optimizer=adam_opt,
							   metric=['accuracy'])

		# build discriminator
		img_dim = (X_l, X_m, n_channel)
		self.discriminator = self._build_discriminator(img_dim)
		self.generator.compile(loss='binary_crossentropy',
							   optimizer=adam_opt,
							   metric=['accuracy'])


		z = Input(shape=Z_dim)
		img = self.generator(z)
		#self.discriminator.trainable = False
		validity = self.discriminator(img)

		self.stackedGAN = Model(z, validity)
		self.stackedGAN.compile(loss='binary_crossentropy',
								optimizer=adam_opt,
								metric=['accuracy'])





	def _build_generator(self, Z_vector_dim):
		# generator builder
		generator = Sequential()
		for lv in xrange(len(g_filter_sizes)-1): # construction all layers in generator but not the last
			if lv == 0:
				generator.add(Conv2DTranspose(filters=g_filter_depths[lv],
							 kernel_size=g_filter_sizes[lv],
							 padding="same",
							 activation='relu',
							 strides=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros(),
							 input_shape=Z_vector_dim
							 )
					 )
			else:
				generator.add(Conv2DTranspose(filters=g_filter_depths[lv],
							 kernel_size=g_filter_sizes[lv],
							 padding="same",
							 activation='relu',
							 strides=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros(),
							 )
					 )
			generator.add(BatchNormalization(beta_initializer='zeros',
										 gamma_initializer=RandomNormal(mean=1., stddev=0.02)))

		# add the last layer of the generator
		generator.add(Conv2DTranspose(filters=g_filter_depths[lv+1],
							 kernel_size=g_filter_sizes[lv+1],
							 padding="same",
							 activation='tanh',
							 strides=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros()
							 )
				 )

		generator.summary()

		Z_vector = Input(shape=Z_vector_dim)

		X = generator(Z_vector)

		return Model(Z_vector, X)

	def _build_discriminator(self, img_dim):
		discriminator = Sequential()
		for lv in xrange(len(d_filter_sizes)-1):
			if lv == 0:
				discriminator.add(Conv2D(filters=d_filter_depths[lv],
							 kernel_size=d_filter_sizes[lv],
							 padding="same",
							 strides=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros(),
							 input_shape=img_dim
							 )
					 )
			else:
				discriminator.add(Conv2D(filters=d_filter_depths[lv],
							 kernel_size=d_filter_sizes[lv],
							 padding="same",
							 strides=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros(),
							 )
					 )
			discriminator.add(LeakyReLU(alpha=0.2))
			discriminator.add(BatchNormalization(beta_initializer='zeros',
										 gamma_initializer=RandomNormal(mean=1., stddev=0.02)))

		#add the last layer to discriminator
		discriminator.add(Conv2D(filters=d_filter_depths[lv+1],
							 kernel_size=d_filter_sizes[lv+1],
							 padding="same",
							 activation='sigmoid',
							 strides=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros()
							 )
				 )
		discriminator.summary()

		X = Input(shape=img_dim)
		prediction = discriminator(X)

		return Model(X, prediction)


	def train(self):
		# image data loading and preparation

		img_collection=sio.loadmat(data_path)[data_var_name] #load img collections

		half_batch_size = int(batch_size / 2) # fake and real data will be of half_batch_size each
		for minibatch_epoch in epoch_num:

			self.discriminator.trainable = True
			# update the discriminator
			for discriminator_step in xrange(ratio_btwn_D_G):
				#create minibatch
				Z_batch = np.random.normal(0, 1, (half_batch_size, Z_l, Z_m, Z_d)) # this distribution is subject to change according to the assumption made
				X_batch = sample_cropped_img(img_collection, half_batch_size, X_l, X_m, n_channel)
				fake_img_batch = self.generator.predict(Z_batch)

				# mix them together and shuffle
				minibatch_X = np.concatenate((fake_img_batch, X_batch), axis=0)
				minibatch_Y = np.concatenate((np.zeros((half_batch_size,1)), np.ones((half_batch_size,1))), axis=0)

				sequence = np.random.permutation(batch_size)

				minibatch_X = minibatch_X[sequence, :, :, :]
				minibatch_Y = minibatch_Y[sequence, :, :, :]

				#d_loss_fake = self.discriminator.train_on_batch(fake_img_batch, np.zeros((half_batch_size, 1)))
				#d_loss_real = self.discriminator.train_on_batch(X_batch, np.ones((half_batch_size, 1)))
				#d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

				d_loss = self.discriminator.train_on_batch(minibatch_X, minibatch_Y)

			# update the generator
			self.discriminator.trainable = False

			Z_batch = np.random.normal(0, 1, (batch_size, Z_l, Z_m, Z_d))

			g_loss = self.stackedGAN.train_on_batch(Z_batch, np.ones(batch_size,1))

			print("Minibatch Epoch %d: [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (minibatch_epoch, d_loss[0], 100*d_loss[1], g_loss))

		dir_name = './model/'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+' '+str(now.hour)+':'+str(now.minute)
		if not os.path.isdir(dir_name):
			os.mkdir(dir_name)
		self.generator.save(dir_name+'/generator.h5')
		self.discriminator.save(dir_name+'/discriminator.h5')
		self.stackedGAN.save(dir_name+'/allTogether.h5')
















Model = SGAN()






