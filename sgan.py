from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Input, GlobalAveragePooling2D
from keras.initializers import Zeros, RandomNormal, RandomUniform
from keras.optimizers import Adam
from keras.regularizers import l2
from utils import *
import numpy as np
import scipy.io as sio
import os
import datetime
import cPickle as pickle
import h5py
# timestamp now
now = datetime.datetime.now()

# input tensor dimensions
Z_l = 2
Z_m = 2
Z_d = 100

# image data specification
data_path = './data/data.mat' #needs to be a .mat file
data_var_name = 'IMG'

# image channel num

n_channel = 1

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
epoch_num = int(1e5)
D_steps = 1 # in each epoch, train discriminator for D_steps times
G_steps = 5 # in each epoch, train generator for G_steps times
mix_minibatch = False

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
							   metrics=['accuracy'])

		# build discriminator
		img_dim = (X_l, X_m, n_channel)
		self.discriminator = self._build_discriminator(img_dim)
		self.discriminator.compile(loss='binary_crossentropy',
							   optimizer=adam_opt,
							   metrics=['accuracy'])


		z = Input(shape=Z_dim)
		img = self.generator(z)
		#self.discriminator.trainable = False
		validity = self.discriminator(img)

		self.stackedGAN = Model(z, validity)
		self.stackedGAN.compile(loss='binary_crossentropy',
								optimizer=adam_opt,
								metrics=['accuracy'])





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
		discriminator.add(GlobalAveragePooling2D())
		discriminator.summary()

		X = Input(shape=img_dim)
		prediction = discriminator(X)

		return Model(X, prediction)


	def train(self):
		# image data loading and preparation
		self.recorder = {'g_loss':[],
						 'd_loss':[],
						 'd_acc':[],
						 'g_acc':[]}
		# img_collection=sio.loadmat(data_path)[data_var_name] #load img collections
		print('===> Loading data...')
		img_collection_data = h5py.File(data_path, 'r')
		img_collection = np.transpose(img_collection_data[data_var_name])
		print('===> Data loaded!')

		half_batch_size = int(batch_size / 2) # fake and real data will be of half_batch_size each
		for minibatch_epoch in xrange(epoch_num):
			if minibatch_epoch % 1 == 0:
				print('===> Mini_epoch:', minibatch_epoch)
			self.discriminator.trainable = True
			# update the discriminator
			for discriminator_step in xrange(D_steps):
				#create minibatch
				Z_batch = np.random.normal(0, 1, (half_batch_size, Z_l, Z_m, Z_d)) # this distribution is subject to change according to the assumption made
				X_batch = sample_cropped_img(img_collection, half_batch_size, X_l, X_m, n_channel)
				fake_img_batch = self.generator.predict(Z_batch)

				# mix them together and shuffle
				if mix_minibatch:
					minibatch_X = np.concatenate((fake_img_batch, X_batch), axis=0)
					minibatch_Y = np.concatenate((np.zeros((half_batch_size,1)), np.ones((half_batch_size,1))), axis=0)

					sequence = np.random.permutation(batch_size)

					minibatch_X = minibatch_X[sequence]
					minibatch_Y = minibatch_Y[sequence]
					d_loss = self.discriminator.train_on_batch(minibatch_X, minibatch_Y)
				else:
					# doesn't apply mixing of minibatch, train separately

					d_loss_fake = self.discriminator.train_on_batch(fake_img_batch, np.zeros((half_batch_size, 1)))
					d_loss_real = self.discriminator.train_on_batch(X_batch, np.ones((half_batch_size, 1)))
					d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

				

			# update the generator
			self.discriminator.trainable = False
			for generator_step in xrange(G_steps):
				Z_batch = np.random.normal(0, 1, (batch_size, Z_l, Z_m, Z_d))

				g_loss = self.stackedGAN.train_on_batch(Z_batch, np.ones((batch_size,1)))

			print("Minibatch Epoch %d: [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (minibatch_epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
			self.recorder['d_loss'].append(d_loss[0])
			self.recorder['d_acc'].append(100*d_loss[1])
			self.recorder['g_loss'].append(g_loss[0])
			self.recorder['g_acc'].append(100*g_loss[1])


		dir_name = './model/'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+' '+str(now.hour)+':'+str(now.minute)
		if not os.path.isdir(dir_name):
			os.mkdir(dir_name)
		self.generator.save(dir_name+'/generator.h5')
		self.discriminator.save(dir_name+'/discriminator.h5')
		self.stackedGAN.save(dir_name+'/allTogether.h5')

		with open(dir_name+'/training_history.pickle', 'wb') as f:
			pickle.dump(self.recorder, f)















Model = SGAN()






