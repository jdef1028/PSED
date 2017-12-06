from __future__ import print_function
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from keras.initializers import Zeros, RandomNormal, RandomUniform
from keras.optimizers import Adam
from keras.regularizers import l2
# input tensor dimensions
Z_l = 2
Z_m = 2
Z_d = 100

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

#batch parameters
batch_size = 64
# regularization penalty parameter
regularizers_weight = 0.02
# optimization paramters
adam_opt = Adam(lr=0.0005, beta_1=0.5, epsilon=1e-7)

class SGAN(object):
	def __init__(self):
		self._build_sgan()

	def _build_sgan(self):

		# build generator
		self.generator = Sequential()
		for lv in xrange(len(g_filter_sizes)-1): # construction all layers in generator but not the last
			if lv == 0:
				self.generator.add(Conv2D(filters=g_filter_depths[lv],
							 kernel_size=g_filter_sizes[lv],
							 padding="same",
							 activation='relu',
							 dilation_rate=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros(),
							 input_shape=(Z_d, Z_l, Z_m)
							 )
					 )
			else:
				self.generator.add(Conv2D(filters=g_filter_depths[lv],
							 kernel_size=g_filter_sizes[lv],
							 padding="same",
							 activation='relu',
							 dilation_rate=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros(),
							 )
					 )
			self.generator.add(BatchNormalization(beta_initializer='zeros',
										 gamma_initializer=RandomNormal(mean=1., stddev=0.02)))

		# add the last layer of the generator
		self.generator.add(Conv2D(filters=g_filter_depths[lv+1],
							 kernel_size=g_filter_sizes[lv+1],
							 padding="same",
							 activation='tanh',
							 dilation_rate=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros()
							 )
				 )
		generator_output = self.generator.layers[-1].output

		print("=== Generator configured ====")

		# build discriminator
		self.discriminator = Sequential()
		for lv in xrange(len(d_filter_sizes)-1):
			if lv == 0:
				self.discriminator.add(Conv2DTranspose(filters=g_filter_depths[lv],
							 kernel_size=g_filter_sizes[lv],
							 padding="same",
							 dilation_rate=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros(),
							 input_shape=(n_channel, X_l, X_m)
							 )
					 )
			else:
				self.discriminator.add(Conv2DTranspose(filters=g_filter_depths[lv],
							 kernel_size=g_filter_sizes[lv],
							 padding="same",
							 dilation_rate=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros(),
							 )
					 )
				self.discriminator.add(LeakyReLU(alpha=0.2))
			self.discriminator.add(BatchNormalization(beta_initializer='zeros',
										 gamma_initializer=RandomNormal(mean=1., stddev=0.02)))

		self.discriminator.add(Conv2DTranspose(filters=g_filter_depths[lv+1],
							 kernel_size=g_filter_sizes[lv+1],
							 padding="same",
							 activation='sigmoid',
							 dilation_rate=(2, 2),
							 kernel_initializer=RandomNormal(stddev=0.02),
							 kernel_regularizer=l2(regularizers_weight),
							 bias_initializer=Zeros()
							 )
				 )
		discriminator_output = self.discriminator.layers[-1].output

		print("=== Discriminator configured === ")


		




Model = SGAN()






