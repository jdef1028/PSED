import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
def lossPlot(path):
	with open(path+'/training_history.pickle', 'r') as f:
		data = pickle.load(f)
	x = range(len(data['d_loss']))
	plt.figure(1)
	plt.plot(x[100:], data['d_loss'][100:])
	plt.plot(x[100:], data['g_loss'][100:])
	plt.legend(['d_loss', 'g_loss'])
	plt.show()

	plt.figure(2)
	plt.plot(x[10:], data['d_acc'][10:])
	plt.plot(x[10:], data['g_acc'][10:])
	plt.legend(['d_acc', 'g_acc'])
	plt.show()
if __name__=='__main__':
	lossPlot('./metric/test1')
