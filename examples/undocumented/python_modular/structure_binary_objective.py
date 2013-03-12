#!/usr/bin/env python
#!/usr/bin/env python

import numpy as np

from shogun.Features 	import RealFeatures
from shogun.Loss     	import HingeLoss
from shogun.Structure	import MulticlassModel, MulticlassSOLabels, RealNumber, ResultSet

def gen_data(num_classes,num_samples,dim):
	np.random.seed(0)
	covs = np.array([[[0., -1. ], [2.5,  .7]],
			 [[3., -1.5], [1.2, .3]],
			 [[ 2,  0  ], [ .0,  1.5 ]]])
	X = np.r_[np.dot(np.random.randn(num_samples, dim), covs[0]) + np.array([0, 10]),
		  np.dot(np.random.randn(num_samples, dim), covs[1]) + np.array([-10, -10]),
		  np.dot(np.random.randn(num_samples, dim), covs[2]) + np.array([10, -10])];
	Y = np.hstack((np.zeros(num_samples), np.ones(num_samples), 2*np.ones(num_samples)))
	return X, Y

# Number of classes
M = 3
# Number of samples of each class
N = 50
# Dimension of the data
dim = 2

traindat, label_traindat = gen_data(M,N,dim)

X = np.array([[-10., -4., 6., 5.]])
Y = np.array([1., 1., 0., 0.])
print X
print Y

labels = MulticlassSOLabels(Y)
features = RealFeatures(X)
model = MulticlassModel(features,labels)
print '#classes =',labels.get_num_classes()
print '#feature vectors =',features.get_num_vectors()
print 'features\' dimension =', features.get_num_features()
print 'psi dimension should be =',labels.get_num_classes()*features.get_num_features()

C = np.array([0.01, 0.10, 1., 100])
N = features.get_num_vectors()
w1 = np.arange(-3.,5.,.1)
w2 = np.arange(-2.,3.,.1)
W1,W2 = np.meshgrid(w1,w2)
O = np.zeros(W1.shape)		# objective
V = []

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()

for c in range(C.shape[0]):
	ax = fig.add_subplot(2,2,c+1)

	for i in range(O.shape[0]):
		for j in range(O.shape[1]):
			w = np.array([W1[i][j], W2[i][j]])
			scores = 0.
			for vec_idx in range(features.get_num_vectors()):
				result = model.argmax(w,vec_idx)
				scores += result.score
			# Compute objective
			O[i][j] = 1./2.*np.linalg.norm(w)**2 + C[c]/N*scores


	#ax = fig.gca(projection='3d')
	#surf = ax.plot_surface(W1, W2, O, rstride=1, cstride=1, linewidth=0)
	#cs = plt.contourf(W1,W2,O)
	#plt.clabel(cs, inline=1, fontsize=10)
	#plt.show()

	from pylab import pcolor, contour, axis, show, colorbar

	c = pcolor(W1, W2, O, shading='interp')
	ax.contour(W1, W2, O, 15, linewidths=1, colors='black', hold=True)
	axis('tight')
	colorbar(c)

show()

'''
parameter_list = [[X,y]]

def so_multiclass (fm_train_real=traindat,label_train_multiclass=label_traindat):
	labels = MulticlassSOLabels(label_train_multiclass)
	features = RealFeatures(fm_train_real.T)

	model = MulticlassModel(features, labels)
	loss = HingeLoss()
	sosvm = PrimalMosekSOSVM(model, loss, labels)
	sosvm.train()

	out = sosvm.apply()
	count = 0
	for i in xrange(out.get_num_labels()):
		yi_pred = RealNumber.obtain_from_generic(out.get_label(i))
		if yi_pred.value == label_train_multiclass[i]:
			count = count + 1

	print "Correct classification rate: %0.2f" % ( 100.0*count/out.get_num_labels() )

if __name__=='__main__':
	print('KNN')
	so_milticlass(*parameter_list[0])
'''
