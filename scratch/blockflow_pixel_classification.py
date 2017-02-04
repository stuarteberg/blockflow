import sys
import os
import subprocess
import numpy as np
import vigra
import h5py
from networkx.drawing import nx_pydot

from blockflow import global_graph, ReadArray, PixelFeatures, FilterSpec, PredictPixels

from lazyflow.classifiers import ParallelVigraRfLazyflowClassifier

# hdf5, all features, prediction
# 
# parameter mechanism, training
#
os.chdir(os.path.dirname(__file__))

zebra = vigra.impex.readImage('zebra.jpg').withAxes('yxc').mean(axis=-1)
vigra.impex.writeImage(zebra, 'zebra-gray.png')
subprocess.call('open zebra-gray.png', shell=True)
print zebra.shape

with h5py.File('zebra-pc.ilp', 'r') as ilp:
    classifier = ParallelVigraRfLazyflowClassifier.deserialize_hdf5(ilp['PixelClassification/ClassifierForests'])

filter_specs = [ FilterSpec('GaussianSmoothing', 0.3),
                 FilterSpec('GaussianSmoothing', 0.7),

                 FilterSpec('LaplacianOfGaussian', 0.7),
                 FilterSpec('LaplacianOfGaussian', 1.6),
                 FilterSpec('LaplacianOfGaussian', 3.5),

                 FilterSpec('GaussianGradientMagnitude', 0.7),
                 FilterSpec('GaussianGradientMagnitude', 1.6),
                 FilterSpec('GaussianGradientMagnitude', 3.5),

                 FilterSpec('DifferenceOfGaussians', 0.7),
                 FilterSpec('DifferenceOfGaussians', 1.6),
                 FilterSpec('DifferenceOfGaussians', 3.5),

                 FilterSpec('StructureTensorEigenvalues', 0.7),
                 FilterSpec('StructureTensorEigenvalues', 1.6),
                 FilterSpec('StructureTensorEigenvalues', 3.5),

                 FilterSpec('HessianOfGaussianEigenvalues', 0.7),
                 FilterSpec('HessianOfGaussianEigenvalues', 1.6),
                 FilterSpec('HessianOfGaussianEigenvalues', 3.5), ]

read_array = ReadArray()
pf = PixelFeatures()
predict = PredictPixels()

input_op = read_array(zebra[None, None, ..., None])
pf_op = pf(input_op, filter_specs)
predict_op = predict(pf_op, classifier)

box = [(0,0,0,0,0), (1,1,400,600,10)] # actual image is smaller than this, but that's okay.

# Construct call graph
with global_graph.register_calls():
    predict_op.dry_pull(box)

# Visualize graph
dot_path = 'blockflow-graph.dot'
nx_pydot.write_dot(global_graph.dag, dot_path)
subprocess.call('dot -Tpng -o {}.png {}'.format(dot_path, dot_path), shell=True)
subprocess.call('open {}.png'.format(dot_path), shell=True)


features = pf_op.pull( box )
print features.shape
#features = features.view(np.ndarray)
np.save('zebra-features.npy', features)

predictions = predict_op.pull( box )
print predictions.shape

# Normalize to uint8 range
predictions = (255*predictions).astype(np.uint8)

if predictions.shape[-1] == 2:
    # Make RGB image from RG image, so PNG can be used.
    result = np.concatenate((predictions, np.zeros_like(predictions[...,0:1])), axis=-1)

# Write filtered image and open
result = vigra.taggedView(result, 'tzyxc')
vigra.impex.writeImage(result[0,0,...], 'zebra-predictions.png')
subprocess.call('open zebra-predictions.png', shell=True)
