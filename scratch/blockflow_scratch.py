import sys
import os
import subprocess
import numpy as np
import vigra
from networkx.drawing import nx_pydot

from blockflow import global_graph, ReadArray, GaussianSmoothing, LaplacianOfGaussian, DifferenceOfGaussians, \
    GaussianGradientMagnitude, HessianOfGaussianEigenvalues, StructureTensorEigenvalues, \
    PixelFeatures

os.chdir(os.path.dirname(__file__))

#a = np.zeros((1,1,100,100,1), dtype=np.float32)
subprocess.call('open zebra.jpg', shell=True)
zebra = vigra.impex.readImage('zebra.jpg').withAxes('yxc').mean(axis=-1)
print zebra.shape

# Prepare functions
read_array = ReadArray()
gassian_smoothing = GaussianSmoothing()
difference_of_gaussians = DifferenceOfGaussians()
laplacian_of_gaussian = LaplacianOfGaussian()
ggm = GaussianGradientMagnitude()
hge = HessianOfGaussianEigenvalues()
ste = StructureTensorEigenvalues()

SCALE = 3.0
input_op = read_array(zebra[None, None, ..., None])
gs_op = gassian_smoothing(input_op, SCALE)
dog_op = difference_of_gaussians(input_op, SCALE)
log_op = laplacian_of_gaussian(input_op, SCALE)
ggm_op = ggm(input_op, SCALE)
hge_op = hge(input_op, SCALE)
ste_op = ste(input_op, SCALE)


box = [(0,0,0,0,0), (1,1,400,600,1)] # actual image is smaller than this, but that's okay.

# Construct call graph
with global_graph.register_calls():
    gs_op.dry_pull(box)
    dog_op.dry_pull(box)
    log_op.dry_pull(box)
    ggm_op.dry_pull(box)
    hge_op.dry_pull(box)
    ste_op.dry_pull(box)

# Visualize graph
dot_path = 'blockflow-graph.dot'
nx_pydot.write_dot(global_graph.dag, dot_path)
subprocess.call('dot -Tpng -o {}.png {}'.format(dot_path, dot_path), shell=True)
subprocess.call('open {}.png'.format(dot_path), shell=True)


def pull_and_show(op, name):
    # Compute DoG filter
    result = op.pull( box )
    print result.shape
    
    # Normalize to uint8 range
    result = (255*(result - result.min()) / (result.max() - result.min())).astype(np.uint8)

    if result.shape[-1] == 2:
        # Make RGB image from RG image, so PNG can be used.
        result = np.concatenate((result, np.zeros_like(result[...,0:1])), axis=-1)
    
    # Write filtered image and open
    result = vigra.taggedView(result, 'tzyxc')
    vigra.impex.writeImage(result[0,0,...], '{}-zebra.png'.format(name))
    subprocess.call('open {}-zebra.png'.format(name), shell=True)

pull_and_show(gs_op, 'GS')
pull_and_show(dog_op, 'DoG')
pull_and_show(log_op, 'LoG')
pull_and_show(ggm_op, 'GGM')
pull_and_show(hge_op, 'HGE')
pull_and_show(ste_op, 'STE')
