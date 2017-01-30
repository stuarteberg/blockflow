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
input_proxy = read_array(zebra[None, None, ..., None])
gs_proxy = gassian_smoothing(input_proxy, SCALE)
dog_proxy = difference_of_gaussians(input_proxy, SCALE)
log_proxy = laplacian_of_gaussian(input_proxy, SCALE)
ggm_proxy = ggm(input_proxy, SCALE)
hge_proxy = hge(input_proxy, SCALE)
ste_proxy = ste(input_proxy, SCALE)


box = [(0,0,0,0,0), (1,1,400,600,1)] # actual image is smaller than this, but that's okay.

# Construct call graph
with global_graph.register_calls():
    gs_proxy.dry_pull(box)
    dog_proxy.dry_pull(box)
    log_proxy.dry_pull(box)
    ggm_proxy.dry_pull(box)
    hge_proxy.dry_pull(box)
    ste_proxy.dry_pull(box)

# Visualize graph
dot_path = 'blockflow-graph.dot'
nx_pydot.write_dot(global_graph.dag, dot_path)
subprocess.call('dot -Tpng -o {}.png {}'.format(dot_path, dot_path), shell=True)
subprocess.call('open {}.png'.format(dot_path), shell=True)


def pull_and_show(proxy, name):
    # Compute DoG filter
    result = proxy.pull( box )
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

pull_and_show(gs_proxy, 'GS')
pull_and_show(dog_proxy, 'DoG')
pull_and_show(log_proxy, 'LoG')
pull_and_show(ggm_proxy, 'GGM')
pull_and_show(hge_proxy, 'HGE')
pull_and_show(ste_proxy, 'STE')
