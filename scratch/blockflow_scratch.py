import sys
import os
import subprocess
import numpy as np
import vigra
from networkx.drawing import nx_pydot

from blockflow import ReadArray, DifferenceOfGaussians, global_graph


os.chdir(os.path.dirname(__file__))

#a = np.zeros((1,1,100,100,1), dtype=np.float32)
subprocess.call('open zebra.jpg', shell=True)
zebra = vigra.impex.readImage('zebra.jpg').withAxes('yxc').mean(axis=-1)
print zebra.shape

# Prepare functions
read_array = ReadArray()
difference_of_gaussians = DifferenceOfGaussians()

input_proxy = read_array(zebra[None, None, ..., None])
dog_proxy = difference_of_gaussians(input_proxy, 2.0, 0.66)

box = [(0,0,0,0,0), (1,1,400,600,1)] # actual image is smaller than this, but that's okay.

# Construct call graph
with global_graph.register_calls():
    dog_proxy.dry_run(box)

# Visualize graph
dot_path = 'blockflow-graph.dot'
nx_pydot.write_dot(global_graph.dag, dot_path)
subprocess.call('dot -Tpng -o {}.png {}'.format(dot_path, dot_path), shell=True)
subprocess.call('open {}.png'.format(dot_path), shell=True)

# Compute DoG filter
result = dog_proxy.pull( box )
print result.shape

# Write filtered image and open
result = vigra.taggedView(result, 'tzyxc')
vigra.impex.writeImage(result[0,0,...,0], 'filtered-zebra.png')
subprocess.call('open filtered-zebra.png', shell=True)
