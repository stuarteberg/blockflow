from functools import wraps
from itertools import starmap

import numpy as np
import vigra


class BlockflowArray(np.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, box=None):
        obj = np.ndarray.__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.box = box
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        orig_box = getattr(obj, 'box', None)
        self.box = orig_box
        
        # We're creating a new array using an existing array as a template, but if the array was generated
        # via a broadcasting ufunc, then the box might not be copied from the correct array.
        # If it's wrong, just remove the box attribute.
        #
        # FIXME: We might be able to handle cases like this automatically 
        #        via __array_wrap__() or __array_prepare__()
        if orig_box is not None:
            if tuple(orig_box[1] - orig_box[0]) == self.shape:
                self.box = orig_box

def slicing(box):
    return tuple( starmap( slice, zip(box[0], box[1]) ) )

def shape_to_box(shape):
    return np.array( ((0,)*len(shape), shape) )

def box_intersection(box_a, box_b):
    intersection = np.array(box_a)
    intersection[0] = np.maximum( box_a[0], box_b[0] )
    intersection[1] = np.minimum( box_a[1], box_b[1] )

    intersection_within_a = intersection - box_a[0]
    intersection_within_b = intersection - box_b[0]
    intersection_global = intersection

    return intersection_global, intersection_within_a, intersection_within_b
    

class Proxy(object):
    
    def __init__(self, dry_run_func, execute_func, *args, **kwargs):
        self.dry_run_func = dry_run_func
        self.execute_func = execute_func
        self.args = args
        self.kwargs = kwargs
        assert 'result_box' not in kwargs, "The result_box should not be passed explicitly.  Use foo.pull(box)"

    def dry_run(self, *args, **kwargs):
        assert 'result_box' in kwargs, 'result_box must be passed as a keyword arg to dry_run()'
        self.dry_run_func(*args, **kwargs)
    
    def pull(self, box):
        box = np.asarray(box)
        assert box.ndim == 2 and box.shape[0] == 2 and box.shape[1] <= 5
        kwargs = {'result_box': box}
        kwargs.update(self.kwargs)
        result_data = self.execute_func(*self.args, **kwargs)
        assert isinstance(result_data, BlockflowArray)
        assert result_data.box is not None
        return result_data

def dry_run_pointwise_1(input_proxy, *args, **kwargs):
    return input_proxy.dry_run(*args, **kwargs)

def proxy(dry_run_func):
    def decorator(execute_func):
        @wraps(execute_func)
        def wrapper(*args, **kwargs):
            return Proxy(dry_run_func, execute_func, *args, **kwargs)
        return wrapper
    return decorator
    
WINDOW_SIZE = 2.0

@proxy(lambda arr, result_box: None)
def read_array(arr, result_box=None):
    full_array_box = shape_to_box(arr.shape)
    valid_box, _, _ = box_intersection( full_array_box, result_box )
    result = arr[slicing(valid_box)].view(BlockflowArray)
    result.box = valid_box
    return result

def vigra_filter_5d(filter_func, input_data, sigma, window_size, box_5d):
    """
    Utility function.
    Given a 5D array (tzyxc), and corresponding output box,
    compute the given filter over the spatial dimensions.
    
    (It doesn't suffice to simply drop the 't' axis and run the filter,
    because singleton spatial dimensions would cause trouble.)
    """
    # FIXME: Loop over time slices
    input_data = vigra.taggedView(input_data, 'tzyxc')
    assert box_5d.shape == (2,5)

    nonsingleton_axes = (np.array(input_data.shape) != 1).nonzero()[0]
    box_3d = box_5d[:, nonsingleton_axes]
    
    squeezed_input = input_data.squeeze()
    result = filter_func(squeezed_input, sigma, roi=box_3d.tolist(), window_size=window_size)
    result = result.withAxes(*'tzyxc')
    return result

@proxy(dry_run_pointwise_1)
def gaussian( input_proxy, sigma, result_box=None ):
    padding = np.ceil(np.array(sigma)*WINDOW_SIZE).astype(np.int64)

    # Ask for the fully padded input
    requested_box = result_box.copy()
    requested_box[0, 1:4] -= padding
    requested_box[1, 1:4] += padding
    input_data = input_proxy.pull(requested_box)

    # The result is tagged with a box.
    # If we asked for too much (wider than the actual image),
    # then this box won't match what we requested.
    retrieved_box = input_data.box
    intersected, result_box_within_retrieved, _ = box_intersection(retrieved_box, result_box)

    filtered = vigra_filter_5d(vigra.filters.gaussianSmoothing, input_data, sigma, WINDOW_SIZE, result_box_within_retrieved)
    filtered = filtered.view(BlockflowArray)
    filtered.box = intersected
    return filtered

@proxy(dry_run_pointwise_1)
def difference_of_gaussians(input_proxy, sigma_1, sigma_2, result_box=None):
    a = gaussian(input_proxy, sigma_1).pull(result_box)
    b = gaussian(input_proxy, sigma_2).pull(result_box)
    
    # For pointwise numpy ufuncs, the result is already cast as 
    # a BlockflowArray, with the box already initialized.
    # Nothing extra needed here.
    return a - b 

