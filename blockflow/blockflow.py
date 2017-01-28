from functools import wraps
from itertools import starmap

import numpy as np
import vigra


class BlockflowArray(np.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, roi=None):
        obj = np.ndarray.__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.roi = roi
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        orig_roi = getattr(obj, 'roi', None)
        self.roi = orig_roi
        
        # We're creating a new array using an existing array as a template, but if the array was generated
        # via a broadcasting ufunc, then the roi might not be copied from the correct array.
        # If it's wrong, just remove the roi attribute.
        #
        # FIXME: We might be able to handle cases like this automatically 
        #        via __array_wrap__() or __array_prepare__()
        if orig_roi is not None:
            if tuple(orig_roi[1] - orig_roi[0]) == self.shape:
                self.roi = orig_roi

def slicing(roi):
    return tuple( starmap( slice, zip(roi[0], roi[1]) ) )

def shape_to_roi(shape):
    return np.array( ((0,)*len(shape), shape) )

def roi_intersection(roi_a, roi_b):
    intersection = np.array(roi_a)
    intersection[0] = np.maximum( roi_a[0], roi_b[0] )
    intersection[1] = np.minimum( roi_a[1], roi_b[1] )

    intersection_within_a = intersection - roi_a[0]
    intersection_within_b = intersection - roi_b[0]
    intersection_global = intersection

    return intersection_global, intersection_within_a, intersection_within_b
    

class Proxy(object):
    
    def __init__(self, dry_run_func, execute_func, *args, **kwargs):
        self.dry_run_func = dry_run_func
        self.execute_func = execute_func
        self.args = args
        self.kwargs = kwargs
        assert 'result_roi' not in kwargs, "The result_roi should not be passed explicitly.  Use foo.pull(roi)"

    def dry_run(self, *args, **kwargs):
        assert 'result_roi' in kwargs, 'result_roi must be passed as a keyword arg to dry_run()'
        self.dry_run_func(*args, **kwargs)
    
    def pull(self, roi):
        roi = np.asarray(roi)
        assert roi.ndim == 2 and roi.shape[0] == 2 and roi.shape[1] <= 5
        kwargs = {'result_roi': roi}
        kwargs.update(self.kwargs)
        result_data = self.execute_func(*self.args, **kwargs)
        assert isinstance(result_data, BlockflowArray)
        assert result_data.roi is not None
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

@proxy(lambda arr, result_roi: None)
def read_array(arr, result_roi=None):
    full_array_roi = shape_to_roi(arr.shape)
    valid_roi, _, _ = roi_intersection( full_array_roi, result_roi )
    result = arr[slicing(valid_roi)].view(BlockflowArray)
    result.roi = valid_roi
    return result

def vigra_filter_5d(filter_func, input_data, sigma, window_size, roi_5d):
    """
    Utility function.
    Given a 5D array (tzyxc), and corresponding output roi,
    compute the given filter over the spatial dimensions.
    
    (It doesn't suffice to simply drop the 't' axis and run the filter,
    because singleton spatial dimensions would cause trouble.)
    """
    # FIXME: Loop over time slices
    input_data = vigra.taggedView(input_data, 'tzyxc')
    assert roi_5d.shape == (2,5)

    nonsingleton_axes = (np.array(input_data.shape) != 1).nonzero()[0]
    roi_3d = roi_5d[:, nonsingleton_axes]
    
    squeezed_input = input_data.squeeze()
    result = filter_func(squeezed_input, sigma, roi=roi_3d.tolist(), window_size=window_size)
    result = result.withAxes(*'tzyxc')
    return result

@proxy(dry_run_pointwise_1)
def gaussian( input_proxy, sigma, result_roi=None ):
    padding = np.ceil(np.array(sigma)*WINDOW_SIZE).astype(np.int64)

    # Ask for the fully padded input
    requested_roi = result_roi.copy()
    requested_roi[0, 1:4] -= padding
    requested_roi[1, 1:4] += padding
    input_data = input_proxy.pull(requested_roi)

    # The result is tagged with a roi.
    # If we asked for too much (wider than the actual image),
    # then this roi won't match what we requested.
    retrieved_roi = input_data.roi
    intersected, result_roi_within_retrieved, _ = roi_intersection(retrieved_roi, result_roi)

    filtered = vigra_filter_5d(vigra.filters.gaussianSmoothing, input_data, sigma, WINDOW_SIZE, result_roi_within_retrieved)
    filtered = filtered.view(BlockflowArray)
    filtered.roi = intersected
    return filtered

@proxy(dry_run_pointwise_1)
def difference_of_gaussians(input_proxy, sigma_1, sigma_2, result_roi=None):
    a = gaussian(input_proxy, sigma_1).pull(result_roi)
    b = gaussian(input_proxy, sigma_2).pull(result_roi)
    
    # For pointwise numpy ufuncs, the result is already cast as 
    # a BlockflowArray, with the roi already initialized.
    # Nothing extra needed here.
    return a - b 

