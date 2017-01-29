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
    
class Operator(object):
    
    def __call__(self, *args, **kwargs):
        return Proxy(self, *args, **kwargs)

    def dry_run(self, *args, **kwargs):
        raise NotImplementedError()

    def execute(self, *args, **kwargs):
        raise NotImplementedError()

class Proxy(object):
    
    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs
        assert 'req_box' not in kwargs, \
            "The req_box should not be passed to operators explicitly.  Use foo.pull(box)"

    def dry_run(self, *args, **kwargs):
        assert 'req_box' in kwargs, \
            'req_box must be passed as a keyword arg to dry_run()'
        self.op.dry_run(*args, **kwargs)
    
    def pull(self, box):
        box = np.asarray(box)
        assert box.ndim == 2 and box.shape[0] == 2 and box.shape[1] <= 5
        kwargs = {'req_box': box}
        kwargs.update(self.kwargs)
        result_data = self.op.execute(*self.args, **kwargs)
        assert isinstance(result_data, BlockflowArray)
        assert result_data.box is not None
        return result_data


class ReadArray(Operator):
    
    def dry_run(self, arr, req_box):
        pass
    
    def execute(self, arr, req_box=None):
        full_array_box = shape_to_box(arr.shape)
        valid_box, _, _ = box_intersection( full_array_box, req_box )
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

class Gaussian(Operator):

    WINDOW_SIZE = 2.0
    
    def dry_run(self, input_proxy, sigma, req_box):
        upstream_req_box = self._get_upstream_box(sigma, req_box)
        input_proxy.dry_run(upstream_req_box)

    def execute(self, input_proxy, sigma, req_box=None):
        # Ask for the fully padded input
        upstream_req_box = self._get_upstream_box(sigma, req_box)
        input_data = input_proxy.pull(upstream_req_box)
    
        # The result is tagged with a box.
        # If we asked for too much (wider than the actual image),
        # then this box won't match what we requested.
        upstream_actual_box = input_data.box
        result_box, req_box_within_upstream, _ = box_intersection(upstream_actual_box, req_box)
    
        filtered = vigra_filter_5d(vigra.filters.gaussianSmoothing, input_data, sigma, self.WINDOW_SIZE, req_box_within_upstream)
        filtered = filtered.view(BlockflowArray)
        filtered.box = result_box
        return filtered

    def _get_upstream_box(self, sigma, req_box):
        padding = np.ceil(np.array(sigma)*self.WINDOW_SIZE).astype(np.int64)
        upstream_req_box = req_box.copy()
        upstream_req_box[0, 1:4] -= padding
        upstream_req_box[1, 1:4] += padding
        return upstream_req_box
        

class DifferenceOfGaussians(Operator):

    def __init__(self):
        self.gaussian_1 = Gaussian()
        self.gaussian_2 = Gaussian()

    def dry_run(self, input_proxy, sigma_1, sigma_2, req_box):
        self.gaussian_1(input_proxy, sigma_1).dry_run()
        self.gaussian_2(input_proxy, sigma_2).dry_run()

    def execute(self, input_proxy, sigma_1, sigma_2, req_box=None):
        a = self.gaussian_1(input_proxy, sigma_1).pull(req_box)
        b = self.gaussian_2(input_proxy, sigma_2).pull(req_box)
        
        # For pointwise numpy ufuncs, the result is already cast as 
        # a BlockflowArray, with the box already initialized.
        # Nothing extra needed here.
        return a - b 



