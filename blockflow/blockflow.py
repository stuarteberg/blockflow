from itertools import starmap
from functools import wraps

import numpy as np
import networkx as nx
import vigra
import contextlib
import collections

BOX_MIN = np.iinfo(np.int64).min
BOX_MAX = np.iinfo(np.int64).max

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
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
    
    def __call__(self, *args, **kwargs):
        return Proxy(self, *args, **kwargs)

    def dry_execute(self, *args, **kwargs):
        raise NotImplementedError()

    def execute(self, *args, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        return self.name

class Proxy(object):
    
    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs
        assert 'req_box' not in kwargs, \
            "The req_box should not be passed to operators explicitly.  Use foo.pull(box)"

    def dry_pull(self, req_box):
        if global_graph.mode == 'registration_dry_run':
            global_graph.dag.add_node(self.op)
            if global_graph.op_callstack:
                caller = global_graph.op_callstack[-1]
                global_graph.dag.add_edge(self.op, caller)
                
            global_graph.op_callstack.append(self.op)
            try:
                box = np.asarray(req_box)
                assert box.ndim == 2 and box.shape[0] == 2 and box.shape[1] <= 5
                kwargs = {'req_box': box}
                kwargs.update(self.kwargs)
                self.op.dry_execute(*self.args, **kwargs)
            finally:
                global_graph.op_callstack.pop()
    
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
    
    def dry_execute(self, arr, req_box):
        pass
    
    def execute(self, arr, req_box=None):
        full_array_box = shape_to_box(arr.shape)
        valid_box, _, _ = box_intersection( full_array_box, req_box )
        result = arr[slicing(valid_box)].view(BlockflowArray)
        result.box = valid_box
        return result
    
def wrap_filter_5d(filter_func):
    """
    Decorator.
    
    Given a 5D array (tzyxc), and corresponding output box,
    compute the given filter over the spatial dimensions.
    
    (It doesn't suffice to simply drop the 't' axis and run the filter,
    because singleton spatial dimensions would cause trouble.)
    """
    @wraps(filter_func)
    def wrapper(input_data, scale, box_5d):
        input_data = vigra.taggedView(input_data, 'tzyxc')
        assert box_5d.shape == (2,5)
        assert box_5d[1,0] - box_5d[0,0] == 1, \
            "FIXME: Can't handle multiple time slices yet.  (Add a loop to this function.)"

        # Find the non-singleton axes, so we can keep only them
        # but also keep channel, no matter what    
        input_shape_nochannel = np.array(input_data.shape[:-1])
        nonsingleton_axes = (input_shape_nochannel != 1).nonzero()[0]
        nonsingleton_axes = tuple(nonsingleton_axes) + (4,) # Keep channel
        box = box_5d[:, nonsingleton_axes] # Might be a 2D OR 3D box

        # Squeeze, but keep channel                
        squeezed_input = input_data.squeeze()
        if 'c' not in squeezed_input.axistags.keys():
            squeezed_input = squeezed_input.insertChannelAxis(-1)

        result = filter_func(squeezed_input, scale, box=box)
        result = result.withAxes(*'tzyxc')
        return result
    return wrapper


class ConvolutionalFilter(Operator):

    WINDOW_SIZE = 2.0 # Subclasses may override this
    
    def __init__(self, name=None):
        super(ConvolutionalFilter, self).__init__(name)
        self.filter_func_5d = wrap_filter_5d(self.filter_func)
    
    def filter_func(self, input_data, scale, box):
        """
        input_data: array data whose axes are one of the following: zyxc, yxc, zxc, zyc
        scale: filter scale (sigma)
        box: Not 5D.  Either 4D or 3D, depending on the dimensionality of input_data
        """
        raise NotImplementedError("Convolutional Filter '{}' must override filter_func()"
                                  .format(self.__class__.__name__))
    
    def dry_execute(self, input_proxy, scale, req_box):
        upstream_req_box = self._get_upstream_box(scale, req_box)
        input_proxy.dry_pull(upstream_req_box)

    def execute(self, input_proxy, scale, req_box=None):
        # Ask for the fully padded input
        upstream_req_box = self._get_upstream_box(scale, req_box)
        input_data = input_proxy.pull(upstream_req_box)
    
        # The result is tagged with a box.
        # If we asked for too much (wider than the actual image),
        # then this box won't match what we requested.
        upstream_actual_box = input_data.box
        result_box, req_box_within_upstream, _ = box_intersection(upstream_actual_box, req_box)
    
        filtered = self.filter_func_5d(input_data, scale, req_box_within_upstream)
        filtered = filtered.view(BlockflowArray)
        filtered.box = result_box
        return filtered

    def _get_upstream_box(self, sigma, req_box):
        padding = np.ceil(np.array(sigma)*self.WINDOW_SIZE).astype(np.int64)
        upstream_req_box = req_box.copy()
        upstream_req_box[0, 1:4] -= padding
        upstream_req_box[1, 1:4] += padding
        return upstream_req_box
        
class GaussianSmoothing(ConvolutionalFilter):    
    def filter_func(self, input_data, scale, box):
        return vigra.filters.gaussianSmoothing(input_data, sigma=scale, window_size=self.WINDOW_SIZE, roi=box[:,:-1].tolist())

class LaplacianOfGaussian(ConvolutionalFilter):
    def filter_func(self, input_data, scale, box):
        return vigra.filters.laplacianOfGaussian(input_data, scale=scale, window_size=self.WINDOW_SIZE, roi=box[:,:-1].tolist())

class GaussianGradientMagnitude(ConvolutionalFilter):
    def filter_func(self, input_data, scale, box):
        return vigra.filters.gaussianGradientMagnitude(input_data, sigma=scale, window_size=self.WINDOW_SIZE, roi=box[:,:-1].tolist())

class HessianOfGaussianEigenvalues(ConvolutionalFilter):
    def filter_func(self, input_data, scale, box):
        return vigra.filters.hessianOfGaussianEigenvalues(input_data, scale=scale, window_size=self.WINDOW_SIZE, roi=box[:,:-1].tolist())

class StructureTensorEigenvalues(ConvolutionalFilter):
    def filter_func(self, input_data, scale, box):
        inner_scale = scale
        outer_scale = scale / 2.0
    
        return vigra.filters.structureTensorEigenvalues(input_data,
                                                        innerScale=inner_scale,
                                                        outerScale=outer_scale,
                                                        window_size=self.WINDOW_SIZE,
                                                        roi=box[:,:-1].tolist())

class DifferenceOfGaussians(ConvolutionalFilter):
    def filter_func(self, input_data, scale, box):
        sigma_1 = scale
        sigma_2 = 0.66*scale
    
        smoothed_1 = vigra.filters.gaussianSmoothing(input_data, sigma=sigma_1, window_size=self.WINDOW_SIZE, roi=box[:,:-1].tolist())
        smoothed_2 = vigra.filters.gaussianSmoothing(input_data, sigma=sigma_2, window_size=self.WINDOW_SIZE, roi=box[:,:-1].tolist())
        
        # In-place subtraction
        np.subtract( smoothed_1, smoothed_2, out=smoothed_1 )
        return smoothed_1

class DifferenceOfGaussiansComposite(Operator):
    """
    Alternative implementation of DifferenceOfGaussians,
    but using internal operators for the two smoothing operations.
    """

    def __init__(self, name=None):
        super(DifferenceOfGaussiansComposite, self).__init__(name)
        self.gaussian_1 = GaussianSmoothing('Gaussian-1')
        self.gaussian_2 = GaussianSmoothing('Gaussian-2')

    def dry_execute(self, input_proxy, scale, req_box):
        self.gaussian_1(input_proxy, scale).dry_pull(req_box)
        self.gaussian_2(input_proxy, scale*0.66).dry_pull(req_box)

    def execute(self, input_proxy, scale, req_box=None):
        a = self.gaussian_1(input_proxy, scale).pull(req_box)
        b = self.gaussian_2(input_proxy, scale*0.66).pull(req_box)
        
        # For pointwise numpy ufuncs, the result is already cast as 
        # a BlockflowArray, with the box already initialized.
        # Nothing extra needed here.
        return a - b 

FilterSpec = collections.namedtuple( 'FilterSpec', 'name scale' )
FilterNames = { 'GaussianSmoothing': GaussianSmoothing,
                'LaplacianOfGaussian': LaplacianOfGaussian,
                'GaussianGradientMagnitude': GaussianGradientMagnitude,
                'DifferenceOfGaussians': DifferenceOfGaussians,
                #'DifferenceOfGaussians': DifferenceOfGaussiansComposite,
                'HessianOfGaussianEigenvalues': HessianOfGaussianEigenvalues,
                'StructureTensorEigenvalues': StructureTensorEigenvalues }

class PixelFeatures(Operator):
    def __init__(self, name=None):
        Operator.__init__(self, name)
        self.feature_ops = {} # (name, scale) : op
    
    def dry_execute(self, input_proxy, filter_specs, req_box):
        for spec in filter_specs:
            feature_op = self._get_filter_op(spec)
            feature_op(input_proxy, spec.scale).dry_pull(req_box)

    def execute(self, input_proxy, filter_specs, req_box=None):
        # FIXME: This requests all channels, no matter what.
        results = []
        for spec in filter_specs:
            feature_op = self._get_filter_op(spec)
            feature_data = feature_op(input_proxy, spec.scale).pull(req_box)
            results.append(feature_data)

        stacked_data = np.concatenate(results, axis=-1)
        stacked_data = stacked_data.view(BlockflowArray)
        stacked_data.box = feature_data.box
        stacked_data.box[:,-1] = req_box[:,-1]
        return stacked_data

    def _get_filter_op(self, spec):
        try:
            feature_op = self.feature_ops[spec]
        except KeyError:
            feature_op = self.feature_ops[spec] = FilterNames[spec.name]()
        return feature_op

class PredictPixels(Operator):
    def dry_execute(self, features_proxy, classifier, req_box):
        upstream_box = req_box.copy()
        upstream_box[:,-1] = (BOX_MIN,BOX_MAX) # Request all features
        features_proxy.dry_pull(upstream_box)
    
    def execute(self, features_proxy, classifier, req_box):
        upstream_box = req_box.copy()
        upstream_box[:,-1] = (BOX_MIN,BOX_MAX) # Request all features
        feature_vol = features_proxy.pull(req_box)
        prod = np.prod(feature_vol.shape[:-1])
        feature_matrix = feature_vol.reshape((prod, feature_vol.shape[-1]))
        probabilities_matrix = classifier.predict_probabilities( feature_matrix )
        
        # TODO: Somehow check for correct number of channels, in case the classifier returned fewer classes than we expected
        #       (See lazyflow for example)        
        probabilities_vol = probabilities_matrix.reshape(feature_vol.shape[:-1] + (-1,))

        # Extract only the channel range that was originally requested
        ch_start, ch_stop = req_box[:,-1]
        probabilities_vol = probabilities_vol[..., ch_start:ch_stop]
        probabilities_vol = probabilities_vol.view(BlockflowArray)
        probabilities_vol.box = np.append(feature_vol.box[:,:-1], req_box[:,-1:], axis=1)
        return probabilities_vol

class Graph(object):
    MODES = ['uninitialized', 'registration_dry_run', 'block_flow_dry_run', 'executable']
    
    def __init__(self):
        self.op_callstack = []
        self.dag = nx.DiGraph()
        self.mode = 'uninitialized'

    @contextlib.contextmanager
    def register_calls(self):
        assert len(self.op_callstack) == 0
        self.mode = 'registration_dry_run'
        yield
        assert len(self.op_callstack) == 0
        self.mode = 'executable'

global_graph = Graph()
