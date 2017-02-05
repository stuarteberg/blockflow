from functools import wraps

import numpy as np
import networkx as nx
import vigra
import collections
from contextlib import contextmanager

from .box import Box

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

class DryArray(BlockflowArray):
    def __new__(cls, shape=(), dtype=float, buffer=None, offset=0, strides=None, order=None, box=None):
        assert shape == () or np.prod(shape) == 0, "DryArray must have empty shape"
        obj = BlockflowArray.__new__(cls, shape, dtype, buffer, offset, strides, order, box=box)
        return obj


@contextmanager
def readonly_array(a):
    a = np.asanyarray(a)
    writeable = a.flags['WRITEABLE']
    a.flags['WRITEABLE'] = False
    yield a
    a.flags['WRITEABLE'] = writeable



class Operator(object):
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
    
    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        assert 'req_box' not in kwargs, \
            "The req_box should not be passed to operators explicitly.  Use foo.pull(box)"
        return self

    def dry_pull(self, box):
        with readonly_array(box) as box:
            assert box.ndim == 2 and box.shape[0] == 2 and box.shape[1] <= 5
            kwargs = {'req_box': box}
            kwargs.update(self.kwargs)

            if global_graph.mode == 'registration_dry_run':
                global_graph.dag.add_node(self)
                if global_graph.op_callstack:
                    caller = global_graph.op_callstack[-1]
                    global_graph.dag.add_edge(self, caller)
    
                global_graph.op_callstack.append(self)
                try:
                    return self.dry_execute(*self.args, **kwargs)
                finally:
                    global_graph.op_callstack.pop()

    def pull(self, box):
        with readonly_array(box) as box:
            assert box.ndim == 2 and box.shape[0] == 2 and box.shape[1] <= 5
            kwargs = {'req_box': box}
            kwargs.update(self.kwargs)
            result_data = self.execute(*self.args, **kwargs)
            assert isinstance(result_data, BlockflowArray)
            assert result_data.box is not None
            return result_data

    def dry_execute(self, *args, **kwargs):
        raise NotImplementedError()

    def execute(self, *args, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        return self.name


class ReadArray(Operator):
    
    def dry_execute(self, arr, req_box):
        return DryArray(box=self._clip_box(arr, req_box))
    
    def execute(self, arr, req_box=None):
        clipped_box = self._clip_box(arr, req_box)
        result = arr[clipped_box.slicing()].view(BlockflowArray)
        result.box = clipped_box
        return result

    def _clip_box(self, arr, req_box):
        full_array_box = Box.from_shape(arr.shape)
        valid_box = full_array_box.intersection(req_box)
        return valid_box
        
    
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
    
    def num_channels_for_input_box(self, box):
        # Default implementation: One output channel per input channel,
        # regardless of dimensions
        return box[1,'c'] - box[0,'c']

    def num_channels_for_input_box_vector_valued(self, box):
        """
        For vector-valued filters whose output channels is N*C
        """
        shape_zyx = box[1,'zyx'] - box[0,'zyx']
        ndim = (shape_zyx > 1).sum()
        channels = box.to_shape()[-1]
        return ndim*channels
    
    def dry_execute(self, input_op, scale, req_box):
        upstream_req_box = self._get_upstream_box(scale, req_box)
        empty_data = input_op.dry_pull(upstream_req_box)
        n_channels = self.num_channels_for_input_box(empty_data.box)
        box = empty_data.box.copy()
        box[:,-1] = (0, n_channels)
        box = box.intersection(req_box)
        return DryArray(box=box)

    def execute(self, input_op, scale, req_box=None):
        # Ask for the fully padded input
        upstream_req_box = self._get_upstream_box(scale, req_box)
        input_data = input_op.pull(upstream_req_box)
    
        # The result is tagged with a box.
        # If we asked for too much (wider than the actual image),
        # then this box won't match what we requested.
        upstream_actual_box = input_data.box
        result_box, req_box_within_upstream = upstream_actual_box.intersection(req_box, True)
    
        filtered = self.filter_func_5d(input_data, scale, req_box_within_upstream)
        filtered = filtered.view(BlockflowArray)
        filtered.box = result_box
        return filtered

    def _get_upstream_box(self, sigma, req_box):
        padding = np.ceil(np.array(sigma)*self.WINDOW_SIZE).astype(np.int64)
        upstream_req_box = req_box.copy()
        upstream_req_box[0, 'zyx'] -= padding
        upstream_req_box[1, 'zyx'] += padding
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
    num_channels_for_input_box = ConvolutionalFilter.num_channels_for_input_box_vector_valued
    def filter_func(self, input_data, scale, box):
        return vigra.filters.hessianOfGaussianEigenvalues(input_data, scale=scale, window_size=self.WINDOW_SIZE, roi=box[:,:-1].tolist())


class StructureTensorEigenvalues(ConvolutionalFilter):
    num_channels_for_input_box = ConvolutionalFilter.num_channels_for_input_box_vector_valued
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

    def dry_execute(self, input_op, scale, req_box):
        empty_1 = self.gaussian_1(input_op, scale).dry_pull(req_box)
        empty_2 = self.gaussian_2(input_op, scale*0.66).dry_pull(req_box)
        assert (empty_1.box == empty_2.box).all()
        return empty_1

    def execute(self, input_op, scale, req_box=None):
        a = self.gaussian_1(input_op, scale).pull(req_box)
        b = self.gaussian_2(input_op, scale*0.66).pull(req_box)
        
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
    
    def dry_execute(self, input_op, filter_specs, req_box):
        n_channels = 0
        for spec in filter_specs:
            feature_op = self._get_filter_op(spec)
            empty = feature_op(input_op, spec.scale).dry_pull(req_box)
            n_channels += empty.box[1, 'c']

        box = empty.box.copy()
        box[:,-1] = (0, n_channels) 
        
        # Restrict to requested channels
        box = box.intersection(req_box)
        return DryArray(box=box)

    def execute(self, input_op, filter_specs, req_box=None):
        # FIXME: This requests all channels, no matter what.
        results = []
        for spec in filter_specs:
            feature_op = self._get_filter_op(spec)
            feature_data = feature_op(input_op, spec.scale).pull(req_box)
            results.append(feature_data)

        stacked_data = np.concatenate(results, axis=-1)
        
        # Select only the requested channels
        stacked_data = stacked_data[..., slice(*req_box[:,-1])]

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
    def dry_execute(self, features_op, classifier, req_box):
        upstream_box = req_box.copy()
        upstream_box[:,-1] = (Box.MIN,Box.MAX) # Request all features
        empty_feats = features_op.dry_pull(upstream_box)

        out_box = empty_feats.box.copy()
        out_box[:,-1] = (0, len(classifier.known_classes))
        out_box = out_box.intersection(req_box)
        return DryArray(dtype=np.float32, box=out_box)
    
    def execute(self, features_op, classifier, req_box):
        upstream_box = req_box.copy()
        upstream_box[:,-1] = (Box.MIN,Box.MAX) # Request all features
        feature_vol = features_op.pull(upstream_box)
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

    @contextmanager
    def register_calls(self):
        assert len(self.op_callstack) == 0
        self.mode = 'registration_dry_run'
        yield
        assert len(self.op_callstack) == 0
        self.mode = 'executable'

global_graph = Graph()
