import sys
from itertools import starmap
import numpy as np

class Box(np.ndarray):
    """
    A special-purpose subclass of numpy.ndarray, for manipulating box-shaped ROIs.
    
    Can be used as a dict key.

    Note: When a Box is first added to a set or dictionary (or anything else that calls __hash__),
          it is marked read-only, and attempts to modify the Box afterwards will raise an error from numpy:
          "ValueError: assignment destination is read-only"
    
    As a convenience, the second axis can be can optionally be indexed with strings instead of ints.
    
        # Equivalent to: box[0,1] = z_start
        box[0,'z'] = z_start
        
        # Equivalent to: box[1,1:4] = (z_stop, y_stop, x_stop)
        box[1,'zyx'] = (z_stop, y_stop, x_stop)
    """
    ORDER = 'tzyxc'
    SHAPE = (2, len(ORDER))
    DTYPE = np.int64
    MIN = np.iinfo(DTYPE).min
    MAX = np.iinfo(DTYPE).max
    
    DefaultBox = np.array([[MIN, MIN, MIN, MIN, MIN],
                           [MAX, MAX, MAX, MAX, MAX]])
    

    def intersection(self, other, return_relative=False):
        """
        Compute the intersection of this Box with another Box.

        return_relative:
            If False, just return the intersection in global coordinates.
            If True, also return the intersection in relative coordinates
            (relative to this Box's start).
        
        Returns:
            Either global_intersection or (global_intersection, relative_intersection),
            depending on return_relative
        """
        global_intersection = Box()
        global_intersection[0] = np.maximum( self[0], other[0] )
        global_intersection[1] = np.minimum( self[1], other[1] )
    
        if return_relative:
            relative_box = global_intersection - self[0]
            return global_intersection, relative_box
        else:
            return global_intersection


    def slicing(self):
        """
        Convert the box to a slicing that can be used with ndarray.__getitem__()
        """
        return tuple( starmap( slice, zip(*self) ) )


    @classmethod
    def from_shape(cls, shape):
        assert len(shape) == Box.SHAPE[1], \
            "Can't create Box: shape is the wrong length: '{}'".format(shape)
        box = Box()
        box[0] = 0
        box[1] = shape
        return box

    def to_shape(self):
        return tuple(self[1] - self[0])

    def is_valid(self):
        """
        A Box is invalid if any of it's side lengths are 0 or negative,
        and thus the box does not describe an array slicing.
        (Note that by default, boxes are created in a VALID state.)
        """
        return (self[1] > self[0]).all()


    def __new__(cls, copy_from=DefaultBox):
        """
        Constructor.
        Creates a box from a copy of the given ndarray
        if given, or a default box otherwise.
        """
        obj = np.ndarray.__new__(cls, Box.SHAPE, Box.DTYPE)
        copy_from = np.asanyarray(copy_from)
        assert copy_from.dtype == Box.DTYPE, "Box requires {} dtype".format(Box.DTYPE.__name__)
        assert copy_from.shape == Box.SHAPE, "Box must have shape {}".format(Box.SHAPE)
        obj.flat[:] = copy_from.flat[:]
        return obj    


    def __array_prepare__(self, out_arr, context=None):
        # If the ufunc would return an array that can't be considered a box,
        # Then return a plain ndarray instead
        # Example:
        #     shape = np.diff(box, axis=0)[0]
        #     assert type(shape) == np.ndarray
        if out_arr.shape != Box.SHAPE or out_arr.dtype != Box.DTYPE:
            out_arr = out_arr.view(np.ndarray)
            return np.ndarray.__array_prepare__(self, out_arr, context).view(np.ndarray)
        return np.ndarray.__array_prepare__(self, out_arr, context)


    def __array_wrap__(self, out_arr, context=None):
        # If the ufunc would return an array that can't be considered a box,
        # Then return a plain ndarray instead
        # Example:
        #     shape = np.diff(box, axis=0)[0]
        #     assert type(shape) == np.ndarray
        if out_arr.shape != Box.SHAPE or out_arr.dtype != Box.DTYPE:
            out_arr = out_arr.view(np.ndarray)
            return np.ndarray.__array_wrap__(self, out_arr, context).view(np.ndarray)
        return np.ndarray.__array_wrap__(self, out_arr, context)


    def __array_finalize__(self, obj):
        # Called in multiple contexts, including a.view(Box)
        if obj is None:
            return
        # This prevents bad behavior, but also forbids some valid operations, e.g. box.transpose()
        # For cases like that, you may need a workaround, e.g. np.array(box).transpose()
        assert obj.shape == Box.SHAPE, "Can't view as Box. Bad shape: {}".format(obj.shape)
        assert obj.dtype == Box.DTYPE, "Can't view as Box. Bad dtype: {}".format(obj.dtype)


    def __hash__(self):
        assert self.base is None, \
            "This box was created from a view and therefore cannot be hashed safely. "\
            "To add it to a set/dict, copy it first."
        
        # Once a box is hashed for the first time, we assume it might be used in a set or dict key.
        # Therefore, we forbid writing to it, which would invalidate the hash.
        # Note that copying a Box (or any ndarray) resets the 'WRITEABLE' flag to True in the copy.
        self.flags['WRITEABLE'] = False
        return hash(bytes(np.getbuffer(self.ravel())))

    
    def __eq__(self, other):
        """
        Override __eq__ so that Box can be stored in hashing containers.
        """
        return np.ndarray.__eq__(self, other).view(Box._ComparisonResult)


    def __ne__(self, other):
        return np.ndarray.__ne__(self, other).view(Box._ComparisonResult)

    
    def __getitem__(self, slicing):
        if isinstance(slicing, tuple):
            if len(slicing) > 2:
                # FIXME: This check forbids a few valid cases, e.g. box[None,:,'t'],
                #        but that seems rare and there are workarounds, e.g. box[None][:,0]
                raise IndexError("Too many indexes: {}".format(slicing))
            if len(slicing) == 2 and isinstance(slicing[1], str):
                slicing = self._convert_slicing_args(slicing)
        result = np.ndarray.__getitem__(self, slicing)
        
        # By definition, Boxes only come in one size.
        # If the selection isn't box-shaped, then return a plain ndarray, not a Box
        if result.shape != Box.SHAPE:
            return result.view(np.ndarray)
        return result


    def __setitem__(self, slicing, value):
        if isinstance(slicing, tuple):
            if len(slicing) > 2:
                raise IndexError("Too many indexes: {}".format(slicing))
            if len(slicing) == 2 and isinstance(slicing[1], str):
                slicing = self._convert_slicing_args(slicing)
        np.ndarray.__setitem__(self, slicing, value)

    
    def _convert_slicing_args(self, args):
        """
        Helper for __getitem__ and __setitem__ to support syntax like this:
        
        box[:, 'c'] = (c_start, c_stop)
        box[1, 'zyx'] = (z_stop, x_stop, y_stop)
        """
        first, second = args
        if isinstance(second, str):
            try:
                if len(second) == 1:
                    # E.g. replace 't' with 0
                    second = Box.ORDER.index(second)
                else:
                    # E.g. replace 'zyx' with slice(1,4)
                    start = Box.ORDER.index(second)
                    second = slice(start, start + len(second))
            except:
                raise IndexError('"{}" is not a valid string to slice into a Box. '
                                 'Must be a substring of "{}"'.format(second, Box.ORDER))
        return (first, second)


    def __str__(self):
        starts = map(Box._coord_to_str, self[0])
        stops  = map(Box._coord_to_str, self[1])
        start_lens = map(len, starts)
        stop_lens  = map(len, stops)
        col_widths = map(max, zip(start_lens, stop_lens))

        starts = starmap(lambda s,l,w: ' '*(w-l) + s, zip(starts, start_lens, col_widths))
        stops  = starmap(lambda s,l,w: ' '*(w-l) + s, zip(stops,  stop_lens,  col_widths))
        return '[[{}],\n [{}]]'.format(', '.join(starts), ', '.join(stops))


    def __repr__(self):
        start_line, stop_line = str(self).split('\n')
        return 'Box({}\n    {})'.format(start_line, stop_line)


    @staticmethod
    def _coord_to_str(c):
        if c == Box.MIN:
            return 'Box.MIN'
        if c == Box.MAX:
            return 'Box.MAX'
        return str(c)


    class _ComparisonResult(np.ndarray):
        """
        Returned by Box.__eq__, to support convenient usage like this:
        
        if box1 == box2:
            print("Yes, they're equal")
        
        without hitting this familiar numpy complaint:
        
            ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        
        This is essential to allow us to use Box instances in sets and dicts.
        """
        if sys.version_info.major == 2:
            def __nonzero__(self):
                return bool(self.all())
        else:
            def __bool__(self):
                return bool(self.all())
