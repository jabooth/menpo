from __future__ import division
from functools import partial
import numpy as np

from menpo.base import MenpoNativeFunctionalityError

from ..base import winitfeature
try:
    from ..windowiterator import WindowIterator, WindowIteratorResult
except ImportError:
    raise MenpoNativeFunctionalityError()


@winitfeature
def hog(pixels, mode='dense', algorithm='dalaltriggs', num_bins=9,
        cell_size=8, block_size=2, signed_gradient=True, l2_norm_clip=0.2,
        window_height=1, window_width=1, window_unit='blocks',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding=True, verbose=False):
    r"""
    Extracts Histograms of Oriented Gradients (HOG) features from the input
    image.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    mode : {``dense``, ``sparse``}, optional
        The ``sparse`` case refers to the traditional usage of HOGs, so
        predefined parameters values are used.

        The ``sparse`` case of ``dalaltriggs`` algorithm sets
        ``window_height = window_width = block_size`` and
        ``window_step_horizontal = window_step_vertical = cell_size``.

        The ``sparse`` case of ``zhuramanan`` algorithm sets
        ``window_height = window_width = 3 * cell_size`` and
        ``window_step_horizontal = window_step_vertical = cell_size``.

        In the ``dense`` case, the user can choose values for `window_height`,
        `window_width`, `window_unit`, `window_step_vertical`,
        `window_step_horizontal`, `window_step_unit` and `padding` to customize
        the HOG calculation.
    window_height : `float`, optional
        Defines the height of the window. The metric unit is defined by
        `window_unit`.
    window_width : `float`, optional
        Defines the width of the window. The metric unit is defined by
        `window_unit`.
    window_unit : {``blocks``, ``pixels``}, optional
        Defines the metric unit of the `window_height` and `window_width`
        parameters.
    window_step_vertical : `float`, optional
        Defines the vertical step by which the window is moved, thus it
        controls the features' density. The metric unit is defined by
        `window_step_unit`.
    window_step_horizontal : `float`, optional
        Defines the horizontal step by which the window is moved, thus it
        controls the features' density. The metric unit is defined by
        `window_step_unit`.
    window_step_unit : {``pixels``, ``cells``}, optional
        Defines the metric unit of the `window_step_vertical` and
        `window_step_horizontal` parameters.
    padding : `bool`, optional
        If ``True``, the output image is padded with zeros to match the input
        image's size.
    algorithm : {``dalaltriggs``, ``zhuramanan``}, optional
        Specifies the algorithm used to compute HOGs. ``dalaltriggs`` is the
        implementation of [1] and ``zhuramanan`` is the implementation of [2].
    cell_size : `float`, optional
        Defines the cell size in pixels. This value is set to both the width
        and height of the cell. This option is valid for both algorithms.
    block_size : `float`, optional
        Defines the block size in cells. This value is set to both the width
        and height of the block. This option is valid only for the
        ``dalaltriggs`` algorithm.
    num_bins : `float`, optional
        Defines the number of orientation histogram bins. This option is
        valid only for the ``dalaltriggs`` algorithm.
    signed_gradient : `bool`, optional
        Flag that defines whether we use signed or unsigned gradient angles.
        This option is valid only for the ``dalaltriggs`` algorithm.
    l2_norm_clip : `float`, optional
        Defines the clipping value of the gradients' L2-norm. This option is
        valid only for the ``dalaltriggs`` algorithm.
    verbose : `bool`, optional
        Flag to print HOG related information.

    Returns
    -------
    hog : :map:`Image` or subclass or ``(X, Y, ..., Z, K)`` `ndarray`
        The HOG features image. It has the same type as the input ``pixels``.
        The output number of channels in the case of ``dalaltriggs`` is
        ``K = num_bins * block_size *block_size`` and ``K = 31`` in the case of
        ``zhuramanan``.

    Raises
    ------
    ValueError
        HOG features mode must be either dense or sparse
    ValueError
        Algorithm must be either dalaltriggs or zhuramanan
    ValueError
        Number of orientation bins must be > 0
    ValueError
        Cell size (in pixels) must be > 0
    ValueError
        Block size (in cells) must be > 0
    ValueError
        Value for L2-norm clipping must be > 0.0
    ValueError
        Window height must be >= block size and <= image height
    ValueError
        Window width must be >= block size and <= image width
    ValueError
        Window unit must be either pixels or blocks
    ValueError
        Horizontal window step must be > 0
    ValueError
        Vertical window step must be > 0
    ValueError
        Window step unit must be either pixels or cells

    References
    ----------
    .. [1] N. Dalal and B. Triggs, "Histograms of oriented gradients for human
        detection", Proceedings of the IEEE Conference on Computer Vision and
        Pattern Recognition (CVPR), 2005.
    .. [2] X. Zhu, D. Ramanan. "Face detection, pose estimation and landmark
        localization in the wild", Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2012.
    """
    # TODO: This is a temporary fix
    # flip axis
    pixels = np.rollaxis(pixels, 0, len(pixels.shape))

    # Parse options
    if mode not in ['dense', 'sparse']:
        raise ValueError("HOG features mode must be either dense or sparse")
    if algorithm not in ['dalaltriggs', 'zhuramanan']:
        raise ValueError("Algorithm must be either dalaltriggs or zhuramanan")
    if num_bins <= 0:
        raise ValueError("Number of orientation bins must be > 0")
    if cell_size <= 0:
        raise ValueError("Cell size (in pixels) must be > 0")
    if block_size <= 0:
        raise ValueError("Block size (in cells) must be > 0")
    if l2_norm_clip <= 0.0:
        raise ValueError("Value for L2-norm clipping must be > 0.0")
    if mode == 'dense':
        if window_unit not in ['pixels', 'blocks']:
            raise ValueError("Window unit must be either pixels or blocks")
        window_height_temp = window_height
        window_width_temp = window_width
        if window_unit == 'blocks':
            window_height_temp = window_height * block_size * cell_size
            window_width_temp = window_width * block_size * cell_size
        if (window_height_temp < block_size * cell_size or
                    window_height_temp > pixels.shape[0]):
            raise ValueError("Window height must be >= block size and <= "
                             "image height")
        if (window_width_temp < block_size*cell_size or
                    window_width_temp > pixels.shape[1]):
            raise ValueError("Window width must be >= block size and <= "
                             "image width")
        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0")
        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0")
        if window_step_unit not in ['pixels', 'cells']:
            raise ValueError("Window step unit must be either pixels or cells")

    # Correct input image_data
    pixels = np.asfortranarray(pixels)
    pixels *= 255.

    # Dense case
    if mode == 'dense':
        # Iterator parameters
        if algorithm == 'dalaltriggs':
            algorithm = 1
            if window_unit == 'blocks':
                block_in_pixels = cell_size * block_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical *
                                                 cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal *
                                                   cell_size)
        elif algorithm == 'zhuramanan':
            algorithm = 2
            if window_unit == 'blocks':
                block_in_pixels = 3 * cell_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical *
                                                 cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal *
                                                   cell_size)
        iterator = WindowIterator(pixels, window_height, window_width,
                                  window_step_horizontal,
                                  window_step_vertical, padding)
    # Sparse case
    else:
        # Create iterator
        if algorithm == 'dalaltriggs':
            algorithm = 1
            window_size = cell_size * block_size
            step = cell_size
        else:
            algorithm = 2
            window_size = 3 * cell_size
            step = cell_size
        iterator = WindowIterator(pixels, window_size, window_size, step,
                                  step, False)
    # Print iterator's info
    if verbose:
        print(iterator)
    # Compute HOG
    hog_descriptor = iterator.HOG(algorithm, num_bins, cell_size, block_size,
                                  signed_gradient, l2_norm_clip, verbose)
    # TODO: This is a temporal fix
    # flip axis
    hog_descriptor = WindowIteratorResult(
        np.ascontiguousarray(np.rollaxis(hog_descriptor.pixels, -1)),
        hog_descriptor.centres)
    return hog_descriptor


sparse_hog = partial(hog, mode='sparse')
sparse_hog.__name__ = 'sparse_hog'
sparse_hog.__doc__ = hog.__doc__


# TODO: Needs fixing ...
@winitfeature
def lbp(pixels, radius=None, samples=None, mapping_type='riu2',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding=True, verbose=False,
        skip_checks=False):
    r"""
    Extracts Local Binary Pattern (LBP) features from the input image. The
    output image has ``N * C`` number of channels, where ``N`` is the number of
    channels of the original image and ``C`` is the number of radius/samples
    values combinations that are used in the LBP computation.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    radius : `int` or `list` of `int` or ``None``, optional
        It defines the radius of the circle (or circles) at which the sampling
        points will be extracted. The radius (or radii) values must be greater
        than zero. There must be a radius value for each samples value, thus
        they both need to have the same length. If ``None``, then
        ``[1, 2, 3, 4]`` is used.
    samples : `int` or `list` of `int` or ``None``, optional
        It defines the number of sampling points that will be extracted at each
        circle. The samples value (or values) must be greater than zero. There
        must be a samples value for each radius value, thus they both need to
        have the same length. If ``None``, then ``[8, 8, 8, 8]`` is used.
    mapping_type : {``u2``, ``ri``, ``riu2``, ``none``}, optional
        It defines the mapping type of the LBP codes. Select ``u2`` for
        uniform-2 mapping, ``ri`` for rotation-invariant mapping, ``riu2`` for
        uniform-2 and rotation-invariant mapping and ``none`` to use no mapping
        and only the decimal values instead.
    window_step_vertical : `float`, optional
        Defines the vertical step by which the window is moved, thus it controls
        the features density. The metric unit is defined by `window_step_unit`.
    window_step_horizontal : `float`, optional
        Defines the horizontal step by which the window is moved, thus it
        controls the features density. The metric unit is defined by
        `window_step_unit`.
    window_step_unit : {``pixels``, ``window``}, optional
        Defines the metric unit of the `window_step_vertical` and
        `window_step_horizontal` parameters.
    padding : `bool`, optional
        If ``True``, the output image is padded with zeros to match the input
        image's size.
    verbose : `bool`, optional
        Flag to print LBP related information.
    skip_checks : `bool`, optional
        If ``True``, do not perform any validation of the parameters.

    Returns
    -------
    lbp : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        The ES features image. It has the same type and shape as the input
        ``pixels``. The output number of channels is
        ``C = len(radius) * len(samples)``.

    Raises
    ------
    ValueError
        Radius and samples must both be either integers or lists
    ValueError
        Radius and samples must have the same length
    ValueError
        Radius must be > 0
    ValueError
        Radii must be > 0
    ValueError
        Samples must be > 0
    ValueError
        Mapping type must be u2, ri, riu2 or none
    ValueError
        Horizontal window step must be > 0
    ValueError
        Vertical window step must be > 0
    ValueError
        Window step unit must be either pixels or window

    References
    ----------
    .. [1] T. Ojala, M. Pietikainen, and T. Maenpaa, "Multiresolution gray-scale
        and rotation invariant texture classification with local binary
        patterns", IEEE Transactions on Pattern Analysis and Machine
        Intelligence, vol. 24, num. 7, p. 971-987, 2002.
    """
    if radius is None:
        radius = range(1, 5)
    if samples is None:
        samples = [8]*4

    # TODO: This is a temporal fix
    # flip axis
    pixels = np.rollaxis(pixels, 0, len(pixels.shape))

    if not skip_checks:
        # Check parameters
        if ((isinstance(radius, int) and isinstance(samples, list)) or
                (isinstance(radius, list) and isinstance(samples, int))):
            raise ValueError("Radius and samples must both be either integers "
                             "or lists")
        elif isinstance(radius, list) and isinstance(samples, list):
            if len(radius) != len(samples):
                raise ValueError("Radius and samples must have the same "
                                 "length")

        if isinstance(radius, int) and radius < 1:
            raise ValueError("Radius must be > 0")
        elif isinstance(radius, list) and sum(r < 1 for r in radius) > 0:
            raise ValueError("Radii must be > 0")

        if isinstance(samples, int) and samples < 1:
            raise ValueError("Samples must be > 0")
        elif isinstance(samples, list) and sum(s < 1 for s in samples) > 0:
            raise ValueError("Samples must be > 0")

        if mapping_type not in ['u2', 'ri', 'riu2', 'none']:
            raise ValueError("Mapping type must be u2, ri, riu2 or "
                             "none")

        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0")

        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0")

        if window_step_unit not in ['pixels', 'window']:
            raise ValueError("Window step unit must be either pixels or "
                             "window")

    # Correct input image_data
    pixels = np.asfortranarray(pixels)

    # Parse options
    radius = np.asfortranarray(radius)
    samples = np.asfortranarray(samples)
    window_height = np.uint32(2 * radius.max() + 1)
    window_width = window_height
    if window_step_unit == 'window':
        window_step_vertical = np.uint32(window_step_vertical * window_height)
        window_step_horizontal = np.uint32(window_step_horizontal *
                                           window_width)
    if mapping_type == 'u2':
        mapping_type = 1
    elif mapping_type == 'ri':
        mapping_type = 2
    elif mapping_type == 'riu2':
        mapping_type = 3
    else:
        mapping_type = 0

    # Create iterator object
    iterator = WindowIterator(pixels, window_height, window_width,
                              window_step_horizontal, window_step_vertical,
                              padding)

    # Print iterator's info
    if verbose:
        print(iterator)

    # Compute LBP
    lbp_descriptor = iterator.LBP(radius, samples, mapping_type, verbose)

    # TODO: This is a temporary fix
    # flip axis
    lbp_descriptor = WindowIteratorResult(
        np.ascontiguousarray(np.rollaxis(lbp_descriptor.pixels, -1)),
        lbp_descriptor.centres)
    return lbp_descriptor
