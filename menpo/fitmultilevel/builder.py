from __future__ import division
import abc
import numpy as np

from menpo.model.pca import PCAModel
from menpo.shape import mean_pointcloud
from menpo.transform import Scale, Translation, GeneralizedProcrustesAnalysis
from menpo.visualize import print_dynamic, progress_bar_str


# COMPATIBILITY ONLY
# this works on every image at once - we generally don't want this going
# forward
def normalization_wrt_reference_shape(images, group, label,
                                      normalization_diagonal, verbose=False):
    r"""
    Function that normalizes the images sizes with respect to the reference
    shape (mean shape) scaling. This step is essential before building a
    deformable model.

    The normalization includes:
    1) Computation of the reference shape as the mean shape of the images'
       landmarks.
    2) Scaling of the reference shape using the normalization_diagonal.
    3) Rescaling of all the images so that their shape's scale is in
       correspondence with the reference shape's scale.

    Parameters
    ----------
    images: list of :class:`menpo.image.MaskedImage`
        The set of landmarked images from which to build the model.

    group : string
        The key of the landmark set that should be used. If None,
        and if there is only one set of landmarks, this set will be used.

    label: string
        The label of of the landmark manager that you wish to use. If no
        label is passed, the convex hull of all landmarks is used.

    normalization_diagonal: int
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If int, it ensures that the mean shape is scaled so that the
        diagonal of the bounding box containing it matches the
        normalization_diagonal value.
        If None, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    verbose: bool, Optional
        Flag that controls information and progress printing.

        Default: False

    Returns
    -------
    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to
        a consistent object size.
    normalized_images : :map:`MaskedImage` list
        A list with the normalized images.
    """
    # the reference_shape is the mean shape of the images' landmarks
    if verbose:
        print_dynamic('- Computing reference shape')
    reference_shape = calculate_reference_shape([i.landmarks for i in images],
                                                group, label,
                                                normalization_diagonal)
    # normalize the scaling of all images wrt the reference_shape size
    normalized_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic('- Normalizing images size: {}'.format(
                progress_bar_str((c + 1.) / len(images),
                                 show_bar=False)))
        normalized_images.append(i.rescale_to_reference_shape(
            reference_shape, group=group, label=label))

    if verbose:
        print_dynamic('- Normalizing images size: Done\n')
    return reference_shape, normalized_images


# COMPATIBILITY ONLY
# this works on every image at once - we generally don't want this going
# forward
def create_pyramid(images, n_levels, downscale, pyramid_on_features,
                   features):
    r"""
    Function that creates a generator function for Gaussian pyramid. The
    pyramid can be created either on the feature space or the original
    (intensities) space.

    Parameters
    ----------
    images: list of :class:`menpo.image.Image`
        The set of landmarked images from which to build the AAM.
    n_levels: int
        The number of multi-resolution pyramidal levels to be used.
    downscale: float
        The downscale factor that will be used to create the different
        pyramidal levels.
    pyramid_on_features: boolean
        If True, the features are extracted at the highest level and the
        pyramid is created on the feature images.
        If False, the pyramid is created on the original (intensities)
        space.
    features: list of size 1 with str or function/closure or None
        The feature type to be used in case pyramid_on_features is enabled.
    verbose: bool, Optional
        Flag that controls information and progress printing.

        Default: False

    Returns
    -------
    generator: function
        The generator function of the Gaussian pyramid.
    """
    return [pyramid_of_feature_images(i, n_levels, downscale,
                                      pyramid_on_features, features) for i in images]


from functools import partial
from menpo.feature import no_op
from menpo.transform import PiecewiseAffine


class AAMBuild(object):

    def __init__(self, landmarks, group=None, label=None, diagonal=None,
                 n_levels=3, downscale=2, pyramid_on_features=False,
                 features=no_op, transform=PiecewiseAffine):
        reference_shape = calculate_reference_shape(landmarks, group,
                                                    label, diagonal=diagonal)
        self.normalize = partial(normalize_wrt_reference_shape,
                                 reference_shape, group, label)
        # todo prepare frames (new func)
        self.pyramid = partial(pyramid_of_warped_feature_images, n_levels,
                               downscale, pyramid_on_features, features, group,
                               label, reference_shape, transform)

    def __call__(self, image):
        return self.pyramid(self.normalize(image))


def normalize_wrt_reference_shape(reference_shape, group, label, image):
    return image.rescale_to_reference_shape(reference_shape, group=group,
                                            label=label)


def calculate_reference_shape(landmarks, group, label,
                              diagonal=None):
    shapes = [lms[group][label] for lms in landmarks]
    reference_shape = mean_pointcloud(shapes)
    # fix the reference_shape's diagonal length if asked
    if diagonal:
        x, y = reference_shape.range()
        scale = diagonal / np.sqrt(x**2 + y**2)
        Scale(scale, reference_shape.n_dims).apply_inplace(reference_shape)
    return reference_shape


def pyramid_of_warped_feature_images(n_levels, downscale, pyramid_on_features,
                                     features, group, label, frames,
                                     transforms, image):
    # images is a generator that yields each of the pyramid levels in
    # turn, with features applied as dictated by pyramid_on_features.
    images = pyramid_of_feature_images(n_levels, downscale,
                                       pyramid_on_features, features, image)
    # we want to return a generator that yields these images warped
    return warp_images_to_frame(group, label, frames, transforms, images)


def pyramid_of_feature_images(n_levels, downscale, pyramid_on_features,
                              features, image):
    if pyramid_on_features:
        # compute highest level feature
        feature_image = features[0](image)
        # create pyramid on the feature image
        return feature_image.gaussian_pyramid(n_levels=n_levels,
                                              downscale=downscale)
    else:
        # create pyramid on intensities image
        # feature will be computed per level
        pyramid_generator = image.gaussian_pyramid(n_levels=n_levels,
                                                   downscale=downscale)
        return feature_images(pyramid_generator, features)


# adds feature extraction to a generator of images
def feature_images(images, features):
    for feature, level in zip(features, images):
        yield feature(level)


# adds warping behavior to a generator of images
def warp_images_to_frame(group, label, frames, transforms, images):
    for f_t_i in zip(frames, transforms, images):
        yield warp_image_to_frame(group, label, *f_t_i)


def warp_image_to_frame(group, label, frame, transform, image):
    # update the transform to point to the new landmarks
    transform.set_target(image.landmarks[group][label])
    # warp the image
    return image.warp_to(frame, transform)


def build_shape_model(shapes, max_components):
    r"""
    Builds a shape model given a set of shapes.

    Parameters
    ----------
    shapes: list of :map:`PointCloud`
        The set of shapes from which to build the model.
    max_components: None or int or float
        Specifies the number of components of the trained shape model.
        If int, it specifies the exact number of components to be retained.
        If float, it specifies the percentage of variance to be retained.
        If None, all the available components are kept (100% of variance).

    Returns
    -------
    shape_model: :class:`menpo.model.pca`
        The PCA shape model.
    """
    # centralize shapes
    centered_shapes = [Translation(-s.centre).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    aligned_shapes = [s.aligned_source for s in gpa.transforms]

    # build shape model
    shape_model = PCAModel(aligned_shapes)
    if max_components is not None:
        # trim shape model if required
        shape_model.trim_components(max_components)

    return shape_model


class DeformableModelBuilder(object):
    r"""
    Abstract class with a set of functions useful to build a Deformable Model.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def build(self, images, group=None, label=None):
        r"""
        Builds a Multilevel Deformable Model.
        """
