from collections import namedtuple
import wrapt
from menpo.rasterize import (GLRasterizer, model_to_clip_transform,
                             dims_3to2, dims_2to3)
from menpo.transform import (AlignmentSimilarity, ThinPlateSplines,
                             optimal_cylindrical_unwrap, Transform)


class FlattenInterpolater(Transform):
    r"""
    Class for bringing landmarked :map:`TriMesh` instances into dense
    correspondence by flattening the mesh into a 2D space.

    Parameters
    ----------

    target : :map:`PointCloud`
        The sparse target that input :map:`TriMesh` landmarks will be
        aligned against

    flatten : func -> :map:`Transform`
        Function that when provided with ``target`` returns a
        :map:`Transform` that will serve to flatten the mesh into a 2D
        space.

    interp : :map:`Alignment`
        The 2D alignment that should be used to interpolate the flattened
        meshes.

    """
    def __init__(self, tgt, flatten=optimal_cylindrical_unwrap,
                 interp=ThinPlateSplines):
        # Transform to flatten the mesh ready for interpolation
        self.flattener = flatten(tgt)
        self.interpolator = interp
        # Save out the rigid target
        self.tgt = tgt
        # Prepare the 2D/3d flattened targets
        self.f_tgt_3d = self.flattener.apply(tgt)
        self.dims_3to2, self.dims_2to3 = dims_3to2(), dims_2to3()
        self.f_tgt_2d = self.dims_3to2.apply(self.f_tgt_3d)

    def _apply(self, x, group=None, label='all'):
        r"""
        Return a version of the mesh that has been flattened and interpolated
        to be aligned with the target of this :map:`FlattenInterp`.

        Parameters
        ----------

        x : :map:`Transformable`
            The mesh that should be placed in correspondence
        group : `string`, optional
            The group label of the landmarks that should be used for
            interpolation. By default None, which will select the only
            available group if only one landmark group is present

            Default None
        label : `string`, optional
            The label of a particular set of landmarks, e.g. 'left_eye'.

            Default 'all'

        Returns
        -------

        :map:`TriMesh`
            A version of the mesh flattened and interpolated in 2D to be in
            correspondence with the original target.

        """
        # 2. Flatten the mesh, and warp it to align with the flattened target
        f_3d = self.flattener.apply(x)
        f_2d = self.dims_3to2.apply(f_3d)

        # 3. Warp the 2D flatted target to be in dense correspondence
        w_2d = self.interpolator(f_2d.landmarks[group][label],
                                 self.f_tgt_2d).apply(f_2d)

        # 4. Append on the Z dim + set it to what it was in the flattened case
        w_3d = self.dims_2to3.apply(w_2d)
        w_3d.points[:, 2] = f_3d.points[:, 2]
        return w_3d


FlattenRasterizerResult = namedtuple('FlattenRasterizerResult',
                                     'rgb_image shape_image')


class FlattenRasterizer(object):

    def __init__(self, tgt, transform, image_width=1000, clip_space_scale=0.9):
        self.dims_3to2, self.dims_2to3 = dims_3to2(), dims_2to3()
        self.transform = transform
        self.tgt = tgt
        tgt_test = tgt.copy()
        tgt_test.landmarks['test_flatten'] = tgt_test
        # find where the tgt landmarks end up in the flattened space (in 3D)
        f_tgt_3d = self.transform.apply(tgt_test, group='test_flatten',
                                        label=None)

        # Need to decide the appropriate size of the image - check the
        # ratio of the flatted 2D mesh and use it to infer height
        r_h, r_w = dims_3to2().apply(f_tgt_3d).range()
        ratio_w_to_h = (1.0 * r_w) / r_h
        image_height = int(ratio_w_to_h * image_width)

        # Build the rasterizer providing the clip space transform and shape
        cs_transform = model_to_clip_transform(f_tgt_3d,
                                               xy_scale=clip_space_scale)
        self.r = GLRasterizer(projection_matrix=cs_transform.h_matrix,
                              width=image_width, height=image_height)

        # Save out where the target landmarks land in the image
        self.i_tgt_2d = self.r.model_to_image_transform.apply(f_tgt_3d)

    def __call__(self, mesh, group=None, label='all'):
        r"""
        Use a flattened warped mesh to build back a TriMesh in dense
        correspondence.

        Parameters
        ----------

        mesh: :class:`TriMesh`
            The original (rigidly aligned) TriMesh that we want to place in
            dense correspondence

        flattened_warped_mesh: :class:`TriMesh`


        """
        f_mesh = self.transform.apply(mesh, group=group, label=label)
        return FlattenRasterizerResult(
            *self.r.rasterize_mesh_with_f3v_interpolant(
            f_mesh, per_vertex_f3v=f_mesh.points))


AlignFlattenRasterizerResult = namedtuple('AlignFlattenRasterizerResult',
                                'aligned_mesh rgb_image shape_image')


@wrapt.decorator
def rigid_align(wrapped, _, args, kwargs):
    def _execute(mesh, *args, **kwargs):
        tgt = wrapped.tgt
        group = kwargs.get('group')
        label = kwargs.get('label')
        aligned_mesh = AlignmentSimilarity(mesh.landmarks[group][label],
                                           tgt).apply(mesh)
        return AlignFlattenRasterizerResult(aligned_mesh,
                                            *wrapped(aligned_mesh,
                                                     *args, **kwargs))
    return _execute(*args, **kwargs)


def AlignFlattenRasterizer(tgt, transform, image_width=1000,
                           clip_space_scale=0.9):
    return rigid_align(FlattenRasterizer(tgt, transform,
                                         image_width=image_width,
                                         clip_space_scale=clip_space_scale))