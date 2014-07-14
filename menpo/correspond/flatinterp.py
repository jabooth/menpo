from menpo.rasterize import (GLRasterizer, model_to_clip_transform,
                             dims_3to2, dims_2to3)
from menpo.transform import (AlignmentSimilarity, ThinPlateSplines,
                             optimal_cylindrical_unwrap, Transform)


class FlattenInterp(Transform):
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
    def __init__(self, target, flatten=optimal_cylindrical_unwrap,
                 interp=ThinPlateSplines):
        # Transform to flatten the mesh ready for interpolation
        self.flattener = flatten(target)
        self.interpolator = interp
        # Save out the rigid target
        self.r_tgt = target
        # Prepare the 2D/3d flattened targets
        self.f_tgt_3d = self.flattener.apply(target)
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
        w_2d = self.interpolator(f_2d.landmarks[group][label].lms,
                                 self.f_tgt_2d).apply(f_2d)

        # 4. Append on the Z dim + set it to what it was in the flattened case
        # TODO we should really re-add the z axis on the landmarks here too
        w_3d = self.dims_2to3.apply(w_2d)
        w_3d.points[:, 2] = f_3d.points[:, 2]
        return w_3d


class AlignFlattenInterp(object):
    r"""
    Combines a :map:`FlattenInterp` with an initial rigid alignment of the
    subject to the target. Useful for constructing dense correspondence.

    Parameters
    ----------

    target : :map:`PointCloud`
        The landmarks that input :map:`TriMesh` landmarks will be aligned
        against

    flatten_interp : :map:`Transform`
        A transform that will be used to flatten and interpolate input
        meshes. See :map:`FlattenInterp` for details.

    """
    def __init__(self, target, flatten_interp, edge_dead_zone=0.1):
        self.target = target
        self.fi = flatten_interp
        self.edge_dead_zone = edge_dead_zone

    def __call__(self, mesh, group=None, label='all'):
        r"""
        Return a version of the mesh that has been placed in correspondence
        by the rules of this TriMeshCorresponder.

        Parameters
        ----------

        mesh : :class:`TriMesh`
            The mesh that should be placed in correspondence
        group: string, optional
            The group label of the landmarks that should be used for
            interpolation. By default None, which will select the only
            available group if only one landmark group is present

            Default None
        label: string, optional
            The label of a particular set of landmarks, e.g. 'left_eye'.

            Default 'all'

        Returns
        -------

        :class:`TriMesh` A trimesh in correspondence with this corresponder.
        """
        # 1. Rigidly align the new mesh to the target
        r_aligned_mesh = AlignmentSimilarity(mesh.landmarks[group][label].lms,
                                             self.target).apply(mesh)

        w_3d = self.fi.apply(r_aligned_mesh, group=group, label=label)

        # 5. Break any connections at the back of the cylinder
        # w_3d.trilist = prune_trilist(w_3d, dead_zone=self.edge_dead_zone)

        return r_aligned_mesh, w_3d


class AFIRasterizer(object):
    r"""
    Object for bringing landmarked TriMeshes into dense correspondence by
    flattening the mesh into a 2D atlas and performing an interpolation in
    the 2D space. The resultant interpolated shape is sampled at a set interval
    producing a mesh with a particular number of points, particular
    connectivity, and particular semantic meaning assigned to each point.

    In other words, if a set of TriMeshes is passed through a particular
    :class:`TriMeshCorresponder`, the returned the resulting meshes
    will be in dense correspondence.

    Parameters
    ----------

    target: :class:`PointCloud`
        The landmarks that input :class:`TriMesh` landmarks will be aligned
        against

    interpolator: :class:`Alignment`
        The 2D alignment that should be used to interpolate the flattened
        meshes

    shape_image_width: integer
        The desired width of the shape_image in pixels. Note that the exact
        width may differ slightly. The height of the shape image will be
        automatically selected based on the ratio of the flattened target.

    """
    def __init__(self, afi, image_width=1000, clip_space_scale=0.9):

        self.afi = afi

        # Need to decide the appropriate size of the image - check the
        # ratio of the flatted 2D mesh and use it to infer height
        r_h, r_w = self.afi.fi.f_tgt_2d.range()
        ratio_w_to_h = (1.0 * r_w) / r_h
        image_height = int(ratio_w_to_h * image_width)

        # Build the rasterizer providing the clip space transform and shape
        cs_transform = model_to_clip_transform(self.afi.fi.f_tgt_3d,
                                               xy_scale=clip_space_scale)
        self.r = GLRasterizer(projection_matrix=cs_transform.h_matrix,
                              width=image_width, height=image_height)

        # Save out where the target landmarks land in the image
        self.i_tgt_2d = self.r.model_to_image_transform.apply(
            self.afi.fi.f_tgt_3d)

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
        r_mesh, afi_mesh = self.afi(mesh, group=group, label=label)
        return self.r.rasterize_mesh_with_f3v_interpolant(
            afi_mesh, per_vertex_f3v=r_mesh.points)
