import numpy as np
from scipy.ndimage.morphology import binary_dilation

from menpo.shape import TriMesh, TexturedTriMesh
from menpo.image import MaskedImage, BooleanImage
from menpo.rasterize import GLRasterizer
from menpo.transform import AlignmentSimilarity, ThinPlateSplines
from menpo.rasterize.transform import (ExtractNDims, AppendNDims,
                                       model_to_clip_transform,
                                       optimal_cylindrical_unwrap)


def sample_points_and_trilist(mask, sampling_rate=1):
    r"""
    Returns sampling indices in x and y along with a trilist that joins the
    together the points.

    Parameters
    ----------

    mask: :class:`BooleanImage`
        The mask that should be sampled from

    sampling_rate: integer, optional
        The spacing of the grid

        Default 1

    Returns
    -------

    sample_x, sample_y, trilist: All ndarray
        To be used on a pixel array of an shape Image as follows
        points = img.pixels[sample_x, sample_y, :]
        points and trilist are now useful to construct a TriMesh

    """
    x, y = np.meshgrid(np.arange(0, mask.height, sampling_rate),
                       np.arange(0, mask.width, sampling_rate))
    # use the mask's mask to filter which points should be
    # selectable
    sampler_in_mask = mask.mask[x, y]
    sample_x, sample_y = x[sampler_in_mask], y[sampler_in_mask]
    # build a cheeky TriMesh to get hassle free trilist
    tm = TriMesh(np.vstack([sample_x, sample_y]).T)
    return sample_x, sample_y, tm.trilist.copy()


def prune_trilist(radius, cy_unwrapped, dead_zone=0.1):
    t = cy_unwrapped.trilist
    bound = (1 - dead_zone) * np.pi * radius
    x = cy_unwrapped.points[:, 0]
    # find which vertices are in the left and right
    lhs, rhs = np.where(x < - bound)[0], np.where(x > bound)[0]
    in_set = lambda tl, s: np.in1d(tl.ravel(), s).reshape([-1, 3]).any(axis=1)
    bad_tris = np.logical_and(in_set(t, lhs), in_set(t, rhs))
    return t[~bad_tris]


class TriMeshCorresponder(object):
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
    def __init__(self, target, interpolator=ThinPlateSplines,
                 image_width=1000, clip_space_scale=0.9,
                 mask_dilation=20, sampling_rate=1,
                 cy_dead_zone=0.1):

        self.f_interp = FlattenInterpolator(target, interpolator=interpolator,
                                            cy_dead_zone=cy_dead_zone)

        # 1. RASTERIZE

        # Need to decide the appropriate size of the image - check the
        # ratio of the flatted 2D mesh and use it to infer height
        r_h, r_w = self.f_interp.f_target_2d.range()
        ratio_w_to_h = (1.0 * r_w) / r_h
        image_height = int(ratio_w_to_h * image_width)

        # Build the rasterizer providing the clip space transform and shape
        cs_transform = model_to_clip_transform(self.f_interp.f_target_3d,
                                               xy_scale=clip_space_scale)
        self.r = GLRasterizer(projection_matrix=cs_transform.h_matrix,
                              width=image_width,
                              height=image_height)

        # 2. SAMPLING

        # Save out where the target landmarks land in the image
        self.img_tgt_2d = self.r.model_to_image_transform.apply(
            self.f_interp.f_target_3d)

        # make an example of the output we expect, and attach the lms
        sample_output = MaskedImage.blank((self.r.height, self.r.width))
        sample_output.landmarks['correspondence_target'] = self.img_tgt_2d
        # create the mask we will want and save it for attaching on outputs
        sample_output.constrain_mask_to_landmarks()
        if mask_dilation != 0:
            self.mask = BooleanImage(binary_dilation(sample_output.mask.mask,
                                                     iterations=mask_dilation))
        else:
            self.mask = sample_output.mask
        self.mask.landmarks = sample_output.landmarks

        self.sampling_rate = sampling_rate
        # attach the sampling arrays
        self.s_x, self.s_y, self.trilist = sample_points_and_trilist(
            self.mask, self.sampling_rate)

    def correspond(self, mesh, group=None, label='all'):
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
        return self.generate_correspondence_mesh(*self.f_interp(mesh,
                                                                group=group,
                                                                label=label))

    def generate_correspondence_mesh(self, f_interp_mesh, mesh):
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
        # build the rgb and shape images
        rgb_image, shape_image = self.r.rasterize_mesh_with_f3v_interpolant(
            f_interp_mesh, per_vertex_f3v=mesh.points)
        # rgb_image.mask = self.mask.copy()
        # shape_image.mask = self.mask.copy()
        if isinstance(mesh, TexturedTriMesh):
            tm = self._extract_textured_trimesh(shape_image, rgb_image)
        else:
            tm = self._extract_trimesh(shape_image)
        tm.landmarks = mesh.landmarks
        return tm, rgb_image, shape_image

    def _extract_trimesh(self, shape_image):
        sampled = shape_image.pixels[self.s_x, self.s_y, :]
        return TriMesh(sampled, trilist=self.trilist)

    def _extract_textured_trimesh(self, shape_image, rgb_image):
        sampled = shape_image.pixels[self.s_x, self.s_y, :]
        # TODO generate tcoords and return TexturedTriMesh
        return TexturedTriMesh(sampled, sampled[:, :2], rgb_image,
                               trilist=self.trilist)


class FlattenInterpolator(object):
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
    def __init__(self, target, interpolator=ThinPlateSplines,
                 cy_dead_zone=0.1):
        self.target = target
        self.interpolator = interpolator
        self.cy_dead_zone = cy_dead_zone

        # Transform to flatten the mesh ready for interpolation
        self.flattener, self.radius = optimal_cylindrical_unwrap(self.target)

        # Prepare the flattened target in 2D and 3D
        self.f_target_3d = self.flattener.apply(self.target)
        self.f_target_2d = ExtractNDims(2).apply(self.f_target_3d)

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
        alignment = AlignmentSimilarity(mesh.landmarks[group][label].lms,
                                        self.target)
        rigid_aligned_mesh = alignment.apply(mesh)

        # 2. Flatten the mesh, and warp it to align with the flattened target
        f_3d = self.flattener.apply(rigid_aligned_mesh)
        f_2d = ExtractNDims(2).apply(f_3d)

        # 3. Warp the 2D flatted target to be in dense correspondence
        tps_transform = self.interpolator(f_2d.landmarks[group][label].lms,
                                          self.f_target_2d)
        w_2d = tps_transform.apply(f_2d)

        # 4. Append on the Z dim + set it to what it was in the flattened case
        w_3d = AppendNDims(1).apply(w_2d)
        w_3d.points[:, 2] = f_3d.points[:, 2]
        # 5. Break any connections at the back of the cylinder
        w_3d.trilist = prune_trilist(self.radius, w_3d,
                                     dead_zone=self.cy_dead_zone)
        return w_3d, rigid_aligned_mesh
