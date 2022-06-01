# A copy of napari.layers.shapes._shapes_models.shape
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from napari.utils.translations import trans

from ._bounding_box_utils import (
    is_collinear,
    path_to_mask,
    poly_to_mask,
    triangulate_edge,
    triangulate_face, find_bbox_corners, rectangle_to_box, find_corners,
)

LOG_DEBUG = True
class BoundingBox(ABC):
    """Class for a single bounding box
    Parameters
    ----------
    data : (N, D) array
        Vertices specifying the bounding box.
    edge_width : float
        thickness of  edges.
    z_index : int
        Specifier of z order priority. Bounding boxes with higher z order are displayed
        ontop of others.
    dims_order : (D,) list
        Order that the dimensions are to be rendered in.
    ndisplay : int
        Number of displayed dimensions.

    Attributes
    ----------
    data : (N, D) array
        Vertices specifying the bounding box.
    data_displayed : (N, 2) or (N, 3) array
        Vertices of the bounding box that are currently displayed.
    edge_width : float
        thickness of lines and edges.
    z_index : int
        Specifier of z order priority. Bounding boxes with higher z order are displayed
        ontop of others.
    dims_order : (D,) list
        Order that the dimensions are rendered in.
    ndisplay : int
        Number of dimensions to be displayed.
    slice_key : (2, M) array
        Min and max values of the M non-displayed dimensions, useful for
        slicing multidimensional bounding boxes.

    Notes
    -----
    _box : np.ndarray
        9x2 array of vertices of the interaction box. The first 8 points are
        the corners and midpoints of the box in clockwise order starting in the
        upper-left corner. The last point is the center of the box
    _face_vertices : np.ndarray
        Qx2 array of vertices of all triangles for the bounding box face
    _face_triangles : np.ndarray
        Px3 array of vertex indices that form the triangles for the bounding box face
    _edge_vertices : np.ndarray
        Rx2 array of centers of vertices of triangles for the bounding box edge.
        These values should be added to the scaled `_edge_offsets` to get the
        actual vertex positions. The scaling corresponds to the width of the
        edge
    _edge_offsets : np.ndarray
        Sx2 array of offsets of vertices of triangles for the bounding box edge. For
        These values should be scaled and added to the `_edge_vertices` to get
        the actual vertex positions. The scaling corresponds to the width of
        the edge
    _edge_triangles : np.ndarray
        Tx3 array of vertex indices that form the triangles for the bounding box edge
    _filled : bool
        Flag if array is filled or not.
    _use_face_vertices : bool
        Flag to use face vertices for mask generation.
    """

    def __init__(
        self,
        data,
        *,
        edge_width=1,
        z_index=0,
        dims_order=None,
        ndisplay=2,
    ):

        self._dims_order = dims_order or list(range(2))
        self._ndisplay = ndisplay
        self.slice_key = None

        self._face_vertices = np.empty((0, self.ndisplay))
        self._face_triangles = np.empty((0, 3), dtype=np.uint32)
        self._edge_vertices = np.empty((0, self.ndisplay))
        self._edge_offsets = np.empty((0, self.ndisplay))
        self._edge_triangles = np.empty((0, 3), dtype=np.uint32)
        self._box = np.empty((9, 2))

        self._filled = True
        self.data = data
        self._use_face_vertices = False
        self.edge_width = edge_width
        self.z_index = z_index
        self.name = 'bounding box'

    @property
    def data(self):
        # user writes own docstring
        return self._data

    @data.setter
    def data(self, data):
        data = np.array(data).astype(float)
        if len(self.dims_order) != data.shape[1]:
            self._dims_order = list(range(data.shape[1]))

        if len(data) == 2 and data.shape[1] == 2:
            data = find_bbox_corners(data)

        if len(data) != 2**len(self.dims_order):
            raise ValueError(
                trans._(
                    "Data shape does not match a rectangle. Rectangle expects four corner vertices, {number} provided.",
                    deferred=True,
                    number=len(data),
                )
            )

        self._data = data
        self._update_displayed_data()

    def _update_displayed_data(self):
        """Update the data that is to be displayed."""
        # Add four boundary lines and then two triangles for each


        if self.ndisplay == 2:
            self._set_meshes(self.data_displayed, face=False)
            self._face_vertices = self.data_displayed
            self._face_triangles = np.array([[0, 1, 2], [0, 2, 3]])
        else:
            ordered_vertices = self.data_displayed[
                [
                    0, 1, 3, 2, 0, 4, 6, 2, 3, 7, 6, 4, 5, 7, 3, 1, 5
                ]
            ]
            self._set_meshes(ordered_vertices, closed=False, face=False)
            self._face_triangles = np.array([[0, 0, 0]])
            # self._face_triangles = np.array([
            #     [0, 1, 2],
            #     [2, 3, 1],
            #     [0, 4, 1],
            #     [1, 4, 5],
            #     [0, 2, 6],
            #     [0, 6, 4],
            #     [7, 4, 6],
            #     [7, 5, 4],
            #     [7, 6, 2],
            #     [7, 2, 3],
            #     [7, 3, 1],
            #     [7, 1, 5]
            # ])
            # self._face_triangles = np.append(self._face_triangles, np.fliplr(self._face_triangles), 0)

        if self.ndisplay == 2:
            self._box = rectangle_to_box(self.data_displayed)

        data_not_displayed = self.data[:, self.dims_not_displayed]
        self.slice_key = np.round(
            [
                np.min(data_not_displayed, axis=0),
                np.max(data_not_displayed, axis=0),
            ]
        ).astype('int')

    @property
    def ndisplay(self):
        """int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        if self.ndisplay == ndisplay:
            return
        self._ndisplay = ndisplay
        self._update_displayed_data()

    @property
    def dims_order(self):
        """(D,) list: Order that the dimensions are rendered in."""
        return self._dims_order

    @dims_order.setter
    def dims_order(self, dims_order):
        if self.dims_order == dims_order:
            return
        self._dims_order = dims_order
        self._update_displayed_data()

    @property
    def dims_displayed(self):
        """tuple: Dimensions that are displayed."""
        return self.dims_order[-self.ndisplay :]

    @property
    def dims_not_displayed(self):
        """tuple: Dimensions that are not displayed."""
        return self.dims_order[: -self.ndisplay]

    @property
    def data_displayed(self):
        """(N, 2) or (N, 3) array: Vertices of the bounding box that are currently displayed."""
        displayed = np.unique(self.data[:, self.dims_displayed], axis=0)
        if len(self.dims_displayed) == 2:
            return find_corners(displayed)
        return displayed

    @data_displayed.setter
    def data_displayed(self, new_value):
        data = self.data[:, self.dims_displayed]
        data_sorted = data.copy()
        idx_sort = np.arange(len(data))
        for d in range(data.shape[1]):
            sub_idx_sort = data_sorted[:, -1 - d].argsort(kind='mergesort')
            data_sorted = data_sorted[sub_idx_sort]
            idx_sort = idx_sort[sub_idx_sort]
        vals, idx_start, count = np.unique(data_sorted, return_counts=True, return_index=True, axis=0)
        uniqe_idx = np.split(idx_sort, idx_start[1:])
        for val, idx_list in zip(new_value, uniqe_idx):
            self._data[idx_list[:, np.newaxis], self.dims_displayed] = val


    @property
    def edge_width(self):
        """float: thickness of lines and edges."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width):
        self._edge_width = edge_width

    @property
    def z_index(self):
        """int: z order priority of bounding box. Bounding boxes with higher z order displayed
        ontop of others.
        """
        return self._z_index

    @z_index.setter
    def z_index(self, z_index):
        self._z_index = z_index

    def _set_meshes(self, data, closed=True, face=True, edge=True):
        """Sets the face and edge meshes from a set of points.

        Parameters
        ----------
        data : np.ndarray
            Nx2 or Nx3 array specifying the bounding box to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not
        face : bool
            Bool which determines if the face need to be traingulated
        edge : bool
            Bool which determines if the edge need to be traingulated
        """
        if edge:
            centers, offsets, triangles = triangulate_edge(data, closed=closed)
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
        else:
            self._edge_vertices = np.empty((0, self.ndisplay))
            self._edge_offsets = np.empty((0, self.ndisplay))
            self._edge_triangles = np.empty((0, 3), dtype=np.uint32)

        if face:
            clean_data = np.array(
                [
                    p
                    for i, p in enumerate(data)
                    if i == 0 or not np.all(p == data[i - 1])
                ]
            )

            if not is_collinear(clean_data[:, -2:]):
                if clean_data.shape[1] == 2:
                    vertices, triangles = triangulate_face(clean_data)
                elif len(np.unique(clean_data[:, 0])) == 1:
                    val = np.unique(clean_data[:, 0])
                    vertices, triangles = triangulate_face(clean_data[:, -2:])
                    exp = np.expand_dims(np.repeat(val, len(vertices)), axis=1)
                    vertices = np.concatenate([exp, vertices], axis=1)
                else:
                    triangles = []
                    vertices = []
                if len(triangles) > 0:
                    self._face_vertices = vertices
                    self._face_triangles = triangles
                else:
                    self._face_vertices = np.empty((0, self.ndisplay))
                    self._face_triangles = np.empty((0, 3), dtype=np.uint32)
            else:
                self._face_vertices = np.empty((0, self.ndisplay))
                self._face_triangles = np.empty((0, 3), dtype=np.uint32)
        else:
            self._face_vertices = np.empty((0, self.ndisplay))
            self._face_triangles = np.empty((0, 3), dtype=np.uint32)

    def transform(self, transform):
        """Performs a linear transform on the bounding box

        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        """

        self._box = self._box @ transform.T
        self._data[:, self.dims_displayed] = (
            self._data[:, self.dims_displayed] @ transform.T
        )
        self._face_vertices = self._face_vertices @ transform.T

        points = self.data_displayed

        centers, offsets, triangles = triangulate_edge(
            points, closed=True
        )
        self._edge_vertices = centers
        self._edge_offsets = offsets
        self._edge_triangles = triangles

    def shift(self, shift):
        """Performs a 2D shift on the bounding box

        Parameters
        ----------
        shift : np.ndarray
            length 2 array specifying shift of bounding boxes.
        """
        shift = np.array(shift)
        self._face_vertices = self._face_vertices + shift
        self._edge_vertices = self._edge_vertices + shift
        self._box = self._box + shift
        self.data_displayed = self.data_displayed + shift

    def to_mask(self, mask_shape=None, zoom_factor=1, offset=[0, 0]):
        # TODO: might not be needed, if so, remove
        """Convert the bounding box vertices to a boolean mask.

        Set points to `True` if they are lying inside the shape if the shape is
        filled, or if they are lying along the boundary of the shape if the
        shape is not filled. Negative points or points outside the mask_shape
        after the zoom and offset are clipped.

        Parameters
        ----------
        mask_shape : (D,) array
            Shape of mask to be generated. If non specified, takes the max of
            the displayed vertices.
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.

        Returns
        -------
        mask : np.ndarray
            Boolean array with `True` for points inside the shape
        """
        if mask_shape is None:
            mask_shape = np.round(self.data_displayed.max(axis=0)).astype(
                'int'
            )

        if len(mask_shape) == 2:
            embedded = False
            shape_plane = mask_shape
        elif len(mask_shape) == self.data.shape[1]:
            embedded = True
            shape_plane = [mask_shape[d] for d in self.dims_displayed]
        else:
            raise ValueError(
                trans._(
                    "mask shape length must either be 2 or the same as the dimensionality of the shape, expected {expected} got {received}.",
                    deferred=True,
                    expected=self.data.shape[1],
                    received=len(mask_shape),
                )
            )

        if self._use_face_vertices:
            data = self._face_vertices
        else:
            data = self.data_displayed

        data = data[:, -len(shape_plane) :]

        if self._filled:
            mask_p = poly_to_mask(shape_plane, (data - offset) * zoom_factor)
        else:
            mask_p = path_to_mask(shape_plane, (data - offset) * zoom_factor)

        # If the mask is to be embedded in a larger array, compute array
        # and embed as a slice.
        if embedded:
            mask = np.zeros(mask_shape, dtype=bool)
            slice_key = [0] * len(mask_shape)
            j = 0
            for i in range(len(mask_shape)):
                if i in self.dims_displayed:
                    slice_key[i] = slice(None)
                else:
                    slice_key[i] = slice(
                        self.slice_key[0, j], self.slice_key[1, j] + 1
                    )
                j += 1
            displayed_order = np.array(copy(self.dims_displayed))
            displayed_order[np.argsort(displayed_order)] = list(
                range(len(displayed_order))
            )
            mask[tuple(slice_key)] = mask_p.transpose(displayed_order)
        else:
            mask = mask_p

        return mask
