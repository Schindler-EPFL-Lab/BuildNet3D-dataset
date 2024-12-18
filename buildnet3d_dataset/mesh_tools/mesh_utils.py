import numpy as np


def triangle_face_normals(
    AB: np.ndarray,
    AC: np.ndarray,
    A_normal: np.ndarray,
    B_normal: np.ndarray,
    C_normal: np.ndarray,
) -> np.ndarray:
    """
    Computes cross product of vectors `AB` and `AC` so that it is aligned with the
    normals of each vertex. `AB` and `AC` are both (n, 3) dimension arrays where the
    first axis is each triangle face and the second axis are the x, y, and z components
    of each vector created from triangle ABC. `A_normal`, `B_normal`, and `C_normal`
    are each (n, 3) dimensional arrays where the first axis represents each triangle
    face and the second axis are the x, y, and z components of the normals of each
    vertex.

    :returns: (n, 3) array representing the x, y, and z components of each triangle
    face's normal vector aligned with the normals of its corresponding verticies.
    """
    face_normals = np.cross(AB, AC)
    vertex_normals = np.stack([A_normal, B_normal, C_normal], axis=2)
    angle_check = check_angles(face_normals, vertex_normals)

    if np.any(angle_check):
        mask = np.any(angle_check, axis=1)
        face_normals[mask, :] = face_normals[mask, :] * -1
        angle_check_2 = check_angles(face_normals, vertex_normals)
        if np.any(angle_check_2):
            count = np.sum(np.any(angle_check, axis=1))
            raise RuntimeError(
                f"{count} meshes were not able to be corrected to match vertex normals"
            )

    return face_normals


def check_angles(face_normals: np.ndarray, vertex_normals: np.ndarray) -> np.ndarray:
    """
    Checks to see if the face normal is on the same side (within 90 degrees) of the
    vertex normals. `face_normals` is an array of shape (n, 3) where the first axis is
    each face and the second axis are the x, y, and z components of the face's normal
    `vertex_normals` is an array of shape (n, 3, 3) where the first axis represents a
    face, the second axis are the x, y, and z components of the normals, and the third
    axis are the vertices (v1, v2, v3).

    The angle between the two vectors is found from the formula:
    cos(theta) = dotproduct(A, B) / (|A|*|B|)
    where theta represents the angle between vector A and vector B

    :return: Numpy array of shape (n, 3) where the first axis is each face and the
    second axis are its verticies. the array contains boolean values with `True` if a
    vertex normal is more than 90 degreeses from the face normal.
    """
    face_normals_cube = np.stack([face_normals, face_normals, face_normals], axis=2)
    dot_prod = npsumdot(face_normals_cube, vertex_normals, axis=1)
    face_normals_mag = np.linalg.norm(face_normals, axis=1).reshape(
        (face_normals.shape[0], 1)
    )
    vertex_normals_mag = np.linalg.norm(vertex_normals, axis=1)
    divisor = np.multiply(face_normals_mag, vertex_normals_mag)
    cos_angles = np.divide(dot_prod, divisor)
    cos_angles[np.isclose(cos_angles, 1)] = 1
    cos_angles[np.isclose(cos_angles, -1)] = -1
    angles = np.arccos(cos_angles)
    return angles > np.pi / 2


def npsumdot(arr1: np.ndarray, arr2: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Takes the dot product of each vector between `arr1` and `arr2` where the reduction
    in dimension is along `axis`.

    `arr1` and `arr2` should be the same shape otherwise numpy broadcasting rules may
    provide unexpected results.
    """
    ret_arr = np.multiply(arr1, arr2)
    ret_arr = np.sum(ret_arr, axis=axis)
    return ret_arr


def find_if_line_and_plane_intersect(
    vertices_1: np.ndarray,
    vertices_2: np.ndarray,
    plane_points: np.ndarray,
    domain: np.ndarray,
) -> np.ndarray:
    """
    `vertices_1` and `vertices_2` are nx3 arrays that describe a line in space and
    `plane_points` is a 3x3 array where the rows are 3 vertices that describe the
    plane of interest. `domain` is a 3x2 array which describes the x, y, and z extents
    of the plane.

    The intersection is determined by finding the equation of the plane and the line
    and evaluating if the line lies on the plane or if the line intersects the plane
    within 1 length of the line vector. The equation of the plane is determined using
    the normal vector of the plane and on of the points making the formula
    ax + by +cz + d = 0. a, b, and c are the x, y, and z components of the normal
    vector respectively and d is solved for by subtituting one of the points into the
    formula. The equation of the line is found using 2 points in space where the
    general form of the equation is p0 + (p1-p0)t = p_new where p0and p1 are the points
    provided in `vertices_1` and `vertices_2` and p_new is another point on the line
    when moving from p0 t times.

    :return: n length array with boolean indicating if the pairs of points in
    `vertices_1` and  `vertices_2` rest on or cross the plane described by
    `plane_points`
    """

    # Get normal
    AB = plane_points[1, :] - plane_points[0, :]
    AC = plane_points[2, :] - plane_points[0, :]
    normal = np.cross(AB, AC)

    # Check if normal is not zero
    if np.all(normal == 0):
        RuntimeError("plane points provided create line or point.")

    d = -np.dot(normal, plane_points[0, :])

    # First check if the vector between the two points are on the plane by taking
    # dot product with plane normal
    vector = vertices_2 - vertices_1
    vector_on_plane = np.isclose(vector @ normal.reshape(3, 1), 0, atol=1e-6).squeeze(1)

    # check to see that both vertices are on the plane
    vertices_1_in_domain = np.all(
        (vertices_1 >= domain[:, 0]) & (vertices_1 <= domain[:, 1]), axis=1
    )

    vertices_2_in_domain = np.all(
        (vertices_2 >= domain[:, 0]) & (vertices_2 <= domain[:, 1]), axis=1
    )

    line_on_plane_in_domain = (
        vector_on_plane & vertices_1_in_domain & vertices_2_in_domain
    )

    results = line_on_plane_in_domain

    # if all points were on the plane return results
    if vector_on_plane.sum() == len(results):
        return results

    other_points = ~vector_on_plane

    # Now check if any vertex pairs are on either side of the plane
    # line equations:
    # x = x0 + dx*t; dx = x1-x0
    # y = y0 + dy*t; dy = y1-y0
    # z = z0 + dz*t; dz = z1-z0
    #
    # plane equation:
    # a*x + b*y + c*z + d = 0
    #
    # subbing in
    # a*(x0 + dx*t) + b*(y0 + dy*t) + c*(z0 + dz*t) + d = 0
    #
    # if u = [x, y, z], du = [dx, dy, dz], and n = [a, b, c]
    # the above can be rewritten as:
    #
    # dotproduct(n, u) + dotproduct(n, du)*t + d = 0
    # t = -(dotproduct(n, u) + d) / dotproduct(n, du)

    du = vertices_2[other_points, :] - vertices_1[other_points, :]
    t = -1 * np.divide(
        np.matmul(vertices_1[other_points, :], normal.reshape((3, 1))) + d,
        np.matmul(du, normal.reshape((3, 1))),
    )

    # Create all True array and evaluate if line crosses
    line_crosses_plane = np.ones_like(t, dtype=bool)
    line_crosses_plane[t < 0] = False
    line_crosses_plane[t > 1] = False

    results[other_points] = line_crosses_plane.squeeze(1)

    return results
