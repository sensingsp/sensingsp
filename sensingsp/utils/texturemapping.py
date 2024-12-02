import bpy
import numpy as np
import mathutils

def interpolate_3d_from_uv_bpy(obj, u0, v0):
    """
    Given a UV coordinate (u0, v0), calculate the corresponding (x, y, z)
    in the 3D mesh using Blender's bpy API, if the UV coordinate lies on the mesh.

    Parameters:
    - obj: The Blender object containing the mesh.
    - u0: float, U coordinate of the UV map.
    - v0: float, V coordinate of the UV map.

    Returns:
    - (x, y, z): Tuple of 3D coordinates if (u0, v0) is within the mesh, else None.
    """
    
    # Ensure the object has a mesh and UV map
    if not obj.data or not obj.data.uv_layers:
        print("The object has no mesh or UV map.")
        return None
    
    # Get the UV layer and the mesh data
    uv_layer = obj.data.uv_layers.active.data
    mesh = obj.data
    
    # Loop through the polygons (faces) in the mesh
    for poly in mesh.polygons:
        # Get the vertices and UV coordinates for the polygon
        poly_verts = poly.vertices
        uv_coords = [uv_layer[loop_idx].uv for loop_idx in poly.loop_indices]

        # Ensure that we are only working with triangular faces (3 vertices)
        if len(uv_coords) >= 3:  # Make sure it has at least 3 vertices
            # Limit to first 3 UVs in case more are present
            uv_triangle = uv_coords[:3]  # Use only first 3 UVs for triangle calculation

            # Check if (u0, v0) is inside the UV triangle of this face using barycentric coordinates
            bary_coords = barycentric_coordinates(uv_triangle, mathutils.Vector((u0, v0)))
            
            if bary_coords is not None and all(0 <= b <= 1 for b in bary_coords):
                # Interpolate the 3D coordinates using the barycentric coordinates
                vertex_coords = [mesh.vertices[poly_verts[i]].co for i in range(3)]
                interpolated_3d = mathutils.Vector((0, 0, 0))
                
                for i in range(3):
                    interpolated_3d += vertex_coords[i] * bary_coords[i]
                
                return tuple(interpolated_3d)
    
    # If no face contains the UV coordinate, return None
    return None


def barycentric_coordinates(uv_triangle, uv_point):
    """
    Calculate the barycentric coordinates of a 2D point relative to a triangle.

    Parameters:
    - uv_triangle: List of 3 UV coordinates (u, v) of the triangle's vertices.
    - uv_point: UV coordinate of the point.

    Returns:
    - Barycentric coordinates (lambda1, lambda2, lambda3) or None if not inside the triangle.
    """
    if len(uv_triangle) != 3:
        return None  # Must be a triangle with 3 vertices

    a, b, c = uv_triangle
    p = uv_point

    v0 = b - a
    v1 = c - a
    v2 = p - a

    # Compute dot products
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return None  # Degenerate triangle

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    if u < 0 or v < 0 or w < 0:
        return None  # Outside the triangle

    return np.array([u, v, w])


def uv_map_image_to_sphere(image, x,y,z):
    u = 0.5 + (np.arctan2(z, x) / (2 * np.pi)) 
    v = 0.5 + (np.arcsin(y) / np.pi) 
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    img_height, img_width = image.shape[:2]
    pixel_x = (u * (img_width - 1)).astype(np.int32)
    pixel_y = (v * (img_height - 1)).astype(np.int32)
    vertex_colors = image[pixel_y, pixel_x]
    return vertex_colors
