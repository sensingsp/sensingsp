import bpy
import numpy as np
from mathutils import Vector, Matrix

def add_lidar(isuite, ilidar, location, rotation):
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'LidarPlane_{isuite}_{ilidar}_{0}'
    empty.parent = Suite_obj
    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(0, 0, 0))
    rx = bpy.context.object
    rx.name = f'Lidar_{isuite}_{ilidar}_{1:05}'
    rx.parent = empty
    # rx.data.lens = 10

def pointcloud(LidarObject):
        
    az = Vector((0.0, 0.0, -1.0))
    az.normalize()
    local_x = Vector((1.0, 0.0, 0.0))
    rotation_axis = local_x.cross(az)
    rotation_axis.normalize()
    angle = local_x.angle(az)
    matrix_x2z = Matrix.Rotation(angle, 4, rotation_axis)
    
    pointcloud = []
    d = isosphere_directions(azimuth_Lim=360,azimuth_N=600, elevation_Lim=50, elevation_N=10)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    global_location = LidarObject.matrix_world.to_translation()
    transformation_matrix = LidarObject.matrix_world.to_3x3() 
    for local_dir in d:
        global_dir = transformation_matrix @ ( matrix_x2z @ local_dir)
        result, location, normal, face_index, hit_obj, matrix = bpy.context.scene.ray_cast(depsgraph, global_location, global_dir)
        if result:
            pointcloud.append([location.x,location.y,location.z])
    
    return np.array(pointcloud)
    
    

def isosphere_directions(azimuth_Lim=360,azimuth_N=64, elevation_Lim=120, elevation_N=28):
    directions = []
    azimuthv = np.linspace(-azimuth_Lim/2,azimuth_Lim/2,azimuth_N)
    elevationv = np.arcsin(np.linspace(-np.sin(np.radians(elevation_Lim/2)),np.sin(np.radians(elevation_Lim/2)),elevation_N))
    for elevation in elevationv:
        for azimuth in azimuthv:
            # Convert degrees to radians
            elev_rad = elevation
            azim_rad = np.radians(azimuth)

            # Spherical to Cartesian conversion
            x = np.cos(elev_rad) * np.cos(azim_rad)
            y = np.cos(elev_rad) * np.sin(azim_rad)
            z = np.sin(elev_rad)

            # Append direction vector
            directions.append(Vector((x, y, z)))

    return directions

# d = isosphere_directions()

# from matplotlib import pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(d[:,0], d[:,1], d[:,2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()