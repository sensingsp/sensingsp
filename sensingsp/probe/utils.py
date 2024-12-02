import bpy
import numpy as np
from mathutils import Vector
def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return (x, y, z)
def add_probe(isuite=0, iprobe=0, location=Vector((0, 0, 0)), rotation=Vector((np.pi/2, 0, -np.pi/2)),scale=Vector((1, 1, 0)), grids=[10,10,1]):
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'ProbePlane_{isuite}_{iprobe}'
    empty.parent = Suite_obj
    fromInput = False
    if "Simulation Settings" in bpy.data.objects:
        option = bpy.data.objects["Simulation Settings"]["Add Probe"]
        parts = option.split(',')
        type = int(parts[0])
        distance = float(parts[1])
        N1 = int(parts[2])
        N2 = int(parts[3])
        # elif type==2:
        #     parts2 = parts[1].split(';')
        #     R=float(parts2[0])
        #     distance = float(parts[1])
        #     N1 = int(parts[2])
        #     N2 = int(parts[3])
            
        fromInput = True
    s = 0.05
    if fromInput:
        if type==1:
            for i1 in range(N1):
                for i2 in range(N2):
                    bpy.ops.object.camera_add(location=((i1-N1/2)*distance, (i2-N1/2)*distance, 0))
                    tx = bpy.context.object
                    tx.scale = (s*distance, s*distance, s*distance)
                    tx.name = f'Probe_Element_{isuite}_{iprobe}_{i1}_{i2}'
                    tx.parent = empty
        else:
            if N1 == 1:
                azimuths = np.array([0])
            else:
                azimuths = np.linspace(0, 2 * np.pi, N1)  # Azimuth angles from 0 to 2*pi

            if N2 == 1:
                elevations = np.array([0])
            else:
                elevations = np.linspace(-np.pi / 2, np.pi / 2, N2) 
            for i1 in range(N1):
                for i2 in range(N2):
                    cartesian_coords = sph2cart(azimuths[i1], elevations[i2], distance)
                    bpy.ops.object.camera_add(location=cartesian_coords)
                    tx = bpy.context.object
                    tx.scale = (s*distance, s*distance, s*distance)
                    tx.name = f'Probe_Element_{isuite}_{iprobe}_{i1}_{i2}'
                    tx.parent = empty
            
    else:
        Lambda = scale[0]/grids[0]
        for i1 in range(grids[0]):
            for i2 in range(grids[1]):
                bpy.ops.object.camera_add(location=((i1-grids[0]/2)*Lambda/2, (i2-grids[1]/2)*Lambda/2, 0))
                tx = bpy.context.object
                tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
                tx.name = f'Probe_Element_{isuite}_{iprobe}_{i1}_{i2}'
                tx.parent = empty