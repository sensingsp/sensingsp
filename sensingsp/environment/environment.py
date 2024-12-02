import bpy
from mathutils import Vector

def constantVelocity(obj):
    for fcurve in obj.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'LINEAR'

def colorobject(obj,rgba=[1,0,0,1]):
    material = bpy.data.materials.new(name="RedMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (rgba[0], rgba[1], rgba[2], rgba[3])  # RGBA
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def add_cube(location, direction, scale, subdivision):
    """
    Adds a cube to the scene at the specified location, with the specified direction, scale, and subdivision.

    :param location: Tuple of the location to place the cube (x, y, z).
    :param direction: Vector of the direction the cube should face.
    :param scale: Tuple of the scale of the cube (x, y, z).
    :param subdivision: Integer specifying the number of subdivisions for the cube.
    :return: The created cube object.
    """
    # Add a cube
    bpy.ops.mesh.primitive_cube_add(location=location, scale=(1, 1, 1))
    cube = bpy.context.object

    # Rotate the cube to face the specified direction
    direction = direction.normalized()
    up = Vector((0, 0, 1))
    rotation_axis = up.cross(direction)
    rotation_angle = up.angle(direction)

    if rotation_axis.length == 0:
        # If direction is parallel to up, handle edge case
        rotation_axis = Vector((1, 0, 0))

    cube.rotation_mode = 'AXIS_ANGLE'
    cube.rotation_axis_angle = (rotation_angle, rotation_axis.x, rotation_axis.y, rotation_axis.z)

    # Apply scale
    cube.scale = scale

    # Apply the rotation and scale
    bpy.context.view_layer.update()

    if subdivision > 0:
        # Subdivide the cube
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=subdivision)
        bpy.ops.object.mode_set(mode='OBJECT')

    return cube

def add_plane(location, direction, scale, subdivision):
    """
    Adds a plane to the scene at the specified location, with the specified direction, scale, and subdivision.

    :param location: Tuple of the location to place the plane (x, y, z).
    :param direction: Vector of the direction the plane should face.
    :param scale: Tuple of the scale of the plane (x, y, z).
    :param subdivision: Integer specifying the number of subdivisions for the plane.
    :return: The created plane object.
    """
    # Add a plane at the specified location
    bpy.ops.mesh.primitive_plane_add(location=location, scale=(1, 1, 1))
    plane = bpy.context.object

    # Calculate rotation needed to face the direction
    direction = direction.normalized()
    up = Vector((0, 0, 1))
    rotation_axis = up.cross(direction)
    rotation_angle = up.angle(direction)

    # Check for the parallel case, where no rotation is needed
    if rotation_axis.length == 0:
        rotation_axis = Vector((1, 0, 0))

    plane.rotation_mode = 'AXIS_ANGLE'
    plane.rotation_axis_angle = (rotation_angle, rotation_axis.x, rotation_axis.y, rotation_axis.z)

    # Apply the rotation and then scale
    bpy.context.view_layer.update()
    plane.scale = scale
    bpy.context.view_layer.update()

    # Apply subdivision if requested
    if subdivision > 0:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=subdivision)
        bpy.ops.object.mode_set(mode='OBJECT')

    # Apply transformations (optional, but generally a good practice)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    return plane
class BlenderSuiteFinder:
    def find_suite_information(self):
        suite_planes = self.find_suite_planes()
        info = []
        for sp in suite_planes:
            suite_info={}
            suite_info['Radar'] = []
            radar_objs = self.find_radar_planes(sp)
            for radar_obj in radar_objs:
                radarinfo={}
                radarinfo['GeneralRadarSpec_Object'] = radar_obj
                radarinfo['TX'] = []
                radarinfo['RX'] = []
                txs = self.find_tx(radar_obj)
                rxs = self.find_rx(radar_obj)
                for tx in txs:
                  radarinfo['TX'].append(tx)
                for rx in rxs:
                  radarinfo['RX'].append(rx)
                suite_info['Radar'].append(radarinfo)

            suite_info['RIS'] = []
            ris_objs = self.find_ris_planes(sp)
            for ris_obj in ris_objs:
              risElements = self.find_risElement(ris_obj)
              risinfo = []
              for riselement in risElements:
                risinfo.append(riselement)
              suite_info['RIS'].append(risinfo)

            suite_info['Probe'] = []
            probe_objs = self.find_probe_planes(sp)
            for probe_obj in probe_objs:
              probeSurface = self.find_probeSurface(probe_obj)
              suite_info['Probe'].append(probeSurface)
              
            
            suite_info['Lidar'] = []
            lidar_objs = self.find_lidar_planes(sp)
            for lidar_obj in lidar_objs:
                lidar = []
                lidarfound = False
                for obj in bpy.data.objects:
                    if obj.parent == lidar_obj:
                        if obj.name.startswith("Lidar_") and obj.type == 'CAMERA':
                            lidarfound = True
                            lidar = obj
                            break
                if lidarfound:  
                    suite_info['Lidar'].append(lidar)
                    
            suite_info['Camera'] = []
            Camera_objs = self.find_camera_planes(sp)
            for Camera_obj in Camera_objs:
                Camera = []
                Camerafound = False
                for obj in bpy.data.objects:
                    if obj.parent == Camera_obj:
                        if obj.name.startswith("Camera_") and obj.type == 'CAMERA':
                            Camerafound = True
                            Camera = obj
                            break
                if Camerafound:  
                    suite_info['Camera'].append(Camera)
              
            
            info.append(suite_info)
        return info
    def find_suite_planes(self):
        suite_planes = []
        for obj in bpy.data.objects:
            if obj.name.startswith("SuitePlane_") and obj.type == 'EMPTY':
                suite_planes.append(obj)
        return suite_planes

    def find_camera_planes(self, Suite_obj):
        camera_planes = []
        for obj in bpy.data.objects:
            if obj.parent == Suite_obj:
                if obj.name.startswith("CameraPlane_") and obj.type == 'EMPTY':
                    camera_planes.append(obj)
        return camera_planes
    def find_lidar_planes(self, Suite_obj):
        lidar_planes = []
        for obj in bpy.data.objects:
            if obj.parent == Suite_obj:
                if obj.name.startswith("LidarPlane_") and obj.type == 'EMPTY':
                    lidar_planes.append(obj)
        return lidar_planes

    def find_radar_planes(self, Suite_obj):
        radar_planes = []
        for obj in bpy.data.objects:
            if obj.parent == Suite_obj:
                if obj.name.startswith("RadarPlane_") and obj.type == 'EMPTY':
                    radar_planes.append(obj)
        return radar_planes

    def find_tx(self, radar_obj):
        tx_planes = []
        for obj in bpy.data.objects:
            if obj.parent == radar_obj:
                if obj.name.startswith("TX_") and obj.type == 'LIGHT':
                    tx_planes.append(obj)
        return tx_planes

    # def find_rx(self, radar_obj):
    #     rx_planes = []
    #     for obj in bpy.data.objects:
    #         if obj.parent == radar_obj:
    #             if obj.name.startswith("RX_") and obj.type == 'CAMERA':
    #                 rx_planes.append(obj)
    #     return rx_planes

    def find_rx(self, radar_obj):
        rx_planes = []
        for obj in bpy.data.objects:
            parent = obj.parent
            check = False
            while parent is not None:
                if parent == radar_obj:
                    check = True
                    break
                parent = parent.parent
            if check:
                if obj.name.startswith("RX_") and obj.type == 'CAMERA':
                    rx_planes.append(obj)
        return rx_planes
    def find_risElement(self, ris_obj):
        risElements = []
        for obj in bpy.data.objects:
            if obj.parent == ris_obj:
                if obj.name.startswith("RIS_Element_") and (obj.type == 'LIGHT'):
                    risElements.append(obj)
        return risElements
    def find_ris_planes(self, Suite_obj):
        ris_planes = []
        for obj in bpy.data.objects:
            if obj.parent == Suite_obj:
                if obj.name.startswith("RISPlane_") and obj.type == 'EMPTY':
                    ris_planes.append(obj)
        return ris_planes

    def find_probe_planes(self, Suite_obj):
        probe_planes = []
        for obj in bpy.data.objects:
            if obj.parent == Suite_obj:
                if obj.name.startswith("ProbePlane_") and obj.type == 'EMPTY':
                    probe_planes.append(obj)
        return probe_planes

    def find_probeSurface(self, probe_obj):
        probe = []
        for obj in bpy.data.objects:
            if obj.parent == probe_obj:
                if obj.name.startswith("Probe_") and obj.type == 'CAMERA':#obj.type == 'MESH':
                    probe.append(obj)
        return probe

