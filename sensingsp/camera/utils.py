import bpy
import numpy as np
from mathutils import Vector
from PIL import Image
import tempfile
import os

def add_camera(isuite, icamera, location, rotation,lens = 10,resolution=[1920,1080]):
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'CameraPlane_{isuite}_{icamera}_{0}'
    empty.parent = Suite_obj
    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(0, 0, 0))
    rx = bpy.context.object
    rx.name = f'Camera_{isuite}_{icamera}_{1:05}'
    rx.parent = empty
    rx.data.lens = lens
    rx["Resolution x"]=resolution[0]
    rx["Resolution y"]=resolution[1]
    rx["Resolution percentage"]=100


def render(cameraObject):
    bpy.context.scene.camera = cameraObject
    bpy.context.scene.render.resolution_x = cameraObject["Resolution x"]
    bpy.context.scene.render.resolution_y = cameraObject["Resolution y"]
    bpy.context.scene.render.resolution_percentage = cameraObject["Resolution percentage"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, "rendered_image.png")
        bpy.context.scene.render.filepath = temp_filepath
        bpy.context.scene.world.use_nodes = True
        bg = bpy.context.scene.world.node_tree.nodes['Background']
        bg.inputs['Color'].default_value = (1, 1, 1, 1)  # RGB color and alpha
        # Perform the rendering
        bpy.ops.render.render(write_still=True)
        
        # Open the rendered image and convert it to a NumPy array
        image = Image.open(temp_filepath)
        image_np = np.array(image)
    
    # bpy.context.scene.render.filepath = f"{render_imade_dir}/rendered_image.png"
    # bpy.ops.render.render(write_still=True)
    # image = Image.open(bpy.context.scene.render.filepath)
    # image_np = np.array(image)
    return image_np