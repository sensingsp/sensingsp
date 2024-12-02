import bpy

# Function to create a material with the car texture
def create_car_material(image_path):
    # Create a new material
    mat = bpy.data.materials.new(name="CarMaterial")
    mat.use_nodes = True
    
    # Get the material's node tree
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    
    # Load the texture
    tex_image.image = bpy.data.images.load(image_path)
    
    # Connect the texture to the Base Color of the shader
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    
    return mat

# Function to create a wheel
def create_wheel(location):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=0.2, location=location)
    wheel = bpy.context.object
    wheel.rotation_euler = (1.5708, 0, 0)
    bpy.ops.object.shade_smooth()
    return wheel

# Function to create the car body
def create_body(location):
    bpy.ops.mesh.primitive_cube_add(size=2.5, location=location)
    body = bpy.context.object
    body.scale[2] = 0.5
    body.scale[1] = 1.0
    body.scale[0] = 2.0
    return body

# Function to create the car roof
def create_roof(location):
    bpy.ops.mesh.primitive_cube_add(size=1.5, location=location)
    roof = bpy.context.object
    roof.scale[2] = 0.5
    roof.scale[1] = 0.7
    roof.scale[0] = 1.2
    return roof

# Function to create the entire car
def create_car(location=(0, 0, 0), material=None):
    # Create car body
    body = create_body((location[0], location[1], location[2] + 1))
    
    # Apply material to the body
    if material:
        if body.data.materials:
            body.data.materials[0] = material
        else:
            body.data.materials.append(material)
    
    # Create car roof
    roof = create_roof((location[0], location[1], location[2] + 2))
    
    # Apply material to the roof
    if material:
        if roof.data.materials:
            roof.data.materials[0] = material
        else:
            roof.data.materials.append(material)
    
    # Create wheels
    wheel1 = create_wheel((location[0] + 1.5, location[1] + 1.2, location[2] + 0.5))
    wheel2 = create_wheel((location[0] + 1.5, location[1] - 1.2, location[2] + 0.5))
    wheel3 = create_wheel((location[0] - 1.5, location[1] + 1.2, location[2] + 0.5))
    wheel4 = create_wheel((location[0] - 1.5, location[1] - 1.2, location[2] + 0.5))
    
    # Combine all parts into a single object
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    roof.select_set(True)
    wheel1.select_set(True)
    wheel2.select_set(True)
    wheel3.select_set(True)
    wheel4.select_set(True)
    bpy.ops.object.join()
    
    # Move the entire car to the specified location (optional)
    bpy.context.object.location = location
    
    # Set the name of the car object
    bpy.context.object.name = "SimpleCar"
    
    # Center the origin of the car
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
    
    # Return the car object
    return bpy.context.object

# # Path to your car texture image
# image_path = "/path/to/your/car_texture_image.jpg"

# # Create the car material
# car_material = create_car_material(image_path)

# # Example usage: create a car at the origin with the car material
# car = create_car(location=(0, 0, 0), material=car_material)
