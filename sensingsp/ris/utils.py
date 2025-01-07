import bpy
import numpy as np
from mathutils import Vector
from scipy.io import loadmat

def add_ris(isuite=0, iris=0, location=Vector((0, 0, 0)), rotation=Vector((np.pi/2, 0, -np.pi/2)), f0=70e9, N1=8, N2=8,
              passive_active='passive',steeringPhase_az_el=[0,0]):
    fromFile = False
    if "Simulation Settings" in bpy.data.objects:
        option = bpy.data.objects["Simulation Settings"]["Add Ris"]
        parts = option.split(',')
        # Convert parts to appropriate types
        input_or_File = int(parts[0])
        if input_or_File == 1:
            N1 = int(parts[1])
            N2 = int(parts[2])
        if input_or_File == 2:
            file = parts[3]
            data = loadmat(file)
            coordinates = data["P"]
            amplitude  = data["amp"]
            phase = data["phase"]
            fromFile = True
    Lambda = 3e8 / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RISPlane_{isuite}_{iris}'
    empty["Center_Frequency_GHz"] = f0/1e9
    empty["passive active"] = passive_active
    empty["steeringPhase_az_el"] = steeringPhase_az_el
    # empty["moein_boolean"] = True
    empty.parent = Suite_obj

    s = 0.05
    Type = 'AREA'
    
    if fromFile:
        for i in range(coordinates.shape[0]):
            bpy.ops.object.light_add(type=Type, radius=1, location=(coordinates[i,0],coordinates[i,1],coordinates[i,2]))
            tx = bpy.context.object
            tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
            NF = 80
            tx.name = f'RIS_Element_{isuite}_{iris}_{i}_{0}_{NF}'
            tx['amplitude']=amplitude[0,i]/1.0
            tx['phase']=phase[0,i]/1.0
            tx.parent = empty

    else:
        for i1 in range(N1):
            for i2 in range(N2):
                bpy.ops.object.light_add(type=Type, radius=1, location=(i1*Lambda/2, i2*Lambda/2, 0))
                tx = bpy.context.object
                tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
                NF = 80
                tx.name = f'RIS_Element_{isuite}_{iris}_{i1}_{i2}_{NF}'
                tx['amplitude']=1
                tx['phase']=0
                tx.parent = empty

def add_ris_mat(isuite=0, iris=0, location=Vector((0, 0, 0)), rotation=Vector((np.pi/2, 0, -np.pi/2)), f0=70e9, pos=np.array([]),passive_active='passive'):
    Lambda = 3e8 / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RISPlane_{isuite}_{iris}'
    empty["Center_Frequency_GHz"] = 70
    empty["passive active"] = passive_active
    empty["steeringPhase_az_el"] = []
    empty.parent = Suite_obj

    s = 0.05
    Type = 'AREA'
    N1=pos.shape[0]
    for i1 in range(N1):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[i1,0], pos[i1,1], 0))
        # bpy.ops.object.empty_add(type='PLAIN_AXES', location=(pos[i1, 0], pos[i1, 1], 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        NF = 80
        tx.name = f'RIS_Element_{isuite}_{iris}_{i1}_{0}_{NF}'
        tx['amplitude']=pos[i1,4]
        tx['phase']=pos[i1,3]
        tx.parent = empty
