import bpy
import numpy as np
from mathutils import Vector, Euler
import copy
import sensingsp as ssp

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r
def azel_fromRotMatrix_dir(Mat,Dir):
    inv = Mat.inverted()
    A=Euler(np.radians((90,0,-90)), 'XYZ').to_quaternion()
    dir = inv @ Dir
    dir = A @ dir
    az, el, r = cart2sph(dir.x, dir.y, dir.z)
    return az , el
def dir_from_azel_matrix(azimuth, elevation, matrix):
    x, y, z = ssp.utils.sph2cart(1,azimuth, elevation)  # Assuming unit vector
    
    # Create a vector from Cartesian coordinates
    local_dir = Vector((x, y, z))
    
    # Define the reverse transformation quaternion
    reverse_quaternion = Euler(np.radians((90, 0, -90)), 'XYZ').to_quaternion()
    
    # Apply the reverse transformation
    local_dir = reverse_quaternion.inverted() @ local_dir
    
    # Transform by the rotation matrix
    direction = matrix @ local_dir
    
    return direction

def azel_fromRotMatrix_dir_test(Mat,Dir):
    local_direction = Mat.inverted() @ Dir
    azimuth = np.arctan2(local_direction.y, local_direction.x)
    
    # Calculate elevation (angle from the XY plane)
    hypotenuse = np.sqrt(local_direction.x**2 + local_direction.y**2 + local_direction.z**2)
    elevation = np.arcsin(local_direction.z / hypotenuse)
    return azimuth , elevation

def calculate_reflected_direction( incident_direction, normal):
    incident_direction.normalize()
    normal.normalize()
    reflected_direction = incident_direction - 2 * (normal.dot(incident_direction)) * normal
    return reflected_direction

def calculate_refracted_direction(incident_direction, normal, n1, n2):
    I = incident_direction.normalized()
    N = normal.normalized()
    if I.dot(N) > 0:
        N = -N
    eta = n1 / n2
    cos_theta_i = -I.dot(N)
    k = 1 - eta ** 2 * (1 - cos_theta_i ** 2)
    if k < 0:
        return None
    else:
        cos_theta_t = float(np.sqrt(k))
        return eta * I + (eta * cos_theta_i - cos_theta_t) * N
class SourceType:
    def __init__(self, source_type, ids):
        self.source_type = source_type  # Can be 'TX', 'scatter', 'RIS'
        self.ids = ids  # List of identifiers (e.g., ['isuite', 'iTX'])


    def __repr__(self):
        return f"SourceType(source_type={self.source_type}, ids={self.ids})"

class RaySet:
    def __init__(self, source_type, source_ray=None):
        self.source_type = source_type  # Instance of SourceType
        self.source_ray = source_ray  # Reference to another RaySet object, optional
        # self.directions_powers = []  # List to store (direction, power) tuples
        self.metaInformation = []

    def root(self,Suite_Position):
      if self.source_type.source_type=='scatter':
        v=self.source_type.ids[1]
        return Vector3D(v.x,v.y,v.z)
      if self.source_type.source_type=='TX':
        v=Suite_Position[self.source_type.ids[0]]['Radar'][self.source_type.ids[1]]['TX-Position'][self.source_type.ids[2]]
        return Vector3D(v.x,v.y,v.z)
      if self.source_type.source_type=='RIS':
        v=Suite_Position[self.source_type.ids[0]]['RIS'][self.source_type.ids[1]]['Position'][self.source_type.ids[2]]
        return Vector3D(v.x,v.y,v.z)

    def __repr__(self):
        return f"SourceType(source_type={self.source_type}, metaInformation={self.metaInformation})"

class Path4WavePropagation:
    def __init__(self, transmitter=None, receiver=None, middle_elements=[]):
        self.transmitter = transmitter
        self.receiver = receiver
        self.middle_elements = middle_elements
    def add_middle_element(self, element):
        self.middle_elements.append(element)
    def __repr__(self):
        return f"Path4WavePropagation(TX={self.transmitter}, MiddleElements={self.middle_elements}, RX={self.receiver})"
# rayset1 = RaySet(SourceType("TX", [0,0,0]))
# rayset1.add_direction_power(Vector((1,0,0)), 0.8)
# rayset1.add_direction_power(Vector((0,1,0)), 0.1)

# rayset2 = RaySet(SourceType("Scatter", [110,20]),rayset1)
# rayset1.add_direction_power(Vector((0,0,1)), 0.2)

# print(rayset2)
class RayTracingFunctions:
  # def __init__(self):
        # self.ScatterReflection = [5,10] 
  def check_line_of_sight(self,start_point, end_point, depsgraph):
      direction = (end_point - start_point).normalized()
      distance = (end_point - start_point).length
      result, location, normal, face_index, hit_obj, matrix = bpy.context.scene.ray_cast(depsgraph, start_point, direction)
      # print((location - start_point).length , distance)
      # 2.672077784734505 2.672077807040983
      d2=(location - start_point).length
      mustbeP = abs(distance - d2 )
     
      #     return False : An object is blocking the line of sight, 
      if result: # and  mustbeP>ssp.config.EpsilonDistanceforHitTest: 
          if d2+ssp.config.EpsilonDistanceforHitTest<distance:
            if ( location - end_point ).length<ssp.config.EpsilonDistanceforHitTest:
              return True
            return False
          # if ( location - end_point ).length<ssp.config.EpsilonDistanceforHitTest:
          #   return True
          # return False  # An object is blocking the line of sight
      return True  # No object is blocking the line of sight

  def check_line_of_sight_checkID(self,start_point, end_point_hash, depsgraph):
      direction = (end_point_hash[0] - start_point).normalized()
      result, location, normal, face_index, hit_obj, matrix = bpy.context.scene.ray_cast(depsgraph, start_point, direction)
      if result and hash(hit_obj) == end_point_hash[1] and face_index == end_point_hash[2]:
        return True
      return False
  def check_line_of_sight_ID(self,start_point, end_point_hash, depsgraph,epsilon = 1e-4):
      direction = (end_point_hash[0] - start_point).normalized()
      result, location, normal, face_index, hit_obj, matrix = bpy.context.scene.ray_cast(depsgraph, start_point, direction)
      if result and hash(hit_obj) == end_point_hash[1] and face_index == end_point_hash[2]:
        reflected_direction = calculate_reflected_direction(direction, normal)
        start_reflected_point = location + reflected_direction * epsilon
        return True, reflected_direction, location, start_reflected_point
      return False, [], [],[]
  def LOS_RX_Target_RIS(self,startPoint,Suite_Position,isuite,iradar,ScattersGeo,depsgraph):
      o={'Target':[],'RX':[],'RIS':[]}
      # print("Check 24 7: 24 ==",len(ScattersGeo))
      for itarget,target in enumerate(ScattersGeo):
        if self.check_line_of_sight(startPoint, target[0], depsgraph):
          o['Target'].append([itarget,target[1],target[2]])
      #   else:
      #     print("Why ", startPoint, target[0])
      # print("Check 2 24 7: 24 ==",len(o['Target']))
      for iris , ris in enumerate(Suite_Position[isuite]['RIS']): # can be for all
        for iriselement ,ris_element_position in enumerate(ris['Position']):
          if self.check_line_of_sight(startPoint, ris_element_position, depsgraph):
            o['RIS'].append([isuite,iris,iriselement])
      RadarRX_only_fromitsTX = ssp.config.RadarRX_only_fromitsTX
      if RadarRX_only_fromitsTX:
        for irx, rx in enumerate(Suite_Position[isuite]['Radar'][iradar]['RX-Position']): # can be for all
          if self.check_line_of_sight(startPoint, rx, depsgraph):
            o['RX'].append([isuite,iradar,irx])
      else:
        for isuite0 in range(len(Suite_Position)):
          for iradar0 in range(len(Suite_Position[isuite0]['Radar'])):
            Radar_TX_RX_isolation=ssp.config.Radar_TX_RX_isolation
            if Radar_TX_RX_isolation==0: 
              for irx, rx in enumerate(Suite_Position[isuite0]['Radar'][iradar0]['RX-Position']): 
                if self.check_line_of_sight(startPoint, rx, depsgraph):
                  o['RX'].append([isuite0,iradar0,irx])
            else: # Radar TX RX isolation
              if not(isuite0==isuite and iradar0==iradar):
                for irx, rx in enumerate(Suite_Position[isuite0]['Radar'][iradar0]['RX-Position']): 
                  if self.check_line_of_sight(startPoint, rx, depsgraph):
                    o['RX'].append([isuite0,iradar0,irx])
      return o

  def LOS_RX_Target_RIS_4ris(self,startPoint,Suite_Position,isuite,iris_inp,ScattersGeo,depsgraph):
      o={'Target':[],'RX':[],'RIS':[]}
      for itarget,target in enumerate(ScattersGeo):
        if self.check_line_of_sight(startPoint, target[0], depsgraph):
          o['Target'].append([itarget,target[1],target[2]])
      for iris , ris in enumerate(Suite_Position[isuite]['RIS']):
        if iris == iris_inp:
          continue
        for iriselement ,ris_element_position in enumerate(ris['Position']):
          if self.check_line_of_sight(startPoint, ris_element_position, depsgraph):
            o['RIS'].append([isuite,iris,iriselement])
      for iradar in range(len(Suite_Position[isuite]['Radar'])):
        for irx, rx in enumerate(Suite_Position[isuite]['Radar'][iradar]['RX-Position']): 
          if self.check_line_of_sight(startPoint, rx, depsgraph):
            o['RX'].append([isuite,iradar,irx])
      return o

  def LOS_RX_RIS_4scatter(self,startPoint,Suite_Position,isuite,iradar,depsgraph):
      o={'RX':[],'RIS':[]}
      # print("Moein",isuite)
      for iris , ris in enumerate(Suite_Position[isuite]['RIS']):
        for iriselement ,ris_element_position in enumerate(ris['Position']):
          if self.check_line_of_sight(startPoint, ris_element_position, depsgraph):
            o['RIS'].append([isuite,iris,iriselement])
      RadarRX_only_fromscatters_itsTX = ssp.config.RadarRX_only_fromscatters_itsTX 
      if RadarRX_only_fromscatters_itsTX:
        for irx, rx in enumerate(Suite_Position[isuite]['Radar'][iradar]['RX-Position']): # can be for all
          if self.check_line_of_sight(startPoint, rx, depsgraph):
            o['RX'].append([isuite,iradar,irx])
      else:
        for isuite0 in range(len(Suite_Position)):
          for iradar0 in range(len(Suite_Position[isuite0]['Radar'])):
            for irx, rx in enumerate(Suite_Position[isuite0]['Radar'][iradar0]['RX-Position']): 
              if self.check_line_of_sight(startPoint, rx, depsgraph):
                o['RX'].append([isuite0,iradar0,irx])
            
      return o

  def random_normalized_vectors_around_direction(self,reflected_direction, deviation_angle_degrees, num_vectors):
      # Convert reflected_direction to numpy array and normalize
      base_direction = np.array(reflected_direction)
      base_direction /= np.linalg.norm(base_direction)

      # Convert deviation angle to radians
      deviation_angle = np.radians(deviation_angle_degrees)

      # Create orthogonal basis
      if np.abs(base_direction[2]) < 0.9:
          ortho_vector = np.array([0, 0, 1])
      else:
          ortho_vector = np.array([1, 0, 0])

      orthogonal1 = np.cross(base_direction, ortho_vector)
      orthogonal1 /= np.linalg.norm(orthogonal1)
      orthogonal2 = np.cross(base_direction, orthogonal1)

      # Generate random vectors around the base direction
      random_vectors = []
      for _ in range(num_vectors):
          angle1 = np.random.uniform(-deviation_angle, deviation_angle)
          angle2 = np.random.uniform(0, 2 * np.pi)
          deviation = (np.cos(angle1) * base_direction +
                      np.sin(angle1) * (np.cos(angle2) * orthogonal1 + np.sin(angle2) * orthogonal2))
          random_vector = Vector(deviation)
          random_vectors.append(random_vector.normalized())

      return random_vectors
  def rayset_gen_TX(self,itx,Suite_Position,isuite,iradar,ScattersGeo,depsgraph):
    tx = Suite_Position[isuite]['Radar'][iradar]['TX-Position'][itx]
    RX_Target_RIS = self.LOS_RX_Target_RIS(tx,Suite_Position,isuite,iradar,ScattersGeo,depsgraph)
    rayset = RaySet(SourceType("TX", [isuite,iradar,itx]))
    rayset.metaInformation={'RX':[],'RIS':[],'Target':[]}
    DirectRX = 1
    if DirectRX:
      rayset.metaInformation['RX'] = RX_Target_RIS['RX']
    else:
      rayset.metaInformation['RX'] = []
    rayset.metaInformation['RIS'] = RX_Target_RIS['RIS']
    for targetLOS in RX_Target_RIS['Target']:
        txdir = Suite_Position[isuite]['Radar'][iradar]['TX-Direction'][itx]
        dir = ScattersGeo[targetLOS[0]][0]-tx
        az,el = azel_fromRotMatrix_dir(txdir,dir)
        backLobe_Block = False
        if backLobe_Block :
          if az > np.pi/2:
            continue
          if az < -np.pi/2:
            continue

        result, reflected_direction, hit_point, start_point = self.check_line_of_sight_ID(tx, ScattersGeo[targetLOS[0]], depsgraph)
        if result==False:
          continue

        deviation_angle_degrees = ScattersGeo[targetLOS[0]][7] #self.ScatterReflection[1] 
        num_vectors = ScattersGeo[targetLOS[0]][6] #self.ScatterReflection[0]
        # deviation_angle_degrees = 0
        # num_vectors = 1
        random_vectors = self.random_normalized_vectors_around_direction(reflected_direction, deviation_angle_degrees, num_vectors)
        # print(random_vectors)
        epsilon = 1e-4
        start_reflected_points=[]
        for random_vector in random_vectors:
          start_reflected_point = hit_point + random_vector * epsilon
          start_reflected_points.append(start_reflected_point)
        rayset.metaInformation['Target'].append([targetLOS,hit_point,start_reflected_points,random_vectors])# be kodom scatter mikhore, kojash, chejori barmigarde
    return rayset

  def rayset_gen_RIS(self,irisElement,Suite_Position,isuite,iris,ScattersGeo,depsgraph):
    risPos = Suite_Position[isuite]['RIS'][iris]['Position'][irisElement]

    RX_Target_RIS = self.LOS_RX_Target_RIS_4ris(risPos,Suite_Position,isuite,iris,ScattersGeo,depsgraph)
    rayset = RaySet(SourceType("RIS", [isuite,iris,irisElement]))
    rayset.metaInformation={'RX':[],'RIS':[],'Target':[]}
    rayset.metaInformation['RX'] = RX_Target_RIS['RX']
    rayset.metaInformation['RIS'] = RX_Target_RIS['RIS']
    for targetLOS in RX_Target_RIS['Target']:
        result, reflected_direction, hit_point, start_point = self.check_line_of_sight_ID(risPos, ScattersGeo[targetLOS[0]], depsgraph)
        if result==False:
          continue
        deviation_angle_degrees = 0
        num_vectors = 1
        random_vectors = self.random_normalized_vectors_around_direction(reflected_direction, deviation_angle_degrees, num_vectors)
        epsilon = 1e-4
        start_reflected_points=[]
        for random_vector in random_vectors:
          start_reflected_point = hit_point + random_vector * epsilon
          start_reflected_points.append(start_reflected_point)
        rayset.metaInformation['Target'].append([targetLOS,hit_point,start_reflected_points,random_vectors])# be kodom scatter mikhore, kojash, chejori barmigarde
    return rayset
  def rayset_gen_Scatter(self,ScatterIndex_HitP_SreflectionPoint_ReflectedDirectionVectors,Suite_Position,isuite,iradar,depsgraph):
    rayset = RaySet(SourceType("scatter", [ScatterIndex_HitP_SreflectionPoint_ReflectedDirectionVectors[0],ScatterIndex_HitP_SreflectionPoint_ReflectedDirectionVectors[1]]))

    rayset.metaInformation={'RX':[],'RIS':[],'Target':[]}
    # print(ScatterIndex_HitP_SreflectionPoint_ReflectedDirectionVectors)
    # print("Moein",isuite)
    RX_Target_RIS = self.LOS_RX_RIS_4scatter(ScatterIndex_HitP_SreflectionPoint_ReflectedDirectionVectors[2][0],Suite_Position,isuite,iradar,depsgraph)
    rayset.metaInformation['RX'] = RX_Target_RIS['RX']
    rayset.metaInformation['RIS'] = RX_Target_RIS['RIS']
    # print( ScatterIndex_HitP_SreflectionPoint_ReflectedDirectionVectors[3])
    for i,direction in enumerate(ScatterIndex_HitP_SreflectionPoint_ReflectedDirectionVectors[3]):
      # print(ScatterIndex_HitP_SreflectionPoint_ReflectedDirectionVectors[2], direction)
      sp = ScatterIndex_HitP_SreflectionPoint_ReflectedDirectionVectors[2][i]
      result, location, normal, face_index, hit_obj, matrix = bpy.context.scene.ray_cast(depsgraph, sp, direction)
      if result:
        reflected_direction = calculate_reflected_direction(direction, normal)
        if 0:
          mesh = hit_obj.data
          normal_obj = mesh.polygons[face_index].normal
          normal_world = hit_obj.matrix_world.to_3x3() @ normal_obj
          normal_world.normalize()
          dot_value = normal.dot(direction)
          if dot_value > 0:
              print("Direction is on the same side as the normal (points away from the face).")
          elif dot_value < 0:
              print("Direction is against the normal (points into the face).")
          else:
              print("Direction is perpendicular to the normal.")
          # refracted_direction = calculate_refracted_direction(direction, normal, n1, n2)
      
        num_vectors=1
        if "Backscatter N" in hit_obj:
            num_vectors = hit_obj["Backscatter N"]
        deviation_angle_degrees=0
        if "Backscatter Dev (deg)" in hit_obj:
            deviation_angle_degrees = hit_obj["Backscatter Dev (deg)"]
        
        random_vectors = self.random_normalized_vectors_around_direction(reflected_direction, deviation_angle_degrees, num_vectors)
        epsilon = ssp.config.RayTracing_ReflectionPointEpsilon
        start_reflected_points=[]
        for random_vector in random_vectors:
          start_reflected_point = location + random_vector * epsilon
          start_reflected_points.append(start_reflected_point)
        rayset.metaInformation['Target'].append([[-1,hash(hit_obj),face_index],location,start_reflected_points,random_vectors])# be kodom scatter mikhore, kojash, chejori barmigarde
        # rayset.metaInformation['Target'].append([location,hit_obj,face_index])
    return rayset


  def RXPath(self,rayset):
    paths=[]
    for isirirx in rayset.metaInformation['RX']:
      path = Path4WavePropagation(None,None,[])
      path.receiver = isirirx
      # print(rayset)
      rayset0 = copy.deepcopy(rayset)
      # rayset0 = rayset
      # print("________")
      while 1:
        if rayset0.source_type.source_type=='TX':
          path.transmitter = rayset0.source_type.ids
          break
        # print(rayset0.source_type)
        path.middle_elements.append(rayset0.source_type)
        rayset0 = rayset0.source_ray

      # print(path)
      # print(len(path.middle_elements))

      paths.append(path)
      # print(paths[0])
      # print(len(paths[0].middle_elements))
      # print('...........')
    return paths

# rayTracingFunctions = RayTracingFunctions()

# # Example usage
# reflected_direction = Vector((1, 0, 0))
# deviation_angle_degrees = 0
# num_vectors = 10
# random_vectors = rayTracingFunctions.random_normalized_vectors_around_direction(reflected_direction, deviation_angle_degrees, num_vectors)

# # Print the random vectors
# for i, vec in enumerate(random_vectors):
#     print(f"Vector {i+1}: {vec}:{vec.length}")

