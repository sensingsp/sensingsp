import numpy as np
from matplotlib import pyplot as plt
import bpy
from mathutils import Vector

# import matplotlib
# matplotlib.use('qtagg')
import sensingsp as ssp

def plot_tiangles(Triangles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
       
    for obj in Triangles:
        for triangle in obj:
            x,y,z=[],[],[]
            x.append(triangle[0][0])
            x.append(triangle[1][0])
            x.append(triangle[2][0])
            x.append(triangle[0][0])
            y.append(triangle[0][1])
            y.append(triangle[1][1])
            y.append(triangle[2][1])
            y.append(triangle[0][1])
            z.append(triangle[0][2])
            z.append(triangle[1][2])
            z.append(triangle[2][2])
            z.append(triangle[0][2])
            # if k>100:
            #     break
            ax.plot(x,y,z,'-k')
        
    
    
    return fig,ax



def visualize_scenario():
  Triangles = ssp.utils.exportBlenderTriangles()
  Suite_Position,ScattersGeo = ssp.utils.export_radar_positions()
  fig,ax = plot_tiangles(Triangles)
  for isuite,suite in enumerate(Suite_Position):
    for iradar,radar in enumerate(suite['Radar']):
      for itx,tx in enumerate(radar['TX-Position']):
        ax.scatter(tx[0],tx[1],tx[2],c='r',marker='x')
      for irx,rx in enumerate(radar['RX-Position']):
          ax.scatter(rx[0],rx[1],rx[2],c='b',marker='x')
  
  ax.set_aspect('equal', 'box')
  plt.show()
  
  

def visualizeProcessingOutputs(ProcessingOutputs):
  
  grid_points , grid_velocities , all_outputs = ProcessingOutputs
  plotScatters = 1

  fig = plt.figure()
  plt.plot(all_outputs,'.')
  all_outputs = np.array(all_outputs)
  fig = plt.figure()

  plt.hist(all_outputs, bins=30, edgecolor='black', alpha=0.7)
  plt.xlabel('Output Value')
  plt.ylabel('Frequency')
  plt.title('Histogram of Detector on Grids')
  plt.grid(True)
  # plt.show()

  max_output = np.max(all_outputs)
  threshold = 0.3 * max_output
  # threshold = .1 /5
  filtered_indices = np.where(all_outputs > threshold)[0]
  filtered_points = grid_points[filtered_indices]
  filtered_outputs = all_outputs[filtered_indices]
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  sc = ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], c=filtered_outputs, cmap='viridis',s=2)
  if plotScatters:
    for __ in ssp.lastScatterInfo:
      _=__[0]
      ax.scatter(_[0],_[1],_[2],c='k',marker='x',s=20)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  sc = ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], c=filtered_outputs, cmap='viridis',s=2)
  if plotScatters:
    for __ in ssp.lastScatterInfo:
      _=__[0]
      ax.scatter(_[0],_[1],_[2],c='k',marker='x',s=20)
  ax.view_init(elev=0, azim=45)
  for i in range(4):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], c=filtered_outputs, cmap='viridis',s=1)
    if plotScatters:
      for __ in ssp.lastScatterInfo:
        _=__[0]
        ax.scatter(_[0],_[1],_[2],c='k',marker='x',s=20)
    for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      for iradar,specifications in enumerate(radarSpecifications):
        global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
        for _ in global_location_TX:
          ax.scatter(_[0],_[1],_[2],c='r',marker='x')
        for _ in global_location_RX:
          ax.scatter(_[0],_[1],_[2],c='b',marker='x')
    plt.colorbar(sc)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if i==1:
      ax.view_init(elev=30, azim=45)
    if i==2:
      ax.view_init(elev=0, azim=45)
    if i==3:
      ax.view_init(elev=90, azim=45)
    # plt.show()

  for i in range(len(ssp.RadarSpecifications)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    radarSpecifications=ssp.RadarSpecifications[i]
    for iradar,specifications in enumerate(radarSpecifications):
      global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
      for _ in global_location_TX:
        ax.scatter(_[0],_[1],_[2],c='r',marker='x')
      for _ in global_location_RX:
        ax.scatter(_[0],_[1],_[2],c='b',marker='x')
  
  plt.show()

def visualize_radar_path_d_drate_amp(path_d_drate_amp, option = 0):
  Channel_d_fd_amp=np.array([[float(d),float(dr),float(a),len(m)] for d,dr,a,m in path_d_drate_amp[0][0][0][0][0][0]])
  if option == 0:
    fig, ax = plt.subplots(1,1)
    ax = fig.add_subplot(1, 1, 1 , projection='3d')
    ax.scatter(Channel_d_fd_amp[:,0],Channel_d_fd_amp[:,1],20*np.log10(Channel_d_fd_amp[:,2]))
    ind = np.where(Channel_d_fd_amp[:,3]>1)
    ax.scatter(Channel_d_fd_amp[ind,0],Channel_d_fd_amp[ind,1],20*np.log10(Channel_d_fd_amp[ind,2]),color='red')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance Rate (m/s)') 
    ax.set_zlabel('Amplitude (dB)')
    # ax.set_box_aspect([1,1,1])  
    plt.show()
    # plt.draw()
    # plt.pause(.1)
  elif option == 1:
    return Channel_d_fd_amp
  
def plot_continuous_curve(vectors, parent_empty=None, curve_name="Curve"):
    curve_data = bpy.data.curves.new(name=curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(vectors) - 1)  # Add additional points (already has one by default)

    # Set the spline points to the given vectors
    for i, vec in enumerate(vectors):
        polyline.points[i].co = (*vec, 1)
    curve_obj = bpy.data.objects.new(curve_name, curve_data)
    bpy.context.collection.objects.link(curve_obj)
    if parent_empty:
        curve_obj.parent = parent_empty

    return curve_obj


def plot_pattern(pat, az, el, patax):
    M = np.max(pat)
    MaxMindB = 20
    m = M / 10**(MaxMindB / 10)
    
    pat[pat < m] = m
    pat = 10 * np.log10(pat)  # Convert to dB
    pat -= np.min(pat)        # Normalize to 0 dB max
    if np.max(pat)>0:
      pat /= np.abs(np.max(pat))        # Normalize to 0 dB max

    g = 2

    for i in range(az.shape[0]):
        v=[]
        for j in range(az.shape[1]):
            x, y, z = ssp.utils.sph2cart(g*pat[i,j], az[i,j], el[i,j])
            v.append(Vector((x, y, z)))
        ray = ssp.visualization.plot_continuous_curve(v, patax)
        ray["Max"] = M
    for j in range(az.shape[1]):
        v=[]
        for i in range(az.shape[0]):
            x, y, z = ssp.utils.sph2cart(g*pat[i,j], az[i,j], el[i,j])
            v.append(Vector((x, y, z)))
        ray = ssp.visualization.plot_continuous_curve(v, patax)
        ray["Max"] = M

def plot_pattern_button():
  r = bpy.context.object
  suite_information = ssp.environment.BlenderSuiteFinder().find_suite_information()
  
  geocalculator = ssp.raytracing.BlenderGeometry()
  Suite_Position,ScattersGeo,HashFaceIndex_ScattersGeo,ScattersGeoV = geocalculator.get_Position_Velocity(bpy.context.scene, suite_information, bpy.context.scene.frame_current, 1)
  r=suite_information[0]['Radar'][0]['GeneralRadarSpec_Object']
  for i,suite in enumerate(suite_information):
    for j,radar in enumerate(suite['Radar']):
      if r == radar['GeneralRadarSpec_Object']:
        _radar = Suite_Position[i]['Radar'][j]
        if 'TX-PatternType' in _radar:
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
            patax = bpy.context.object
            patax.name = f'Plot Pattern'
            patax.location = r.location
            itx=0
            az0 = np.linspace(-np.pi, np.pi, 120)
            el0 = np.linspace(-np.pi/2,  np.pi/2, 100)
            az, el = np.meshgrid(az0, el0)
            pat=np.zeros_like(az)
            for i in range(az.shape[0]):
                for j in range(az.shape[1]):
                        x,y,z=ssp.utils.sph2cart(1,az[i,j],el[i,j])
                        dir = Vector((x,y,z))
                        gain = ssp.raytracing.antenna_gain(_radar['TX-PatternType'][itx],_radar['TX-Direction'][itx],_radar['TX-PatternMaxGain'][itx],
                                                           _radar['TX-PatternAzimuthBeamWidth'][itx],
                                                           _radar['TX-PatternElevationBeamWidth'][itx],
                                                           _radar['TX-PatternType'][itx],dir)
                        pat[i,j]=gain
        
            plot_pattern(pat, az, el, patax)