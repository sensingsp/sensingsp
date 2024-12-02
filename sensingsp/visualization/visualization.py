import numpy as np
from matplotlib import pyplot as plt
import bpy
# import matplotlib
# matplotlib.use('qtagg')
import sensingsp as ssp
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
  Channel_d_fd_amp=np.array([[float(d),float(dr),float(a)] for d,dr,a,m in path_d_drate_amp[0][0][0][0][0][0]])
  if option == 0:
    fig, ax = plt.subplots(1,1)
    ax = fig.add_subplot(1, 1, 1 , projection='3d')
    ax.scatter(Channel_d_fd_amp[:,0],Channel_d_fd_amp[:,1],Channel_d_fd_amp[:,2])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance Rate (m/s)') 
    ax.set_zlabel('Amplitude (v)')
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
