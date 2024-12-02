import cv2
import numpy as np
def get_video_resolution(standard):
    resolutions = {
        "16K": (15360, 8640),
        "12K": (11520, 6480),
        "10K": (10240, 5760),
        "8K": (7680, 4320),
        "6K": (6144, 3160),
        "5K": (5120, 2880),
        "4K UHD": (3840, 2160),  # Commonly referred to as 4K
        "4K DCI": (4096, 2160),  # Digital Cinema Initiatives standard
        "1440p": (2560, 1440),  # Quad HD (QHD)
        "1080p": (1920, 1080),  # Full HD (FHD)
        "1080i": (1920, 1080),  # Interlaced Full HD
        "720p": (1280, 720),  # HD
        "576p": (720, 576),  # SD for PAL systems
        "480p": (854, 480),  # SD for NTSC systems
        "360p": (640, 360),  # Standard YouTube resolution
        "240p": (426, 240),  # Lower resolution for mobile or low bandwidth
        "144p": (256, 144),  # Minimum resolution for YouTube
        "NTSC": (720, 480),  # Standard definition in the USA
        "PAL": (720, 576),  # Standard definition in Europe and Asia
        "SVGA": (800, 600),  # Super Video Graphics Array
        "XGA": (1024, 768),  # Extended Graphics Array
        "WXGA": (1280, 800),  # Widescreen Extended Graphics Array
        "SXGA": (1280, 1024),  # Super Extended Graphics Array
        "HD+": (1600, 900),  # High Definition Plus
        "UXGA": (1600, 1200),  # Ultra Extended Graphics Array
        "FHD+": (2160, 1440),  # Full HD Plus
        "QHD+": (3200, 1800),  # Quad HD Plus
        "UHD+": (5120, 2880),  # Ultra HD Plus
    }
    
    return resolutions.get(standard, (None, None))
def create_grid_video(video_count, NW, NH, video_directory,fps=30, output_filename="grid_video_4k.avi",
                      output_width = 3840, output_height = 2160):

    # Initialize video readers
    videos = [cv2.VideoCapture(f"{video_directory}/radar_video_{i+1}.avi") for i in range(video_count)]

    # Read the first frame to get video dimensions
    ret, frame0 = videos[0].read()
    # fps = videos[0].get(cv2.CAP_PROP_FPS)
    if not ret:
        print("Failed to read video")
        return
    f0_flag = True
    videos[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_height, frame_width = frame0.shape[:2]

    # Calculate new frame dimensions for grid layout
    tile_width = output_width // NW
    tile_height = output_height // NH

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))
    frameNumer = 0
    while True:
        frameNumer +=1
        # print(f'frameNumer = {frameNumer}')
        grid_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        all_videos_read = True

        for i in range(NH):
            for j in range(NW):
                idx = i * NW + j
                if idx >= video_count:
                    break
                if f0_flag:
                   f0_flag=False
                   ret, frame = True,frame0
                else:
                    ret, frame = videos[idx].read()
                if ret:
                    all_videos_read = False
                    resized_frame = cv2.resize(frame, (tile_width, tile_height))
                    grid_frame[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width] = resized_frame

        if all_videos_read:
            break

        out.write(grid_frame)

    for video in videos:
        video.release()
    out.release()

def addframe_GridVideoWriters(images,videos_WH,videos):
  for i_image,image in enumerate(images):
    image = cv2.resize(image, (videos_WH[i_image][1], videos_WH[i_image][0]))
    videos[i_image].write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def firsttime_init_GridVideoWriters(images,video_directory,fps):
  videos = []
  videos_WH=[]

  for i_image,image in enumerate(images):
    frame_height, frame_width = image.shape[0], image.shape[1]  # Adjust as necessary
    videos_WH.append([frame_height, frame_width])
    #
    videos.append(cv2.VideoWriter(f"{video_directory}/radar_video_{i_image+1}.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height)))
  return videos,videos_WH  

def captureFig(fig):
  fig.canvas.draw()
  image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
  return image.reshape(fig.canvas.get_width_height()[::-1] + (3,))