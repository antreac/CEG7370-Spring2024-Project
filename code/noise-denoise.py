import numpy as np 
import cv2
import time 

# read video 
video_path = 'C:/Users/Calvin/Documents/Distributed Computing/Denoising/short_video.mov'
cap = cv2.VideoCapture(video_path)

# error handling 
if not cap.isOpened():
    print("Unable to open the video")
    exit()

# video properties 
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# add noise to the video 
def add_noise(image, scale=0.2):
    noise = np.random.randn(*image.shape).astype(np.uint8) * scale
    noisy_image = cv2.add(image.astype(np.int16), noise.astype(np.int16))
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


# Output video for denoised frames
output_path = 'seq_output_video_denoised.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Output video for noisy frames
noisy_output_path = 'seq_output_video_noisy.mp4'
noisy_out = cv2.VideoWriter(noisy_output_path, fourcc, fps, (width, height))

start_time = time.time()

# process the video
while True:
    ret, frame = cap.read()
    if not ret:
        break 

    # noise
    noisy_frame = add_noise(frame)

    # Write noisy frame to video
    noisy_out.write(noisy_frame)


    denoised_frames = cv2.fastNlMeansDenoisingColored(noisy_frame, 10,10, 7, 21)

    out.write(denoised_frames)

    


end_time = time.time()
total_time = end_time - start_time
print("Total time: {:.2f} seconds".format(total_time))

# Release resources
cap.release()
out.release()
noisy_out.release()
cv2.destroyAllWindows()


# Parallel
# import numpy as np 
# import cv2
# import multiprocessing
# import time 

# # Function to add noise to a frame
# def add_noise(image):
#     mean = 0
#     sigma = 25
#     noisy_image = image.copy()
#     cv2.randn(noisy_image, mean, sigma)
#     noisy_image = cv2.add(image, noisy_image)
#     noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
#     return noisy_image

# # Function to denoise a frame
# def denoise_frame(frame):
#     return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

# # Function to process frames in parallel
# def process_frames(frames):
#     with multiprocessing.Pool(processes=5) as pool:  # Limiting to 5 processes
#         denoised_frames = pool.map(denoise_frame, frames)
#     return denoised_frames



# if __name__ == '__main__':
#     # Read video
#     video_path = 'C:/Users/Calvin/Documents/Distributed Computing/Denoising/short_video.mov'
#     cap = cv2.VideoCapture(video_path)

#     # Check if video is opened successfully
#     if not cap.isOpened():
#         print("Unable to open the video")
#         exit()

#     # Get video properties
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Output video
#     output_path = 'output_video_denoised.mp4'
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Process video frames
#     start_time = time.time()
#     batch_size = 10  # Adjust batch size as needed
#     frames_batch = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Add noise to frame
#         noisy_frame = add_noise(frame)
        
#         # Add frame to the batch
#         frames_batch.append(noisy_frame)

#         # If batch size is reached, process the batch
#         if len(frames_batch) == batch_size:
#             denoised_frames_batch = process_frames(frames_batch)

#             # Write denoised frames to output video
#             for denoised_frame in denoised_frames_batch:
#                 out.write(denoised_frame)

#             # Clear the batch
#             frames_batch = []

#     end_time = time.time()
#     total_time = end_time - start_time
#     print("Total Time: {:.2f} seconds".format(total_time))

#     # Release resources
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
