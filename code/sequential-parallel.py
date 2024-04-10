# Sequential Code 

# import cv2
# import os
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import time 

# # Function to extract features from frames (adapt as needed)
# def extract_features(frame):
#     # Example: Color histogram as a feature vector
#     hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     return cv2.normalize(hist, None).flatten()

# # Folder path containing frames
# folder_path = r"C:\Users\Calvin\Documents\CEG7370-Spring2024-Project\data"

# start_time = time.time()

# # Load frames and extract features
# frames = []
# features = []
# for filename in os.listdir(folder_path):
#     filepath = os.path.join(folder_path, filename)
#     if filename.endswith((".jpg", ".png")):
#         try:
#             frame = cv2.imread(filepath)
#             frames.append(frame)
#             features.append(extract_features(frame))
#         except Exception as e:
#             print(f"Error processing frame {filename}: {e}")

# # Number of clusters (adjust as needed)
# num_clusters = 5

# # Perform K-Means clustering
# kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# kmeans.fit(features)
# frame_labels = kmeans.labels_

# # Group frames by cluster
# frame_groups = {}
# for i, label in enumerate(frame_labels):
#     frame_groups.setdefault(label, []).append(frames[i])

# # Select representative frames (keyframes) from each cluster for video summarization
# representative_frames = []
# for label, group_frames in frame_groups.items():
#     # Check if there are frames in the cluster
#     if len(group_frames) > 0:
#         # Compute the centroid of the cluster in feature space
#         cluster_features = [features[i] for i, f in enumerate(group_frames) if frame_labels[i] == label]
#         cluster_centroid = np.mean(cluster_features, axis=0)

#         # Find the index of the frame closest to the centroid
#         closest_frame_idx = min(range(len(group_frames)), key=lambda i: np.linalg.norm(features[i] - cluster_centroid))

#         # Add the closest frame to the representative frames list
#         representative_frames.append(group_frames[closest_frame_idx])

# # Zip representative_frames with their corresponding cluster labels
# zipped_frames_labels = zip(representative_frames, frame_labels)

# # Sort based on the cluster labels
# sorted_frames_labels = sorted(zipped_frames_labels, key=lambda x: x[1])

# # Extract the sorted frames
# sorted_representative_frames = [frame_label[0] for frame_label in sorted_frames_labels]
# # Save the paths to the keyframe images for summarization
# keyframe_paths = []
# for i, frame in enumerate(representative_frames):
#     keyframe_path = f"keyframe_{i+1}.jpg"
#     cv2.imwrite(keyframe_path, frame)  # Save keyframe as an image
#     keyframe_paths.append(keyframe_path)
#     # Display and save the representative frames as keyframes for video summarization
#     # cv2.imshow(f"Keyframe {i+1}", frame)
#     # cv2.waitKey(0)  # Wait for a key press before closing each window

# cv2.destroyAllWindows()

# # Display the summary of the video
# print("Video Summary:")
# for i, frame_path in enumerate(keyframe_paths, start=1):
#     print(f"Keyframe {i}: {frame_path}")
#     # Add additional information if needed, such as cluster number, color histogram, etc.

# end_time = time.time()
# total_time = end_time - start_time
# print(f"Totale Sequential Code Time: {total_time}")

###################################################
# Parallel Code
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import multiprocessing
import time

# Function to extract features from frames (adapt as needed)
def extract_features(frame):
    # Example: Color histogram as a feature vector
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, None).flatten()

# Function to process frames in parallel
def process_frame(filepath):
    try:
        frame = cv2.imread(filepath)
        features = extract_features(frame)
        return features
    except Exception as e:
        print(f"Error processing frame {filepath}: {e}")
        return None

if __name__ == "__main__":
    # Folder path containing frames
    folder_path = r"C:\Users\Calvin\Documents\CEG7370-Spring2024-Project\data"
    
    start_time = time.time()

    # Get list of frame file paths
    frame_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
                   if filename.endswith((".jpg", ".png"))]

    # Number of clusters (adjust as needed)
    num_clusters = 5

    # Process frames in parallel
    with multiprocessing.Pool(processes=5) as pool:  # Limiting to 5 CPU cores
        features = pool.map(process_frame, frame_paths)

    # Remove None values (failed processing)
    features = [feature for feature in features if feature is not None]

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(features)
    frame_labels = kmeans.labels_

    # Group frames by cluster
    frame_groups = {}
    for filepath, label in zip(frame_paths, frame_labels):
        frame_groups.setdefault(label, []).append(filepath)

    # Select representative frames (keyframes) from each cluster for video summarization
    representative_frames = []
    for group_filepaths in frame_groups.values():
        # Compute the centroid of the cluster in feature space
        cluster_features = [features[frame_paths.index(filepath)] for filepath in group_filepaths]
        cluster_centroid = np.mean(cluster_features, axis=0)

        # Find the index of the frame closest to the centroid
        closest_frame_idx = min(range(len(group_filepaths)),
                                 key=lambda i: np.linalg.norm(features[frame_paths.index(group_filepaths[i])] - cluster_centroid))

        # Add the closest frame to the representative frames list
        representative_frames.append(cv2.imread(group_filepaths[closest_frame_idx]))

    # Save the paths to the keyframe images for summarization
    keyframe_paths = []
    for i, frame in enumerate(representative_frames):
        keyframe_path = f"keyframe_{i+1}.jpg"
        cv2.imwrite(keyframe_path, frame)  # Save keyframe as an image
        keyframe_paths.append(keyframe_path)

    # Display the summary of the video
    print("Video Summary:")
    for i, frame_path in enumerate(keyframe_paths, start=1):
        print(f"Keyframe {i}: {frame_path}")
        # Add additional information if needed, such as cluster number, color histogram, etc.

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Parallel Code Time: {total_time} seconds")
