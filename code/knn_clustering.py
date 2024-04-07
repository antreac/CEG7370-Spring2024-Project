import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to extract features from frames (adapt as needed)
def extract_features(frame):
    # Example: Color histogram as a feature vector
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, None).flatten()

# Folder path containing frames
folder_path = "/Users/andreachristou/Documents/Spring2024/DC/PhaseII/data"

# Load frames and extract features
frames = []
features = []
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    if filename.endswith((".jpg", ".png")):
        try:
            frame = cv2.imread(filepath)
            frames.append(frame)
            features.append(extract_features(frame))
        except Exception as e:
            print(f"Error processing frame {filename}: {e}")

# Number of clusters (adjust as needed)
num_clusters = 5

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(features)
frame_labels = kmeans.labels_

# Group frames by cluster
frame_groups = {}
for i, label in enumerate(frame_labels):
    frame_groups.setdefault(label, []).append(frames[i])

# Visualize cluster groupings (using sampling)
for label, group_frames in frame_groups.items():
    print(f"Cluster {label}:")

    # Display a sample of 5 frames from the cluster
    for i, frame in enumerate(group_frames[:5]):
        cv2.imshow(f"Frame {i+1} (Cluster {label})", frame)
        cv2.waitKey(0)  # Wait for a key press before closing each window

    # Calculate average color histogram for the cluster
    cluster_hist = np.mean(np.vstack(features[i] for i, f in enumerate(group_frames) if frame_labels[i] == label), axis=0)  # Average features

    # Find the index of the most frequent color intensity
    most_frequent_color_idx = np.argmax(cluster_hist)

    # Map the index to the corresponding color channel (assuming BGR order)
    color_channel_names = ['Blue', 'Green', 'Red']
    most_frequent_color_channel = color_channel_names[most_frequent_color_idx % 3]

    # Print information about the most common color
    print(f"\tMost Common Color Channel: {most_frequent_color_channel}")
    print(f"\tMost Frequent Color Intensity: {most_frequent_color_idx}")

    # Visualize the histogram
    plt.figure()
    plt.title(f"Cluster {label} Average Color Histogram")
    plt.plot(cluster_hist)
    plt.xlabel("Color Intensity")
    plt.ylabel("Frequency")
    plt.xlim(0, 256)
    plt.show()

cv2.destroyAllWindows()
