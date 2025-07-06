#-------------------------------------------------------------------------------------------------------#
#   Although k-means clustering can group bounding boxes in a dataset, 
#   many datasets have boxes with similar dimensions, resulting in 9 clusters that are not very distinct.
#   Such anchor boxes may not be optimal for model training. 
#   Different feature layers are designed to handle different anchor sizes: 
#   smaller feature layers work better with larger anchors, and vice versa.
#   The original YOLO anchor boxes are already well-proportioned across different scales,
#   so clustering may not provide significant benefits in some cases.
#-------------------------------------------------------------------------------------------------------#
import glob
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def calculate_aspect_ratio(box, cluster):
    """Calculate the aspect ratio between boxes and clusters"""
    ratios_box_to_cluster = box / cluster
    ratios_cluster_to_box = cluster / box
    ratios = np.concatenate([ratios_box_to_cluster, ratios_cluster_to_box], axis=-1)
    return np.max(ratios, axis=-1)

def average_ratio(boxes, clusters):
    """Calculate the average aspect ratio between boxes and clusters"""
    return np.mean([np.min(calculate_aspect_ratio(boxes[i], clusters)) for i in range(boxes.shape[0])])

def kmeans(boxes, k):
    """
    Perform k-means clustering on bounding boxes to generate anchor boxes
    
    Args:
        boxes: Array of bounding box dimensions (width, height)
        k: Number of clusters (anchor boxes) to generate
        
    Returns:
        clusters: Array of generated anchor box dimensions
        nearest_indices: Indices of the nearest cluster for each box
    """
    # Number of boxes
    num_boxes = boxes.shape[0]
    
    # Distance matrix (boxes x clusters)
    distance = np.empty((num_boxes, k))
    
    # Cluster assignments from previous iteration
    previous_cluster_assignments = np.zeros((num_boxes,))

    np.random.seed()

    # Initialize cluster centers by randomly selecting k boxes
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]

    iteration = 0
    while True:
        # Compute distance between each box and each cluster
        for i in range(num_boxes):
            distance[i] = calculate_aspect_ratio(boxes[i], clusters)
        
        # Find the nearest cluster for each box
        nearest_indices = np.argmin(distance, axis=1)

        # Check convergence
        if (previous_cluster_assignments == nearest_indices).all():
            break
        
        # Update cluster centers to be the median of the boxes in each cluster
        for j in range(k):
            boxes_in_cluster = boxes[nearest_indices == j]
            if len(boxes_in_cluster) > 0:
                clusters[j] = np.median(boxes_in_cluster, axis=0)

        previous_cluster_assignments = nearest_indices
        
        # Print progress every 5 iterations
        if iteration % 5 == 0:
            print(f'Iteration: {iteration}. Average ratio: {average_ratio(boxes, clusters):.2f}')
        
        iteration += 1

    return clusters, nearest_indices

def load_data(path):
    """
    Load bounding box dimensions from XML annotations
    
    Args:
        path: Path to directory containing XML annotation files
        
    Returns:
        data: Array of normalized bounding box dimensions (width, height)
    """
    data = []
    
    # Process each XML file in the directory
    for xml_file in tqdm(glob.glob(f'{path}/*.xml'), desc="Loading annotations"):
        try:
            tree = ET.parse(xml_file)
            height = int(tree.findtext('./size/height'))
            width = int(tree.findtext('./size/width'))
            
            # Skip invalid images
            if height <= 0 or width <= 0:
                continue
            
            # Extract bounding boxes from each object
            for obj in tree.iter('object'):
                xmin = float(obj.findtext('bndbox/xmin')) / width
                ymin = float(obj.findtext('bndbox/ymin')) / height
                xmax = float(obj.findtext('bndbox/xmax')) / width
                ymax = float(obj.findtext('bndbox/ymax')) / height

                # Store width and height of the bounding box
                data.append([xmax - xmin, ymax - ymin])
                
        except Exception as e:
            print(f"Skipping file {xml_file} due to error: {e}")
    
    return np.array(data)

if __name__ == '__main__':
    np.random.seed(0)
    
    # Configuration
    input_shape = [640, 640]  # Model input size
    num_anchors = 9           # Number of anchor boxes to generate
    
    # Path to annotation directory
    annotation_path = 'VOCdevkit/VOC2007/Annotations'
    
    # Load annotation data
    print('Loading annotation data...')
    box_dimensions = load_data(annotation_path)
    print('Annotation loading complete.')
    
    # Perform k-means clustering to generate anchor boxes
    print('Generating anchor boxes via k-means clustering...')
    clusters, nearest_indices = kmeans(box_dimensions, num_anchors)
    print('K-means clustering complete.')
    
    # Convert normalized dimensions back to input shape scale
    original_box_dimensions = box_dimensions * np.array([input_shape[1], input_shape[0]])
    anchor_boxes = clusters * np.array([input_shape[1], input_shape[0]])

    # Visualize the clustering results
    plt.figure(figsize=(10, 8))
    plt.title('K-means Clustering of Bounding Boxes for YOLO Anchor Generation')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    for j in range(num_anchors):
        # Plot boxes assigned to this cluster
        plt.scatter(
            original_box_dimensions[nearest_indices == j][:, 0], 
            original_box_dimensions[nearest_indices == j][:, 1],
            alpha=0.5, label=f'Cluster {j+1}'
        )
        
        # Plot cluster center
        plt.scatter(anchor_boxes[j][0], anchor_boxes[j][1], marker='x', c='black', s=100)
    
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig("kmeans_for_anchors.jpg")
    plt.close()
    print('Visualization saved as kmeans_for_anchors.jpg')

    # Sort anchor boxes by area (width * height)
    anchor_boxes = anchor_boxes[np.argsort(anchor_boxes[:, 0] * anchor_boxes[:, 1])]
    
    # Print results
    print(f'Average aspect ratio: {average_ratio(original_box_dimensions, anchor_boxes):.2f}')
    print('Generated anchor boxes (width, height):')
    for i, box in enumerate(anchor_boxes):
        print(f"  Anchor {i+1}: ({int(box[0])}, {int(box[1])})")

    # Save anchor boxes to file
    with open("yolo_anchors.txt", 'w') as f:
        anchor_strings = [f"{int(box[0])},{int(box[1])}" for box in anchor_boxes]
        f.write(", ".join(anchor_strings))
    
    print('Anchor boxes saved to yolo_anchors.txt')