import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from matplotlib import pyplot as plt

def load_embeddings(embedding_file):
    """
    Load embeddings from a file.
    """
    embeddings = np.load(embedding_file, allow_pickle=True).item()
    return embeddings

def cluster_embeddings(embeddings, num_clusters):
    """
    Perform clustering on embeddings using K-Means.
    """
    embedding_list = np.array(list(embeddings.values())).squeeze()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding_list)
    return labels, kmeans

def map_labels_to_images(embeddings, labels):
    """
    Map cluster labels to their corresponding images.
    """
    image_label_map = {}
    for i, image_name in enumerate(embeddings.keys()):
        image_label_map[image_name] = labels[i]
    return image_label_map

def annotate_and_count(image_dir, image_label_map, output_path):
    """
    Annotate images with product labels and count the number of products in each category.
    """
    label_counts = {}
    annotated_image = None

    for image_name, label in image_label_map.items():
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.putText(
            image, f"Label: {label}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        if annotated_image is None:
            annotated_image = image
        else:
            annotated_image = np.hstack((annotated_image, image))

        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    # Save annotated image
    os.makedirs(output_path, exist_ok=True)
    annotated_image_path = os.path.join(output_path, "annotated_images.png")
    cv2.imwrite(annotated_image_path, annotated_image)
    print(f"Annotated images saved to {annotated_image_path}")

    # Save label counts
    for label, count in label_counts.items():
        print(f"Label {label}: {count} items")
    
    return label_counts

def plot_clustered_images(image_dir, image_label_map, output_path):
    """
    Create a grid visualization of clustered images.
    """
    labels = sorted(set(image_label_map.values()))
    fig, axs = plt.subplots(len(labels), 5, figsize=(15, len(labels) * 3))

    for label_idx, label in enumerate(labels):
        images = [image_name for image_name, lbl in image_label_map.items() if lbl == label]
        for i, image_name in enumerate(images[:5]):  # Limit to 5 images per cluster
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axs[label_idx, i].imshow(image)
            axs[label_idx, i].set_title(f"Label: {label}")
            axs[label_idx, i].axis('off')

        for i in range(len(images), 5):
            axs[label_idx, i].axis('off')

    plt.tight_layout()
    plt_path = os.path.join(output_path, "cluster_visualization.png")
    plt.savefig(plt_path)
    print(f"Cluster visualization saved to {plt_path}")
    plt.show()

def main(args):
    """
    Main function for clustering embeddings and labeling products.
    """
    # Load embeddings
    embeddings = load_embeddings(args.embedding_file)
    print(f"Loaded {len(embeddings)} embeddings.")

    # Perform clustering
    labels, kmeans = cluster_embeddings(embeddings, args.num_clusters)
    print(f"Clustering completed with {args.num_clusters} clusters.")

    # Map labels to images
    image_label_map = map_labels_to_images(embeddings, labels)

    # Annotate and count products
    label_counts = annotate_and_count(args.image_dir, image_label_map, args.output_dir)

    # Plot clustered images
    plot_clustered_images(args.image_dir, image_label_map, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster embeddings and annotate images with product labels.")
    parser.add_argument("--embedding_file", type=str, required=True, help="Path to the file containing embeddings.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory with images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output files (annotated images, visualizations).")
    parser.add_argument("--num_clusters", type=int, default=5, help="Number of clusters for K-Means.")
    args = parser.parse_args()
    main(args)


# python cluster.py --embedding_file embeddings.npy --image_dir path_to_images --output_dir output_directory --num_clusters 5
