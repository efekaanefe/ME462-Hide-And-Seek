import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import insightface  # Replace face_recognition with InsightFace
from scipy.spatial.distance import cosine
import time  # For timing operations
import torch  # Add torch for GPU detection

# Suppress unnecessary warnings and logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["INSIGHTFACE_LOG_LEVEL"] = "ERROR"

# Check if seaborn is available, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Initialize ArcFace model
print("Loading ArcFace model...")
try:
    face_model = insightface.app.FaceAnalysis(name='buffalo_l')
    face_model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)  # Use GPU if available
except Exception as e:
    print(f"Error loading ArcFace model: {e}")
    print("Please install InsightFace: pip install insightface onnxruntime-gpu")
    face_model = None

def load_encodings_from_known_people(known_people_dir="known_people"):
    """Load face encodings from the known_people directory using ArcFace"""
    # Try both current directory and app subdirectory
    if not os.path.exists(known_people_dir) and os.path.exists(os.path.join("app", known_people_dir)):
        known_people_dir = os.path.join("app", known_people_dir)
        print(f"Using known_people directory from app subdirectory: {known_people_dir}")
    
    if not os.path.exists(known_people_dir):
        print(f"Known people directory not found: {known_people_dir}")
        return {}, [], [], []
    
    if face_model is None:
        print("ArcFace model not available. Cannot load encodings.")
        return {}, [], [], []
    
    # Data structures to store encodings and metadata
    all_encodings = []
    all_names = []
    all_image_paths = []
    name_to_encodings = {}
    
    start_time = time.time()
    print("Extracting ArcFace embeddings from known people...")
    
    for person_name in os.listdir(known_people_dir):
        person_dir = os.path.join(known_people_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        encodings = []
        image_paths = []
        
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(person_dir, img_file)
            try:
                # Load image
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                # Detect faces using ArcFace
                faces = face_model.get(frame)
                
                if not faces:
                    print(f"No face found in {img_path}")
                    continue
                
                # Use the face with highest detection score
                best_face = max(faces, key=lambda x: x.det_score) if len(faces) > 1 else faces[0]
                
                # Get and normalize the embedding
                embedding = best_face.embedding
                normalized_embedding = embedding / np.linalg.norm(embedding)
                
                encodings.append(normalized_embedding)
                image_paths.append(img_path)
                
                # Add to global lists
                all_encodings.append(normalized_embedding)
                all_names.append(person_name)
                all_image_paths.append(img_path)
                
                print(f"Successfully extracted features from {img_path}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if encodings:
            name_to_encodings[person_name] = {
                'encodings': encodings,
                'image_paths': image_paths
            }
            print(f"Loaded {len(encodings)} encodings for {person_name}")
        else:
            print(f"Warning: No valid encodings extracted for {person_name}")
    
    elapsed_time = time.time() - start_time
    print(f"Completed feature extraction in {elapsed_time:.2f} seconds")
    
    return name_to_encodings, all_encodings, all_names, all_image_paths

def simple_pca(X, n_components=3):
    """Simple PCA implementation supporting up to 3 components"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    
    # Select top n_components eigenvectors (up to 3)
    n_components = min(n_components, 3)  # Ensure at most 3 components
    components = eigenvectors[:, :n_components]
    
    # Project data onto principal components
    X_pca = np.dot(X_centered, components)
    
    # Calculate explained variance ratio
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_pca, explained_variance

def visualize_encodings_pca(encodings, names, title="PCA of ArcFace Embeddings", dim=3):
    """Visualize encodings using PCA for dimensionality reduction with 2D or 3D plots"""
    if not encodings:
        print("No encodings to visualize")
        return
    
    if dim not in [2, 3]:
        print("Dimension must be 2 or 3")
        dim = 3  # Default to 3D
    
    # Convert to numpy array
    X = np.array(encodings)
    
    # Perform PCA
    X_pca, explained_variance = simple_pca(X, n_components=dim)
    
    # Get unique names and assign colors
    unique_names = list(set(names))
    
    # Create a color map - use seaborn if available or generate manually
    if HAS_SEABORN:
        colors = sns.color_palette("husl", len(unique_names))
    else:
        # Simple color generation without seaborn
        cmap = plt.cm.get_cmap('hsv', len(unique_names) + 1)
        colors = [cmap(i) for i in range(len(unique_names))]
    
    name_to_color = {name: colors[i] for i, name in enumerate(unique_names)}
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # Plot based on dimension
    if dim == 2:
        # 2D plot
        ax = fig.add_subplot(111)
        
        # Plot each point
        for i, (x, y) in enumerate(X_pca):
            name = names[i]
            color = name_to_color[name]
            ax.scatter(x, y, c=[color], label=name, alpha=0.7)
        
        ax.set_xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")
        ax.grid(alpha=0.3)
        
    else:
        # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each point
        for i, (x, y, z) in enumerate(X_pca):
            name = names[i]
            color = name_to_color[name]
            ax.scatter(x, y, z, c=[color], label=name, alpha=0.7, s=100)  # Larger points
        
        ax.set_xlabel(f"PC1 ({explained_variance[0]:.2%})")
        ax.set_ylabel(f"PC2 ({explained_variance[1]:.2%})")
        ax.set_zlabel(f"PC3 ({explained_variance[2]:.2%})")
        
        # Add rotation capability
        ax.view_init(elev=30, azim=45)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    ax.set_title(title)
    
    # Show the total explained variance
    total_var = sum(explained_variance)
    fig.suptitle(f"Total Explained Variance: {total_var:.2%}", y=0.92)  # Adjusted position
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout to avoid title overlap
    # plt.savefig(f"arcface_embeddings_pca_{dim}d.png")
    plt.show()
    
    return fig, ax  # Return the figure and axis for potential further modifications

def calculate_distance(vector1, vector2, metric='cosine'):
    """Calculate distance between two vectors using specified metric
    For ArcFace, cosine similarity is the preferred method."""
    if metric == 'euclidean':
        return np.linalg.norm(vector1 - vector2)
    elif metric == 'cosine':
        # For ArcFace, we use 1-similarity as distance metric
        # Dot product of normalized vectors gives cosine similarity
        similarity = np.dot(vector1, vector2)
        return 1 - similarity
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def calculate_distance_stats(name_to_encodings, metric='cosine'):
    """Calculate intra-class and inter-class distances using specified metric"""
    # Intra-class distances (same person)
    intra_class_distances = []
    for person, data in name_to_encodings.items():
        encodings = data['encodings']
        if len(encodings) > 1:
            for i in range(len(encodings)):
                for j in range(i+1, len(encodings)):
                    distance = calculate_distance(
                        np.array(encodings[i]), 
                        np.array(encodings[j]),
                        metric
                    )
                    intra_class_distances.append(distance)
    
    # Inter-class distances (different people)
    inter_class_distances = []
    person_names = list(name_to_encodings.keys())
    if len(person_names) > 1:
        for i in range(len(person_names)):
            for j in range(i+1, len(person_names)):
                person1 = person_names[i]
                person2 = person_names[j]
                
                for enc1 in name_to_encodings[person1]['encodings']:
                    for enc2 in name_to_encodings[person2]['encodings']:
                        distance = calculate_distance(
                            np.array(enc1), 
                            np.array(enc2),
                            metric
                        )
                        inter_class_distances.append(distance)
    
    return intra_class_distances, inter_class_distances

def plot_distance_histogram_with_threshold(intra_distances, inter_distances, title, metric='cosine'):
    """Plot histogram of distances with suggested threshold"""
    plt.figure(figsize=(10, 6))
    
    # Calculate a good threshold between distributions
    threshold = (np.max(intra_distances) + np.min(inter_distances)) / 2
    
    if HAS_SEABORN:
        sns.histplot(intra_distances, color='blue', label='Same Person', alpha=0.5, kde=True)
        sns.histplot(inter_distances, color='red', label='Different People', alpha=0.5, kde=True)
    else:
        plt.hist(intra_distances, color='blue', label='Same Person', alpha=0.5, bins=20)
        plt.hist(inter_distances, color='red', label='Different People', alpha=0.5, bins=20)
    
    # Add threshold line
    plt.axvline(x=threshold, color='green', linestyle='--', 
               label=f'Threshold: {threshold:.4f}')
    
    plt.title(title)
    plt.xlabel(f'{metric.capitalize()} Distance')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig(f"arcface_{metric}_threshold.png")
    plt.show()
    
    if metric == 'cosine':
        similarity_threshold = 1 - threshold
        print(f"Recommended {metric} distance threshold: {threshold:.4f}")
        print(f"(Use {similarity_threshold:.4f} as similarity threshold)")
    else:
        print(f"Recommended {metric} distance threshold: {threshold:.4f}")
    
    return threshold

def main():
    print("Loading ArcFace embeddings...")
    name_to_encodings, all_encodings, all_names, all_image_paths = load_encodings_from_known_people()
    
    if not all_encodings:
        print("No encodings found. Please add face images to the 'known_people' directory.")
        return
    
    print(f"Loaded {len(all_encodings)} total encodings from {len(name_to_encodings)} people")
    
    # Visualize with PCA in 3D only
    print("Visualizing embeddings with 3D PCA...")
    visualize_encodings_pca(all_encodings, all_names, title="3D PCA of ArcFace Embeddings", dim=3)
    
    # Calculate distance statistics for Cosine distance (preferred for ArcFace)
    print("\nCalculating Cosine distance statistics (recommended for ArcFace)...")
    cosine_intra, cosine_inter = calculate_distance_stats(name_to_encodings, 'cosine')
    
    if cosine_intra and cosine_inter:
        avg_intra = np.mean(cosine_intra)
        avg_inter = np.mean(cosine_inter)
        print(f"Average Cosine distance: Same person = {avg_intra:.4f}, Different people = {avg_inter:.4f}")
        
        # Plot distribution with threshold
        cosine_threshold = plot_distance_histogram_with_threshold(
            cosine_intra, 
            cosine_inter, 
            'ArcFace Cosine Distance Distribution', 
            'cosine'
        )
    
    # Calculate and display distance statistics for Euclidean distance
    print("\nCalculating Euclidean distance statistics...")
    euclidean_intra, euclidean_inter = calculate_distance_stats(name_to_encodings, 'euclidean')
    
    if euclidean_intra and euclidean_inter:
        avg_intra = np.mean(euclidean_intra)
        avg_inter = np.mean(euclidean_inter)
        print(f"Average Euclidean distance: Same person = {avg_intra:.4f}, Different people = {avg_inter:.4f}")
        
        # Plot distribution with threshold
        euclidean_threshold = plot_distance_histogram_with_threshold(
            euclidean_intra, 
            euclidean_inter, 
            'ArcFace Euclidean Distance Distribution', 
            'euclidean'
        )
    
    # Print comparison summary
    if euclidean_intra and euclidean_inter and cosine_intra and cosine_inter:
        euclidean_ratio = np.mean(euclidean_inter) / np.mean(euclidean_intra)
        cosine_ratio = np.mean(cosine_inter) / np.mean(cosine_intra)
        
        print("\nDistance metric comparison:")
        print(f"Euclidean inter/intra ratio: {euclidean_ratio:.2f} [higher is better]")
        print(f"Cosine inter/intra ratio: {cosine_ratio:.2f} [higher is better]")
        
        if euclidean_ratio > cosine_ratio:
            print("Euclidean distance provides better separation for this dataset")
        else:
            print("Cosine distance provides better separation for this dataset (recommended for ArcFace)")
            
    print("\nVisualization images saved to current directory.")

if __name__ == "__main__":
    main()
