import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
from scipy.spatial.distance import cosine

# Check if seaborn is available, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def load_encodings_from_known_people(known_people_dir="known_people"):
    """Load face encodings from the known_people directory"""
    if not os.path.exists(known_people_dir):
        print(f"Known people directory not found: {known_people_dir}")
        return {}, [], [], []
    
    # Data structures to store encodings and metadata
    all_encodings = []
    all_names = []
    all_image_paths = []
    name_to_encodings = {}
    
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
                # Load and convert image
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                    
                # Convert to RGB (face_recognition expects RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get face encoding using face_recognition
                face_encodings = face_recognition.face_encodings(rgb_frame)
                
                if face_encodings:
                    encoding = face_encodings[0]
                    encodings.append(encoding)
                    image_paths.append(img_path)
                    
                    # Add to global lists
                    all_encodings.append(encoding)
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
    
    return name_to_encodings, all_encodings, all_names, all_image_paths

def simple_pca(X, n_components=2):
    """Simple PCA implementation without requiring scikit-learn"""
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
    
    # Select top n_components eigenvectors
    components = eigenvectors[:, :n_components]
    
    # Project data onto principal components
    X_pca = np.dot(X_centered, components)
    
    # Calculate explained variance ratio
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_pca, explained_variance

def visualize_encodings_pca(encodings, names, title="PCA of Face Encodings"):
    """Visualize encodings using PCA for dimensionality reduction"""
    if not encodings:
        print("No encodings to visualize")
        return
    
    # Convert to numpy array
    X = np.array(encodings)
    
    # Perform PCA
    X_pca, explained_variance = simple_pca(X, n_components=2)
    
    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    
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
    
    # Plot each point
    for i, (x, y) in enumerate(X_pca):
        name = names[i]
        color = name_to_color[name]
        plt.scatter(x, y, c=[color], label=name, alpha=0.7)
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.title(title)
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.2%} variance)")
    plt.grid(alpha=0.3)
    
    # Show the total explained variance
    total_var = sum(explained_variance)
    plt.suptitle(f"Total Explained Variance: {total_var:.2%}", y=0.92)
    
    plt.tight_layout()
    # plt.savefig("face_encodings_pca.png")
    plt.show()

def calculate_distance(vector1, vector2, metric='euclidean'):
    """Calculate distance between two vectors using specified metric"""
    if metric == 'euclidean':
        return np.linalg.norm(vector1 - vector2)
    elif metric == 'cosine':
        # Cosine similarity is 1 when vectors are identical, so we use 1-similarity as distance
        return cosine(vector1, vector2)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def calculate_distance_stats(name_to_encodings, metric='euclidean'):
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

def plot_distance_histogram(intra_distances, inter_distances, title, metric='euclidean'):
    """Plot histogram of distances"""
    plt.figure(figsize=(10, 6))
    if HAS_SEABORN:
        sns.histplot(intra_distances, color='blue', label='Same Person', alpha=0.5, kde=True)
        sns.histplot(inter_distances, color='red', label='Different People', alpha=0.5, kde=True)
    else:
        plt.hist(intra_distances, color='blue', label='Same Person', alpha=0.5, bins=20)
        plt.hist(inter_distances, color='red', label='Different People', alpha=0.5, bins=20)
    
    plt.title(title)
    plt.xlabel(f'{metric.capitalize()} Distance')
    plt.ylabel('Frequency')
    plt.legend()
    # plt.savefig(f"distance_distribution_{metric}.png")
    plt.tight_layout()

def main():
    print("Loading face encodings...")
    name_to_encodings, all_encodings, all_names, all_image_paths = load_encodings_from_known_people()
    
    if not all_encodings:
        print("No encodings found. Please add face images to the 'known_people' directory.")
        return
    
    print(f"Loaded {len(all_encodings)} total encodings from {len(name_to_encodings)} people")
    
    # Visualize with PCA
    print("Visualizing encodings with PCA...")
    visualize_encodings_pca(all_encodings, all_names)
    
    # Calculate and display distance statistics for Euclidean distance
    print("\nCalculating Euclidean distance statistics...")
    euclidean_intra, euclidean_inter = calculate_distance_stats(name_to_encodings, 'euclidean')
    
    if euclidean_intra:
        avg_intra = np.mean(euclidean_intra)
        print(f"Average Euclidean distance between same person: {avg_intra:.4f} [0.3-0.5 is good]")
    
    if euclidean_inter:
        avg_inter = np.mean(euclidean_inter)
        print(f"Average Euclidean distance between different people: {avg_inter:.4f} [0.7+ is good]")
    
    if euclidean_intra and euclidean_inter:
        # Plot distributions
        plot_distance_histogram(
            euclidean_intra, 
            euclidean_inter, 
            'Euclidean Distance Distribution', 
            'euclidean'
        )
        plt.show()
    
    # Calculate and display distance statistics for Cosine distance
    print("\nCalculating Cosine distance statistics...")
    cosine_intra, cosine_inter = calculate_distance_stats(name_to_encodings, 'cosine')
    
    if cosine_intra:
        avg_intra = np.mean(cosine_intra)
        print(f"Average Cosine distance between same person: {avg_intra:.4f} [0.0-0.2 is good]")
    
    if cosine_inter:
        avg_inter = np.mean(cosine_inter)
        print(f"Average Cosine distance between different people: {avg_inter:.4f} [0.3+ is good]")
    
    if cosine_intra and cosine_inter:
        # Plot distributions
        plot_distance_histogram(
            cosine_intra, 
            cosine_inter, 
            'Cosine Distance Distribution', 
            'cosine'
        )
        plt.show()
    
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
            print("Cosine distance provides better separation for this dataset")

if __name__ == "__main__":
    main()
