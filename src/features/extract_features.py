"""
Handcrafted feature extraction module for the Indian traditional painting classification project.

This script parses raw images and computes mathematical features:
- Mean R, G, B colors and global color variance
- Edge density via Canny edge detection
- Horizontal symmetry via Mean Squared Error comparison
- Texture entropy via Shannon Entropy

It outputs these characteristics matched to their file path and textual labels
directly into `data/features.csv`.
"""

import os
import cv2
import numpy as np
import pandas as pd

def compute_color_features(img: np.ndarray) -> tuple:
    """
    Computes standard RGB color means and global color variance.
    
    Args:
        img (np.ndarray): Original image natively loaded by OpenCV in BGR.
        
    Returns:
        tuple: (mean_r, mean_g, mean_b, color_variance)
    """
    # OpenCV natively opens in BGR mode; map it properly to RGB.
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate independent channel averages
    mean_r = np.mean(rgb_img[:, :, 0])
    mean_g = np.mean(rgb_img[:, :, 1])
    mean_b = np.mean(rgb_img[:, :, 2])
    
    # Calculate single global variance for structural diversity mapping
    color_variance = np.var(rgb_img)
    
    return mean_r, mean_g, mean_b, color_variance

def compute_edge_density(gray_img: np.ndarray) -> float:
    """
    Measures ratio of image area representing explicit 'edges' using Canny tracking.
    
    Args:
        gray_img (np.ndarray): Image loaded or cast into grayscale (single channel).
        
    Returns:
        float: Normalized fractional ratio of edge_pixels / total_pixels.
    """
    # Default Canny thresholds set to 100/200; produces an identically-sized binary map
    edges = cv2.Canny(gray_img, 100, 200)
    
    # In binary Canny maps, 'edges' are denoted by numerical 255
    edge_pixels = np.count_nonzero(edges == 255)
    
    # Calculate fractional footprint matching the absolute bounds
    height, width = gray_img.shape
    total_pixels = height * width
    
    if total_pixels == 0:
        return 0.0
        
    edge_density = edge_pixels / total_pixels
    return float(edge_density)

def compute_symmetry(gray_img: np.ndarray) -> float:
    """
    Evaluates geometric horizontal symmetry using pixel MSE differences.
    
    Args:
        gray_img (np.ndarray): Image loaded or cast into grayscale.
        
    Returns:
        float: Standardized similarity bounds `1 - (MSE / MAX_POSSIBLE_MSE)`, mapped [0, 1].
    """
    # Execute a horizontal matrix flip (flipCode=1)
    flipped_img = cv2.flip(gray_img, 1)
    
    # Calculate direct pixel intensity divergence leveraging native numpy float computations
    mse = np.mean((gray_img.astype("float") - flipped_img.astype("float")) ** 2)
    
    # Grayscale image domains max out at intensity 255; the absolute worst-case divergence
    # pixel-to-pixel would be 255 (e.g. comparing pitch black to pure white).
    max_possible_mse = 255 ** 2
    
    # Normalize score projecting into bounded similarity spectrum 
    symmetry_score = 1 - (mse / max_possible_mse)
    return float(symmetry_score)

def compute_texture_entropy(gray_img: np.ndarray) -> float:
    """
    Projects Shannon Entropy calculated against standard grayscaled histograms.
    
    Args:
        gray_img (np.ndarray): Image loaded or cast into grayscale.
        
    Returns:
        float: Computed structural image entropy tracking texture density distributions.
    """
    # Initialize a 1D sequence containing the 256 pixel distributions
    histogram, _ = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))
    
    # Normalize histogram to operate strictly on probability vectors
    probabilities = histogram / np.sum(histogram)
    
    # Retain strictly non-zero distributions for numeric logarithm stability 
    # and forcibly inject an arbitrarily small offset.
    p = probabilities[probabilities > 0]
    p = p + 1e-10
    
    # Native Numpy translation of iterative Shannon Entropy
    texture_entropy = -np.sum(p * np.log2(p))
    return float(texture_entropy)

def generate_feature_table():
    """
    Iterates over target data dictionaries, aggregates standard image descriptors
    from valid `.jpg`, `.jpeg`, `.png` bounds into memory, and natively writes dataframe
    results mapped back into `data/features.csv`.
    """
    root_dir = "data/raw"
    features_list = []
    
    print("Initializing feature extraction...")
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    total_images = 0
    unique_classes = set()

    for class_folder in os.listdir(root_dir):
        # Ignore strictly configured macOS hidden paths
        if class_folder.startswith("."):
            continue
            
        class_path = os.path.join(root_dir, class_folder)
        
        if not os.path.isdir(class_path):
            continue
            
        unique_classes.add(class_folder)

        for file_name in os.listdir(class_path):
            if file_name.startswith("."):
                continue
                
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in valid_extensions:
                continue

            file_path = os.path.join(class_path, file_name)
            
            # Load raw BGR Image
            img = cv2.imread(file_path)
            
            if img is None:
                continue
                
            # Cast single gray sequence explicitly referenced by structural computations
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Compute extraction routines
            mean_r, mean_g, mean_b, color_var = compute_color_features(img)
            edge_den = compute_edge_density(gray_img)
            symmetry = compute_symmetry(gray_img)
            entropy = compute_texture_entropy(gray_img)

            # Aggregate to target dict sequence
            features_list.append({
                "image_path": file_path,
                "mean_r": mean_r,
                "mean_g": mean_g,
                "mean_b": mean_b,
                "color_variance": color_var,
                "edge_density": edge_den,
                "symmetry_score": symmetry,
                "texture_entropy": entropy,
                "label": class_folder
            })
            
            total_images += 1
            if total_images % 50 == 0:
                print(f"Processed {total_images} images...")

    print("\n--- Extraction Complete ---")
    
    df = pd.DataFrame(features_list)
    
    print(f"Total Images Processed: {total_images}")
    print(f"Total Unique Classes: {len(unique_classes)}")
    print(f"DataFrame Shape: {df.shape}")
    
    output_path = "data/features.csv"
    df.to_csv(output_path, index=False)
    print(f"Features natively dumped to => {output_path}")

if __name__ == "__main__":
    generate_feature_table()
