import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_hog(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        return

    # Initialize the HOG descriptor
    hog = cv2.HOGDescriptor()

    # Compute the HOG descriptors
    hog_descriptors = hog.compute(image)

    # Display HOG visualization
    hog_visualization = visualize_hog(image, hog)

    # Show results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("HOG Visualization")
    plt.imshow(hog_visualization, cmap='gray')
    plt.axis('off')

    plt.show()

def visualize_hog(image, hog_descriptor):
    # Parameters for visualization
    cell_size = (8, 8)  # Default OpenCV HOG cell size
    bins = 9

    # Calculate gradient magnitude and angle
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    angle = angle % 180  # Map angles to 0-180 degrees

    # Replace NaN and Inf values in magnitude
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)

    # Prepare an empty image to draw HOG visualization
    vis = np.zeros_like(image, dtype=np.float32)

    # Divide image into cells
    cell_x, cell_y = cell_size
    for y in range(0, image.shape[0] - cell_y, cell_y):
        for x in range(0, image.shape[1] - cell_x, cell_x):
            # Extract magnitude and angle for the cell
            cell_mag = magnitude[y:y + cell_y, x:x + cell_x]
            cell_ang = angle[y:y + cell_y, x:x + cell_x]

            # Compute histogram for the cell
            hist, _ = np.histogram(
                cell_ang,
                bins=bins,
                range=(0, 180),
                weights=cell_mag
            )

            # Avoid division by zero in normalization
            max_hist = np.max(hist)
            if max_hist == 0:
                max_hist = 1

            # Draw histogram as lines
            center_x, center_y = x + cell_x // 2, y + cell_y // 2
            for bin_idx in range(bins):
                theta = np.deg2rad(bin_idx * 180 / bins)
                dx = int(np.cos(theta) * hist[bin_idx] / max_hist * cell_x / 2)
                dy = int(np.sin(theta) * hist[bin_idx] / max_hist * cell_y / 2)
                cv2.line(
                    vis,
                    (center_x - dx, center_y - dy),
                    (center_x + dx, center_y + dy),
                    (255,),
                    1
                )

    return vis

# Example usage
image_path = "/home/shibuviswam/FDP/10.jpg"  # Replace with your image path
compute_hog(image_path)

