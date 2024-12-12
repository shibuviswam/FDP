import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_detection(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        return

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction

    # Compute magnitude of gradients
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Convert to uint8 for visualization
    sobel_x = np.uint8(np.absolute(sobel_x))
    sobel_y = np.uint8(np.absolute(sobel_y))
    sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))

    # Display results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Sobel X")
    plt.imshow(sobel_x, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Sobel Y")
    plt.imshow(sobel_y, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Sobel Magnitude")
    plt.imshow(sobel_magnitude, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
image_path = "/home/shibuviswam/FDP/10.jpg"  # Replace with your image path
sobel_edge_detection(image_path)

