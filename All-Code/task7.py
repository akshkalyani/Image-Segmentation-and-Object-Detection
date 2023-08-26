import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the synthetic and real images
synthetic_img = cv2.imread('synthetic_image_2.png')
real_img = cv2.imread('real_image_2.jpg')

# Convert the images to RGB
synthetic_img_rgb = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2RGB)
real_img_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)

# Set the parameters for CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply CLAHE to the images
synthetic_img_clahe = clahe.apply(synthetic_img_rgb[:, :, 0])
real_img_clahe = clahe.apply(real_img_rgb[:, :, 0])

# Display the results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(synthetic_img_clahe, cmap='gray')
axs[0].set_title("Synthetic image after CLAHE")
axs[1].imshow(real_img_clahe, cmap='gray')
axs[1].set_title("Real image after CLAHE")
plt.show()
