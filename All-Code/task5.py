import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the synthetic and real images
synthetic_img = cv2.imread('synthetic_image_2.png')
real_img = cv2.imread('real_image_2.jpg')

# Convert the images from BGR to RGB
synthetic_img_rgb = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2RGB)
real_img_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)

# Define the number of bins for each channel
bins = 256

# Define the range of pixel values for each channel
range = [0, 256]

# Compute the histograms for each channel of the synthetic image
synthetic_hist_r = cv2.calcHist([synthetic_img_rgb], [0], None, [bins], range)
synthetic_hist_g = cv2.calcHist([synthetic_img_rgb], [1], None, [bins], range)
synthetic_hist_b = cv2.calcHist([synthetic_img_rgb], [2], None, [bins], range)

# Compute the histograms for each channel of the real image
real_hist_r = cv2.calcHist([real_img_rgb], [0], None, [bins], range)
real_hist_g = cv2.calcHist([real_img_rgb], [1], None, [bins], range)
real_hist_b = cv2.calcHist([real_img_rgb], [2], None, [bins], range)

# Concatenate the histograms of the three channels for both images
synthetic_hist = np.concatenate(
    (synthetic_hist_r, synthetic_hist_g, synthetic_hist_b))
real_hist = np.concatenate((real_hist_r, real_hist_g, real_hist_b))

# Normalize the histograms
synthetic_hist_norm = synthetic_hist / np.sum(synthetic_hist)
real_hist_norm = real_hist / np.sum(real_hist)

# Compute the Chi-Square distance between the histograms
dist = cv2.compareHist(synthetic_hist_norm, real_hist_norm, cv2.HISTCMP_CHISQR)

print("Histogram distance: ", dist)

# Plot the histograms for the synthetic image
plt.figure(figsize=(8, 6))
plt.title("RGB Histogram of Synthetic Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(synthetic_hist_r, color='r', label='Red channel')
plt.plot(synthetic_hist_g, color='g', label='Green channel')
plt.plot(synthetic_hist_b, color='b', label='Blue channel')
plt.legend()
plt.show()

# Plot the histograms for the real image
plt.figure(figsize=(8, 6))
plt.title("RGB Histogram of Real Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(real_hist_r, color='r', label='Red channel')
plt.plot(real_hist_g, color='g', label='Green channel')
plt.plot(real_hist_b, color='b', label='Blue channel')
plt.legend()
plt.show()
