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

# Compute the difference between the histograms
hist_diff = real_hist - synthetic_hist

# Plot the difference histogram
plt.plot(hist_diff)
plt.title("Histogram Difference")
plt.show()

# Equalize the synthetic image histogram to match the real image histogram
synthetic_img_eq = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2YCrCb)
synthetic_img_eq[:, :, 0] = cv2.equalizeHist(synthetic_img_eq[:, :, 0])
synthetic_img_eq = cv2.cvtColor(synthetic_img_eq, cv2.COLOR_YCrCb2BGR)

# Equalize the real image histogram to match the synthetic image histogram
real_img_eq = cv2.cvtColor(real_img, cv2.COLOR_BGR2YCrCb)
real_img_eq[:, :, 0] = cv2.equalizeHist(real_img_eq[:, :, 0])
real_img_eq = cv2.cvtColor(real_img_eq, cv2.COLOR_YCrCb2BGR)


# Display the images
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(synthetic_img_rgb)
axs[0].set_title("Synthetic Image")
axs[1].imshow(synthetic_img_eq)
axs[1].set_title("Synthetic Image - Equalized")
axs[2].imshow(real_img_eq)
axs[2].set_title("Real Image - Equalized")
plt.show()
