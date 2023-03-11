import cv2
import numpy as np

# Read the input image
input_image = cv2.imread("inp.png", cv2.IMREAD_GRAYSCALE)
print(input_image.shape)
# Get the kernel from user

kernel = np.array([[2,3,4],[4,3,5],[1,2,3]])

# Normalize the kernel
kernel = kernel / np.sum(kernel)

# Convolve the input image with the kernel
output_image = cv2.filter2D(input_image, -1, kernel)

# Show the input and output images side by side
cv2.imshow("Input image", input_image)
cv2.imshow("Output image", output_image)
cv2.waitKey(0)

# Save the output image
cv2.imwrite("output_image.png", output_image)
