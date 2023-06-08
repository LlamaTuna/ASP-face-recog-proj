import dlib
import cv2
import time


#This tests if CUDA is working as expected.
#On windows you'll need the zlibwapi.dll in you system $PATH like System32
#https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows
# Load the CNN face detector model


if dlib.DLIB_USE_CUDA:
    print("Dlib is compiled to use CUDA")
else:
    print("Dlib is not compiled to use CUDA")


cnn_face_detector = dlib.cnn_face_detection_model_v1("./tests/mmod_human_face_detector.dat")

# Load an image
image = cv2.imread("./tests/face1.jpeg")

# Convert the image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face detection using the CNN model
start_time = time.time()
detected_faces = cnn_face_detector(image_rgb, 1)
end_time = time.time()

# Calculate and print the time taken for face detection
time_taken = end_time - start_time
print("Time taken for face detection: {:.4f} seconds".format(time_taken))

# Print the number of faces detected
print("Number of faces detected:", len(detected_faces))
