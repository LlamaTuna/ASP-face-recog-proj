import os
import cv2
import sys
import hashlib
import numpy as np
from mtcnn import MTCNN
from scipy.spatial import distance
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input

output_folder = "./output/"
face_detector = MTCNN()
resnet_model = ResNet152(weights='imagenet', include_top=False)

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def resize_image_with_aspect_ratio(img, size):
    # Get the aspect ratio
    aspect_ratio = img.shape[1] / img.shape[0]
    target_aspect_ratio = size[1] / size[0]

    # If the aspect ratios are equal, we don't need to do anything
    if aspect_ratio == target_aspect_ratio:
        return cv2.resize(img, size)
    elif aspect_ratio < target_aspect_ratio:
        # If the aspect ratio of the image is less than the target aspect ratio
        # then we need to add padding to the width
        scale_factor = size[0] / img.shape[0]
        new_width = int(img.shape[1] * scale_factor)
        rescaled_img = cv2.resize(img, (new_width, size[0]))
        pad_width = size[1] - new_width
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad
        padded_img = cv2.copyMakeBorder(rescaled_img, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    else:
        # If the aspect ratio of the image is greater than the target aspect ratio
        # then we need to add padding to the height
        scale_factor = size[1] / img.shape[1]
        new_height = int(img.shape[0] * scale_factor)
        rescaled_img = cv2.resize(img, (size[1], new_height))
        pad_height = size[0] - new_height
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        padded_img = cv2.copyMakeBorder(rescaled_img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    return padded_img

def convert_image_to_vector(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature_vector = resnet_model.predict(img)
    return feature_vector.flatten()



def save_faces_from_folder(folder_path, face_detector, output_folder, progress_callback=None):
    face_data = {}
    valid_extensions = ['.png', '.jpeg', '.jpg', '.bmp']
    processed_images = set()  # Keep track of processed images
    processed_faces = set()  # Keep track of processed faces

    # Use os.walk to traverse directories
    image_paths = [os.path.join(root, name)
                   for root, dirs, files in os.walk(folder_path)
                   for name in files
                   if os.path.splitext(name)[-1].lower() in valid_extensions]

    num_images = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        image_name = os.path.basename(image_path)

        try:
            img = cv2.imread(image_path)
            assert img is not None, f"Image at {image_path} is None"
        except AssertionError as e:
            print(e)
            continue

        # Using MTCNN for face detection
        try:
            detected_faces = face_detector.detect_faces(img)
            faces = [(face['box'][1], face['box'][0]+face['box'][2], face['box'][1]+face['box'][3], face['box'][0]) for face in detected_faces]
        except Exception as e:
            print(f"Error detecting faces in {image_name}: {e}")
            continue

        if len(faces) > 0:
            try:
                img_hash = hashlib.sha256(open(image_path, 'rb').read()).hexdigest()

                # If we have processed this image, continue to the next
                if img_hash in processed_images:
                    continue
                
                # Mark this image as processed
                processed_images.add(img_hash)

                face_data[img_hash] = {"file_name": image_name, "faces": []}
                for (top, right, bottom, left) in faces:
                    face_img = img[top:bottom, left:right]
                    resized_face_img = resize_image_with_aspect_ratio(face_img, (224, 224))  # or another size if you prefer
                    
                    # Here we convert the face image into a feature vector
                    face_vector = convert_image_to_vector(resized_face_img)

                    face_img_hash = hashlib.sha256(face_vector.tobytes()).hexdigest()
                    
                    # If we have processed this face, continue to the next
                    if face_img_hash in processed_faces:
                        continue
                    
                    # Mark this face as processed
                    processed_faces.add(face_img_hash)

                    face_data[img_hash]["faces"].append(face_vector)
                    image_output_path = os.path.join(output_folder, f"{img_hash}_{len(face_data[img_hash]['faces'])}.png")
                    cv2.imwrite(image_output_path, resized_face_img)

            except Exception as e:
                print(f"Error processing face data in {image_name}: {e}")
                continue

        if progress_callback:
            progress_callback(idx / num_images * 100)

    return face_data



def find_matching_face(image_path, face_data, face_detector, threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image from {image_path}")

    # Now using MTCNN for face detection
    detected_faces = face_detector.detect_faces(img)
    matching_faces = []

    for face in detected_faces:
        # Get face box
        left, top, width, height = face['box']
        right, bottom = left + width, top + height

        face_img = img[top:bottom, left:right]  # define face_img first
        face_img = resize_image_with_aspect_ratio(face_img, (224, 224))  # then resize it

        # Convert face image to vector
        face_vector = convert_image_to_vector(face_img)

        for img_hash, stored_data in face_data.items():
            stored_faces = stored_data["faces"]
            for i, stored_face in enumerate(stored_faces):
                if stored_face.size == 0:  # Add this check for empty vectors
                    continue
                
                # Compare vectors instead of images
                similarity = distance.cosine(face_vector, stored_face)

                if similarity < threshold:
                    matching_faces.append((img_hash, stored_data["file_name"], stored_face, similarity, f"{img_hash}_{i+1}.npy"))

    return matching_faces