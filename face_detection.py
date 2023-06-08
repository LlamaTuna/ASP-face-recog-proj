import os
import sys
import cv2
import hashlib
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input

output_folder = "./output/"
face_detector = MTCNN()
resnet_model = ResNet152(weights='imagenet', include_top=False)

def convert_image_format(image_path, output_format='png'):
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        raise ValueError(f"Unable to read image from {image_path}")
    output_path = os.path.splitext(image_path)[0] + '.' + output_format
    cv2.imwrite(output_path, img)
    return output_path


def save_faces_from_folder(folder_path, face_cascade, output_folder, progress_callback=None):
    face_data = {}
    valid_extensions = ['.png', '.jpeg', '.jpg', '.bmp']

    # Use os.walk to traverse directories
    image_paths = [os.path.join(root, name)
                   for root, dirs, files in os.walk(folder_path)
                   for name in files
                   if os.path.splitext(name)[-1].lower() in valid_extensions]

    num_images = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        image_name = os.path.basename(image_path)

        try:
            converted_image_path = convert_image_format(image_path, output_format='png')
        except ValueError as e:
            print(f"Error converting image {image_name}: {e}")
            continue

        try:
            img = cv2.imread(converted_image_path)
            assert img is not None, f"Image at {converted_image_path} is None"
        except AssertionError as e:
            print(e)
            continue

        # Now using MTCNN for face detection
        try:
            detected_faces = face_detector.detect_faces(img)
            faces = [(face['box'][1], face['box'][0]+face['box'][2], face['box'][1]+face['box'][3], face['box'][0]) for face in detected_faces]
        except Exception as e:
            print(f"Error detecting faces in {image_name}: {e}")
            continue

        if len(faces) > 0:
            try:
                img_hash = hashlib.sha256(open(converted_image_path, 'rb').read()).hexdigest()
                face_data[img_hash] = {"file_name": image_name, "faces": []}
                for (top, right, bottom, left) in faces:
                    face_img = img[top:bottom, left:right]
                    face_data[img_hash]["faces"].append(face_img)
                    output_path = os.path.join(output_folder, f"{img_hash}_{len(face_data[img_hash]['faces'])}.png")
                    cv2.imwrite(output_path, face_img)
            except Exception as e:
                print(f"Error processing face data in {image_name}: {e}")
                continue

        if progress_callback:
            progress_callback(idx / num_images * 100)

    return face_data

def find_matching_face(image_path, face_data, threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image from {image_path}")

    # Resize the image before face detection
    target_size = (224, 224)
    img = cv2.resize(img, target_size)

    # Now using MTCNN for face detection
    detected_faces = face_detector.detect_faces(img)
    matching_faces = []

    for face in detected_faces:
        # Get face box
        left, top, width, height = face['box']
        right, bottom = left + width, top + height
        face_img = img[top:bottom, left:right]

        # Resize the face to the input size expected by ResNet152 (224,224)
        face_img = cv2.resize(face_img, target_size)
        face_img = preprocess_input(face_img)  # preprocess for ResNet152

        for img_hash, stored_data in face_data.items():
            stored_faces = stored_data["faces"]
            for i, stored_face in enumerate(stored_faces):
                if stored_face.size == 0:  # Add this check for empty images
                    continue
                stored_face_resized = cv2.resize(stored_face, target_size)
                similarity = np.mean(np.abs(face_img.astype(np.float32) - stored_face_resized.astype(np.float32))) / 255.0

                if similarity < threshold:
                    matching_faces.append((img_hash, stored_data["file_name"], stored_face, similarity, f"{img_hash}_{i+1}.png"))

    return matching_faces
