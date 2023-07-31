from PyQt6.QtCore import QThread, pyqtSignal
from face_detection import save_faces_from_folder, find_matching_face, face_detector  # Import face_detector
import logging
import traceback
import threading

class FaceProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    processing_done = pyqtSignal(tuple)  # Add this custom signal
    error_signal = pyqtSignal(str)  # new signal for errors

    def __init__(self, input_folder, output_folder, image_to_search):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_to_search = image_to_search

    def run(self):
        try:
            self.face_data = save_faces_from_folder(
                folder_path=self.input_folder,
                output_folder=self.output_folder,
                face_detector=face_detector,  # Pass face_detector here
                progress_callback=self.update_progress
            )
            matching_faces = find_matching_face(self.image_to_search, self.face_data, face_detector)  # Pass face_detector here too
            self.processing_done.emit((matching_faces, self.face_data))  # Emit the custom signal with face_data
        except Exception as e:
            error_message = f"An error occurred during face processing: {str(e)}"
            logging.error(error_message)
            logging.error(traceback.format_exc())
            self.error_signal.emit(error_message)  # emit error_signal

    def update_progress(self, progress):
        print(f"update_progress called in FaceProcessingThread with: {progress}, current thread: {threading.current_thread()}")
        self.progress_signal.emit(progress)

