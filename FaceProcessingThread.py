from PyQt6.QtCore import QThread, pyqtSignal
from face_detection import save_faces_from_folder, find_matching_face, face_detector  # Import face_detector
import logging
import traceback
import threading
from PyQt6.QtCore import pyqtSignal

class FaceProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    processing_done = pyqtSignal(tuple)
    error_signal = pyqtSignal(str)
    partial_result_signal = pyqtSignal(object)

    def __init__(self, input_folder, output_folder, image_to_search):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_to_search = image_to_search
        self.cancel = False  # Initialize the cancel flag
    
    def cancel_processing(self):
        """Set the cancel flag to stop processing."""
        self.cancel = True

    def run(self):
        try:
            if not self.cancel:
                self.face_data = save_faces_from_folder(
                    folder_path=self.input_folder,
                    output_folder=self.output_folder,
                    face_detector=face_detector,
                    progress_callback=self.update_progress,
                    cancel_flag=lambda: self.cancel

                )
            if not self.cancel:
                matching_faces = find_matching_face(self.image_to_search, self.face_data, face_detector)
                self.processing_done.emit((matching_faces, self.face_data))
                
        except Exception as e:
            error_message = f"An error occurred during face processing: {str(e)}"
            logging.error(error_message)
            logging.error(traceback.format_exc())
            self.error_signal.emit(error_message)

    def update_progress(self, progress):
        print(f"update_progress called in FaceProcessingThread with: {progress}, current thread: {threading.current_thread()}")
        self.progress_signal.emit(int(progress))
