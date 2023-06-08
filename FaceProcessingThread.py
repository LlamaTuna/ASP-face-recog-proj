from PyQt5.QtCore import QThread, pyqtSignal
from face_detection import save_faces_from_folder, find_matching_face

class FaceProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    processing_done = pyqtSignal(list)  # Add this custom signal

    def __init__(self, input_folder, output_folder, image_to_search):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_to_search = image_to_search

    def run(self):
        face_data = save_faces_from_folder(folder_path=self.input_folder, output_folder=self.output_folder, face_cascade=None, progress_callback=self.update_progress)
        matching_faces = find_matching_face(self.image_to_search, face_data)
        self.processing_done.emit(matching_faces)  # Emit the custom signal instead

    def update_progress(self, progress):
        self.progress_signal.emit(progress)
