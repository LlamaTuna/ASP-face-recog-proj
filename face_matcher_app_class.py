import os
import cv2
import sys
import types
from gui_init import initUI
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from face_detection import save_faces_from_folder, find_matching_face
from gui_elements import NumericTableWidgetItem, MatchTableWidgetItem
from PyQt6.QtGui import QAction
from FaceProcessingThread import FaceProcessingThread
import logging

try:
    logging.basicConfig(filename=r'.\debug.log',
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.debug('Logging setup successful')
    print("Logging setup successful.")
except Exception as e:
    print(f"Logging setup failed: {str(e)}")

logger = logging.getLogger()

class FaceMatcherApp(QMainWindow):
    def __init__(self, face_data=None):
        super().__init__()
        self.face_data = face_data
        self.dark_theme_enabled = False
        self.worker = None
        self.initUI = types.MethodType(initUI, self)  # This binds the initUI function as an instance method
        self.initUI()
        self.result_table.cellDoubleClicked.connect(self.open_image_in_default_viewer)

    def create_menu_bar(self):
        try:
            menubar = self.menuBar()

            # Create File menu
            file_menu = menubar.addMenu('File')

            # Create Exit action
            exit_action = QAction('Exit', self)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

            # Create View menu
            view_menu = menubar.addMenu('View')

            # Create Toggle Dark Theme action
            toggle_dark_theme_action = QAction('Toggle Dark Theme', self)
            toggle_dark_theme_action.triggered.connect(self.toggle_dark_theme)
            view_menu.addAction(toggle_dark_theme_action)
        except Exception as e:
            logging.exception("An error occurred while creating the menu bar")
            raise e

    def get_matched_face_by_row(self, row):
        if row >= 0 and row < self.result_table.rowCount():
            resized_image_name = self.result_table.item(row, 4).text()
            matched_face_path = os.path.join(self.output_folder_edit.text(), resized_image_name)
            matched_face = cv2.imread(matched_face_path)

            return matched_face
        return None

    def previous_matched_face(self):
        try:
            current_row = self.result_table.currentRow()
            if current_row > 0:
                self.result_table.selectRow(current_row - 1)
                matched_face = self.get_matched_face_by_row(current_row - 1)
                if matched_face is not None:
                    similarity = float(self.result_table.item(current_row - 1, 1).text().rstrip('%')) / 100
                    original_image_name = self.result_table.item(current_row - 1, 2).text()
                    self.display_matched_face(matched_face, similarity, original_image_name)
        except Exception as e:
            logging.exception("An error occurred while navigating to the previous matched face")
            raise e

    def next_matched_face(self):
        try:
            current_row = self.result_table.currentRow()
            if current_row < self.result_table.rowCount() - 1:
                self.result_table.selectRow(current_row + 1)
                matched_face = self.get_matched_face_by_row(current_row + 1)
                if matched_face is not None:
                    similarity = float(self.result_table.item(current_row + 1, 1).text().rstrip('%')) / 100
                    original_image_name = self.result_table.item(current_row + 1, 2).text()
                    self.display_matched_face(matched_face, similarity, original_image_name)
        except Exception as e:
            logging.exception("An error occurred while navigating to the next matched face")
            raise e

    def on_result_table_selection_changed(self):
        try:
            current_row = self.result_table.currentRow()
            if current_row != -1:
                print("Selection changed")  # debug print
                img_hash = self.result_table.item(current_row, 3).text()
                print(f"Selected image hash: {img_hash}")

                # access face_data using the selected image hash
                face_info = self.face_data.get(img_hash, {})

                # print the face_info to the console
                print(face_info)

                # Also, you can display it in any widget you want like QTextEdit
                self.console_output.append("\nSelected face:")
                self.console_output.append(f"\nImage file: {face_info.get('file_name', 'N/A')}")

                # EXIF data
                self.console_output.append("\nEXIF data:")
                exif_data = face_info.get('exif_data', {})
                for k, v in exif_data.items():
                    self.console_output.append(f"{k}: {v}")

                self.display_selected_matched_face()
        except Exception as e:
            logging.exception("An error occurred while handling the selection change in the result table")
            raise e

    def on_table_selection_changed_and_display_face(self):  # renamed this method
        try:
            self.display_selected_matched_face()
        except Exception as e:
            logging.exception("An error occurred while handling the selection change in the result table and displaying the face")
            raise e

    # function amended for #GUI themes CB
    def toggle_dark_theme(self):
        try:
            if self.dark_theme_enabled:
                self.dark_theme_enabled = False
                if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                    # Running in a PyInstaller bundle
                    bundle_dir = sys._MEIPASS
                else:
                    # Running in a normal Python environment
                    bundle_dir = os.path.dirname(os.path.abspath(__file__))

                light_theme_path = os.path.join(bundle_dir, "styles", "light_theme.qss")
                self.setStyleSheet(load_stylesheet(light_theme_path)) # CB removed replace
            else:
                self.dark_theme_enabled = True
                if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                    # Running in a PyInstaller bundle
                    bundle_dir = sys._MEIPASS
                else:
                    # Running in a normal Python environment
                    bundle_dir = os.path.dirname(os.path.abspath(__file__))

                dark_theme_path = os.path.join(bundle_dir, "styles", "dark_theme.qss")
                self.setStyleSheet(load_stylesheet(dark_theme_path)) # CB removed replace
        except Exception as e:
            logging.exception("An error occurred while toggling the dark theme")
            raise e

    def browse_input_folder(self):
        try:
            print("Browsing input folder")
            input_folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
            if input_folder:
                self.input_folder_edit.setText(input_folder)
            print("Finished browsing input folder")
        except Exception as e:
            logging.exception("An error occurred while browsing the input folder")
            raise e

    def browse_output_folder(self):
        try:
            print("Browsing output folder")
            output_folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
            if output_folder:
                self.output_folder_edit.setText(output_folder)
            print("Finished browsing output  folder")
        except Exception as e:
            logging.exception("An error occurred while browsing the output folder")
            raise e

    def browse_image_to_search(self):
        try:
            print("Browsing image to search")
            file_path, _ = QFileDialog.getOpenFileName(self, 'Select Image to Search', '', 'Image files (*.png *.jpeg *.jpg *.bmp *.tiff)')
            if file_path:
                self.image_to_search_edit.setText(file_path)
                self.load_image_thumbnail(file_path)
            print("Finished browsing image to search")
        except Exception as e:
            logging.exception("An error occurred while browsing the image to search")
            raise e

    def load_image_thumbnail(self, file_path):
        try:
            print("Loading image thumbnail")
            image = QImage(file_path)
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_preview_label.setPixmap(scaled_pixmap)
            print("Finished loading image thumbnail")
        except Exception as e:
            logging.exception("An error occurred while loading the image thumbnail")
            raise e

    def display_matched_face(self, matched_face, similarity, original_image_name):
        try:
            if matched_face is None:
                print("Error: Matched face is None. Cannot display.")
                return
            
            print("Displaying matched face")
            height, width, _ = matched_face.shape
            bytes_per_line = width * 3
            matched_face = cv2.cvtColor(matched_face, cv2.COLOR_BGR2RGB)
            q_image = QImage(matched_face.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            if pixmap.isNull():
                print("Error: QPixmap is null. Cannot display.")
                return
                
            scaled_pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
            self.matched_face_label.setPixmap(scaled_pixmap)
            self.similarity_original_image_label.setText(f"Similarity: {similarity * 100:.2f}% | Original Image: {original_image_name}")
            print("Finished displaying matched face")
        except Exception as e:
            logging.exception("An error occurred while displaying the matched face")
            raise e

    def find_match(self):
        try:
            logging.debug("Starting find_match")
            input_folder = self.input_folder_edit.text()
            output_folder = self.output_folder_edit.text()
            image_to_search = self.image_to_search_edit.text()

            if not input_folder or not output_folder or not image_to_search:
                QMessageBox.critical(self, "Error", "Please select all required folders and files.")
                return

            self.face_processing_thread = FaceProcessingThread(input_folder, output_folder, image_to_search)
            self.face_processing_thread.processing_done.connect(self.on_face_processing_done)
            self.face_processing_thread.progress_signal.connect(self.update_progress_bar)
            self.face_processing_thread.error_signal.connect(self.show_error_message)
            self.face_processing_thread.start()
            logging.debug("FaceProcessingThread started successfully")
            logging.debug("Finished find_match")
        except Exception as e:
            logging.exception("An error occurred while finding the match")
            raise e

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)
        
    def on_face_processing_done(self, result):
        try:
            matching_faces, face_data = result
            self.face_data = face_data
            logging.debug("Processing finished")
            
            if len(matching_faces) > 0:
                # Setup the table columns
                columns = ['Match', 'Similarity', 'Original Image File', 'Latitude', 'Longitude', 'Device', 'Date', 'Time', 'Resized Image Name']
                self.result_table.setColumnCount(len(columns))
                self.result_table.setHorizontalHeaderLabels(columns)
                self.result_table.setRowCount(len(matching_faces))

                for i, (img_hash, original_image_name, face_vector, similarity, resized_image_name) in enumerate(matching_faces):
                    self.result_table.setItem(i, 0, MatchTableWidgetItem(f"Match {i + 1}"))
                    self.result_table.setItem(i, 1, NumericTableWidgetItem(f"{similarity * 100:.2f}%"))
                    self.result_table.setItem(i, 2, QTableWidgetItem(original_image_name))
                    
                    exif_data = face_data.get(img_hash, {}).get('exif_data', {})
                    latitude = exif_data.get('GPSInfo', {}).get('Latitude', '')
                    longitude = exif_data.get('GPSInfo', {}).get('Longitude', '')
                    device = f"{exif_data.get('Make', '')} {exif_data.get('Model', '')}".strip()
                    date = exif_data.get('DateDigitized', '')
                    time = exif_data.get('TimeDigitized', '')

                    self.result_table.setItem(i, 3, QTableWidgetItem(str(latitude)))
                    self.result_table.setItem(i, 4, QTableWidgetItem(str(longitude)))
                    self.result_table.setItem(i, 5, QTableWidgetItem(device))
                    self.result_table.setItem(i, 6, QTableWidgetItem(date))
                    self.result_table.setItem(i, 7, QTableWidgetItem(time))
                    self.result_table.setItem(i, 8, QTableWidgetItem(resized_image_name))
                    # self.result_table.setColumnHidden(8, True)
                    print(f"Storing resized_image_name for row {i}: {resized_image_name}")

                self.result_table.resizeColumnsToContents()
            else:
                self.result_table.setRowCount(0)
                self.result_table.setColumnCount(0)
        except Exception as e:
            logging.exception("An error occurred while handling the face processing done event")
            raise e

    def display_selected_matched_face(self):
        try:
            current_row = self.result_table.currentRow()
            if current_row != -1:
                print("Displaying selected matched face")
                
                # Fetch the 'Resized Image Name' from the table
                resized_image_name = self.result_table.item(current_row, 8).text()
                
                # Replace the .npy extension with .png
                # image_file_name = resized_image_name.replace('.npy', '.png')
                
                matched_face_path = os.path.join(self.output_folder_edit.text(), resized_image_name)
                
                # Print the matched face path for debugging
                print(f"Attempting to read image from: {matched_face_path}")
                
                # Check if the file exists before trying to read
                if not os.path.exists(matched_face_path):
                    print(f"Image file does not exist at: {matched_face_path}")
                    return

                matched_face = cv2.imread(matched_face_path)

                similarity = float(self.result_table.item(current_row, 1).text().replace('%', '')) / 100.0

                # Fetch the 'Original Image Name' from the table CB
                original_image_name = self.result_table.item(current_row, 2).text()

                self.display_matched_face(matched_face, similarity, original_image_name)
                print(f"Retrieved resized_image_name for row {current_row}: {resized_image_name}")

                print("Finished displaying selected matched face")
        except Exception as e:
            logging.exception("An error occurred while displaying the selected matched face")
            raise e

    def update_progress_bar(self, progress):
        print(f"update_progress_bar in face_matcher_app_class called with: {progress}")
        try:
            self.progress_bar.setValue(int(progress))
        except Exception as e:
            logging.exception("An error occurred while updating the progress bar")
            raise e

    def on_result_table_selection_changed(self):
        try:
            self.display_selected_matched_face()
        except Exception as e:
            logging.exception("An error occurred while handling the selection change in the result table")
            raise e

    def open_image_in_default_viewer(self, row, column):
        if column == 2:  # Assuming the 'Original Image File' is in column 2
            original_image_name = self.result_table.item(row, 2).text()
            
            # Assuming the image is in the input folder, construct the full path
            full_path = os.path.join(self.input_folder_edit.text(), original_image_name)
            
            # Open the image using the default viewer
            if os.path.exists(full_path):
                os.startfile(full_path)
            else:
                QMessageBox.critical(self, "Error", f"File not found: {full_path}")


def load_stylesheet(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        logging.exception("An error occurred while loading the stylesheet")
        raise e
