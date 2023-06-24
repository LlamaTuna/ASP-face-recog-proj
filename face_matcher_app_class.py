import os
import cv2
import sys
# from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMainWindow, QSplitter, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QGridLayout, QMessageBox, QScrollArea, QFrame, QTableWidgetItem, QProgressBar, QTableWidget, QTextEdit
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from face_detection import save_faces_from_folder, find_matching_face
from gui_elements import NumericTableWidgetItem, MatchTableWidgetItem
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QStyle
from PyQt6.QtWidgets import QHBoxLayout
from FaceProcessingThread import FaceProcessingThread
from console_output import ConsoleWidget

import logging
import traceback


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

def load_stylesheet(file_path):
    with open(file_path, "r") as file:
        return file.read()

class FaceMatcherApp(QMainWindow):
    def __init__(self, face_data=None):
        super().__init__()
        self.face_data = face_data
        self.dark_theme_enabled = False
        self.worker = None
        self.initUI()

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

            # assume resized_image_name is a numpy filename,
            # change it to an image filename according to image convention
            image_file_name = resized_image_name.replace('.npy', '.png')  # use your actual extension here
            matched_face_path = os.path.join(self.output_folder_edit.text(), image_file_name)
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

    def toggle_dark_theme(self):
        try:
            if self.dark_theme_enabled:
                self.dark_theme_enabled = False
                self.setStyleSheet("")
            else:
                self.dark_theme_enabled = True
                if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                    # Running in a PyInstaller bundle
                    bundle_dir = sys._MEIPASS
                else:
                    # Running in a normal Python environment
                    bundle_dir = os.path.dirname(os.path.abspath(__file__))

                dark_theme_path = os.path.join(bundle_dir, "styles", "dark_theme.qss")
                self.setStyleSheet(load_stylesheet(dark_theme_path).replace('QMainWindow', 'QWidget'))
        except Exception as e:
            logging.exception("An error occurred while toggling the dark theme")
            raise e


    def initUI(self):
        try:
            self.setWindowTitle('Face Matcher')
            self.create_menu_bar()

            main_widget = QWidget()
            self.setCentralWidget(main_widget)

            main_layout = QVBoxLayout(main_widget)

            top_splitter = QSplitter(Qt.Orientation.Horizontal)
            left_panel_widget = QWidget()
            right_panel_widget = QWidget()

            top_splitter.addWidget(left_panel_widget)
            top_splitter.addWidget(right_panel_widget)

            left_panel_layout = QVBoxLayout(left_panel_widget)
            right_panel_layout = QVBoxLayout(right_panel_widget)

            
            # Image to search
            image_to_search_layout = QHBoxLayout()
            self.image_to_search_edit = QLineEdit()
            image_to_search_button = QPushButton('Browse')
            image_to_search_button.clicked.connect(self.browse_image_to_search)
            image_to_search_layout.addWidget(QLabel('Image to search for:'))
            image_to_search_layout.addWidget(self.image_to_search_edit)
            image_to_search_layout.addWidget(image_to_search_button)
            left_panel_layout.addLayout(image_to_search_layout)

            # Input folder
            input_folder_layout = QHBoxLayout()
            self.input_folder_edit = QLineEdit()
            input_folder_button = QPushButton('Browse')
            input_folder_button.clicked.connect(self.browse_input_folder)
            input_folder_layout.addWidget(QLabel('Input folder:'))
            input_folder_layout.addWidget(self.input_folder_edit)
            input_folder_layout.addWidget(input_folder_button)
            left_panel_layout.addLayout(input_folder_layout)

            # Output folder
            output_folder_layout = QHBoxLayout()  
            self.output_folder_edit = QLineEdit()
            output_folder_button = QPushButton('Browse')
            output_folder_button.clicked.connect(self.browse_output_folder)
            output_folder_layout.addWidget(QLabel('Output folder:'))
            output_folder_layout.addWidget(self.output_folder_edit)
            output_folder_layout.addWidget(output_folder_button)
            left_panel_layout.addLayout(output_folder_layout)


            # Find match button
            find_match_button = QPushButton('Find match')
            find_match_button.clicked.connect(self.find_match)
            left_panel_layout.addWidget(find_match_button)

            # Image preview
            self.image_preview_label = QLabel()
            left_panel_layout.addWidget(self.image_preview_label)

            # Matched face thumbnail
            self.matched_face_label = QLabel()
            left_panel_layout.addWidget(self.matched_face_label)

            # image name and similarity text
            self.similarity_original_image_label = QLabel()
            left_panel_layout.addWidget(self.similarity_original_image_label)

            # Add arrow buttons and similarity/original image name label
            self.left_arrow_button = QPushButton()
            self.left_arrow_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))
            self.left_arrow_button.clicked.connect(self.previous_matched_face)

            self.right_arrow_button = QPushButton()
            self.right_arrow_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
            self.right_arrow_button.clicked.connect(self.next_matched_face)

            arrows_layout = QHBoxLayout()
            arrows_layout.addWidget(self.left_arrow_button)
            arrows_layout.addWidget(self.right_arrow_button)
            left_panel_layout.addLayout(arrows_layout)

            # Progress bar
            left_panel_layout.addWidget(QLabel('Progress:'))
            self.progress_bar = QProgressBar()
            left_panel_layout.addWidget(self.progress_bar)
           
            # Console output
            console_widget = ConsoleWidget()
            right_panel_layout.addWidget(console_widget)
            console_widget.start_console_output_thread()

            # Add top_splitter to main_layout
            main_layout.addWidget(top_splitter)

            # Table should be in the main layout, not in the left panel
            self.result_table = QTableWidget(self)
            self.result_table.itemSelectionChanged.connect(self.on_result_table_selection_changed)
            self.result_table.setSortingEnabled(True)
            
            # Add table to main_layout
            main_layout.addWidget(self.result_table)

        except Exception as e:
            logging.exception("An error occurred while initializing the UI")
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
                self.result_table.setColumnCount(5)
                self.result_table.setHorizontalHeaderLabels(['Match', 'Similarity', 'Original Image File', 'Image Hash', 'Resized Image'])
                self.result_table.setRowCount(len(matching_faces))

                for i, (img_hash, original_image_name, face_vector, similarity, resized_image_name) in enumerate(matching_faces):
                    self.result_table.setItem(i, 0, MatchTableWidgetItem(f"Match {i + 1}"))
                    self.result_table.setItem(i, 1, NumericTableWidgetItem(f"{similarity * 100:.2f}%"))
                    self.result_table.setItem(i, 2, QTableWidgetItem(original_image_name))
                    self.result_table.setItem(i, 3, QTableWidgetItem(img_hash))
                    self.result_table.setItem(i, 4, QTableWidgetItem(resized_image_name))

                    if i == 0:
                        matched_face_path = os.path.join(self.output_folder_edit.text(), resized_image_name)
                        matched_face = cv2.imread(matched_face_path)
                        if matched_face is not None:
                            self.display_matched_face(matched_face, similarity, original_image_name)

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
                img_hash = self.result_table.item(current_row, 3).text()
                resized_image_name = self.result_table.item(current_row, 4).text()
                image_file_name = resized_image_name.replace('.npy', '.png')  # Adjust to your actual extension

                matched_face_path = os.path.join(self.output_folder_edit.text(), image_file_name)
                matched_face = cv2.imread(matched_face_path)

                similarity = float(self.result_table.item(current_row, 1).text().replace('%', '')) / 100.0
                original_image_name = self.result_table.item(current_row, 2).text()

                self.display_matched_face(matched_face, similarity, original_image_name)
                print("Finished displaying selected matched face")
        except Exception as e:
            logging.exception("An error occurred while displaying the selected matched face")
            raise e


    def update_progress_bar(self, progress):
        print(f"update_progress_bar called with: {progress}")
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


def load_stylesheet(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        logging.exception("An error occurred while loading the stylesheet")
        raise e




               



# 
# import os
# import cv2
# import sys
# from PyQt6.QtWidgets import QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QGridLayout, QMessageBox, QScrollArea, QFrame, QTableWidgetItem, QProgressBar, QTableWidget, QTextEdit
# from PyQt6.QtGui import QImage, QPixmap
# from PyQt6.QtCore import Qt
# from face_detection import save_faces_from_folder, find_matching_face
# from gui_elements import NumericTableWidgetItem, MatchTableWidgetItem
# from PyQt6.QtGui import QAction
# from PyQt6.QtWidgets import QStyle
# from PyQt6.QtWidgets import QHBoxLayout
# from FaceProcessingThread import FaceProcessingThread
# import logging
# import traceback


# try:
#     logging.basicConfig(filename=r'.\debug.log',
#                         filemode='w',
#                         level=logging.DEBUG,
#                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logging.debug('Logging setup successful')
#     print("Logging setup successful.")
# except Exception as e:
#     print(f"Logging setup failed: {str(e)}")

# logger = logging.getLogger()

# def load_stylesheet(file_path):
#     with open(file_path, "r") as file:
#         return file.read()

# class FaceMatcherApp(QMainWindow):
#     def __init__(self, face_data = None):
#         super().__init__()
#         self.face_data = face_data
#         self.dark_theme_enabled = False
#         self.initUI()

#     def create_menu_bar(self):
#         menubar = self.menuBar()

#         # Create File menu
#         file_menu = menubar.addMenu('File')

#         # Create Exit action
#         exit_action = QAction('Exit', self)
#         exit_action.triggered.connect(self.close)
#         file_menu.addAction(exit_action)

#         # Create View menu
#         view_menu = menubar.addMenu('View')

#         # Create Toggle Dark Theme action
#         toggle_dark_theme_action = QAction('Toggle Dark Theme', self)
#         toggle_dark_theme_action.triggered.connect(self.toggle_dark_theme)
#         view_menu.addAction(toggle_dark_theme_action)

#     def get_matched_face_by_row(self, row):
#         if row >= 0 and row < self.result_table.rowCount():
#             resized_image_name = self.result_table.item(row, 4).text()
            
#             # assume resized_image_name is a numpy filename, 
#             # change it to an image filename according to your convention
#             image_file_name = resized_image_name.replace('.npy', '.png')  # use your actual extension here
#             matched_face_path = os.path.join(self.output_folder_edit.text(), image_file_name)
#             matched_face = cv2.imread(matched_face_path)

#             return matched_face
#         return None


#     def previous_matched_face(self):
#         current_row = self.result_table.currentRow()
#         if current_row > 0:
#             self.result_table.selectRow(current_row - 1)
#             matched_face = self.get_matched_face_by_row(current_row - 1)
#             if matched_face is not None:
#                 similarity = float(self.result_table.item(current_row - 1, 1).text().rstrip('%')) / 100
#                 original_image_name = self.result_table.item(current_row - 1, 2).text()
#                 self.display_matched_face(matched_face, similarity, original_image_name)

#     def next_matched_face(self):
#         current_row = self.result_table.currentRow()
#         if current_row < self.result_table.rowCount() - 1:
#             self.result_table.selectRow(current_row + 1)
#             matched_face = self.get_matched_face_by_row(current_row + 1)
#             if matched_face is not None:
#                 similarity = float(self.result_table.item(current_row + 1, 1).text().rstrip('%')) / 100
#                 original_image_name = self.result_table.item(current_row + 1, 2).text()
#                 self.display_matched_face(matched_face, similarity, original_image_name)


#     def on_result_table_selection_changed(self):
#         current_row = self.result_table.currentRow()
#         if current_row != -1:
#             print("Selection changed")  # debug print
#             img_hash = self.result_table.item(current_row, 3).text()
#             print(f"Selected image hash: {img_hash}")
                    
#             # access face_data using the selected image hash
#             face_info = self.face_data.get(img_hash, {})
                    
#             # print the face_info to the console
#             print(face_info)

#             # Also, you can display it in any widget you want like QTextEdit
#             self.console_output.append("\nSelected face:")
#             self.console_output.append(f"\nImage file: {face_info.get('file_name', 'N/A')}")
                    
#             # EXIF data
#             self.console_output.append("\nEXIF data:")
#             exif_data = face_info.get('exif_data', {})
#             for k, v in exif_data.items():
#                 self.console_output.append(f"{k}: {v}")
                        
#             self.display_selected_matched_face()

            
#     def on_table_selection_changed_and_display_face(self):  # renamed this method
#         self.display_selected_matched_face()
    
#     def toggle_dark_theme(self):
#             if self.dark_theme_enabled:
#                 self.dark_theme_enabled = False
#                 self.setStyleSheet("")
#             else:
#                 self.dark_theme_enabled = True
#                 if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
#                     # Running in a PyInstaller bundle
#                     bundle_dir = sys._MEIPASS
#                 else:
#                     # Running in a normal Python environment
#                     bundle_dir = os.path.dirname(os.path.abspath(__file__))
                
#                 dark_theme_path = os.path.join(bundle_dir, "styles", "dark_theme.qss")
#                 self.setStyleSheet(load_stylesheet(dark_theme_path).replace('QMainWindow', 'QWidget'))



#     def initUI(self):
#         self.setWindowTitle('Face Matcher')
#         self.create_menu_bar()

#         main_widget = QWidget()
#         self.setCentralWidget(main_widget)

#         top_level_layout = QHBoxLayout(main_widget)
#         left_panel_layout = QVBoxLayout()
#         top_level_layout.addLayout(left_panel_layout)


#         # Input folder
#         input_folder_layout = QHBoxLayout()
#         self.input_folder_edit = QLineEdit()
#         input_folder_button = QPushButton('Browse')
#         input_folder_button.clicked.connect(self.browse_input_folder)
#         input_folder_layout.addWidget(QLabel('Input folder:'))
#         input_folder_layout.addWidget(self.input_folder_edit)
#         input_folder_layout.addWidget(input_folder_button)
#         left_panel_layout.addLayout(input_folder_layout)

#         # Output folder
#         output_folder_layout = QHBoxLayout()
#         self.output_folder_edit = QLineEdit()
#         output_folder_button = QPushButton('Browse')
#         output_folder_button.clicked.connect(self.browse_output_folder)
#         output_folder_layout.addWidget(QLabel('Output folder:'))
#         output_folder_layout.addWidget(self.output_folder_edit)
#         output_folder_layout.addWidget(output_folder_button)
#         left_panel_layout.addLayout(output_folder_layout)

#         # Image to search
#         image_to_search_layout = QHBoxLayout()
#         self.image_to_search_edit = QLineEdit()
#         image_to_search_button = QPushButton('Browse')
#         image_to_search_button.clicked.connect(self.browse_image_to_search)
#         image_to_search_layout.addWidget(QLabel('Image to search for:'))
#         image_to_search_layout.addWidget(self.image_to_search_edit)
#         image_to_search_layout.addWidget(image_to_search_button)
#         left_panel_layout.addLayout(image_to_search_layout)

#         # Find match button
#         find_match_button = QPushButton('Find match')
#         find_match_button.clicked.connect(self.find_match)
#         left_panel_layout.addWidget(find_match_button)

#         # Image preview
#         self.image_preview_label = QLabel()
#         left_panel_layout.addWidget(self.image_preview_label)

#         # Matched face thumbnail
#         self.matched_face_label = QLabel()
#         left_panel_layout.addWidget(self.matched_face_label)

#         #image name and similarity text
#         self.similarity_original_image_label = QLabel()
#         left_panel_layout.addWidget(self.similarity_original_image_label)


#         # Add arrow buttons and similarity/original image name label
#         self.left_arrow_button = QPushButton()
#         self.left_arrow_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))

#         self.right_arrow_button = QPushButton()
#         self.right_arrow_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
#         self.right_arrow_button.clicked.connect(self.next_matched_face)

#         arrows_layout = QHBoxLayout()
#         arrows_layout.addWidget(self.left_arrow_button)
#         arrows_layout.addWidget(self.right_arrow_button)
#         left_panel_layout.addLayout(arrows_layout)



#         # Progress bar
#         left_panel_layout.addWidget(QLabel('Progress:'))
#         self.progress_bar = QProgressBar()
#         left_panel_layout.addWidget(self.progress_bar)

#         # Result label
#         self.result_table = QTableWidget(self)
#         self.result_table.itemSelectionChanged.connect(self.on_result_table_selection_changed)
#         self.result_table.setSortingEnabled(True)
#         left_panel_layout.addWidget(self.result_table)

#         # Console output
#         self.console_output = QTextEdit()
#         self.console_output.setReadOnly(True)
#         top_level_layout.addWidget(self.console_output)
        
#         class ConsoleOutput:
#                 def __init__(self, output_widget):
#                     self.output_widget = output_widget

#                 def write(self, s):
#                     self.output_widget.append(s)  # append the output at the end

#                 def flush(self):
#                     pass  # needed for file-like interface

#         console_output = ConsoleOutput(self.console_output)
#         sys.stdout = console_output  # replace the standard output


#     def browse_input_folder(self):
#         print("Browsing input folder")
#         input_folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
#         if input_folder:
#             self.input_folder_edit.setText(input_folder)
#         print("Finished browsing input folder")

#     def browse_output_folder(self):
#         print("Browsing output folder")
#         output_folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
#         if output_folder:
#             self.output_folder_edit.setText(output_folder)
#         print("Finished browsing output folder")

#     def browse_image_to_search(self):
#         print("Browsing image to search")
#         file_path, _ = QFileDialog.getOpenFileName(self, 'Select Image to Search', '', 'Image files (*.png *.jpeg *.jpg *.bmp *.tiff)')
#         if file_path:
#             self.image_to_search_edit.setText(file_path)
#             self.load_image_thumbnail(file_path)
#         print("Finished browsing image to search")

#     def load_image_thumbnail(self, file_path):
#         print("Loading image thumbnail")
#         image = QImage(file_path)
#         pixmap = QPixmap.fromImage(image)
#         scaled_pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
#         self.image_preview_label.setPixmap(scaled_pixmap)
#         print("Finished loading image thumbnail")
        
#     def display_matched_face(self, matched_face, similarity, original_image_name):
#         print("Displaying matched face")
#         height, width, _ = matched_face.shape
#         bytes_per_line = width * 3
#         q_image = QImage(matched_face.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
#         pixmap = QPixmap.fromImage(q_image)
#         scaled_pixmap = pixmap.scaled(100, 100, aspectMode=Qt.AspectRatioMode.KeepAspectRatio)
#         self.matched_face_label.setPixmap(scaled_pixmap)
#         self.similarity_original_image_label.setText(f"Similarity: {similarity * 100:.2f}% | Original Image: {original_image_name}")
#         print("Finished displaying matched face")

#     def find_match(self):
#         logging.debug("Starting find_match")
#         input_folder = self.input_folder_edit.text()
#         output_folder = self.output_folder_edit.text()
#         image_to_search = self.image_to_search_edit.text()

#         if not input_folder or not output_folder or not image_to_search:
#             QMessageBox.critical(self, "Error", "Please select all required folders and files.")
#             return

#         try:
#             self.face_processing_thread = FaceProcessingThread(input_folder, output_folder, image_to_search)
#             self.face_processing_thread.processing_done.connect(self.on_face_processing_done)
#             self.face_processing_thread.progress_signal.connect(self.update_progress_bar)
#             self.face_processing_thread.start()
#             logging.debug("FaceProcessingThread started successfully")
#         except Exception as e:
#             logging.exception("Failed to start FaceProcessingThread")
#             raise e  # re-raise the exception after logging
#         logging.debug("Finished find_match")


#     def on_face_processing_done(self, result):
#         try:
#             matching_faces, face_data = result
#             self.face_data = face_data
#             logging.debug("Processing finished")
#             if len(matching_faces) > 0:
#                 self.result_table.setColumnCount(5)
#                 self.result_table.setHorizontalHeaderLabels(['Match', 'Similarity', 'Original Image File', 'Image Hash', 'Resized Image'])
#                 self.result_table.setRowCount(len(matching_faces))

#                 for i, (img_hash, original_image_name, face_vector, similarity, resized_image_name) in enumerate(matching_faces):
#                     self.result_table.setItem(i, 0, MatchTableWidgetItem(f"Match {i + 1}"))
#                     self.result_table.setItem(i, 1, NumericTableWidgetItem(f"{similarity * 100:.2f}%"))
#                     self.result_table.setItem(i, 2, QTableWidgetItem(original_image_name))
#                     self.result_table.setItem(i, 3, QTableWidgetItem(img_hash))
#                     self.result_table.setItem(i, 4, QTableWidgetItem(resized_image_name))

#                     if i == 0:
#                         matched_face_path = os.path.join(self.output_folder_edit.text(), resized_image_name)
#                         matched_face = cv2.imread(matched_face_path)
#                         if matched_face is not None:
#                             self.display_matched_face(matched_face, similarity, original_image_name)

#                 self.result_table.resizeColumnsToContents()
#             else:
#                 self.result_table.setRowCount(0)
#                 self.result_table.setColumnCount(0)
#         except Exception as e:
#             logging.exception("Error occurred during processing")
#             print(traceback.format_exc())  # Print detailed exception information
#             raise e  # Re-raise the exception after logging


#     def display_selected_matched_face(self):
#         current_row = self.result_table.currentRow()
#         if current_row != -1:
#             print("Displaying selected matched face")
#             img_hash = self.result_table.item(current_row, 3).text()
#             resized_image_name = self.result_table.item(current_row, 4).text()
#             image_file_name = resized_image_name.replace('.npy', '.png')  # Adjust to your actual extension

#             matched_face_path = os.path.join(self.output_folder_edit.text(), image_file_name)
#             matched_face = cv2.imread(matched_face_path)

#             similarity = float(self.result_table.item(current_row, 1).text().replace('%', '')) / 100.0
#             original_image_name = self.result_table.item(current_row, 2).text()

#             self.display_matched_face(matched_face, similarity, original_image_name)
#             print("Finished displaying selected matched face")


#     def update_progress_bar(self, progress):
#         self.progress_bar.setValue(int(progress))

#     def on_result_table_selection_changed(self):
#         self.display_selected_matched_face()

    
# def load_stylesheet(file_path):
#     with open(file_path, "r") as file:
#         return file.read()