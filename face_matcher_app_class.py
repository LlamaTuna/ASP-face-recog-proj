import os
import cv2
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QGridLayout, QMessageBox, QScrollArea, QFrame, QTableWidgetItem, QProgressBar, QTableWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from face_detection import save_faces_from_folder, find_matching_face
from gui_elements import NumericTableWidgetItem, MatchTableWidgetItem
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QStyle
from PyQt5.QtWidgets import QHBoxLayout
from face_detection import save_faces_from_folder, find_matching_face
from FaceProcessingThread import FaceProcessingThread



input_folder = "./faceTests/"
output_folder = "./output/"

class FaceMatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dark_theme_enabled = False
        self.initUI()

    def create_menu_bar(self):
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

    def get_matched_face_by_row(self, row):
        if row >= 0 and row < self.result_table.rowCount():
            resized_image_name = self.result_table.item(row, 4).text()
            matched_face_path = os.path.join(self.output_folder_edit.text(), resized_image_name)
            matched_face = cv2.imread(matched_face_path)
            return matched_face
        return None

    def previous_matched_face(self):
        current_row = self.result_table.currentRow()
        if current_row > 0:
            self.result_table.selectRow(current_row - 1)
            matched_face = self.get_matched_face_by_row(current_row - 1)
            if matched_face is not None:
                similarity = float(self.result_table.item(current_row - 1, 1).text().rstrip('%')) / 100
                original_image_name = self.result_table.item(current_row - 1, 2).text()
                self.display_matched_face(matched_face, similarity, original_image_name)

    def next_matched_face(self):
        current_row = self.result_table.currentRow()
        if current_row < self.result_table.rowCount() - 1:
            self.result_table.selectRow(current_row + 1)
            matched_face = self.get_matched_face_by_row(current_row + 1)
            if matched_face is not None:
                similarity = float(self.result_table.item(current_row + 1, 1).text().rstrip('%')) / 100
                original_image_name = self.result_table.item(current_row + 1, 2).text()
                self.display_matched_face(matched_face, similarity, original_image_name)


    def on_result_table_selection_changed(self):
        self.display_selected_matched_face()
    
    def toggle_dark_theme(self):
        if self.dark_theme_enabled:
            self.dark_theme_enabled = False
            self.setStyleSheet("")
        else:
            self.dark_theme_enabled = True
            current_directory = os.path.dirname(os.path.abspath(__file__))
            dark_theme_path = os.path.join(current_directory, "styles", "dark_theme.qss")
            self.setStyleSheet(load_stylesheet(dark_theme_path))

    def initUI(self):
        self.setWindowTitle('Face Matcher')
        self.create_menu_bar()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        self.progress_bar = QProgressBar()
        layout = QGridLayout(main_widget)
        layout.addWidget(QLabel('Progress:'), 5, 0)
        layout.addWidget(self.progress_bar, 5, 1, 1, 2)  # Update the column span from 3 to 2
                
        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area, 0, 0, 1, 4)

        # Create a container widget for the scroll area
        scroll_container = QWidget()
        scroll_area.setWidget(scroll_container)

        # Create a layout for the container widget
        scroll_layout = QVBoxLayout(scroll_container)

        # Input folder
        self.input_folder_edit = QLineEdit()
        input_folder_button = QPushButton('Browse')
        input_folder_button.clicked.connect(self.browse_input_folder)
        scroll_layout.addWidget(QLabel('Input folder:'))
        scroll_layout.addWidget(self.input_folder_edit)
        scroll_layout.addWidget(input_folder_button)

        # Output folder
        self.output_folder_edit = QLineEdit()
        output_folder_button = QPushButton('Browse')
        output_folder_button.clicked.connect(self.browse_output_folder)
        scroll_layout.addWidget(QLabel('Output folder:'))
        scroll_layout.addWidget(self.output_folder_edit)
        scroll_layout.addWidget(output_folder_button)

        # Image to search
        self.image_to_search_edit = QLineEdit()
        image_to_search_button = QPushButton('Browse')
        image_to_search_button.clicked.connect(self.browse_image_to_search)
        scroll_layout.addWidget(QLabel('Image to search for:'))
        scroll_layout.addWidget(self.image_to_search_edit)
        scroll_layout.addWidget(image_to_search_button)

        # Image preview
        self.image_preview_label = QLabel()
        scroll_layout.addWidget(self.image_preview_label)

        # # Matched face thumbnail
        # self.matched_face_label = QLabel()
        # scroll_layout.addWidget(self.matched_face_label)

        # Find match button
        find_match_button = QPushButton('Find match')
        find_match_button.clicked.connect(self.find_match)
     
        scroll_layout.addWidget(find_match_button)

        # Result label
        self.result_table = QTableWidget(self)
        self.result_table.itemSelectionChanged.connect(self.on_result_table_selection_changed)
        self.result_table.setSortingEnabled(True)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 1)


        # Matched face thumbnail
        self.matched_face_label = QLabel()
        scroll_layout.addWidget(self.matched_face_label)

        # Add arrow buttons and similarity/original image name label
        self.left_arrow_button = QPushButton()
        self.left_arrow_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.left_arrow_button.clicked.connect(self.previous_matched_face)

        self.right_arrow_button = QPushButton()
        self.right_arrow_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.right_arrow_button.clicked.connect(self.next_matched_face)

        arrows_layout = QHBoxLayout()
        arrows_layout.addWidget(self.left_arrow_button)
        arrows_layout.addWidget(self.right_arrow_button)
        scroll_layout.addLayout(arrows_layout)

        self.similarity_original_image_label = QLabel()
        scroll_layout.addWidget(self.similarity_original_image_label)


        layout.addWidget(self.result_table, 7, 0, 1, 3)

        layout.setContentsMargins(10, 10, 10, 10)

    def browse_input_folder(self):
        print("Browsing input folder")
        input_folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder', options=QFileDialog.DontUseNativeDialog)
        if input_folder:
            self.input_folder_edit.setText(input_folder)
        print("Finished browsing input folder")

    def browse_output_folder(self):
        print("Browsing output folder")
        output_folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder', options=QFileDialog.DontUseNativeDialog)
        if output_folder:
            self.output_folder_edit.setText(output_folder)
        print("Finished browsing output folder")

    def browse_image_to_search(self):
        print("Browsing image to search")
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Image to Search', '', 'Image files (*.png *.jpeg *.jpg *.bmp *.tiff)', options=QFileDialog.DontUseNativeDialog)
        if file_path:
            self.image_to_search_edit.setText(file_path)
            self.load_image_thumbnail(file_path)
        print("Finished browsing image to search")

    def load_image_thumbnail(self, file_path):
        print("Loading image thumbnail")
        image = QImage(file_path)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(100, 100, aspectRatioMode=Qt.KeepAspectRatio)
        self.image_preview_label.setPixmap(scaled_pixmap)
        print("Finished loading image thumbnail")
        
    def display_matched_face(self, matched_face, similarity, original_image_name):
        print("Displaying matched face")
        height, width, _ = matched_face.shape
        bytes_per_line = width * 3
        q_image = QImage(matched_face.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(100, 100, aspectRatioMode=Qt.KeepAspectRatio)
        self.matched_face_label.setPixmap(scaled_pixmap)
        self.similarity_original_image_label.setText(f"Similarity: {similarity * 100:.2f}% | Original Image: {original_image_name}")
        print("Finished displaying matched face")

    def find_match(self):
        print("Starting find_match")
        input_folder = self.input_folder_edit.text()
        output_folder = self.output_folder_edit.text()
        image_to_search = self.image_to_search_edit.text()

        if not input_folder or not output_folder or not image_to_search:
            QMessageBox.critical(self, "Error", "Please select all required folders and files.")
            return

        self.face_processing_thread = FaceProcessingThread(input_folder, output_folder, image_to_search)
        self.face_processing_thread.processing_done.connect(self.on_processing_finished)
        self.face_processing_thread.progress_signal.connect(self.update_progress_bar)
        self.face_processing_thread.start()
        print("Finished find_match")


    def on_processing_finished(self, matching_faces):
        print("Processing finished")
        if len(matching_faces) > 0:
            self.result_table.setColumnCount(5)
            self.result_table.setHorizontalHeaderLabels(['Match', 'Similarity', 'Original Image File', 'Original Image Hash', 'Resized Image File'])
            self.result_table.setRowCount(len(matching_faces))

            for i, (img_hash, original_image_name, matched_face, similarity, resized_image_name) in enumerate(matching_faces):
                self.result_table.setItem(i, 0, MatchTableWidgetItem(f"Match {i + 1}"))
                self.result_table.setItem(i, 1, NumericTableWidgetItem(f"{similarity * 100:.2f}%"))
                self.result_table.setItem(i, 2, QTableWidgetItem(original_image_name))
                self.result_table.setItem(i, 3, QTableWidgetItem(img_hash))
                self.result_table.setItem(i, 4, QTableWidgetItem(resized_image_name))

                if i == 0:
                    self.display_matched_face(matched_face, similarity, original_image_name)

            self.result_table.resizeColumnsToContents()
        else:
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)

        print("Finished find_match")


    def display_selected_matched_face(self):
        current_row = self.result_table.currentRow()
        if current_row >= 0:
            img_hash = self.result_table.item(current_row, 3).text()
            resized_image_name = self.result_table.item(current_row, 4).text()
            similarity = float(self.result_table.item(current_row, 1).text().rstrip('%')) / 100
            original_image_name = self.result_table.item(current_row, 2).text()

            matched_face_path = os.path.join(self.output_folder_edit.text(), resized_image_name)
            matched_face = cv2.imread(matched_face_path)

            if matched_face is not None:
                self.display_matched_face(matched_face, similarity, original_image_name)
            else:
                print(f"Error: Could not load matched face image at '{matched_face_path}'")

        print("Finished find_match")

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(int(progress))

    def on_result_table_selection_changed(self):
        self.display_selected_matched_face()

    
def load_stylesheet(file_path):
    with open(file_path, "r") as file:
        return file.read()