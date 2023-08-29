from PyQt6.QtWidgets import  QSplitter, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QProgressBar, QTableWidget
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QStyle
from PyQt6.QtWidgets import QHBoxLayout
from console_output import ConsoleWidget
from PyQt6.QtWidgets import QDoubleSpinBox

import logging
import os
import sys

def initUI(self):
    try:
        self.setWindowTitle('Face Finder')
        self.create_menu_bar()

        # CB - set initial stylesheet to light
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # Running in a PyInstaller bundle
            bundle_dir = sys._MEIPASS
        else:
            # Running in a normal Python environment
            bundle_dir = os.path.dirname(os.path.abspath(__file__))

        light_theme_path = os.path.join(bundle_dir, "styles", "light_theme.qss")
        self.setStyleSheet(load_stylesheet(light_theme_path)) # CB removed replace

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

        # Find match and cancel match buttons
        match_buttons_layout = QHBoxLayout()
        # Find match button
        find_match_button = QPushButton('Find match')
        find_match_button.clicked.connect(self.find_match)
        match_buttons_layout.addWidget(find_match_button)
        # Cancel match button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_face_processing)
        match_buttons_layout.addWidget(self.cancel_button) 
        left_panel_layout.addLayout(match_buttons_layout)

        # Image preview
        self.image_preview_label = QLabel()
        self.image_preview_label.setObjectName('image_preview_label')
        left_panel_layout.addWidget(self.image_preview_label)
    
        # Progress bar
        left_panel_layout.addWidget(QLabel('Progress:'))
        self.progress_bar = QProgressBar()
        left_panel_layout.addWidget(self.progress_bar)

        # Add arrow buttons and similarity/original image name label
        self.left_arrow_button = QPushButton()
        self.left_arrow_button.setObjectName('left_arrow_button')
        self.left_arrow_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))
        self.left_arrow_button.clicked.connect(self.previous_matched_face)

        self.right_arrow_button = QPushButton()
        self.right_arrow_button.setObjectName('right_arrow_button')
        self.right_arrow_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self.right_arrow_button.clicked.connect(self.next_matched_face)

        arrows_layout = QHBoxLayout()
        arrows_layout.addWidget(self.left_arrow_button)
            
        # Matched face thumbnail
        self.matched_face_label = QLabel()
        self.matched_face_label.setObjectName('matched_face_label')
            
        arrows_layout.addWidget(self.matched_face_label)
        arrows_layout.addWidget(self.right_arrow_button)
        left_panel_layout.addLayout(arrows_layout)

        # image name and similarity text
        self.similarity_original_image_label = QLabel()
        self.similarity_original_image_label.setObjectName('similarity_original_image_label')
        self.similarity_original_image_label.setWordWrap(True)
        left_panel_layout.addWidget(self.similarity_original_image_label)
        
        # Console output
        self.console_widget = ConsoleWidget()
        right_panel_layout.addWidget(self.console_widget)
        self.console_widget.start_console_output_thread()

        #ouput selected photos
        # Adjust the range and default value
        self.similarity_threshold_spinbox = QDoubleSpinBox(self)
        self.similarity_threshold_spinbox.setRange(0, 100)  # Range from 0% to 100%
        self.similarity_threshold_spinbox.setSingleStep(1)  # Adjust by 1%
        self.similarity_threshold_spinbox.setValue(90)     # Default to 90%
        self.similarity_threshold_spinbox.setFixedWidth(100)
        self.similarity_threshold_spinbox.setSuffix('%')   # Add a '%' suffix to make it clear that it's a percentage
        similarity_label = QLabel('Enter the minimum similarity percentage for images to copy.')
        right_panel_layout.addWidget(similarity_label) 
        right_panel_layout.addWidget(self.similarity_threshold_spinbox)

        self.select_output_directory_button = QPushButton("Select Output Directory", self)
        self.select_output_directory_button.clicked.connect(self.select_output_directory)
        right_panel_layout.addWidget(self.select_output_directory_button)

        self.copy_photos_button = QPushButton("Copy Matching Images", self)
        self.copy_photos_button.clicked.connect(self.copy_matching_photos)
        right_panel_layout.addWidget(self.copy_photos_button)

        self.output_directory = None  # Initialize to None

        #Add tag buttons here.

        # 1. Output Directory Selection Button
        output_directory_layout = QHBoxLayout()
        tags_label = QLabel('Select an output folder for any tagged images from table.')
        right_panel_layout.addWidget(tags_label)
        self.output_directory_edit = QLineEdit()
        output_directory_button = QPushButton('Browse')
        output_directory_button.clicked.connect(self.browse_output_directory)
        output_directory_layout.addWidget(QLabel('Output Directory:'))
        output_directory_layout.addWidget(self.output_directory_edit)
        output_directory_layout.addWidget(output_directory_button)
        right_panel_layout.addLayout(output_directory_layout)

        # 2. Export Tags Button
        export_tags_button = QPushButton('Copy Tagged Images')
        export_tags_button.clicked.connect(self.export_tagged_photos)
        right_panel_layout.addWidget(export_tags_button)

        # Add top_splitter to main_layout
        main_layout.addWidget(top_splitter)

        # Table should be in the main layout, not in the left panel
        self.result_table = QTableWidget(self)
        self.result_table.itemSelectionChanged.connect(self.on_result_table_selection_changed)
        self.result_table.setSortingEnabled(True)
        self.result_table.verticalHeader().setDefaultSectionSize(12);
        
        # Add table to main_layout
        main_layout.addWidget(self.result_table)

    except Exception as e:
        logging.exception("An error occurred while initializing the UI")
        raise e

def load_stylesheet(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        logging.exception("An error occurred while loading the stylesheet")
        raise e