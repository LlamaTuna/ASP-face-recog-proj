'''
@file main.py
main function
'''

import sys
import os
from PyQt6.QtWidgets import QApplication
from face_matcher_app_class import FaceMatcherApp

def set_up_paths():
    """
    Set up necessary paths and environment variables for the application.
    """
    # No matter if the app is frozen or not, point to the correct directory.
    app_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    pyqt6_path = os.path.join(app_path, 'PyQt6') 
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(pyqt6_path, 'Qt', 'plugins')

    if pyqt6_path in sys.path:
        sys.path.remove(pyqt6_path) 

    return os.path.join(app_path, "styles", "dark_theme.qss")

if __name__ == '__main__':
    """
    Main entry point of the Face Matcher application.
    """
    app = QApplication(sys.argv)
    face_matcher_app = FaceMatcherApp()
    face_matcher_app.show()
    app.exec()