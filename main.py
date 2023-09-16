'''
@file main.py
main function
'''

import sys
import os
from PyQt6.QtWidgets import QApplication  # Changed PyQt5 to PyQt6
from face_matcher_app_class import FaceMatcherApp

# No matter the app is frozen or not, point to the correct directory.
app_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
pyqt6_path = os.path.join(app_path, 'PyQt6') 
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(pyqt6_path, 'Qt', 'plugins')

# If the app is frozen, add face_recognition library path
if getattr(sys, 'frozen', False):
    library_path = os.path.join(app_path, 'face_recognition')
    sys.path.insert(0, library_path)

if pyqt6_path in sys.path:  # Changed PyQt5 to PyQt6
    sys.path.remove(pyqt6_path) 

dark_theme_path = os.path.join(app_path, "styles", "dark_theme.qss")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    face_matcher_app = FaceMatcherApp()
    face_matcher_app.show()
    app.exec()  # Renamed exec_() to exec() and removed sys.exit()
    


