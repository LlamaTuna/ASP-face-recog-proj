import os
import sys
import logging

def load_stylesheet(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        logging.exception("An error occurred while loading the stylesheet")
        raise e

def apply_dark_theme(main_window):
    try:
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_dir = sys._MEIPASS
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        
        dark_theme_path = os.path.join(bundle_dir, "styles", "dark_theme.qss")
        main_window.setStyleSheet(load_stylesheet(dark_theme_path))
    except Exception as e:
        logging.exception("An error occurred while applying the dark theme")
        raise e

def apply_light_theme(main_window):
    try:
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_dir = sys._MEIPASS
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        
        light_theme_path = os.path.join(bundle_dir, "styles", "light_theme.qss")
        main_window.setStyleSheet(load_stylesheet(light_theme_path))
    except Exception as e:
        logging.exception("An error occurred while applying the light theme")
        raise e
