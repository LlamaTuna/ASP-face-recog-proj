from PyQt6.QtWidgets import QTextEdit, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal
import sys

class ConsoleOutputThread(QThread):
    signal = pyqtSignal(str)  # signal to emit text

    def write(self, s):
        self.signal.emit(s)  # emit the signal with the text

    def flush(self):
        pass  # needed for file-like interface

class ConsoleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up layout and widgets
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Console output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        layout.addWidget(self.console_output)

        # ConsoleOutputThread instance
        self.console_output_thread = ConsoleOutputThread()
        self.console_output_thread.signal.connect(self.console_output.append)  # connect signal to append method of QTextEdit

        # Replace the standard output with ConsoleOutputThread instance
        sys.stdout = self.console_output_thread

    def start_console_output_thread(self):
        self.console_output_thread.start()

    def clear_console(self):
        self.text_edit.clear()