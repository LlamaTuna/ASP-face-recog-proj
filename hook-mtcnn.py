'''
@file hook_mtcnn.py
Pyinstaller dependacies for exe compilation
'''

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files('mtcnn')

print(datas)