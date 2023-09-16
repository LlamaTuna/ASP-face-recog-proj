# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
<<<<<<< Updated upstream
    datas=[('ASP-face-recog\\Lib\\site-packages\\mtcnn\\data\\mtcnn_weights.npy', 'mtcnn/data'), ('styles\\dark_theme.qss', 'styles')],
=======
    datas=[('ASP-face-recog\\Lib\\site-packages\\mtcnn\\data\\mtcnn_weights.npy', 'mtcnn/data'), ('C:\\Users\\Saul_T_Lode\\ASP-face-recog-proj\\styles\\dark_theme.qss', 'styles/'), ('C:\\Users\\Saul_T_Lode\\ASP-face-recog-proj\\styles\\light_theme.qss', 'styles/')],
>>>>>>> Stashed changes
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
<<<<<<< Updated upstream
    console=False,
=======
    console=True,
>>>>>>> Stashed changes
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
