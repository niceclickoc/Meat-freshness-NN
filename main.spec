# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
    collect_data_files,
)

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=collect_dynamic_libs("xgboost"),
    datas=(
    collect_data_files('xgboost') +
    [("src/models/*.h5", "models"),
    ("src/models/meta/*.joblib", "models/meta")]
    ),
    hiddenimports=[
        'xgboost', 'xgboost.core', 'xgboost.sklearn',
        'PyQt5.QtGui', 'PyQt5.QtCore', 'PyQt5.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MeatFreshness',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

#app = COLLECT(
#    exe,
#    a.binaries,
#    a.zipfiles,
#    a.datas,
#    strip=False,
#    upx=True,
#    name='MeatFreshness',
#)