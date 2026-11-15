
import os
import shutil
import subprocess
import sys

from PIL import Image

APP_NAME = "EEG_Viewer"
MAIN_FILE = "eeg_viewer.py"
ICON_FILE = "icon.png"


def build():
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    if not os.path.exists(MAIN_FILE):
        raise FileNotFoundError(f"找不到 {MAIN_FILE}")

    if not os.path.exists(ICON_FILE):
        raise FileNotFoundError(f"找不到 {ICON_FILE}")

    icon_ico = "icon.ico"

    Image.open(ICON_FILE).convert("RGBA").save(
        icon_ico,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    )

    for path in ["build", "dist", f"{APP_NAME}.spec"]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name", APP_NAME,
        "--icon", icon_ico,
        "--collect-all", "mne",
        "--collect-all", "scipy",
        "--collect-all", "pyqtgraph",
        MAIN_FILE
    ]

    subprocess.run(command, check=True)

    shutil.rmtree("build", ignore_errors=True)

    if os.path.exists(f"{APP_NAME}.spec"):
        os.remove(f"{APP_NAME}.spec")

    exe = os.path.join("dist", f"{APP_NAME}.exe")

    if not os.path.exists(exe):
        raise FileNotFoundError("EXE 生成失败")

    print(f"\n打包成功: {os.path.abspath(exe)}")
    print(f"文件大小: {os.path.getsize(exe) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    try:
        build()
    except Exception as e:
        print(f"\n打包失败: {e}")
        sys.exit(1)
