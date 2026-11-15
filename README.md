# EEG Viewer

A lightweight EEG visualization and analysis tool built with Python, PySide6, MNE, and PyQtGraph.

## Features

- **EDF/BDF Support**: Read and visualize EEG recordings.
- **EEG Visualization**: Multi-channel waveforms with different colors.
- **Annotations**: View events and double-click to jump to event timestamps.
- **Filtering**: Custom Butterworth bandpass and 50 Hz notch filters.
- **PSD Analysis**: Power spectral density analysis using Welch's method.
- **Navigation**: Adjust time windows, amplitude, and channel selection.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python eeg_viewer.py
```

## Build EXE

```bash
python build_exe.py
```

The executable will be generated at `dist/EEG_Viewer.exe`.

## Requirements

- Python 3.10+
- Windows 10/11
- MNE, NumPy, SciPy, PySide6, PyQtGraph

## Notes

- Original EEG files are not modified.
- Filtering and PSD analysis operate on the selected time window.
- Intended for research and educational use.
