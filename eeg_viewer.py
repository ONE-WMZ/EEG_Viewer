
import os
import sys
import numpy as np
import mne
import pyqtgraph as pg

from scipy.signal import butter, sosfiltfilt, welch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QSlider,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSplitter, QAbstractItemView, QDialog, QCheckBox
)


class EEGViewer(QMainWindow):
    COLORS = [
        "#E53935", "#1E88E5", "#43A047", "#8E24AA",
        "#FB8C00", "#00ACC1", "#D81B60", "#6D4C41",
        "#3949AB", "#7CB342", "#F4511E", "#00897B",
        "#5E35B1", "#C0CA33", "#039BE5", "#8D6E63",
        "#F06292", "#546E7A", "#9E9D24", "#7E57C2",
        "#FF7043", "#26A69A", "#EC407A", "#5C6BC0",
        "#66BB6A", "#FFA726", "#AB47BC", "#29B6F6",
        "#D4E157", "#EF5350", "#42A5F5", "#26A69A"
    ]

    def __init__(self):
        super().__init__()

        self.raw = None
        self.file_path = ""
        self.start = 0.0
        self.event_data = []
        self.channel_colors = {}

        self.setWindowTitle("EEG Viewer - EDF")
        self.resize(1550, 950)

        pg.setConfigOptions(antialias=False, background="w", foreground="k")

        self.init_ui()

    # --------------------------------------------------
    # UI
    # --------------------------------------------------

    def init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Top toolbar
        toolbar = QHBoxLayout()
        layout.addLayout(toolbar)

        open_btn = QPushButton("打开 EDF")
        open_btn.clicked.connect(self.open_edf)
        toolbar.addWidget(open_btn)

        toolbar.addWidget(QLabel("时间窗口(s):"))

        self.window = QDoubleSpinBox()
        self.window.setRange(1, 120)
        self.window.setValue(5)
        self.window.valueChanged.connect(self.update_range)
        toolbar.addWidget(self.window)

        toolbar.addWidget(QLabel("幅值范围(µV):"))

        self.scale = QDoubleSpinBox()
        self.scale.setRange(1, 10000)
        self.scale.setValue(100)
        self.scale.valueChanged.connect(self.plot_eeg)
        toolbar.addWidget(self.scale)

        toolbar.addStretch()

        self.spectrum_btn = QPushButton("频谱分析")
        self.spectrum_btn.clicked.connect(self.show_spectrum)
        toolbar.addWidget(self.spectrum_btn)

        self.event_btn = QPushButton("隐藏事件列表")
        self.event_btn.clicked.connect(self.toggle_events)
        toolbar.addWidget(self.event_btn)

        # Filter toolbar
        filter_bar = QHBoxLayout()
        layout.addLayout(filter_bar)

        self.filter_enabled = QCheckBox("启用滤波")
        self.filter_enabled.setChecked(False)
        self.filter_enabled.stateChanged.connect(self.plot_eeg)
        filter_bar.addWidget(self.filter_enabled)

        filter_bar.addWidget(QLabel("低频(Hz):"))

        self.low_freq = QDoubleSpinBox()
        self.low_freq.setRange(0.1, 100)
        self.low_freq.setValue(0.5)
        self.low_freq.setSingleStep(0.5)
        self.low_freq.valueChanged.connect(self.plot_eeg)
        filter_bar.addWidget(self.low_freq)

        filter_bar.addWidget(QLabel("高频(Hz):"))

        self.high_freq = QDoubleSpinBox()
        self.high_freq.setRange(1, 1000)
        self.high_freq.setValue(50)
        self.high_freq.setSingleStep(1)
        self.high_freq.valueChanged.connect(self.plot_eeg)
        filter_bar.addWidget(self.high_freq)

        self.notch_enabled = QCheckBox("50 Hz 陷波")
        self.notch_enabled.setChecked(True)
        self.notch_enabled.stateChanged.connect(self.plot_eeg)
        filter_bar.addWidget(self.notch_enabled)

        filter_bar.addStretch()

        self.filter_info = QLabel("当前显示：原始 EEG")
        filter_bar.addWidget(self.filter_info)

        # File information
        self.info = QLabel("未加载 EDF 文件")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        # Main splitter
        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter, 1)

        # Left: channel list
        channel_widget = QWidget()
        channel_layout = QVBoxLayout(channel_widget)

        channel_layout.addWidget(QLabel("EEG 通道"))

        channel_buttons = QHBoxLayout()

        select_all = QPushButton("全选")
        select_all.clicked.connect(self.select_all_channels)
        channel_buttons.addWidget(select_all)

        select_none = QPushButton("清空")
        select_none.clicked.connect(self.clear_channels)
        channel_buttons.addWidget(select_none)

        channel_layout.addLayout(channel_buttons)

        self.channels = QListWidget()
        self.channels.setSelectionMode(QAbstractItemView.MultiSelection)
        self.channels.itemSelectionChanged.connect(self.plot_eeg)
        channel_layout.addWidget(self.channels)

        self.splitter.addWidget(channel_widget)

        # Center: EEG waveform
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        plot_layout.addWidget(QLabel("EEG Waveforms"))

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Channels")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setClipToView(True)

        plot_layout.addWidget(self.plot)
        self.splitter.addWidget(plot_widget)

        # Right: Annotation events
        self.event_widget = QWidget()
        event_layout = QVBoxLayout(self.event_widget)

        event_layout.addWidget(QLabel("Annotation Events"))

        self.events = QTableWidget(0, 3)
        self.events.setHorizontalHeaderLabels([
            "事件", "时间(s)", "持续(s)"
        ])

        header = self.events.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)

        self.events.setColumnWidth(0, 100)
        self.events.setColumnWidth(1, 80)
        self.events.setColumnWidth(2, 80)

        self.events.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.events.setSelectionMode(QAbstractItemView.SingleSelection)
        self.events.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.events.cellDoubleClicked.connect(self.jump_to_event)

        event_layout.addWidget(self.events)

        self.splitter.addWidget(self.event_widget)
        self.splitter.setSizes([170, 950, 300])

        # Navigation
        navigation = QHBoxLayout()
        layout.addLayout(navigation)

        prev_btn = QPushButton("◀ 上一页")
        prev_btn.clicked.connect(lambda: self.move(-self.window.value()))
        navigation.addWidget(prev_btn)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.seek)
        navigation.addWidget(self.slider, 1)

        next_btn = QPushButton("下一页 ▶")
        next_btn.clicked.connect(lambda: self.move(self.window.value()))
        navigation.addWidget(next_btn)

        self.time_label = QLabel("0.0 / 0.0 s")
        navigation.addWidget(self.time_label)

    # --------------------------------------------------
    # EDF loading
    # --------------------------------------------------

    def open_edf(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择脑电文件",
            "",
            "EEG Files (*.edf *.bdf);;All Files (*)"
        )

        if not path:
            return

        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return

        if self.raw is not None:
            self.raw.close()

        self.raw = raw
        self.file_path = path
        self.start = 0.0

        self.load_channels()
        self.load_annotations()
        self.update_file_info()
        self.update_range()

        self.statusBar().showMessage(
            f"已加载: {os.path.basename(path)}"
        )

    def update_file_info(self):
        sfreq = self.raw.info["sfreq"]
        duration = self.raw.n_times / sfreq

        self.info.setText(
            f"文件: {os.path.basename(self.file_path)}   |   "
            f"通道: {len(self.raw.ch_names)}   |   "
            f"采样率: {sfreq:.2f} Hz   |   "
            f"时长: {duration:.2f} s   |   "
            f"Annotation: {len(self.raw.annotations)}"
        )

    # --------------------------------------------------
    # Channel management
    # --------------------------------------------------

    def load_channels(self):
        self.channels.blockSignals(True)
        self.channels.clear()
        self.channel_colors.clear()

        for i, name in enumerate(self.raw.ch_names):
            color = self.COLORS[i % len(self.COLORS)]
            self.channel_colors[name] = color

            self.channels.addItem(name)
            item = self.channels.item(self.channels.count() - 1)
            item.setForeground(pg.mkColor(color))
            item.setSelected(True)

        self.channels.blockSignals(False)

    def select_all_channels(self):
        self.channels.selectAll()

    def clear_channels(self):
        self.channels.clearSelection()

    # --------------------------------------------------
    # Annotation
    # --------------------------------------------------

    def load_annotations(self):
        self.events.setRowCount(0)
        self.event_data = []

        if self.raw is None:
            return

        for onset, duration, description in zip(
            self.raw.annotations.onset,
            self.raw.annotations.duration,
            self.raw.annotations.description
        ):
            onset = float(onset)
            duration = float(duration)
            description = str(description)

            self.event_data.append((onset, duration, description))

            row = self.events.rowCount()
            self.events.insertRow(row)

            for col, value in enumerate([
                description,
                f"{onset:.3f}",
                f"{duration:.3f}"
            ]):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.events.setItem(row, col, item)

        self.events.resizeRowsToContents()

    def toggle_events(self):
        visible = self.event_widget.isVisible()
        self.event_widget.setVisible(not visible)
        self.event_btn.setText(
            "显示事件列表" if visible else "隐藏事件列表"
        )

    def jump_to_event(self, row, column):
        if self.raw is None or not 0 <= row < len(self.event_data):
            return

        onset, duration, description = self.event_data[row]

        window = self.window.value()
        total = self.raw.n_times / self.raw.info["sfreq"]

        self.start = max(
            0,
            min(onset - window / 2, max(0, total - window))
        )

        self.slider.setValue(round(self.start * 10))

        self.statusBar().showMessage(
            f"Event: {description} | "
            f"Onset: {onset:.3f}s | "
            f"Duration: {duration:.3f}s"
        )

    # --------------------------------------------------
    # Filtering
    # --------------------------------------------------

    def get_segment_data(self, names, start, stop):
        sfreq = self.raw.info["sfreq"]

        # Read extra samples around the display window
        padding = round(2 * sfreq)
        ext_start = max(0, start - padding)
        ext_stop = min(self.raw.n_times, stop + padding)

        data = self.raw.get_data(
            picks=names,
            start=ext_start,
            stop=ext_stop
        ) * 1e6

        if self.filter_enabled.isChecked():
            nyquist = sfreq / 2
            low = self.low_freq.value()
            high = self.high_freq.value()

            high = min(high, nyquist * 0.99)

            if low >= high:
                raise ValueError(
                    f"滤波频率设置错误：低频必须小于高频，"
                    f"当前 Nyquist={nyquist:.2f} Hz"
                )

            # Notch filter
            if self.notch_enabled.isChecked() and nyquist > 51:
                sos_notch = butter(
                    4,
                    [49, 51],
                    btype="bandstop",
                    fs=sfreq,
                    output="sos"
                )
                data = sosfiltfilt(sos_notch, data, axis=-1)

            # Bandpass filter
            sos_band = butter(
                4,
                [low, high],
                btype="bandpass",
                fs=sfreq,
                output="sos"
            )
            data = sosfiltfilt(sos_band, data, axis=-1)

        # Return only the requested display segment
        offset = start - ext_start
        return data[:, offset:offset + stop - start]

    # --------------------------------------------------
    # Time navigation
    # --------------------------------------------------

    def update_range(self):
        if self.raw is None:
            return

        duration = self.raw.n_times / self.raw.info["sfreq"]
        maximum = max(0, duration - self.window.value())

        self.start = min(self.start, maximum)

        self.slider.blockSignals(True)
        self.slider.setRange(0, round(maximum * 10))
        self.slider.setValue(round(self.start * 10))
        self.slider.blockSignals(False)

        self.plot_eeg()

    def seek(self, value):
        if self.raw is None:
            return

        self.start = value / 10
        self.plot_eeg()

    def move(self, seconds):
        if self.raw is None:
            return

        duration = self.raw.n_times / self.raw.info["sfreq"]
        maximum = max(0, duration - self.window.value())

        self.start = max(0, min(self.start + seconds, maximum))
        self.slider.setValue(round(self.start * 10))

    # --------------------------------------------------
    # EEG visualization
    # --------------------------------------------------

    def plot_eeg(self):
        if self.raw is None:
            return

        names = [
            item.text()
            for item in self.channels.selectedItems()
        ]

        self.plot.clear()

        if not names:
            self.time_label.setText("未选择通道")
            return

        sfreq = self.raw.info["sfreq"]
        duration = self.raw.n_times / sfreq

        start = min(
            int(self.start * sfreq),
            self.raw.n_times - 1
        )

        stop = min(
            start + round(self.window.value() * sfreq),
            self.raw.n_times
        )

        try:
            data = self.get_segment_data(names, start, stop)
        except ValueError as e:
            self.statusBar().showMessage(str(e))
            return

        times = np.arange(start, stop) / sfreq

        # Reduce plotting points
        step = max(1, int(np.ceil(len(times) / 3000)))
        times = times[::step]
        data = data[:, ::step]

        amplitude = self.scale.value()
        spacing = amplitude * 3
        n = len(names)

        ticks = []

        for i, (name, signal) in enumerate(zip(names, data)):
            offset = (n - i - 1) * spacing
            color = self.channel_colors[name]

            self.plot.plot(
                times,
                signal + offset,
                pen=pg.mkPen(color, width=1.2)
            )

            ticks.append((offset, name))

        self.plot.getAxis("left").setTicks([ticks])
        self.plot.setYRange(-spacing, n * spacing, padding=0)

        self.plot.setXRange(
            self.start,
            min(self.start + self.window.value(), duration),
            padding=0
        )

        # Annotation markers
        for onset, duration_event, description in self.event_data:
            if self.start <= onset <= self.start + self.window.value():
                line = pg.InfiniteLine(
                    pos=onset,
                    angle=90,
                    pen=pg.mkPen(
                        "#DB3333",
                        width=1.5,
                        style=Qt.DashLine
                    )
                )
                self.plot.addItem(line)

                text = pg.TextItem(
                    f"{description} ({onset:.2f}s)",
                    color="#DB3333",
                    anchor=(0, 0)
                )

                text.setPos(
                    onset,
                    n * spacing - spacing * 0.4
                )

                self.plot.addItem(text)

        self.time_label.setText(
            f"{self.start:.1f} / {duration:.1f} s"
        )

        if self.filter_enabled.isChecked():
            self.filter_info.setText(
                f"当前显示：{self.low_freq.value():g}–"
                f"{self.high_freq.value():g} Hz"
                + (" + 50 Hz Notch" if self.notch_enabled.isChecked() and sfreq > 102 else "")
            )
        else:
            self.filter_info.setText("当前显示：原始 EEG")

    # --------------------------------------------------
    # PSD spectrum analysis
    # --------------------------------------------------

    def show_spectrum(self):
        if self.raw is None:
            QMessageBox.warning(self, "提示", "请先加载 EDF 文件")
            return

        names = [
            item.text()
            for item in self.channels.selectedItems()
        ]

        if not names:
            QMessageBox.warning(self, "提示", "请先选择 EEG 通道")
            return

        sfreq = self.raw.info["sfreq"]

        start = min(
            int(self.start * sfreq),
            self.raw.n_times - 1
        )

        stop = min(
            start + round(self.window.value() * sfreq),
            self.raw.n_times
        )

        try:
            data = self.get_segment_data(names, start, stop)
        except ValueError as e:
            QMessageBox.warning(self, "滤波参数错误", str(e))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("EEG Power Spectral Density")
        dialog.resize(1050, 700)

        layout = QVBoxLayout(dialog)

        info = QLabel(
            f"Welch PSD | {len(names)} Channels | "
            f"{self.start:.2f}–{min(self.start + self.window.value(), self.raw.n_times / sfreq):.2f} s"
        )
        layout.addWidget(info)

        spectrum = pg.PlotWidget()
        spectrum.showGrid(x=True, y=True, alpha=0.2)
        spectrum.setLabel("bottom", "Frequency", units="Hz")
        spectrum.setLabel("left", "PSD", units="µV²/Hz")
        spectrum.setTitle("Power Spectral Density")

        layout.addWidget(spectrum)

        spectrum.setLogMode(x=False, y=True)

        nperseg = min(data.shape[-1], max(64, int(sfreq * 2)))
        if nperseg < 2:
            QMessageBox.warning(dialog, "提示", "当前数据长度不足以进行频谱分析")
            dialog.close()
            return

        for name, signal in zip(names, data):
            freqs, psd = welch(
                signal,
                fs=sfreq,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2,
                detrend="constant",
                scaling="density"
            )

            spectrum.plot(
                freqs,
                np.maximum(psd, 1e-20),
                pen=pg.mkPen(
                    self.channel_colors[name],
                    width=1.5
                ),
                name=name
            )

        # Frequency band references
        for freq, label in [
            (4, "Theta"),
            (8, "Alpha"),
            (13, "Beta"),
            (30, "Gamma")
        ]:
            if freq < sfreq / 2:
                line = pg.InfiniteLine(
                    pos=freq,
                    angle=90,
                    pen=pg.mkPen(
                        "#888888",
                        width=1,
                        style=Qt.DashLine
                    )
                )
                spectrum.addItem(line)

        spectrum.addLegend()

        layout.addWidget(QLabel(
            "频段参考：Delta 0.5–4 Hz | Theta 4–8 Hz | "
            "Alpha 8–13 Hz | Beta 13–30 Hz | Gamma 30–45 Hz"
        ))

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    # --------------------------------------------------
    # Close
    # --------------------------------------------------

    def closeEvent(self, event):
        if self.raw is not None:
            self.raw.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    viewer = EEGViewer()
    viewer.show()

    sys.exit(app.exec())
