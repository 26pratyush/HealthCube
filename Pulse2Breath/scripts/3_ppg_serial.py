#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time PPG Respiratory Rate Analyzer (patched)

Changes in this build:
- Adds LiveDataHandler.sniff_bytes(n) with pause to safely dump raw bytes (HEX + ASCII).
- Adds ASCII-6 decoder: tries to convert two printable chars -> one 12-bit sample.
- Unifies _ingest_raw_bytes (no duplicates) and sets raw_mode=False once real samples decode.
- Skips feature extraction while truly in RAW-byte mode (to avoid fake metrics).
- Adds a "Sniff" button in the Connection panel.
- Adds get_recent_data() helper to return a last-N-seconds timebase for plotting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import time
import threading
from datetime import datetime, timedelta
import re

# Serial
import serial
import serial.tools.list_ports

# Tk / ttk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Signal processing
from scipy.signal import butter, filtfilt, find_peaks

# Matplotlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Regex for tolerant numeric parsing (ASCII line mode)
NUM_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)')


# =========================
#    Live Data Handler
# =========================
class LiveDataHandler:
    def __init__(self, maxlen=3000):  # ~30s window at 100 Hz
        self.maxlen = maxlen
        self.reset_buffers()
        self.is_connected = False
        self.serial_port = None
        self.read_thread = None
        self.stop_reading = threading.Event()

        # Reader control
        self.pause_read = False  # used by sniff_bytes()

        # File simulation attributes
        self.file_simulation = False
        self.file_data = None
        self.file_index = 0
        self.file_fs = 10.0
        self.file_start_time = None

        # Raw/decoder state
        self.raw_mode = False
        self.numeric_samples = 0
        self.last_mode_check = time.time()
        self._rawbuf = bytearray()

    # ---------- Utilities ----------
    def reset_buffers(self):
        self.timestamps = deque(maxlen=self.maxlen)
        self.pleth_data = deque(maxlen=self.maxlen)
        self.spo2_data = deque(maxlen=self.maxlen)
        self.pr_data = deque(maxlen=self.maxlen)
        self.si_data = deque(maxlen=self.maxlen)
        # File sim
        self.file_index = 0
        self.file_start_time = None

    def get_recent_data(self, seconds: int = 30):
        if len(self.timestamps) == 0:
            return None, None, 0

        cutoff = datetime.now() - timedelta(seconds=seconds)
        ts = list(self.timestamps)
        pv = list(self.pleth_data)

        # Find first index >= cutoff (scan from the end for speed)
        start = 0
        for i in range(len(ts) - 1, -1, -1):
            if ts[i] < cutoff:
                start = i + 1
                break

        ts_recent = ts[start:]
        pv_recent = pv[start:]
        if not ts_recent:
            return None, None, len(pv)

        t0 = ts_recent[0]
        elapsed = np.array([(t - t0).total_seconds() for t in ts_recent], dtype=float)
        return elapsed, pv_recent, len(pv)

    # ---------- Serial Connection ----------
    def connect_serial(self, port, baudrate=115200, device_type="Generic"):
        try:
            self.device_type = device_type
            self.serial_port = serial.Serial(port, baudrate, timeout=1)
            # Helpful toggles
            self.serial_port.dtr = True
            self.serial_port.rts = True
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()

            # Try to kick the device into streaming
            self._initialize_ppg_device()

            self.is_connected = True
            self.stop_reading.clear()
            self.read_thread = threading.Thread(target=self._read_serial_data, daemon=True)
            self.read_thread.start()
            return True
        except Exception as e:
            print(f"Serial connection error: {e}")
            return False

    def _initialize_ppg_device(self):
        """Send simple start/stream commands; tolerate unknown protocols."""
        try:
            device_type = getattr(self, 'device_type', 'Generic')
            print(f"Initializing {device_type} device...")
            time.sleep(1.0)
            if self.serial_port.in_waiting > 0:
                old = self.serial_port.read_all()
                print(f"Cleared {len(old)} bytes from buffer")

            commands = [
                b'START\n', b'S\n', b'start\n', b's\n',
                b'B\n', b'b\n',
                b'STREAM\n', b'stream\n',
                b'RUN\n', b'run\n',
                b'BEGIN\n', b'begin\n',
                b'GO\n', b'go\n',
                b'1\n', b'0\n',
                b'\r\n', b'\n',
            ]
            for i, cmd in enumerate(commands):
                print(f"Sending command {i+1}/{len(commands)}: {cmd!r}")
                self.serial_port.write(cmd); self.serial_port.flush()
                time.sleep(0.5)
                resp = b''
                for _ in range(5):
                    if self.serial_port.in_waiting > 0:
                        resp += self.serial_port.read_all()
                    time.sleep(0.1)
                if resp:
                    s = resp.decode('utf-8', errors='ignore').strip()
                    print(f"Response to {cmd!r}: {s!r} (len={len(resp)})")
                    if len(s) > 5 or any(ch.isdigit() for ch in s):
                        print("Got meaningful response, stopping command sequence")
                        break
            print("Device initialization complete")
        except Exception as e:
            print(f"Device initialization error: {e}")

    def disconnect_serial(self):
        try:
            self.is_connected = False
            self.stop_reading.set()
            if self.read_thread:
                self.read_thread.join(timeout=2)
        except Exception:
            pass
        try:
            if self.serial_port and getattr(self.serial_port, "is_open", False):
                self.serial_port.close()
        except Exception:
            pass
        self.read_thread = None
        self.serial_port = None

    # ---------- Debug Sniffer ----------
    def sniff_bytes(self, n: int = 256):
        """Dump up to n bytes currently buffered (HEX + ASCII)."""
        if not self.serial_port or not self.serial_port.is_open:
            print("sniff_bytes: serial port not open"); return
        self.pause_read = True
        try:
            time.sleep(0.25)  # allow bytes to accrue
            n = min(n, self.serial_port.in_waiting or n)
            if n <= 0:
                print("sniff_bytes: nothing to read")
                return
            buf = self.serial_port.read(n)
            print(f"[sniff_bytes] {len(buf)} bytes")
            print("HEX   :", buf.hex(" "))
            print("ASCII :", "".join(chr(b) if 32 <= b <= 126 else "." for b in buf))
        finally:
            self.pause_read = False

    # ---------- File Simulation ----------
    def load_file_for_simulation(self, file_path):
        try:
            df = pd.read_csv(file_path, sep=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(file_path, delim_whitespace=True, engine="python")
            except Exception as e:
                print(f"Error loading file: {e}")
                return False

        for col in ["Pleth", "SPO2", "PR", "SI"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        fs = None
        if "Time" in df.columns:
            try:
                counts = df["Time"].astype(str).value_counts().to_numpy(dtype=float)
                if counts.size:
                    fs = float(np.median(counts))
            except Exception:
                fs = None
        if not fs or not np.isfinite(fs) or fs <= 0:
            fs = 100.0

        self.file_data = df
        self.file_fs = float(fs)
        self.file_index = 0
        self.file_start_time = None
        print(f"Loaded file: {len(df)} samples at {self.file_fs:.2f} Hz")
        return True

    def start_file_simulation(self):
        if self.file_data is None:
            return False
        self.file_simulation = True
        self.file_index = 0
        self.file_start_time = time.time()
        self.reset_buffers()
        return True

    def stop_file_simulation(self):
        self.file_simulation = False
        self.file_index = 0
        self.file_start_time = None

    def simulate_file_data(self):
        if not self.file_simulation or self.file_data is None:
            return False

        current_time = time.time()
        if self.file_start_time is None:
            self.file_start_time = current_time

        elapsed_time = current_time - self.file_start_time
        target_index = int(elapsed_time * self.file_fs)

        samples_added = 0
        while self.file_index <= target_index and self.file_index < len(self.file_data):
            row = self.file_data.iloc[self.file_index]
            timestamp = datetime.now()
            self.timestamps.append(timestamp)
            pleth = float(row.get('Pleth', 50)) if not pd.isna(row.get('Pleth', 50)) else 50
            spo2 = float(row.get('SPO2', 98)) if not pd.isna(row.get('SPO2', 98)) else 98
            pr   = float(row.get('PR', 70))   if not pd.isna(row.get('PR', 70))   else 70
            si   = float(row.get('SI', 6))    if not pd.isna(row.get('SI', 6))    else 6
            self.pleth_data.append(pleth)
            self.spo2_data.append(spo2)
            self.pr_data.append(pr)
            self.si_data.append(si)
            self.file_index += 1
            samples_added += 1

        if self.file_index >= len(self.file_data):
            self.file_index = 0
            self.file_start_time = current_time
        return samples_added > 0

    # ---------- Serial Reader & Decoders ----------
    def _ascii6_decode_pairs(self, buf: bytearray):
        out = []
        i = 0
        n = len(buf)
        while i + 1 < n:
            a = buf[i]; b = buf[i+1]
            if 0x23 <= a <= 0x7A and 0x23 <= b <= 0x7A:
                v = ((a - 0x23) << 6) | (b - 0x23)   # 12-bit sample
                out.append(float(v))
                i += 2
            else:
                i += 1  # resync
        return i, out

        def _decode_c5d16_frames(self, buf: bytearray):
            """
            Decode 16-bit little-endian samples from frames that look like either:
              (A) 5 bytes:  <HDR0> <HDR1> <STATUS> <LO> <HI>
              (B) 6 bytes:  <SEQ> <HDR0> <HDR1> <STATUS> <LO> <HI>
            Observed headers so far: (0x43,0x5D)='C]', (0x3D,0x5F)='=_', (0x3F,0x5F)='?_'.
            Returns (consumed_bytes, samples_list).
            """
            HEADERS = {(0x43, 0x5D), (0x3D, 0x5F), (0x3F, 0x5F)}
            out = []
            i = 0
            n = len(buf)
            while i < n:
                # Try 6-byte form first (with sequence byte)
                if i + 5 < n and (buf[i+1], buf[i+2]) in HEADERS:
                    lo = buf[i+4]; hi = buf[i+5]
                    out.append(float((hi << 8) | lo))
                    i += 6
                    continue
                # Then 5-byte form (no sequence)
                if i + 4 < n and (buf[i], buf[i+1]) in HEADERS:
                    lo = buf[i+3]; hi = buf[i+4]
                    out.append(float((hi << 8) | lo))
                    i += 5
                    continue
                i += 1
            return n, out


    def _ingest_raw_bytes(self, data: bytes):
        """
        RAW path: try device-specific framed samples first (headers C], =_, ?_),
        then ASCII-6 pairs, else plot raw bytes.
        """
        if not data:
            return
        self._rawbuf.extend(data)

        # 1) Try header-framed samples (5 or 6 byte forms)
        consumed1, samples1 = self._decode_c5d16_frames(self._rawbuf)
        if consumed1:
            del self._rawbuf[:consumed1]
        if samples1:
            self.raw_mode = False  # genuine numeric samples
            now = datetime.now()
            for v in samples1:
                self.timestamps.append(now)
                self.pleth_data.append(v - 2048.0)  # center baseline
                self.spo2_data.append(np.nan)
                self.pr_data.append(np.nan)
                self.si_data.append(np.nan)
            self.numeric_samples += len(samples1)
            return

        # 2) Fallback: printable ASCII-6 pairs → 12-bit samples
        consumed2, samples2 = self._ascii6_decode_pairs(self._rawbuf)
        if consumed2:
            del self._rawbuf[:consumed2]
        if samples2:
            self.raw_mode = False
            now = datetime.now()
            for v in samples2:
                self.timestamps.append(now)
                self.pleth_data.append(v - 2048.0)  # keep centering consistent
                self.spo2_data.append(np.nan)
                self.pr_data.append(np.nan)
                self.si_data.append(np.nan)
            self.numeric_samples += len(samples2)
            return

        # 3) Last resort: draw centered bytes just so user sees activity
        for b in data:
            self.timestamps.append(datetime.now())
            self.pleth_data.append(float(b) - 128.0)
            self.spo2_data.append(np.nan)
            self.pr_data.append(np.nan)
            self.si_data.append(np.nan)

    def _read_serial_data(self):
        while not self.stop_reading.is_set() and self.is_connected:
            try:
                if self.pause_read:
                    time.sleep(0.05); continue

                if not self.serial_port:
                    time.sleep(0.05); continue

                n = self.serial_port.in_waiting
                if n:
                    # Try newline-delimited ASCII first
                    line_bytes = self.serial_port.readline()
                    if line_bytes:
                        line = line_bytes.decode('utf-8', errors='ignore').strip()
                        if any(ch.isdigit() for ch in line):
                            self._parse_data_line(line)
                            self.numeric_samples += 1
                        else:
                            self._ingest_raw_bytes(line_bytes)
                    else:
                        chunk = self.serial_port.read(n)
                        self._ingest_raw_bytes(chunk)

                # Mode control: show RAW banner if no numbers for a while
                now = time.time()
                if not self.raw_mode and (now - self.last_mode_check) > 2.0 and self.numeric_samples < 5:
                    print("No numeric tokens detected — switching to RAW-BYTES fallback mode.")
                    self.raw_mode = True
                if self.raw_mode and self.numeric_samples > 20:
                    # If we eventually see many numeric samples, clear the timer
                    self.last_mode_check = now

                time.sleep(0.01)
            except Exception as e:
                print(f"Serial read error: {e}")
                break

    def _parse_data_line(self, line):
        try:
            parts = [p for p in line.replace(',', '\t').replace(' ', '\t').split('\t') if p]
            def to_float(s):
                try:
                    return float(s)
                except Exception:
                    return np.nan

            pleth = spo2 = pr = si = np.nan

            if len(parts) >= 5 and any(ch.isdigit() for ch in line):
                if parts[0].lower() == "time":
                    return
                pleth = to_float(parts[1]); spo2 = to_float(parts[2])
                pr    = to_float(parts[3]); si   = to_float(parts[4])
            else:
                nums = [float(x) for x in NUM_RE.findall(line)]
                if not nums:
                    return
                if len(nums) >= 4:
                    pleth, spo2, pr, si = nums[0], nums[1], nums[2], nums[3]
                else:
                    pleth = nums[0]; spo2, pr, si = 98.0, 72.0, 6.0

            self.timestamps.append(datetime.now())
            self.pleth_data.append(pleth)
            self.spo2_data.append(spo2)
            self.pr_data.append(pr)
            self.si_data.append(si)
            # ASCII line means we have numeric data -> ensure we exit raw-only mode
            self.raw_mode = False
        except Exception as e:
            print(f"Data parsing error: {e}")

    # ---------- Simulation (no serial) ----------
    def simulate_data(self):
        timestamp = datetime.now()
        t = time.time()
        base_signal = 50 + 15 * np.sin(2 * np.pi * 1.2 * t)   # ~1.2 Hz
        breathing_mod = 5 * np.sin(2 * np.pi * 0.25 * t)      # ~0.25 Hz
        noise = 2 * np.random.randn()
        pleth = base_signal + breathing_mod + noise
        spo2 = 98 + np.random.randn() * 0.5
        pr   = 72 + np.random.randn() * 3
        si   = 6 + np.random.randn() * 1
        self.timestamps.append(timestamp)
        self.pleth_data.append(pleth)
        self.spo2_data.append(spo2)
        self.pr_data.append(pr)
        self.si_data.append(si)


# =========================
#    Signal Processing
# =========================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    if len(data) < 2 * order:
        return data
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data)

def derive_resp_signal_from_ppg(ppg, fs):
    x = np.asarray(ppg, dtype=float)
    if len(x) < 10:
        return x, x
    x = x - np.nanmean(x)
    win = max(int(0.8 * fs), 3)
    win = min(win, len(x) // 2)
    x_smooth = pd.Series(x).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
    resp_signal = butter_lowpass_filter(x_smooth, cutoff=0.5, fs=fs, order=4)
    return x_smooth, resp_signal

def rr_fft(resp_signal, fs, fmin=0.1, fmax=0.5):
    if len(resp_signal) < 20:
        return np.nan, None, None
    t = np.arange(len(resp_signal)) / fs
    target_fs = 5.0
    if t[-1] < 1.0:
        return np.nan, None, None
    t_new = np.arange(0, t[-1], 1.0 / target_fs)
    if len(t_new) < 10:
        return np.nan, None, None
    x = np.interp(t_new, t, resp_signal) - np.mean(resp_signal)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / target_fs)
    psd = (np.abs(X) ** 2) / len(x)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return np.nan, freqs, psd
    peak_freq = freqs[band][np.argmax(psd[band])]
    return float(peak_freq * 60.0), freqs, psd

def rr_time_domain(resp_signal, fs):
    if len(resp_signal) < fs:
        return np.nan, []
    peaks, _ = find_peaks(resp_signal, distance=max(int(fs * 1.5), 5))
    duration_min = len(resp_signal) / fs / 60.0
    rr_bpm = len(peaks) / duration_min if duration_min > 0 else np.nan
    return rr_bpm, peaks

def extract_rr_features_realtime(pleth_data, fs):
    if len(pleth_data) < 10:
        return None, None, None, None, None, None, None
    t = np.arange(len(pleth_data)) / fs
    x_smooth, resp_signal = derive_resp_signal_from_ppg(pleth_data, fs)
    rr_fft_bpm, freqs, psd = rr_fft(resp_signal, fs)
    rr_td_bpm, peaks = rr_time_domain(resp_signal, fs)
    features = {
        "rr_bpm_fft": rr_fft_bpm,
        "rr_bpm_time": rr_td_bpm,
        "breath_count": int(len(peaks)),
        "duration_s": float(len(resp_signal) / fs),
        "fs_hz": fs,
        "data_points": len(pleth_data)
    }
    return features, t, x_smooth, resp_signal, peaks, freqs, psd


# =========================
#    Real-time GUI
# =========================
class RealTimeRRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-time PPG Respiratory Rate Analyzer")
        self.configure(bg="#1f1f1f")
        self.geometry("1200x800")

        self.data_handler = LiveDataHandler()
        self.is_running = False
        self.simulation_mode = True
        self.file_mode = False

        self.setup_styling()
        self.create_control_panel()
        self.create_feature_display()
        self.create_plots()
        self.setup_animation()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styling(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#1f1f1f", foreground="#e8e8e8", fieldbackground="#2a2a2a")
        style.configure("TLabel", background="#1f1f1f", foreground="#e8e8e8", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Dark.TEntry", fieldbackground="#2a2a2a", foreground="#e8e8e8")
        style.configure("Green.TButton", font=("Segoe UI", 10, "bold"), background="#2e7d32", foreground="#ffffff", padding=8)
        style.map("Green.TButton", background=[("active", "#348f38")])
        style.configure("Red.TButton", font=("Segoe UI", 10, "bold"), background="#d32f2f", foreground="#ffffff", padding=8)
        style.map("Red.TButton", background=[("active", "#f44336")])
        style.configure("Blue.TButton", font=("Segoe UI", 10, "bold"), background="#1976d2", foreground="#ffffff", padding=8)
        style.map("Blue.TButton", background=[("active", "#2196f3")])

    def create_control_panel(self):
        control_frame = ttk.Frame(self, padding=12)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        conn_frame = ttk.LabelFrame(control_frame, text="Connection", padding=8)
        conn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        ttk.Label(conn_frame, text="Serial Port:").grid(row=0, column=0, sticky="w", pady=2)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var, width=15)
        self.port_combo.grid(row=1, column=0, sticky="ew", pady=2)
        self.refresh_ports()
        ttk.Button(conn_frame, text="Refresh", command=self.refresh_ports).grid(row=1, column=1, padx=(5,0), pady=2)

        button_frame = ttk.Frame(conn_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(8,0))
        self.connect_btn = ttk.Button(button_frame, text="Connect", style="Green.TButton", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=(0,5))
        self.sim_btn = ttk.Button(button_frame, text="Simulate", style="Blue.TButton", command=self.toggle_simulation)
        self.sim_btn.pack(side=tk.LEFT, padx=(0,5))
        self.file_btn = ttk.Button(button_frame, text="Load File", style="Blue.TButton", command=self.load_file_simulation)
        self.file_btn.pack(side=tk.LEFT, padx=(0,5))
        # Sniff button (prints to console)
        self.sniff_btn = ttk.Button(button_frame, text="Sniff", command=lambda: self.data_handler.sniff_bytes(256))
        self.sniff_btn.pack(side=tk.LEFT)

        status_frame = ttk.LabelFrame(control_frame, text="Status", padding=8)
        status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.status_label = ttk.Label(status_frame, text="● Simulation Mode", style="Status.TLabel", foreground="#4CAF50")
        self.status_label.pack(anchor="w")
        self.data_count_label = ttk.Label(status_frame, text="Data points: 0")
        self.data_count_label.pack(anchor="w")
        self.update_rate_label = ttk.Label(status_frame, text="Update rate: -- Hz")
        self.update_rate_label.pack(anchor="w")
        self.file_info_label = ttk.Label(status_frame, text="", foreground="#9E9E9E")
        self.file_info_label.pack(anchor="w")
        self.file_progress_label = ttk.Label(status_frame, text="", foreground="#9E9E9E")
        self.file_progress_label.pack(anchor="w")

    def create_feature_display(self):
        mid_frame = ttk.Frame(self, padding=(12, 0, 12, 8))
        mid_frame.pack(side=tk.TOP, fill=tk.X)

        features_frame = ttk.LabelFrame(mid_frame, text="Real-time Features", padding=8)
        features_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        style = ttk.Style(self)
        style.configure("Dark.Treeview", background="#2a2a2a", foreground="#e8e8e8", fieldbackground="#2a2a2a", borderwidth=0, font=("Segoe UI", 10))
        style.configure("Dark.Treeview.Heading", background="#1f1f1f", foreground="#e8e8e8", font=("Segoe UI", 10, "bold"))
        style.map("Dark.Treeview", background=[("selected", "#4CAF50")], foreground=[("selected", "#ffffff")])

        self.tree = ttk.Treeview(features_frame, columns=("key", "value"), show="headings", height=6, style="Dark.Treeview")
        self.tree.heading("key", text="Feature"); self.tree.column("key", width=200, anchor="w")
        self.tree.heading("value", text="Value"); self.tree.column("value", width=150, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True)

        insights_frame = ttk.LabelFrame(mid_frame, text="Live Insights", padding=8)
        insights_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        self.insights_text = tk.Text(insights_frame, height=6, width=40, wrap="word",
                                     bg="#2a2a2a", fg="#e8e8e8", insertbackground="#e8e8e8",
                                     relief="flat", padx=8, pady=6)
        self.insights_text.pack(fill=tk.BOTH, expand=True)

    def create_plots(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.fig1 = Figure(figsize=(6, 3), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self._setup_dark_plot(self.fig1, self.ax1)
        tab1 = ttk.Frame(self.notebook); self.notebook.add(tab1, text="Live PPG Signal")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=tab1); self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig2 = Figure(figsize=(6, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self._setup_dark_plot(self.fig2, self.ax2)
        tab2 = ttk.Frame(self.notebook); self.notebook.add(tab2, text="Respiratory Signal")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=tab2); self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig3 = Figure(figsize=(6, 3), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self._setup_dark_plot(self.fig3, self.ax3)
        tab3 = ttk.Frame(self.notebook); self.notebook.add(tab3, text="Frequency Analysis")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=tab3); self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_dark_plot(self, fig, ax):
        fig.patch.set_facecolor("#1f1f1f"); ax.set_facecolor("#1f1f1f")
        ax.tick_params(colors="#e8e8e8")
        ax.xaxis.label.set_color("#e8e8e8"); ax.yaxis.label.set_color("#e8e8e8"); ax.title.set_color("#e8e8e8")
        ax.grid(True, alpha=0.3)

    def setup_animation(self):
        self.after(100, self.update_display)
        self.last_update_time = time.time()
        self.update_count = 0

    def refresh_ports(self):
        try:
            ports = [port.device for port in serial.tools.list_ports.comports()]
        except Exception:
            ports = []
        self.port_combo['values'] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    def load_file_simulation(self):
        file_path = filedialog.askopenfilename(title="Select PPG file for simulation",
                                               filetypes=[("Data files", "*.txt *.csv"), ("All files", "*.*")])
        if not file_path:
            return
        if self.data_handler.load_file_for_simulation(file_path):
            if self.data_handler.is_connected:
                self.data_handler.disconnect_serial()
                self.connect_btn.config(text="Connect", style="Green.TButton")
            self.data_handler.start_file_simulation()
            self.simulation_mode = False
            self.file_mode = True
            self.is_running = True
            filename = Path(file_path).name
            file_samples = len(self.data_handler.file_data)
            file_fs = self.data_handler.file_fs
            duration = file_samples / max(file_fs, 1e-6)
            self.status_label.config(text="● File Simulation", foreground="#FF9800")
            self.file_info_label.config(text=f"File: {filename} ({file_samples} samples, {file_fs:.1f}Hz, {duration:.1f}s)")
            messagebox.showinfo("File Loaded", f"Loaded: {filename}\nSamples: {file_samples}\nRate: {file_fs:.1f} Hz\nDuration: {duration:.1f} s")
        else:
            messagebox.showerror("Load Error", "Failed to load the selected file.")

    def stop_all_modes(self):
        if self.file_mode:
            self.data_handler.stop_file_simulation()
            self.file_mode = False
            self.file_info_label.config(text="")
            self.file_progress_label.config(text="")
        if self.data_handler.is_connected:
            self.data_handler.disconnect_serial()
            self.connect_btn.config(text="Connect", style="Green.TButton")
        self.simulation_mode = False
        self.is_running = False
        self.status_label.config(text="● Stopped", foreground="#f44336")

    def toggle_connection(self):
        if not self.data_handler.is_connected:
            self.stop_all_modes()
            port = self.port_var.get()
            if not port:
                messagebox.showwarning("No Port", "Please select a serial port.")
                return
            baudrate = 115200
            device_type = "Generic"
            if self.data_handler.connect_serial(port, baudrate, device_type):
                self.simulation_mode = False
                self.file_mode = False
                self.connect_btn.config(text="Disconnect", style="Red.TButton")
                self.status_label.config(text=f"● Connected ({device_type})", foreground="#4CAF50")
                self.is_running = True
            else:
                messagebox.showerror("Connection Failed", "Failed to connect to serial port.")
        else:
            self.data_handler.disconnect_serial()
            self.connect_btn.config(text="Connect", style="Green.TButton")
            self.status_label.config(text="● Disconnected", foreground="#f44336")
            self.is_running = False

    def toggle_simulation(self):
        if not self.simulation_mode:
            self.stop_all_modes()
            self.simulation_mode = True
            self.file_mode = False
            self.status_label.config(text="● Simulation Mode", foreground="#2196f3")
            self.is_running = True
        else:
            self.stop_all_modes()

    def update_display(self):
        # In true RAW-byte mode, we only draw the waveform (no features)
        if getattr(self.data_handler, "raw_mode", False):
            self.status_label.config(text="● RAW stream (bytes) — decoding…", foreground="#FFC107")
            # Still update plot area using whatever samples are arriving
            elapsed_time, pleth_data, _ = self.data_handler.get_recent_data(seconds=30)
            if elapsed_time is not None and pleth_data is not None and len(pleth_data) > 2:
                self.ax1.clear(); self._setup_dark_plot(self.fig1, self.ax1)
                self.ax1.plot(elapsed_time, pleth_data, linewidth=1, alpha=0.8, label='Raw (bytes)')
                self.ax1.set_title("Live PPG Signal (RAW bytes view)")
                self.ax1.set_xlabel("Time (s)"); self.ax1.set_ylabel("Amplitude")
                self.ax1.legend(); self.fig1.tight_layout(); self.canvas1.draw_idle()
            self.after(100, self.update_display)
            return

        try:
            if self.is_running:
                if self.simulation_mode:
                    self.data_handler.simulate_data()
                elif self.file_mode:
                    self.data_handler.simulate_file_data()
                # serial mode feeds itself

            elapsed_time, pleth_data, data_count = self.data_handler.get_recent_data(seconds=30)
            if elapsed_time is not None and pleth_data is not None and len(pleth_data) > 10:
                fs = len(pleth_data) / max(elapsed_time[-1], 1.0) if len(elapsed_time) > 1 else 10.0
                fs = float(np.clip(fs, 1.0, 100.0))
                features, t, x_smooth, resp_signal, peaks, freqs, psd = extract_rr_features_realtime(pleth_data, fs)
                if features:
                    self.update_features_display(features)
                    self.update_plots(elapsed_time, pleth_data, x_smooth, resp_signal, peaks, freqs, psd, features)
                    self.update_insights(features)

            total_points = len(self.data_handler.pleth_data)
            self.data_count_label.config(text=f"Data points: {total_points}")

            if self.file_mode and self.data_handler.file_data is not None and self.data_handler.file_fs > 0:
                progress = (self.data_handler.file_index / max(len(self.data_handler.file_data), 1)) * 100
                elapsed = time.time() - self.data_handler.file_start_time if self.data_handler.file_start_time else 0
                remaining_samples = max(len(self.data_handler.file_data) - self.data_handler.file_index, 0)
                remaining_time = remaining_samples / self.data_handler.file_fs
                self.file_progress_label.config(text=f"Progress: {progress:.1f}% ({elapsed:.1f}s elapsed, ~{remaining_time:.1f}s remaining)")

            current_time = time.time()
            self.update_count += 1
            if current_time - self.last_update_time > 1.0:
                rate = self.update_count / (current_time - self.last_update_time)
                self.update_rate_label.config(text=f"Update rate: {rate:.1f} Hz")
                self.last_update_time = current_time
                self.update_count = 0
        except Exception as e:
            print(f"Display update error: {e}")
        finally:
            self.after(100, self.update_display)

    def update_features_display(self, features):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for key, value in features.items():
            if isinstance(value, float):
                display_value = "N/A" if np.isnan(value) else f"{value:.3f}"
            else:
                display_value = str(value)
            self.tree.insert("", "end", values=(key, display_value))

    def update_plots(self, time_data, pleth_data, x_smooth, resp_signal, peaks, freqs, psd, features):
        try:
            self.ax1.clear(); self._setup_dark_plot(self.fig1, self.ax1)
            if len(time_data) > 0:
                self.ax1.plot(time_data, pleth_data, linewidth=1, alpha=0.7, label='Raw PPG')
                if x_smooth is not None and len(x_smooth) == len(time_data):
                    self.ax1.plot(time_data, x_smooth, linewidth=2, label='Smoothed')
            self.ax1.set_title("Live PPG Signal (Last 30s)")
            self.ax1.set_xlabel("Time (s)"); self.ax1.set_ylabel("Amplitude")
            self.ax1.legend(); self.fig1.tight_layout(); self.canvas1.draw_idle()

            self.ax2.clear(); self._setup_dark_plot(self.fig2, self.ax2)
            if resp_signal is not None and len(resp_signal) > 0:
                self.ax2.plot(time_data, resp_signal, linewidth=2, label='Respiratory')
                if peaks is not None and len(peaks) > 0:
                    peak_times = [time_data[i] for i in peaks if i < len(time_data)]
                    peak_values = [resp_signal[i] for i in peaks if i < len(resp_signal)]
                    self.ax2.scatter(peak_times, peak_values, s=50, marker='x', label='Peaks')
            rr_fft_v = features.get('rr_bpm_fft', np.nan)
            rr_time_v = features.get('rr_bpm_time', np.nan)
            self.ax2.set_title(f"Respiratory Signal - FFT: {rr_fft_v:.1f} bpm, Time: {rr_time_v:.1f} bpm")
            self.ax2.set_xlabel("Time (s)"); self.ax2.set_ylabel("Amplitude")
            self.ax2.legend(); self.fig2.tight_layout(); self.canvas2.draw_idle()

            self.ax3.clear(); self._setup_dark_plot(self.fig3, self.ax3)
            if freqs is not None and psd is not None:
                mask = (freqs >= 0.0) & (freqs <= 1.0)
                self.ax3.plot(freqs[mask], psd[mask], linewidth=2)
                resp_band = (freqs >= 0.1) & (freqs <= 0.5)
                if np.any(resp_band):
                    self.ax3.fill_between(freqs[resp_band], psd[resp_band], alpha=0.3)
            self.ax3.set_title("Power Spectral Density (0-1 Hz)")
            self.ax3.set_xlabel("Frequency (Hz)"); self.ax3.set_ylabel("Power")
            self.fig3.tight_layout(); self.canvas3.draw_idle()
        except Exception as e:
            print(f"Plot update error: {e}")

    def update_insights(self, features):
        try:
            self.insights_text.delete("1.0", tk.END)
            insights = []
            rr_fft_v = features.get('rr_bpm_fft', np.nan)
            if not np.isnan(rr_fft_v):
                if 12 <= rr_fft_v <= 20:
                    insights.append(f"✓ RR {rr_fft_v:.1f} bpm - Normal range")
                elif rr_fft_v < 12:
                    insights.append(f"⚠ RR {rr_fft_v:.1f} bpm - Below normal")
                else:
                    insights.append(f"⚠ RR {rr_fft_v:.1f} bpm - Elevated")
            else:
                insights.append("⚠ Unable to calculate RR - Need more data")
            data_points = features.get('data_points', 0)
            duration = features.get('duration_s', 0)
            if duration > 0:
                insights.append(f"📊 {data_points} points over {duration:.1f}s")
            breath_count = features.get('breath_count', 0)
            if breath_count > 0:
                insights.append(f"🫁 {breath_count} breaths detected")
            if not insights:
                insights.append("Waiting for sufficient data...")
            self.insights_text.insert("1.0", "\n".join(insights))
        except Exception as e:
            print(f"Insights update error: {e}")

    def on_closing(self):
        self.is_running = False
        try:
            self.data_handler.disconnect_serial()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = RealTimeRRApp()
    app.mainloop()
