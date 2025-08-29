#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import time
import threading
from datetime import datetime, timedelta

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from scipy.signal import butter, filtfilt, find_peaks

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =========================
#    Data Handler
# =========================
class DataHandler:
    def __init__(self, maxlen=3000):  # ~5 minutes at 10Hz
        self.maxlen = maxlen
        self.reset_buffers()
        
        # File simulation attributes
        self.file_simulation = False
        self.file_data = None
        self.file_index = 0
        self.file_fs = 10.0
        self.file_start_time = None
        
    def reset_buffers(self):
        self.timestamps = deque(maxlen=self.maxlen)
        self.pleth_data = deque(maxlen=self.maxlen)
        self.spo2_data = deque(maxlen=self.maxlen)
        self.pr_data = deque(maxlen=self.maxlen)
        self.si_data = deque(maxlen=self.maxlen)
        
        self.file_index = 0
        self.file_start_time = None
            
    def load_file_for_simulation(self, file_path):
        """Load a .txt file for simulated real-time playback"""
        try:
            df = pd.read_csv(file_path, sep=r"\s+", engine="python")
            
            for col in ["Pleth", "SPO2", "PR", "SI"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Calculate sampling frequency
            times = pd.to_datetime(df["Time"], format="%H:%M:%S")
            duration_sec = (times.iloc[-1] - times.iloc[0]).total_seconds()
            if duration_sec <= 0:
                duration_sec = df["Time"].ne(df["Time"].shift()).cumsum().max() - 1
            
            fs = len(df) / max(duration_sec, 1)
            
            # Store file data
            self.file_data = df
            self.file_fs = max(float(fs), 1.0)  
            self.file_index = 0
            self.file_start_time = None
            
            print(f"Loaded file: {len(df)} samples at {self.file_fs:.2f} Hz")
            return True
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def start_file_simulation(self):
        """Start file simulation mode"""
        if self.file_data is None:
            return False
        
        self.file_simulation = True
        self.file_index = 0
        self.file_start_time = time.time()
        self.reset_buffers()
        return True
    
    def stop_file_simulation(self):
        """Stop file simulation mode"""
        self.file_simulation = False
        self.file_index = 0
        self.file_start_time = None
        
    def get_recent_data(self, seconds=60):
        """Get data from last N seconds"""
        if not self.timestamps:
            return None, None, None
            
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        
        # Convert to numpy arrays for easier processing
        timestamps = list(self.timestamps)
        pleth_data = list(self.pleth_data)
        
        # Filter recent data
        recent_indices = [i for i, ts in enumerate(timestamps) if ts >= cutoff_time]
        
        if len(recent_indices) < 10:  # Need minimum data points
            return None, None, None
            
        recent_timestamps = [timestamps[i] for i in recent_indices]
        recent_pleth = [pleth_data[i] for i in recent_indices]
        
        # Convert timestamps to elapsed seconds
        start_time = recent_timestamps[0]
        elapsed_seconds = [(ts - start_time).total_seconds() for ts in recent_timestamps]
        
        return np.array(elapsed_seconds), np.array(recent_pleth), len(recent_indices)

    def simulate_data(self):
        """Simulate PPG data for testing"""
        timestamp = datetime.now()
        
        # Simulate realistic PPG signal with breathing modulation
        t = time.time()
        base_signal = 50 + 15 * np.sin(2 * np.pi * 1.2 * t) 
        breathing_mod = 5 * np.sin(2 * np.pi * 0.25 * t)     
        noise = 2 * np.random.randn()
        
        pleth = base_signal + breathing_mod + noise
        spo2 = 98 + np.random.randn() * 0.5
        pr = 72 + np.random.randn() * 3
        si = 6 + np.random.randn() * 1
        
        self.timestamps.append(timestamp)
        self.pleth_data.append(pleth)
        self.spo2_data.append(spo2)
        self.pr_data.append(pr)
        self.si_data.append(si)

    def simulate_file_data(self):
        """Simulate data playback from loaded file at original sampling rate"""
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
            
            # Handle potential NaN values
            pleth = float(row.get('Pleth', 50)) if not pd.isna(row.get('Pleth', 50)) else 50
            spo2 = float(row.get('SPO2', 98)) if not pd.isna(row.get('SPO2', 98)) else 98
            pr = float(row.get('PR', 70)) if not pd.isna(row.get('PR', 70)) else 70
            si = float(row.get('SI', 6)) if not pd.isna(row.get('SI', 6)) else 6
            
            self.pleth_data.append(pleth)
            self.spo2_data.append(spo2)
            self.pr_data.append(pr)
            self.si_data.append(si)
            
            self.file_index += 1
            samples_added += 1
            
        # Check if file playback is complete
        if self.file_index >= len(self.file_data):
            self.file_index = 0
            self.file_start_time = current_time
            
        return samples_added > 0


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
    
    if t[-1] < 1.0:  # Less than 1 second of data
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
    if len(resp_signal) < fs:  # Less than 1 second
        return np.nan, []
        
    peaks, _ = find_peaks(resp_signal, distance=max(int(fs * 1.5), 5))
    duration_min = len(resp_signal) / fs / 60.0
    rr_bpm = len(peaks) / duration_min if duration_min > 0 else np.nan
    return rr_bpm, peaks


def extract_rr_features_realtime(pleth_data, fs):
    """Extract features from real-time data"""
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
#    GUI Application
# =========================
class PPGAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PPG Respiratory Rate Analyzer")
        self.configure(bg="#1f1f1f")
        self.geometry("1200x800")
        
        self.data_handler = DataHandler()
        self.is_running = False
        self.simulation_mode = True  
        self.file_mode = False 
        
        # Setup GUI styling
        self.setup_styling()
        
        # Create GUI elements
        self.create_control_panel()
        self.create_feature_display()
        self.create_plots()
        
        self.setup_animation()
        
        # Handle window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_styling(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#1f1f1f", foreground="#e8e8e8", fieldbackground="#2a2a2a")
        style.configure("TLabel", background="#1f1f1f", foreground="#e8e8e8", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))
        
        # Button styles
        style.configure("Green.TButton", font=("Segoe UI", 10, "bold"),
                        background="#2e7d32", foreground="#ffffff", padding=8)
        style.map("Green.TButton", background=[("active", "#348f38")])
        
        style.configure("Red.TButton", font=("Segoe UI", 10, "bold"),
                        background="#d32f2f", foreground="#ffffff", padding=8)
        style.map("Red.TButton", background=[("active", "#f44336")])
        
        style.configure("Blue.TButton", font=("Segoe UI", 10, "bold"),
                        background="#1976d2", foreground="#ffffff", padding=8)
        style.map("Blue.TButton", background=[("active", "#2196f3")])
        
    def create_control_panel(self):
        # Top control panel
        control_frame = ttk.Frame(self, padding=12)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Mode controls
        mode_frame = ttk.LabelFrame(control_frame, text="Data Source", padding=8)
        mode_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        
        # Mode buttons
        button_frame = ttk.Frame(mode_frame)
        button_frame.grid(row=0, column=0, pady=(0, 8))
        
        self.sim_btn = ttk.Button(button_frame, text="Start Simulation", style="Green.TButton", command=self.toggle_simulation)
        self.sim_btn.pack(side=tk.TOP, pady=(0, 5), fill=tk.X)
        
        self.file_btn = ttk.Button(button_frame, text="Load File", style="Blue.TButton", command=self.load_file_simulation)
        self.file_btn.pack(side=tk.TOP, pady=(0, 5), fill=tk.X)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", style="Red.TButton", command=self.stop_all_modes)
        self.stop_btn.pack(side=tk.TOP, fill=tk.X)
        
        # Status panel
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding=8)
        status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.status_label = ttk.Label(status_frame, text="● Simulation Mode", style="Status.TLabel", foreground="#4CAF50")
        self.status_label.pack(anchor="w")
        
        self.data_count_label = ttk.Label(status_frame, text="Data points: 0")
        self.data_count_label.pack(anchor="w")
        
        self.update_rate_label = ttk.Label(status_frame, text="Update rate: -- Hz")
        self.update_rate_label.pack(anchor="w")
        
        # File simulation info
        self.file_info_label = ttk.Label(status_frame, text="", foreground="#9E9E9E")
        self.file_info_label.pack(anchor="w")
        
        # Progress label for file playback
        self.file_progress_label = ttk.Label(status_frame, text="", foreground="#9E9E9E")
        self.file_progress_label.pack(anchor="w")
        
    def create_feature_display(self):
        # Middle section for features and insights
        mid_frame = ttk.Frame(self, padding=(12, 0, 12, 8))
        mid_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Features display
        features_frame = ttk.LabelFrame(mid_frame, text="Real-time Features", padding=8)
        features_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create Treeview for features
        self.tree = ttk.Treeview(features_frame, columns=("key", "value"), show="headings", height=6)
        self.tree.heading("key", text="Feature")
        self.tree.column("key", width=200, anchor="w")
        self.tree.heading("value", text="Value")
        self.tree.column("value", width=150, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Insights display
        insights_frame = ttk.LabelFrame(mid_frame, text="Live Insights", padding=8)
        insights_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        
        self.insights_text = tk.Text(insights_frame, height=6, width=40, wrap="word", 
                                    bg="#2a2a2a", fg="#e8e8e8", insertbackground="#e8e8e8", 
                                    relief="flat", padx=8, pady=6)
        self.insights_text.pack(fill=tk.BOTH, expand=True)
        
    def create_plots(self):
        # Bottom notebook with plots
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        
        # Plot 1: Real-time PPG
        self.fig1 = Figure(figsize=(6, 3), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self._setup_dark_plot(self.fig1, self.ax1)
        
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Live PPG Signal")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=tab1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot 2: Respiratory signal
        self.fig2 = Figure(figsize=(6, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self._setup_dark_plot(self.fig2, self.ax2)
        
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Respiratory Signal")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=tab2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot 3: Frequency domain
        self.fig3 = Figure(figsize=(6, 3), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self._setup_dark_plot(self.fig3, self.ax3)
        
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="Frequency Analysis")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=tab3)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _setup_dark_plot(self, fig, ax):
        fig.patch.set_facecolor("#1f1f1f")
        ax.set_facecolor("#1f1f1f")
        ax.tick_params(colors="#e8e8e8")
        ax.xaxis.label.set_color("#e8e8e8")
        ax.yaxis.label.set_color("#e8e8e8")
        ax.title.set_color("#e8e8e8")
        ax.grid(True, alpha=0.3)
        
    def setup_animation(self):
        # Update plots every 100ms (10 FPS)
        self.after(100, self.update_display)
        self.last_update_time = time.time()
        self.update_count = 0

    def load_file_simulation(self):
        """Load a .txt file for simulated playback"""
        file_path = filedialog.askopenfilename(
            title="Select PPG .txt file for simulation",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.data_handler.load_file_for_simulation(file_path):
                # Stop any current activity
                self.stop_all_modes()
                
                # Start file simulation
                self.data_handler.start_file_simulation()
                self.simulation_mode = False  
                self.file_mode = True
                self.is_running = True
                
                # Update UI
                filename = Path(file_path).name
                file_samples = len(self.data_handler.file_data)
                file_fs = self.data_handler.file_fs
                duration = file_samples / file_fs
                
                self.status_label.config(text="● File Simulation", foreground="#FF9800")
                self.file_info_label.config(text=f"File: {filename} ({file_samples} samples, {file_fs:.1f}Hz, {duration:.1f}s)")
                
                # Update button states
                self.sim_btn.config(text="Start Simulation")
                
                messagebox.showinfo("File Loaded", 
                    f"Loaded: {filename}\n"
                    f"Samples: {file_samples}\n"
                    f"Rate: {file_fs:.1f} Hz\n"
                    f"Duration: {duration:.1f} seconds")
            else:
                messagebox.showerror("Load Error", "Failed to load the selected file.")
                
    def stop_all_modes(self):
        """Stop all simulation and file modes"""
        if self.file_mode:
            self.data_handler.stop_file_simulation()
            self.file_mode = False
            self.file_info_label.config(text="")
            self.file_progress_label.config(text="")
        
        self.simulation_mode = False
        self.is_running = False
        self.status_label.config(text="● Stopped", foreground="#f44336")
        self.sim_btn.config(text="Start Simulation")
            
    def toggle_simulation(self):
        """Toggle simulation mode"""
        if not self.simulation_mode or not self.is_running:
            self.stop_all_modes()
            
            self.simulation_mode = True
            self.file_mode = False
            self.status_label.config(text="● Simulation Mode", foreground="#2196f3")
            self.is_running = True
            self.sim_btn.config(text="Stop Simulation")
            self.data_handler.reset_buffers()
        else:
            self.stop_all_modes()
            
    def update_display(self):
        """Main update loop"""
        try:
            # Generate or get data
            if self.simulation_mode and self.is_running:
                self.data_handler.simulate_data()
            elif self.file_mode and self.is_running:
                self.data_handler.simulate_file_data()
            
            elapsed_time, pleth_data, data_count = self.data_handler.get_recent_data(seconds=30)
            
            if elapsed_time is not None and len(pleth_data) > 10:
                # Estimate sampling frequency
                fs = len(pleth_data) / max(elapsed_time[-1], 1.0) if len(elapsed_time) > 1 else 10.0
                fs = min(max(fs, 1.0), 100.0)
                
                # Extract features
                features, t, x_smooth, resp_signal, peaks, freqs, psd = extract_rr_features_realtime(pleth_data, fs)
                
                if features:
                    self.update_features_display(features)
                    self.update_plots(elapsed_time, pleth_data, x_smooth, resp_signal, peaks, freqs, psd, features)
                    self.update_insights(features)
                    
            # Update status
            total_points = len(self.data_handler.pleth_data)
            self.data_count_label.config(text=f"Data points: {total_points}")
            
            # Show file simulation progress
            if self.file_mode and self.data_handler.file_data is not None:
                progress = (self.data_handler.file_index / len(self.data_handler.file_data)) * 100
                elapsed = time.time() - self.data_handler.file_start_time if self.data_handler.file_start_time else 0
                remaining_samples = len(self.data_handler.file_data) - self.data_handler.file_index
                remaining_time = remaining_samples / self.data_handler.file_fs
                
                progress_text = f"Progress: {progress:.1f}% ({elapsed:.1f}s elapsed, ~{remaining_time:.1f}s remaining)"
                self.file_progress_label.config(text=progress_text)
            
            # Calculate update rate
            current_time = time.time()
            self.update_count += 1
            if current_time - self.last_update_time > 1.0:  # Update rate every second
                rate = self.update_count / (current_time - self.last_update_time)
                self.update_rate_label.config(text=f"Update rate: {rate:.1f} Hz")
                self.last_update_time = current_time
                self.update_count = 0
                
        except Exception as e:
            print(f"Display update error: {e}")
            
        self.after(100, self.update_display)
        
    def update_features_display(self, features):
        """Update the features table"""
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for key, value in features.items():
            if isinstance(value, float):
                if np.isnan(value):
                    display_value = "N/A"
                else:
                    display_value = f"{value:.3f}"
            else:
                display_value = str(value)
            self.tree.insert("", "end", values=(key, display_value))
            
    def update_plots(self, time_data, pleth_data, x_smooth, resp_signal, peaks, freqs, psd, features):
        """Update all plots"""
        try:
            # Plot 1: Live PPG signal
            self.ax1.clear()
            self._setup_dark_plot(self.fig1, self.ax1)
            
            if len(time_data) > 0:
                # Show last 30 seconds
                self.ax1.plot(time_data, pleth_data, 'cyan', linewidth=1, alpha=0.7, label='Raw PPG')
                if x_smooth is not None and len(x_smooth) == len(time_data):
                    self.ax1.plot(time_data, x_smooth, 'yellow', linewidth=2, label='Smoothed')
                
            self.ax1.set_title("Live PPG Signal (Last 30s)")
            self.ax1.set_xlabel("Time (s)")
            self.ax1.set_ylabel("Amplitude")
            self.ax1.legend()
            self.fig1.tight_layout()
            self.canvas1.draw_idle()
            
            # Plot 2: Respiratory signal
            self.ax2.clear()
            self._setup_dark_plot(self.fig2, self.ax2)
            
            if resp_signal is not None and len(resp_signal) > 0:
                self.ax2.plot(time_data, resp_signal, 'lime', linewidth=2, label='Respiratory')
                
                # Mark detected peaks
                if len(peaks) > 0:
                    peak_times = [time_data[i] for i in peaks if i < len(time_data)]
                    peak_values = [resp_signal[i] for i in peaks if i < len(resp_signal)]
                    self.ax2.scatter(peak_times, peak_values, color='red', s=50, marker='x', label='Peaks')
                    
            rr_fft = features.get('rr_bpm_fft', np.nan)
            rr_time = features.get('rr_bpm_time', np.nan)
            title = f"Respiratory Signal - FFT: {rr_fft:.1f} bpm, Time: {rr_time:.1f} bpm"
            self.ax2.set_title(title)
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Amplitude")
            self.ax2.legend()
            self.fig2.tight_layout()
            self.canvas2.draw_idle()
            
            # Plot 3: Frequency domain
            self.ax3.clear()
            self._setup_dark_plot(self.fig3, self.ax3)
            
            if freqs is not None and psd is not None:
                mask = (freqs >= 0.0) & (freqs <= 1.0)
                self.ax3.plot(freqs[mask], psd[mask], 'orange', linewidth=2)
                
                resp_band = (freqs >= 0.1) & (freqs <= 0.5)
                if np.any(resp_band):
                    self.ax3.fill_between(freqs[resp_band], psd[resp_band], alpha=0.3, color='lime')
                    
            self.ax3.set_title("Power Spectral Density (0-1 Hz)")
            self.ax3.set_xlabel("Frequency (Hz)")
            self.ax3.set_ylabel("Power")
            self.fig3.tight_layout()
            self.canvas3.draw_idle()
            
        except Exception as e:
            print(f"Plot update error: {e}")
            
    def update_insights(self, features):
        """Update insights text"""
        try:
            self.insights_text.delete("1.0", tk.END)
            
            insights = []
            
            # RR analysis
            rr_fft = features.get('rr_bpm_fft', np.nan)
            if not np.isnan(rr_fft):
                if 12 <= rr_fft <= 20:
                    insights.append(f"✓ RR {rr_fft:.1f} bpm - Normal range\n")
                elif rr_fft < 12:
                    insights.append(f"⚠ RR {rr_fft:.1f} bpm - Below normal\n")
                else:
                    insights.append(f"⚠ RR {rr_fft:.1f} bpm - Elevated\n")
            else:
                insights.append("⚠ Unable to calculate RR - Need more data\n")
                
            data_points = features.get('data_points', 0)
            duration = features.get('duration_s', 0)
            if duration > 0:
                insights.append(f"📊 {data_points} points over {duration:.1f}s\n")
                
            # Breath count
            breath_count = features.get('breath_count', 0)
            if breath_count > 0:
                insights.append(f"🫁 {breath_count} breaths detected\n")
                
            if not insights:
                insights.append("Waiting for sufficient data...")
                
            self.insights_text.insert("1.0", "".join(insights))
            
        except Exception as e:
            print(f"Insights update error: {e}")
            
    def on_closing(self):
        """Handle application closing"""
        self.is_running = False
        self.stop_all_modes()
        self.destroy()


if __name__ == "__main__":
    app = PPGAnalyzerApp()
    app.mainloop()