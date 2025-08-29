#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

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
#    Core signal helpers
# =========================
def load_ppg_file(path: str):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    for col in ["Pleth", "SPO2", "PR", "SI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    times = pd.to_datetime(df["Time"], format="%H:%M:%S")
    duration_sec = (times.iloc[-1] - times.iloc[0]).total_seconds()
    if duration_sec <= 0:
        duration_sec = df["Time"].ne(df["Time"].shift()).cumsum().max() - 1
    fs = len(df) / max(duration_sec, 1)
    return df, float(fs), float(duration_sec)


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data)


def derive_resp_signal_from_ppg(ppg, fs):
    x = np.asarray(ppg, dtype=float)
    x = x - np.nanmean(x)
    win = max(int(0.8 * fs), 3)
    x_smooth = pd.Series(x).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
    resp_signal = butter_lowpass_filter(x_smooth, cutoff=0.5, fs=fs, order=4)
    return x_smooth, resp_signal


def rr_fft(resp_signal, fs, fmin=0.1, fmax=0.5):
    t = np.arange(len(resp_signal)) / fs
    if len(t) < 2:
        return np.nan, None, None
    target_fs = 5.0
    t_new = np.arange(0, t[-1], 1.0 / target_fs)
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
    peaks, _ = find_peaks(resp_signal, distance=fs * 1.5)
    duration_min = len(resp_signal) / fs / 60.0
    rr_bpm = len(peaks) / duration_min if duration_min > 0 else np.nan
    return rr_bpm, peaks


def breath_interval_stats(peaks, fs):
    if len(peaks) < 2:
        return {"bti_mean_s": np.nan, "bti_sd_s": np.nan, "bti_cv": np.nan}
    intervals_s = np.diff(peaks) / fs
    bti_mean = float(np.mean(intervals_s))
    bti_sd = float(np.std(intervals_s, ddof=1)) if len(intervals_s) > 1 else 0.0
    bti_cv = float(bti_sd / bti_mean) if bti_mean > 0 else np.nan
    return {"bti_mean_s": bti_mean, "bti_sd_s": bti_sd, "bti_cv": bti_cv}


def ie_ratio_from_resp(resp_signal, fs):
    peaks, _ = find_peaks(resp_signal, distance=fs * 1.2)
    troughs, _ = find_peaks(-resp_signal, distance=fs * 1.2)
    if len(peaks) < 2 or len(troughs) < 2:
        return np.nan, np.nan, np.nan
    peak_t = np.array(peaks) / fs
    trough_t = np.array(troughs) / fs
    insp, exp = [], []
    for tr in trough_t:
        nxt_peak = peak_t[peak_t > tr]
        if len(nxt_peak) == 0:
            break
        pk = nxt_peak[0]
        nxt_tr = trough_t[trough_t > pk]
        if len(nxt_tr) == 0:
            break
        tr2 = nxt_tr[0]
        insp.append(pk - tr)
        exp.append(tr2 - pk)
    if not insp or not exp:
        return np.nan, np.nan, np.nan
    insp = np.array(insp); exp = np.array(exp)
    return float(np.median(insp / exp)), float(np.median(insp)), float(np.median(exp))


def resp_power_ratio(freqs, psd, fmin=0.1, fmax=0.5):
    if freqs is None or psd is None:
        return np.nan
    band = (freqs >= fmin) & (freqs <= fmax)
    lf = (freqs >= 0.0) & (freqs <= 1.0)
    p_band = float(np.trapz(psd[band], freqs[band])) if np.any(band) else np.nan
    p_lf = float(np.trapz(psd[lf], freqs[lf])) if np.any(lf) else np.nan
    return float(p_band / p_lf) if p_lf and p_lf > 0 else np.nan


def extract_rr_features(ppg_df, fs):
    t = np.arange(len(ppg_df)) / fs
    x_smooth, resp_signal = derive_resp_signal_from_ppg(ppg_df["Pleth"].values, fs)
    rr_fft_bpm, freqs, psd = rr_fft(resp_signal, fs)
    rr_td_bpm, peaks = rr_time_domain(resp_signal, fs)
    bti = breath_interval_stats(peaks, fs)
    ie_ratio, insp_med, exp_med = ie_ratio_from_resp(resp_signal, fs)
    p_ratio = resp_power_ratio(freqs, psd)
    features = {
        "rr_bpm_fft": rr_fft_bpm,
        "rr_bpm_time": rr_td_bpm,
        "breath_count": int(len(peaks)),
        "duration_s": float(len(resp_signal) / fs),
        "bti_mean_s": bti["bti_mean_s"],
        "bti_sd_s": bti["bti_sd_s"],
        "bti_cv": bti["bti_cv"],
        "ie_ratio_median": ie_ratio,
        "insp_median_s": insp_med,
        "exp_median_s": exp_med,
        "resp_power_ratio": p_ratio,
        "fs_hz": fs,
        "rr_resolution_bpm": float(60.0 / max(len(resp_signal) / fs, 1.0)),
    }
    return features, t, x_smooth, resp_signal, peaks, freqs, psd


def interpret_rr_features(features):
    import math
    out = []
    rr = features.get("rr_bpm_fft", math.nan)
    res = features.get("rr_resolution_bpm", math.nan)
    if not math.isnan(rr):
        if 12 <= rr <= 20:
            out.append(f"RR {rr:.1f} bpm (±{res:.1f}) is within normal resting range (12–20).")
        elif rr < 12:
            out.append(f"RR {rr:.1f} bpm (±{res:.1f}) is below normal (bradypnea).")
        else:
            out.append(f"RR {rr:.1f} bpm (±{res:.1f}) is elevated (tachypnea).")
    ie = features.get("ie_ratio_median", math.nan)
    if not math.isnan(ie):
        if 0.4 <= ie <= 0.6:
            out.append(f"I:E ≈ {ie:.2f} (~1:2) suggests relaxed breathing.")
        elif ie < 0.4:
            out.append(f"I:E {ie:.2f}: short inspiration / long expiration.")
        elif ie > 0.7:
            out.append(f"I:E {ie:.2f}: prolonged inspiration (possible obstructive pattern).")
    cv = features.get("bti_cv", math.nan)
    if not math.isnan(cv):
        out.append(f"Breathing regularity {'is good' if cv < 0.3 else 'shows higher variability'} (CV {cv:.2f}).")
    pr = features.get("resp_power_ratio", math.nan)
    if not math.isnan(pr):
        out.append(f"Respiratory modulation is {'strong' if pr > 0.2 else 'weak'} (power ratio {pr:.2f}).")
    return out


# =========================
#           GUI
# =========================
class RRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PPG Respiratory Rate — Analyzer")
        self.configure(bg="#1f1f1f")
        self.geometry("1100x700")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#1f1f1f", foreground="#e8e8e8", fieldbackground="#2a2a2a")
        style.configure("TLabel", background="#1f1f1f", foreground="#e8e8e8", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Dark.TEntry", fieldbackground="#2a2a2a", foreground="#e8e8e8")
        style.configure("Green.TButton", font=("Segoe UI", 10, "bold"),
                        background="#2e7d32", foreground="#ffffff", padding=8)
        style.map("Green.TButton", background=[("active", "#348f38")])
        style.configure("Grey.TButton", font=("Segoe UI", 10, "bold"),
                        background="#424242", foreground="#ffffff", padding=8)
        style.map("Grey.TButton", background=[("active", "#4f4f4f")])
        style.configure("Tag.TButton", font=("Segoe UI", 9, "bold"),
                        background="#2e7d32", foreground="#d8ffd8", padding=6, relief="flat")

        # Top controls
        top = ttk.Frame(self, padding=12); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Select file to analyze:", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.path_var, width=70, style="Dark.TEntry").grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(top, text="Select file", style="Green.TButton", command=self.pick_file).grid(row=1, column=1, padx=(0, 8))
        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Save CSV & PNGs", variable=self.save_var, takefocus=False).grid(row=1, column=2)
        ttk.Button(top, text="Analyze", style="Grey.TButton", command=self.analyze).grid(row=1, column=3, padx=(8, 0))
        top.columnconfigure(0, weight=1)

        # Middle: tags + features + insights
        mid = ttk.Frame(self, padding=(12, 0, 12, 8)); mid.pack(side=tk.TOP, fill=tk.X)

        '''tags = ttk.Frame(mid); tags.pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(tags, text="Quick info:").pack(anchor="w")
        for txt in ["Mobility", "Continence", "Behaviour", "Complex Care"]:
            ttk.Button(tags, text=txt, style="Tag.TButton").pack(side=tk.LEFT, padx=6, pady=4)'''

        right = ttk.Frame(mid); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Extracted Features:", style="Header.TLabel").pack(anchor="w")

        # Custom style only for features Treeview
        features_style = ttk.Style()
        features_style.theme_use("clam")
        features_style.configure("Features.Treeview",
                                 background="white",
                                 foreground="black",
                                 fieldbackground="white",
                                 rowheight=22)
        features_style.map("Features.Treeview",
                           background=[("selected", "#4CAF50")],
                           foreground=[("selected", "white")])

        # Create Treeview with custom style
        self.tree = ttk.Treeview(right,
                                 columns=("key", "value"),
                                 show="headings",
                                 height=9,
                                 style="Features.Treeview")
        self.tree.heading("key", text="Feature")
        self.tree.column("key", width=240, anchor="w")
        self.tree.heading("value", text="Value")
        self.tree.column("value", width=220, anchor="w")
        self.tree.pack(fill=tk.X, pady=6)

        ttk.Label(right, text="Interpretation:", style="Header.TLabel").pack(anchor="w", pady=(10, 0))
        self.insights = tk.Text(right, height=5, wrap="word", bg="#2a2a2a", fg="#e8e8e8",
                                insertbackground="#e8e8e8", relief="flat", padx=8, pady=6)
        self.insights.pack(fill=tk.X)

        # Bottom notebook with 3 plots
        nb = ttk.Notebook(self); nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.fig1 = Figure(figsize=(5.2, 2.6), dpi=100); self.ax1 = self.fig1.add_subplot(111)
        self._darken(self.fig1, self.ax1)
        tab1 = ttk.Frame(nb); nb.add(tab1, text="Smoothed PPG")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=tab1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig2 = Figure(figsize=(5.2, 2.6), dpi=100); self.ax2 = self.fig2.add_subplot(111)
        self._darken(self.fig2, self.ax2)
        tab2 = ttk.Frame(nb); nb.add(tab2, text="Resp Signal")   
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=tab2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig3 = Figure(figsize=(5.2, 2.6), dpi=100); self.ax3 = self.fig3.add_subplot(111)
        self._darken(self.fig3, self.ax3)
        tab3 = ttk.Frame(nb); nb.add(tab3, text="PSD 0–1 Hz")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=tab3)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _darken(self, fig: Figure, ax):
        fig.patch.set_facecolor("#1f1f1f")
        ax.set_facecolor("#1f1f1f")

    def pick_file(self):
        path = filedialog.askopenfilename(title="Select PPG .txt",
                                          filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self.path_var.set(path)

    def analyze(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please select a PPG .txt file first.")
            return
        try:
            df, fs, _ = load_ppg_file(path)
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load file:\n{e}")
            return

        try:
            feats, t, x_smooth, resp, peaks, freqs, psd = extract_rr_features(df, fs)
        except Exception as e:
            messagebox.showerror("Processing error", f"Failed to process signal:\n{e}")
            return

        # Table
        for row in self.tree.get_children():
            self.tree.delete(row)
        order = ["rr_bpm_fft","rr_bpm_time","breath_count","duration_s","bti_mean_s",
                 "bti_sd_s","bti_cv","ie_ratio_median","insp_median_s","exp_median_s",
                 "resp_power_ratio","fs_hz","rr_resolution_bpm"]
        for k in order:
            v = feats.get(k, np.nan)
            self.tree.insert("", "end", values=(k, f"{v:.4f}" if isinstance(v, float) else v))

        # Insights
        self.insights.delete("1.0", tk.END)
        self.insights.insert(tk.END, "\n".join(f"• {line}" for line in interpret_rr_features(feats)))

        # Plots
        self.ax1.clear(); self._darken(self.fig1, self.ax1)
        self.ax1.plot(t, x_smooth)
        self.ax1.set_title("Smoothed PPG (slow modulation)")
        self.ax1.set_xlabel("Time (s)"); self.ax1.set_ylabel("Pleth (detrended)")
        self.fig1.tight_layout(); self.canvas1.draw_idle()

        self.ax2.clear(); self._darken(self.fig2, self.ax2)
        self.ax2.plot(t, resp, label="Resp")
        if len(peaks) > 0:
            self.ax2.scatter(np.array(peaks)/feats["fs_hz"], resp[peaks], marker="x")
        self.ax2.set_title("Derived Respiration Signal (with peaks)")
        self.ax2.set_xlabel("Time (s)"); self.ax2.set_ylabel("Amplitude (a.u.)")
        self.fig2.tight_layout(); self.canvas2.draw_idle()

        self.ax3.clear(); self._darken(self.fig3, self.ax3)
        if freqs is not None and psd is not None:
            mask = (freqs >= 0.0) & (freqs <= 1.0)
            self.ax3.plot(freqs[mask], psd[mask])
        title = f"PSD (0–1 Hz). FFT RR ≈ {feats['rr_bpm_fft']:.1f} bpm" if not np.isnan(feats["rr_bpm_fft"]) else "PSD (0–1 Hz)"
        self.ax3.set_title(title); self.ax3.set_xlabel("Frequency (Hz)"); self.ax3.set_ylabel("Power")
        self.fig3.tight_layout(); self.canvas3.draw_idle()

    # Optional save
        if self.save_var.get():
            prefix = Path(path).with_suffix("").as_posix()
            report_path = f"{prefix}_report.pdf"
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                import matplotlib.pyplot as plt

                with PdfPages(report_path) as pdf:
                    # -------- Page 1: Features + Interpretation --------
                    fig_feat, axf = plt.subplots(figsize=(8.5, 11))
                    fig_feat.patch.set_facecolor("white")  
                    axf.set_facecolor("white")              
                    axf.axis("off")

                    lines = ["Extracted RR Features\n"]
                    order = ["rr_bpm_fft","rr_bpm_time","breath_count","duration_s","bti_mean_s",
                            "bti_sd_s","bti_cv","ie_ratio_median","insp_median_s","exp_median_s",
                            "resp_power_ratio","fs_hz","rr_resolution_bpm"]

                    for k in order:
                        v = feats.get(k, np.nan)
                        vtxt = f"{v:.4f}" if isinstance(v, float) else str(v)
                        lines.append(f"{k:>20}:  {vtxt}")

                    lines.append("\nInterpretation")
                    for line in interpret_rr_features(feats):
                        lines.append(f"• {line}")

                    axf.text(0.01, 0.99,
                            "\n".join(lines),
                            va="top", ha="left",
                            fontsize=12, color="black",
                            wrap=True)

                    pdf.savefig(fig_feat, bbox_inches="tight", facecolor=fig_feat.get_facecolor())
                    plt.close(fig_feat)

                    # -------- Pages 2–4: existing figures --------
                    pdf.savefig(self.fig1, bbox_inches="tight")
                    pdf.savefig(self.fig2, bbox_inches="tight")
                    pdf.savefig(self.fig3, bbox_inches="tight")

                messagebox.showinfo("Saved", f"Report saved to:\n{report_path}")

            except Exception as e:
                messagebox.showwarning("Save warning", f"Processed but could not save report:\n{e}")


if __name__ == "__main__":
    app = RRApp()
    app.mainloop()
