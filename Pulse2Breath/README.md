# Pulse2Breath

Real-time and playback toolkit to derive respiratory metrics and other features from **photoplethysmography (PPG)** signals. It comes with a desktop UI (**Tk + Matplotlib**) that shows the live PPG waveform, a respiration proxy derived from PPG, and a frequency analysis panel, alongside running feature estimates.

-----

## What This Project Does

This project derives respiration from PPG using a simple pipeline:

  * Smooth the PPG signal.
  * Low-pass the signal to the respiratory band.
  * Estimate the **respiratory rate (RR)** by using both an **FFT peak** and a **time-domain peak count**.

It displays three synchronized views of the last \~30-second window:

  * **PPG** (raw and smoothed).
  * **Respiratory proxy** (low-passed PPG).
  * **PSD (0–1 Hz)** with the respiration band highlighted.

It also computes live features such as RR (FFT and time-domain) and breath count.

The toolkit can run in three different modes:

  * **Simulation mode** (using synthetic PPG data).
  * **File playback** (replays a recorded .txt file as if it were live).
  * **Serial mode** (experimental) to read directly from a USB-to-UART PPG device.

-----

## Current Status: Serial vs. Playback

I attempted to connect to a physical PPG device over a serial connection to plot its stream live. The device reports a baud rate of 115200, but its data framing is non-text/binary and uses changing header bytes. I implemented an auto-decoder that attempts to lock onto 5- or 6-byte frames (with observed patterns like `C]`, `=*`, `?*` for the second header byte) and extracts 16-bit little-endian samples. When the decoding fails, the UI automatically enters a **RAW-bytes** mode, showing only the waveform and skipping feature calculations to prevent displaying misleading numbers.

Because the device's framing is not fully specified and varies, the most reliable method today is **file playback**. You can load a .txt recording from a PPG probe, and the application will replay it in real-time, allowing you to validate plots and features end-to-end.

-----

## Getting Started

There are three versions of the application, all in the form of a single script with a built-in UI using **Streamlit**:

1.  `1_ppg_extraction.py`
2.  `2_ppg_live.py`
3.  `3_ppg_serial.py`

You can run any of these scripts from your IDE (like VS Code) using the command `python app_name.py` or by selecting "Run Without Debugging."

-----

## Differences Between Versions

| Filename | Description |
| :--- | :--- |
| `1_ppg_extraction.py` | This is the most minimalistic and well-tested version. It loads any PPG .txt record, plots PPG, Respiratory, and Power Spectral Density graphs, and extracts customizable features and insights. It also includes an option to save records by ticking a button at the top right. |
| `2_ppg_live.py` | This version offers two key abilities: 1) simulating synthetic PPG signals for testing and 2) loading a pre-recorded .txt file and replaying it in real-time to simulate a live data feed from a PPG probe. |
| `3_ppg_serial.py` | This version was designed to connect to and plot real-time values from a physical PPG probe but faced compatibility issues. When connected to the probe, it can switch to a RAW-bytes-only mode, which plots any received values at the cost of accuracy. |

Videos demonstrating the usage of each mode have also been uploaded.

-----

## Requirements

  * **Python 3.x+**
  * The following Python libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `pyserial`, and `streamlit`.
  * **Tk**, which is included with most Python installations.

You can install the required libraries by running:
`pip install numpy scipy pandas matplotlib pyserial streamlit`

-----

## How to Run the App

Open your terminal or command prompt and use the following command:
`python app_name.py`

-----

## Using the App

### 1\) Pre-Recorded .txt (`1_ppg_extraction.py`)

Load a .txt PPG record from your files and watch it get plotted. You'll also have the option to save the record.

### 2\) File Playback (`2_ppg_live.py`)

1.  Click **Load File**.
2.  Select a .txt file with columns such as:
    ```
    Time,Pleth,SPO2,PR,SI
    0,2050,98,72,6
    1,2057,98,72,6
    2,2068,98,73,6
    ```
    Note: The `Time` column is optional. If it is missing or irregular, the application will use a default sample rate (100 Hz) or infer a rate.

The file will be replayed as if it were a live data stream: plots update approximately every 10 Hz, features update continuously, and a progress indicator shows your position in the file.

### 3\) Serial Device (`3_ppg_serial.py`)

1.  Plug in your USB-UART PPG board (e.g., CP210x).
2.  Click **Refresh**, select the correct **COM port**, and then click **Connect** (defaults to 115200 baud).

<!-- end list -->

  * If your device uses the known binary frames, the app will auto-decode the data, and you'll see realistic pleth and features.
  * If not, the status will show **"RAW stream (bytes) — decoding…"**, plotting a byte-based waveform without features.

**Sniffer:** Use the **Sniff** button to dump the first \~256 bytes (in both HEX and ASCII) to the console. This can help you refine the decoder for new header variants.

-----

## How It Works (High-Level)

  * **Respiratory proxy**: The PPG signal is smoothed and then low-pass filtered (with a cutoff of ≈0.5 Hz) to isolate the baseline wander associated with respiration.
  * **RR (FFT)**: The respiration proxy is resampled to a lower rate, and the dominant frequency in the 0.1–0.5 Hz range is converted into breaths per minute.
  * **RR (time-domain)**: The application also counts peaks with a minimum spacing and converts the result to breaths/min over the current window.
  * **Serial auto-decoder (experimental)**: This feature attempts to parse 16-bit little-endian samples from repeating 5- or 6-byte frames. If decoding fails for a long enough duration, the app stays in RAW mode and does not compute features.

-----

## Keyboard/GUI Notes

  * **Status** (top right) shows the current mode: `Simulation`, `File`, `Connected`, or `RAW decoding`.
  * The **Features table** lists `RR (FFT & time)`, `breath count`, `data points`, and `window length`.
  * **Tabs** include `PPG`, `Respiratory signal`, and `PSD`.
  * **Sniff** dumps HEX/ASCII to the console (and consumes a small portion of the serial buffer).

-----

## Troubleshooting

  * **No plots**: Start with Simulation or File mode first to confirm the UI is working correctly.
  * **Serial connected, but "RAW stream (bytes)"**: Your device is sending an unrecognized binary format. Use the **Sniff** button and share the dump to help extend the decoder.
  * **Flat or "steppy" waveform**: This often indicates a framing or alignment issue. Use **Sniff**, then reconnect.
  * **High or odd COM number**: Some applications may have issues with high COM ports. You can reassign the port in Windows Device Manager (under Port Settings → Advanced).