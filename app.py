from flask import Flask, jsonify, render_template
import telnetlib
import threading
import time
from scipy.signal import find_peaks, savgol_filter, welch
from scipy.stats import entropy
import numpy as np
import statistics
from threading import Lock

app = Flask(__name__)

# create one global lock, near the top of the file
data_lock = Lock()

EKG_HOST = "192.168.235.95"
EKG_PORT = 23
MAX_BUFFER_SIZE = 1000
BATCH_SIZE = 500

ekg_data = []
last_batch_time = None
batch_interval_seconds = None
arrhythmia_detected = False
vfib_detected = False
afib_detected = False

def format_ekg_preview(data):
    preview = ", ".join(str(v) for v in data[:10])
    if len(data) > 10:
        preview += "..."
    return f"[{preview}] ({len(data)} values)"

def read_ekg_data():
    global ekg_data, last_batch_time, batch_interval_seconds
    try:
        print("Connecting to Arduino EKG Telnet server...")
        with telnetlib.Telnet(EKG_HOST, EKG_PORT, timeout=10) as tn:
            tn.write(b"\n")
            print("Connected to Arduino Telnet server.")

            while True:
                try:
                    data = tn.read_very_eager()
                    if data:
                        decoded = data.decode().strip()
                        parts = decoded.split(",")

                        new_values = []
                        for part in parts:
                            try:
                                value = int(part.strip())
                                new_values.append(value)
                            except ValueError:
                                continue

                        if new_values:
                            now = time.time()
                            if last_batch_time is not None:
                                batch_interval_seconds = round(now - last_batch_time, 2)
                                if 2.8 <= batch_interval_seconds <= 3.3:
                                    print(f"⏱ Batch interval: {batch_interval_seconds}s ✅")
                                else:
                                    print(f"⏱ Batch interval: {batch_interval_seconds}s ❌ Skipped for BPM")

                            last_batch_time = now
                            ekg_data.extend(new_values)
                            if len(ekg_data) > MAX_BUFFER_SIZE:
                                ekg_data = ekg_data[-MAX_BUFFER_SIZE:]

                            print("📈 EKG Data:", format_ekg_preview(new_values))

                    time.sleep(0.5)
                except EOFError:
                    print("Connection closed by Arduino.")
                    break
    except Exception as e:
        print(f"❌ Error: {e}")

def get_peaks_with_values(peaks, signal):
    return [(int(p), float(signal[p])) for p in peaks]

def calculate_bpm(EKGvals, timeElapsedMs):
    if len(EKGvals) < 30 or timeElapsedMs is None:
        return None  # Too little data or missing interval
    y = np.array(EKGvals)
    y_smooth = savgol_filter(y, window_length=min(21, len(y) // 2 * 2 + 1), polyorder=3)
    peaks, _ = find_peaks(y_smooth, distance=10, prominence=5)
    valleys, _ = find_peaks(-y_smooth, distance=10, prominence=5)
    cycle_peaks = []
    for i in range(len(valleys) - 1):
        start, end = valleys[i], valleys[i + 1]
        cycle_peak_indices = [p for p in peaks if start < p < end]
        cycle_peaks.extend(cycle_peak_indices)
    peaks_with_values = get_peaks_with_values(cycle_peaks, y_smooth)
    beatsPerSecond = len(peaks_with_values) / 3 / (timeElapsedMs / 1000)
    bpm = beatsPerSecond * 60
    return round(bpm, 1)

def arrhythmia_exists(EKGvals):
    if len(EKGvals) < 30:
        return False
        
    y = np.array(EKGvals)
    y_smooth = savgol_filter(y, window_length=min(21, len(y) // 2 * 2 + 1), polyorder=3)
    peaks, _ = find_peaks(y_smooth, distance=10, prominence=5)
    valleys, _ = find_peaks(-y_smooth, distance=10, prominence=5)
    
    cycle_peaks = []
    for i in range(len(valleys) - 1):
        start, end = valleys[i], valleys[i + 1]
        cycle_peak_indices = [p for p in peaks if start < p < end]
        cycle_peaks.extend(cycle_peak_indices)
    
    if len(cycle_peaks) < 4:  # Need at least 4 peaks to compare
        return False
        
    peaks_with_values = get_peaks_with_values(cycle_peaks, y_smooth)
    peak_distances = [int(peaks_with_values[i][0] - peaks_with_values[i - 3][0]) 
                     for i in range(3, len(peaks_with_values))]
    
    if not peak_distances:
        return False
        
    st_dev = statistics.stdev(peak_distances)
    mean_distance = statistics.mean(peak_distances)
    
    # Thresholds for arrhythmia detection
    return st_dev > 20 or mean_distance < 100 or mean_distance > 180

def detect_vfib_afib(EKGvals):
    global vfib_detected, afib_detected
    
    if len(EKGvals) < 100:  # Need enough data for reliable detection
        return False, False
    
    y = np.array(EKGvals)
    
    # Normalize signal
    y = (y - np.mean(y)) / np.std(y)
    
    # Smooth signal
    y_smooth = savgol_filter(y, window_length=21, polyorder=3)
    
    # Parameters for windowing
    window_size = min(250, len(y_smooth))  # 1 second window at 250 Hz
    stride = window_size // 2
    
    def compute_rms(window):
        return np.sqrt(np.mean(window ** 2))
    
    def compute_spectral_entropy(window, fs=250, nperseg=128):
        f, Pxx = welch(window, fs=fs, nperseg=nperseg)
        Pxx_norm = Pxx / np.sum(Pxx)
        return entropy(Pxx_norm)
    
    def count_r_peaks(signal_segment, height_thresh=0.5, distance=20):
        peaks, _ = find_peaks(signal_segment, height=height_thresh, distance=distance)
        return len(peaks)
    
    # Analyze windows
    vfib_windows = []
    flatline_windows = []
    
    for i in range(0, len(y_smooth) - window_size, stride):
        window = y_smooth[i:i + window_size]
        rms = compute_rms(window)
        ent = compute_spectral_entropy(window)
        peak_count = count_r_peaks(window, height_thresh=0.5, distance=20)
        
        # VFib detection criteria
        if rms < 0.4 and ent > 1.8:
            vfib_windows.append((i, i + window_size))
        # Flatline detection
        elif rms < 0.25 and ent < 1.5:
            flatline_windows.append((i, i + window_size))
    
    # Merge adjacent windows
    def merge_windows(windows, max_gap=1):
        if not windows:
            return []
        merged = []
        current_start, current_end = windows[0]
        for start, end in windows[1:]:
            if start <= current_end + max_gap * stride:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        return merged
    
    vfib_windows_merged = merge_windows(vfib_windows)
    
    # Final VFib detection requires at least 3 seconds of continuous VFib
    vfib_duration_threshold = 3 * window_size
    vfib_detected = any((end - start) >= vfib_duration_threshold 
                     for start, end in vfib_windows_merged)
    
    # AFib is detected if arrhythmia exists but not VFib
    afib_detected = not vfib_detected and arrhythmia_exists(EKGvals)
    
    return vfib_detected, afib_detected

threading.Thread(target=read_ekg_data, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    global arrhythmia_detected, vfib_detected, afib_detected
    
    with data_lock:
        ekg_snapshot = ekg_data[-BATCH_SIZE:]
        interval_snapshot = batch_interval_seconds

    # Only use the interval if it's within a valid range (2.8–3.3 seconds)
    if interval_snapshot and 2.8 <= interval_snapshot <= 3.3:
        bpm = calculate_bpm(
            ekg_snapshot,
            interval_snapshot * 1000
        )
        # Run arrhythmia detection only when we have a valid interval
        arrhythmia_detected = arrhythmia_exists(ekg_snapshot)
        
        # If arrhythmia detected, run VFib/AFib detection
        if arrhythmia_detected:
            vfib_detected, afib_detected = detect_vfib_afib(ekg_snapshot)
            if vfib_detected:
                print("⚠️⚠️⚠️ VENTRICULAR FIBRILLATION DETECTED! ⚠️⚠️⚠️")
            elif afib_detected:
                print("⚠️⚠️⚠️ ATRIAL FIBRILLATION DETECTED! ⚠️⚠️⚠️")
    else:
        bpm = None  # Skip BPM calculation for bad intervals
        vfib_detected = False
        afib_detected = False

    if bpm is not None:
        print(f"❤️ Calculated BPM: {bpm}")
    elif interval_snapshot:
        print(f"⚠️ Skipped BPM calculation: invalid interval {interval_snapshot}s")

    return jsonify({
        "ekg": ekg_snapshot,
        "interval": interval_snapshot,
        "bpm": bpm,
        "arrhythmia": arrhythmia_detected,
        "vfib": vfib_detected,
        "afib": afib_detected
    })

if __name__ == "__main__":
    app.run(debug=True)