from flask import Flask, jsonify, render_template
import telnetlib
import threading
import time
from scipy.signal import find_peaks, savgol_filter
import numpy as np
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

threading.Thread(target=read_ekg_data, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    with data_lock:
        ekg_snapshot = ekg_data[-BATCH_SIZE:]
        interval_snapshot = batch_interval_seconds

    # Only use the interval if it's within a valid range (2.8–3.3 seconds)
    if interval_snapshot and 2.8 <= interval_snapshot <= 3.3:
        bpm = calculate_bpm(
            ekg_snapshot,
            interval_snapshot * 1000
        )
    else:
        bpm = None  # Skip BPM calculation for bad intervals

    if bpm is not None:
        print(f"❤️ Calculated BPM: {bpm}")
    elif interval_snapshot:
        print(f"⚠️ Skipped BPM calculation: invalid interval {interval_snapshot}s")

    return jsonify({
        "ekg": ekg_snapshot,
        "interval": interval_snapshot,
        "bpm": bpm
    })



if __name__ == "__main__":
    app.run(debug=True)
