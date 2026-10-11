"""
Background listener for a BBC micro:bit acting as a 'test strip scanner'.

The micro:bit (MakeCode prototype) emits the line `SCAN COMPLETE` over USB
serial when its P0 pin is touched. This module auto-detects the micro:bit's
COM port (by USB VID) and reads lines on a daemon thread.

Use via Streamlit's @st.cache_resource so only ONE listener exists per
process even when Streamlit hot-reloads the script.
"""

import random
import threading
import time
from collections import deque

import serial
import serial.tools.list_ports

# BBC micro:bit's USB Vendor ID (ARM mbed). Both v1 and v2 use the same VID.
MICROBIT_VID = 0x0D28
SCAN_SIGNAL = "SCAN COMPLETE"
FAIL_SIGNAL = "SCAN FAILED"
BAUDRATE = 115200


def find_microbit_port() -> str | None:
    for p in serial.tools.list_ports.comports():
        if p.vid == MICROBIT_VID:
            return p.device
    return None


class MicrobitListener:
    """Long-lived serial reader. Instantiate once per process via st.cache_resource."""

    def __init__(self) -> None:
        self.port: str | None = None
        self.connected: bool = False
        self.scan_count: int = 0
        self.failed_scan_count: int = 0
        self.last_line: str = ""
        self.last_shade: str = ""        # "dark" | "normal" | "light" | ""
        self.last_red_ratio: float = 0.0 # 0..100, percent of total RGB
        # Pulse sensor — rolling buffer of last 200 samples + derived BPM
        self.pulse_buffer: deque = deque(maxlen=200)
        self.pulse_timestamps: deque = deque(maxlen=200)
        self.last_bpm: int = 0
        self._peak_times: deque = deque(maxlen=10)
        self._pulse_baseline: float = 512.0
        self.error: str | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, name="microbit-listener", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            port = find_microbit_port()
            if port is None:
                with self._lock:
                    self.port = None
                    self.connected = False
                    self.error = "micro:bit not detected — plug it in via USB"
                time.sleep(2)
                continue

            try:
                with serial.Serial(port, BAUDRATE, timeout=1) as ser:
                    with self._lock:
                        self.port = port
                        self.connected = True
                        self.error = None
                    while True:
                        raw = ser.readline()
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        with self._lock:
                            self.last_line = line
                            # Parse fields tolerantly — micro:bit USB serial drops
                            # characters under load, so use fuzzy substring matching
                            # against known shade keywords.
                            line_lower = line.lower()
                            if "hydrat" in line_lower or "dehyd" in line_lower or "dhyd" in line_lower:
                                self.last_shade = "dehydrated"
                            elif "poor" in line_lower or "control" in line_lower or "ontrol" in line_lower:
                                self.last_shade = "poor_control"
                            elif "heal" in line_lower or "ealthy" in line_lower:
                                self.last_shade = "healthy"
                            elif "anem" in line_lower or "nemic" in line_lower:
                                self.last_shade = "anemic"
                            import re
                            ratio_match = re.search(r"ratio=([\d.]+)", line)
                            if ratio_match:
                                try:
                                    self.last_red_ratio = float(ratio_match.group(1))
                                except ValueError:
                                    pass
                            # Pulse sensor: micro:bit streams "PULSE=<value>" ~50Hz
                            pulse_match = re.search(r"PULSE=(\d+)", line)
                            if pulse_match:
                                try:
                                    val = int(pulse_match.group(1))
                                    now = time.time()
                                    self.pulse_buffer.append(val)
                                    self.pulse_timestamps.append(now)
                                    self._pulse_baseline = 0.98 * self._pulse_baseline + 0.02 * val
                                    if len(self.pulse_buffer) >= 6:
                                        recent = list(self.pulse_buffer)[-6:]
                                        threshold = self._pulse_baseline + 30
                                        if (val > threshold
                                                and val == max(recent)
                                                and (not self._peak_times
                                                     or now - self._peak_times[-1] > 0.35)):
                                            self._peak_times.append(now)
                                            if len(self._peak_times) >= 3:
                                                intervals = [
                                                    self._peak_times[i + 1] - self._peak_times[i]
                                                    for i in range(len(self._peak_times) - 1)
                                                ]
                                                avg_interval = sum(intervals) / len(intervals)
                                                if 0.35 <= avg_interval <= 2.0:
                                                    self.last_bpm = int(60.0 / avg_interval)
                                except ValueError:
                                    pass
                            if FAIL_SIGNAL in line:
                                self.failed_scan_count += 1
                            elif SCAN_SIGNAL in line:
                                self.scan_count += 1
            except (serial.SerialException, OSError) as e:
                with self._lock:
                    self.connected = False
                    self.error = f"serial error: {e}"
                time.sleep(2)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "port": self.port,
                "connected": self.connected,
                "scan_count": self.scan_count,
                "failed_scan_count": self.failed_scan_count,
                "last_line": self.last_line,
                "last_shade": self.last_shade,
                "last_red_ratio": self.last_red_ratio,
                "pulse_buffer": list(self.pulse_buffer),
                "pulse_timestamps": list(self.pulse_timestamps),
                "last_bpm": self.last_bpm,
                "error": self.error,
            }

    def simulate_scan(self, shade: str = "") -> None:
        with self._lock:
            self.scan_count += 1
            tag = f" shade={shade}" if shade else ""
            self.last_line = f"{SCAN_SIGNAL} (simulated){tag}"
            if shade:
                self.last_shade = shade


def generate_sample(seed: int | None = None) -> dict:
    """Build a randomized, realistic teen biomarker panel for a 'scanned strip'."""
    rng = random.Random(seed)
    return {
        "activity_minutes": rng.randint(30, 400),
        "sugar_intake": round(rng.uniform(20.0, 150.0), 1),
        "fiber_intake": round(rng.uniform(5.0, 35.0), 1),
        "crp_level": round(rng.uniform(0.2, 8.0), 2),
        "nlr_value": round(rng.uniform(0.8, 5.0), 2),
        "age": rng.randint(12, 19),
        "gender": rng.choice(["Male", "Female"]),
        "bmi": round(rng.uniform(17.0, 32.0), 1),
        "waist": round(rng.uniform(55.0, 105.0), 1),
        "hba1c": round(rng.uniform(4.8, 6.2), 1),
        "systolic_bp": float(rng.randint(95, 135)),
        "diastolic_bp": float(rng.randint(55, 85)),
        "carb_percent": round(rng.uniform(40.0, 65.0), 1),
    }
