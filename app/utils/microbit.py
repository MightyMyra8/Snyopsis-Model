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
                "error": self.error,
            }

    def simulate_scan(self) -> None:
        with self._lock:
            self.scan_count += 1
            self.last_line = f"{SCAN_SIGNAL} (simulated)"


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
