"""Stream lines from the micro:bit on COM4 to stdout with timestamps."""
import sys
import time
import serial

PORT = "COM4"
BAUD = 115200

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    print(f"[listening on {PORT} @ {BAUD}]", flush=True)
    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").rstrip()
        if not line:
            continue
        ts = time.strftime("%H:%M:%S")
        print(f"{ts}  {line}", flush=True)
