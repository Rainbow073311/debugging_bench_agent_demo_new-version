"""Deterministic Keysight DSOX1204G mean-voltage display capture bridge.

Reads one JSON request from stdin and writes one JSON response to stdout.
The capture action preserves the oscilloscope's current measurement source,
adds a display-interval average-voltage measurement for that source, reads the
result, and downloads the current display as PNG over VXI-11. The lightweight
read action only reads CHANNEL2 mean voltage for the probe-search feedback loop.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Protocol


EXPECTED_MODEL = "DSOX1204G"
CAPTURE_ACTION = "capture_current_display"
READ_MEAN_ACTION = "read_mean_voltage"
SUPPORTED_MEASUREMENT = "MEAN"


class InstrumentProtocol(Protocol):
    timeout: float

    def ask(self, command: str) -> str: ...

    def read_raw(self, num: int = -1) -> bytes: ...

    def write(self, command: str) -> int: ...

    def close(self) -> None: ...


def _default_instrument_factory(host: str) -> InstrumentProtocol:
    try:
        import vxi11
    except ImportError as exc:
        raise RuntimeError(
            "python-vxi11 is required; run "
            "`python -m pip install -r requirements-instruments.txt` "
            "from the Inputdemo directory"
        ) from exc
    return vxi11.Instrument(host)


def _normalize_source(raw_source: str) -> str:
    source = raw_source.strip().strip('"').upper()
    aliases = {
        "CHAN1": "CHANNEL1",
        "CHAN2": "CHANNEL2",
        "CHAN3": "CHANNEL3",
        "CHAN4": "CHANNEL4",
    }
    source = aliases.get(source, source)
    allowed = {
        "CHANNEL1",
        "CHANNEL2",
        "CHANNEL3",
        "CHANNEL4",
        "FUNCTION",
        "MATH",
        "WMEMORY1",
        "WMEMORY2",
    }
    if source not in allowed:
        raise RuntimeError(f"Unsupported DSOX1204G measurement source: {raw_source!r}")
    return source


def _select_current_source(
    instrument: InstrumentProtocol,
    raw_sources: str,
    configured_source: str | None = None,
) -> str:
    if configured_source:
        return _normalize_source(configured_source)

    candidates = [
        _normalize_source(item)
        for item in raw_sources.split(",")
        if item.strip()
    ]
    if not candidates:
        raise RuntimeError("DSOX1204G did not report a measurement source")
    if len(candidates) == 1:
        return candidates[0]

    for source in candidates:
        if not source.startswith("CHANNEL"):
            continue
        channel_number = source.removeprefix("CHANNEL")
        displayed = instrument.ask(f":CHANnel{channel_number}:DISPlay?").strip()
        if displayed in {"1", "ON"}:
            return source
    return candidates[0]


def _parse_ieee_block(payload: bytes) -> bytes:
    if not payload.startswith(b"#") or len(payload) < 3:
        raise RuntimeError("DSOX1204G display response is not an IEEE 488.2 block")
    digits_byte = payload[1:2]
    if not digits_byte.isdigit():
        raise RuntimeError("DSOX1204G display response has an invalid block header")
    digits = int(digits_byte)
    if digits <= 0 or len(payload) < 2 + digits:
        raise RuntimeError("DSOX1204G display response has an incomplete block header")
    length_bytes = payload[2 : 2 + digits]
    if not length_bytes.isdigit():
        raise RuntimeError("DSOX1204G display response has an invalid payload length")
    expected = int(length_bytes)
    start = 2 + digits
    end = start + expected
    if len(payload) < end:
        raise RuntimeError(
            f"DSOX1204G display response is incomplete: expected {expected} bytes, "
            f"received {max(0, len(payload) - start)}"
        )
    return payload[start:end]


def capture_current_display(
    payload: dict[str, Any],
    instrument_factory=_default_instrument_factory,
) -> dict[str, Any]:
    config = payload.get("config") or {}
    host = str(config.get("host") or "192.168.2.3")
    transport = str(config.get("transport") or "vxi11").lower()
    measurement_slot = int(config.get("measurementSlot") or 1)
    desired_measurement = str(
        config.get("desiredMeasurement") or SUPPORTED_MEASUREMENT
    ).upper()
    mean_interval = str(config.get("meanInterval") or "DISPLAY").upper()
    timeout_sec = float(config.get("timeoutMs") or 10000) / 1000.0
    screenshot_path = Path(payload["screenshotPath"]).resolve()

    if transport != "vxi11":
        raise ValueError("DSOX1204G bridge currently requires transport=vxi11")
    if measurement_slot != 1:
        raise ValueError("This workflow currently records measurement slot 1")
    if desired_measurement != SUPPORTED_MEASUREMENT:
        raise ValueError("This workflow is locked to the MEAN measurement function")
    if mean_interval not in {"DISPLAY", "CYCLE"}:
        raise ValueError("meanInterval must be DISPLAY or CYCLE")

    started = time.time()
    instrument = instrument_factory(host)
    instrument.timeout = timeout_sec
    try:
        identity = instrument.ask("*IDN?").strip()
        if EXPECTED_MODEL not in identity.upper():
            raise RuntimeError(
                f"Expected {EXPECTED_MODEL}, but connected instrument reported: {identity}"
            )

        source = _select_current_source(
            instrument,
            instrument.ask(":MEASure:SOURce?"),
            config.get("measurementSource"),
        )
        scpi_measurement = "VAVerage"
        instrument.write(f":MEASure:{scpi_measurement} {mean_interval},{source}")

        value = None
        for _ in range(5):
            candidate = float(
                instrument.ask(
                    f":MEASure:{scpi_measurement}? {mean_interval},{source}"
                )
            )
            if math.isfinite(candidate) and abs(candidate) < 1e36:
                value = candidate
                break
            time.sleep(1.0)
        if value is None:
            raise RuntimeError(
                f"DSOX1204G did not return a valid {desired_measurement} result "
                f"for {source}"
            )

        instrument.write(":DISPlay:DATA? PNG,COLor")
        display_block = instrument.read_raw()
        image_bytes = _parse_ieee_block(display_block)
        if len(image_bytes) < 1024 or not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(
                f"DSOX1204G screenshot is not a valid PNG: {len(image_bytes)} bytes"
            )
    finally:
        instrument.close()

    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.write_bytes(image_bytes)
    return {
        "ok": True,
        "instrument": identity,
        "host": host,
        "transport": transport,
        "visaAddress": f"TCPIP::{host}::inst0::INSTR",
        "measurementSlot": measurement_slot,
        "measurement": desired_measurement,
        "scpiMeasurement": scpi_measurement.upper(),
        "meanInterval": mean_interval,
        "source": source,
        "value": value,
        "unit": "V",
        "screenshotPath": str(screenshot_path),
        "screenshotBytes": len(image_bytes),
        "capturedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "durationMs": round((time.time() - started) * 1000),
    }


def read_mean_voltage(
    payload: dict[str, Any],
    instrument_factory=_default_instrument_factory,
) -> dict[str, Any]:
    config = payload.get("config") or {}
    host = str(config.get("host") or "192.168.2.3")
    transport = str(config.get("transport") or "vxi11").lower()
    timeout_sec = float(config.get("timeoutMs") or 10000) / 1000.0
    source = _normalize_source(str(config.get("probeMeasurementSource") or "CHANNEL2"))
    mean_interval = str(config.get("meanInterval") or "DISPLAY").upper()

    if transport != "vxi11":
        raise ValueError("DSOX1204G bridge currently requires transport=vxi11")
    if source != "CHANNEL2":
        raise ValueError("Probe signal search is locked to CHANNEL2")
    if mean_interval not in {"DISPLAY", "CYCLE"}:
        raise ValueError("meanInterval must be DISPLAY or CYCLE")

    started = time.time()
    instrument = instrument_factory(host)
    instrument.timeout = timeout_sec
    try:
        identity = instrument.ask("*IDN?").strip()
        if EXPECTED_MODEL not in identity.upper():
            raise RuntimeError(
                f"Expected {EXPECTED_MODEL}, but connected instrument reported: {identity}"
            )
        value = float(
            instrument.ask(f":MEASure:VAVerage? {mean_interval},{source}")
        )
        if not math.isfinite(value) or abs(value) >= 1e36:
            raise RuntimeError(
                f"DSOX1204G did not return a valid MEAN result for {source}"
            )
    finally:
        instrument.close()

    return {
        "ok": True,
        "instrument": identity,
        "host": host,
        "transport": transport,
        "measurement": SUPPORTED_MEASUREMENT,
        "meanInterval": mean_interval,
        "source": source,
        "value": value,
        "unit": "V",
        "durationMs": round((time.time() - started) * 1000),
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        action = payload.get("action")
        if action == CAPTURE_ACTION:
            result = capture_current_display(payload)
        elif action == READ_MEAN_ACTION:
            result = read_mean_voltage(payload)
        else:
            raise ValueError(f"Unsupported DSOX1204G action: {action}")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
