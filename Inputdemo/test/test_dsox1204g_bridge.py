import importlib.util
import tempfile
import unittest
from pathlib import Path


BRIDGE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "dsox1204g_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("dsox1204g_bridge", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bridge)


class FakeInstrument:
    def __init__(self):
        self.timeout = None
        self.commands = []
        self.closed = False

    def ask(self, command):
        self.commands.append(("ask", command))
        responses = {
            "*IDN?": "KEYSIGHT TECHNOLOGIES,DSOX1204G,CNTEST,02.12",
            ":MEASure:SOURce?": "CHAN2",
            ":MEASure:VAVerage? DISPLAY,CHANNEL2": "3.287654",
        }
        return responses[command]

    def write(self, command):
        self.commands.append(("write", command))
        return len(command)

    def read_raw(self, num=-1):
        self.commands.append(("read_raw", num))
        png = b"\x89PNG\r\n\x1a\n" + (b"x" * 2048)
        length = str(len(png)).encode("ascii")
        return b"#" + str(len(length)).encode("ascii") + length + png + b"\n"

    def close(self):
        self.closed = True


class Dsox1204gBridgeTests(unittest.TestCase):
    def test_fast_mean_read_uses_channel2_without_screenshot(self):
        fake = FakeInstrument()
        result = bridge.read_mean_voltage(
            {
                "config": {
                    "host": "192.168.2.3",
                    "transport": "vxi11",
                    "probeMeasurementSource": "CHANNEL2",
                    "meanInterval": "DISPLAY",
                    "timeoutMs": 5000,
                }
            },
            instrument_factory=lambda host: fake,
        )

        self.assertEqual(result["source"], "CHANNEL2")
        self.assertEqual(result["value"], 3.287654)
        self.assertEqual(result["unit"], "V")
        self.assertNotIn(("write", ":DISPlay:DATA? PNG,COLor"), fake.commands)
        self.assertFalse(any(command[0] == "read_raw" for command in fake.commands))
        self.assertTrue(fake.closed)

    def test_capture_preserves_source_and_returns_mean_png(self):
        fake = FakeInstrument()
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "capture.png"
            result = bridge.capture_current_display(
                {
                    "config": {
                        "host": "192.168.2.3",
                        "transport": "vxi11",
                        "measurementSlot": 1,
                        "desiredMeasurement": "MEAN",
                        "meanInterval": "DISPLAY",
                        "timeoutMs": 5000,
                    },
                    "screenshotPath": str(screenshot),
                },
                instrument_factory=lambda host: fake,
            )

            self.assertEqual(result["measurement"], "MEAN")
            self.assertEqual(result["source"], "CHANNEL2")
            self.assertEqual(result["value"], 3.287654)
            self.assertEqual(result["unit"], "V")
            self.assertTrue(screenshot.read_bytes().startswith(b"\x89PNG"))
            self.assertIn(
                ("write", ":MEASure:VAVerage DISPLAY,CHANNEL2"),
                fake.commands,
            )
            self.assertTrue(fake.closed)

    def test_ieee_block_rejects_incomplete_payload(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            bridge._parse_ieee_block(b"#210abc")

    def test_multiple_measurement_sources_prefer_displayed_channel(self):
        fake = FakeInstrument()
        fake.ask = lambda command: {
            ":CHANnel1:DISPlay?": "0",
            ":CHANnel2:DISPlay?": "1",
        }[command]
        source = bridge._select_current_source(fake, "CHAN1,CHAN2")
        self.assertEqual(source, "CHANNEL2")


if __name__ == "__main__":
    unittest.main()
