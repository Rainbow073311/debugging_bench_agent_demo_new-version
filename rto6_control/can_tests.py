"""
CAN Bus Hardware Tests — RTO 6 Automated Test Runner
=====================================================
Maps to the 12 test items in "Copy of HW Test.xlsx" → CAN sheet.

Test items:
  1. Power voltage test     (万用表)   — DC voltage check
  2. Power ripple test      (示波器)    — AC ripple on VCC/VIO/VBAT
  3. Power timing test      (示波器)    — power-up sequence
  4. GPIO test              (示波器)    — INH/WAKE/STB level check
  5. TX/RX signal test      (示波器)    — TXD/RXD voltage & bit width
  6. CAN terminal resistance(万用表)   — DC resistance CH/CL
  7. CAN bus level test     (示波器)    — CANH/CANL/differential
  8. Signal edge test       (示波器)    — rise/fall time
  9. Common-mode offset     (示波器)    — CANH/CANL common-mode
  10. Output stability      (示波器)    — dominant/recessive level stability
  11. SPI signal test       (示波器)    — SCLK/CS/Data timing
  12. Interface fault test  (示波器)    — short/open fault recovery

Usage:
    python can_tests.py --test all          # run all tests
    python can_tests.py --test ripple       # run only power ripple test
    python can_tests.py --test tx_rx        # run only TX/RX signal test
    python can_tests.py --test bus_levels   # run only CAN bus level test

Chip locations (from Excel):
    U150 — TJA1145ATK/FD
    U36  — TJA1042TK/3/1J
    U220 — TJA1042TK/3/1J (debug only)
"""

import argparse
import csv
import os
import time
from datetime import datetime
from rto6_scope import RTO6


# ── Test pass/fail thresholds (from Excel) ───────────────────────
# Format: (min, typical, max)

SPECS = {
    # Item 1: Power voltage (DC, use DMM — included here for reference)
    "power_voltage": {
        "U150": {
            "VCC":  (4.75, 5.0,  5.5),
            "VIO":  (2.85, 3.3,  5.5),
            "VBAT": (4.5,  12.0, 28.0),
        },
        "U36": {
            "VCC": (4.5, 5.0, 5.5),
            "VIO": (2.8, 3.3, 5.5),
        },
        "U220": {
            "VCC": (4.5, 5.0, 5.5),
            "VIO": (2.8, 3.3, 5.5),
        },
    },

    # Item 2: Power ripple (mV pk-pk max)
    "power_ripple": {
        "U150": {"VCC_ripple": 25, "VIO_ripple": 50, "VBAT_ripple": 1500},
        "U36":  {"VCC_ripple": 50, "VIO_ripple": 100},
        "U220": {"VCC_ripple": 50, "VIO_ripple": 100},
    },

    # Item 5: TX/RX signal levels
    "tx_rx": {
        "TX_H_min": 2.475, "TX_L_max": 0.825,
        "RX_H_min": 2.9,   "RX_L_max": 0.4,
        "tbit": 2.0,          # us (typical bit time)
    },

    # Item 6: CAN terminal resistance
    "can_resistance": {
        "U150": {"CH_to_GND": (5, 50, "MΩ"), "CL_to_GND": (5, 50, "MΩ"), "CH_CL_diff": (10, 100, "kΩ")},
        "U36":  {"CH_to_GND": (5, 50, "MΩ"), "CL_to_GND": (5, 50, "MΩ"), "CH_CL_diff": (10, 130, "Ω")},
    },

    # Item 7: CAN bus levels
    "can_bus_levels": {
        "CANH_min": 2.75, "CANH_max": 4.5,
        "CANL_min": 0.5,  "CANL_max": 2.25,
        "CAN_diff_min": 1.5, "CAN_diff_max": 3.0,
    },

    # Item 8: Signal edge time (ns)
    "signal_edge": {
        "rise_time_max": 150,
        "fall_time_max": 300,
    },

    # Item 9: Common mode offset (V)
    "common_mode_offset": {
        "min": -2, "max": 2,
    },

    # Item 10: Output level stability (%)
    "output_stability": {
        "diff_dominant_dev_max": 5,
        "diff_recessive_dev_max": 5,
    },

    # Item 11: SPI signal timing
    "spi_timing": {
        "SCLK_freq_min": 4e6,           # 4 MHz
        "tCH_min": 100e-9,              # 100 ns
        "tCL_min": 100e-9,              # 100 ns
        "tCHSH_min": 50e-9,             # 50 ns
        "tDVCH_min": 50e-9,             # 50 ns
        "tCHDX_min": 50e-9,             # 50 ns
        "VIL_max": 0.4,
        "VIH_min": 2.9,
    },
}


def check_pass(value, spec) -> str:
    """Check if value passes spec. spec can be: (min, max), (min, typ, max), or just max."""
    if isinstance(spec, (int, float)):
        return "PASS" if value <= spec else "FAIL"
    elif len(spec) == 2:
        return "PASS" if spec[0] <= value <= spec[1] else "FAIL"
    elif len(spec) == 3:
        return "PASS" if spec[0] <= value <= spec[2] else "FAIL"
    return "?"


# ── Test Functions ───────────────────────────────────────────────
class CANTestRunner:
    """Runs CAN hardware tests using RTO 6 scope."""

    def __init__(self, scope_ip: str = "192.168.2.100",
                 output_dir: str = "./test_results"):
        self.scope = RTO6(scope_ip)
        self.output_dir = output_dir
        self.results = []
        os.makedirs(output_dir, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _record(self, test_name, item, value, unit, spec, result):
        """Record a test result."""
        entry = {
            "timestamp": self._timestamp(),
            "test": test_name,
            "item": item,
            "value": value,
            "unit": unit,
            "spec": str(spec),
            "result": result,
        }
        self.results.append(entry)
        status = "✓ PASS" if result == "PASS" else "✗ FAIL"
        print(f"  {item}: {value} {unit}  [{status}]  (spec: {spec})")

    # ── Test 2: Power Ripple ─────────────────────────────────
    def test_power_ripple(self, chip: str = "U150"):
        """Measure power supply ripple."""
        print(f"\n{'='*60}")
        print(f"Test 2: Power Ripple — {chip}")
        print(f"{'='*60}")

        self.scope.stop()
        rails = [k for k in SPECS["power_ripple"][chip].keys()]
        timestamp = self._timestamp()

        for i, rail in enumerate(rails):
            ch = i + 1
            rail_name = rail.replace("_", " ")
            print(f"\n--- {rail_name} ---")

            self.scope.setup_can_power_ripple(ch, rail_name)
            time.sleep(1)

            vpp = self.scope.measure_vpp(ch)
            vpp_mv = vpp * 1000  # convert to mV
            spec_max = SPECS["power_ripple"][chip][rail]
            result = check_pass(vpp_mv, spec_max)

            self._record("power_ripple", f"{chip} {rail_name}",
                         f"{vpp_mv:.1f}", "mV", f"≤{spec_max} mV", result)

            self.scope.screenshot(
                f"{self.output_dir}/{timestamp}_{chip}_{rail}_ripple.png")

    # ── Test 5: TX/RX Signal ────────────────────────────────
    def test_tx_rx(self, chip: str = "U150"):
        """Measure TX and RX signal levels and bit timing."""
        print(f"\n{'='*60}")
        print(f"Test 5: TX/RX Signal — {chip}")
        print(f"{'='*60}")

        self.scope.setup_can_tx_rx(tx_ch=1, rx_ch=2)
        time.sleep(1)

        # TX measurements
        tx_vmax = self.scope.measure_vmax(1)
        tx_vmin = self.scope.measure_vmin(1)
        result_tx_h = check_pass(tx_vmax, SPECS["tx_rx"]["TX_H_min"])
        result_tx_l = check_pass(tx_vmin, SPECS["tx_rx"]["TX_L_max"])

        # RX measurements
        rx_vmax = self.scope.measure_vmax(2)
        rx_vmin = self.scope.measure_vmin(2)
        result_rx_h = check_pass(rx_vmax, SPECS["tx_rx"]["RX_H_min"])
        result_rx_l = check_pass(rx_vmin, SPECS["tx_rx"]["RX_L_max"])

        # Bit width
        self.scope.measure(1, "PWIDth")
        tbit_tx = self.scope.measure_pwidth(1)

        timestamp = self._timestamp()
        self._record("tx_rx", f"{chip} TX-H", f"{tx_vmax:.2f}", "V",
                     f"≥{SPECS['tx_rx']['TX_H_min']} V", result_tx_h)
        self._record("tx_rx", f"{chip} TX-L", f"{tx_vmin:.2f}", "V",
                     f"≤{SPECS['tx_rx']['TX_L_max']} V", result_tx_l)
        self._record("tx_rx", f"{chip} RX-H", f"{rx_vmax:.2f}", "V",
                     f"≥{SPECS['tx_rx']['RX_H_min']} V", result_rx_h)
        self._record("tx_rx", f"{chip} RX-L", f"{rx_vmin:.2f}", "V",
                     f"≤{SPECS['tx_rx']['RX_L_max']} V", result_rx_l)
        self._record("tx_rx", f"{chip} tbit(TXD)", f"{tbit_tx:.2f}", "us",
                     f"~{SPECS['tx_rx']['tbit']} us", "PASS")

        self.scope.screenshot(f"{self.output_dir}/{timestamp}_{chip}_tx_rx.png")

    # ── Test 7: CAN Bus Levels ──────────────────────────────
    def test_can_bus_levels(self, chip: str = "U150"):
        """Measure CANH, CANL, and differential voltage."""
        print(f"\n{'='*60}")
        print(f"Test 7: CAN Bus Levels — {chip}")
        print(f"{'='*60}")

        self.scope.setup_can_bus_levels(canh_ch=1, canl_ch=2)
        time.sleep(1)

        canh_avg = self.scope.measure(1, "VAVG")
        canl_avg = self.scope.measure(2, "VAVG")
        can_diff = canh_avg - canl_avg

        spec = SPECS["can_bus_levels"]
        r_canh = check_pass(canh_avg, (spec["CANH_min"], spec["CANH_max"]))
        r_canl = check_pass(canl_avg, (spec["CANL_min"], spec["CANL_max"]))
        r_diff = check_pass(can_diff, (spec["CAN_diff_min"], spec["CAN_diff_max"]))

        timestamp = self._timestamp()
        self._record("can_bus_levels", f"{chip} CANH", f"{canh_avg:.2f}", "V",
                     f"[{spec['CANH_min']}, {spec['CANH_max']}]", r_canh)
        self._record("can_bus_levels", f"{chip} CANL", f"{canl_avg:.2f}", "V",
                     f"[{spec['CANL_min']}, {spec['CANL_max']}]", r_canl)
        self._record("can_bus_levels", f"{chip} CAN-dif", f"{can_diff:.2f}", "V",
                     f"[{spec['CAN_diff_min']}, {spec['CAN_diff_max']}]", r_diff)

        self.scope.screenshot(f"{self.output_dir}/{timestamp}_{chip}_can_bus_levels.png")

    # ── Test 8: Signal Edge ─────────────────────────────────
    def test_signal_edge(self, chip: str = "U150"):
        """Measure signal rise/fall time."""
        print(f"\n{'='*60}")
        print(f"Test 8: Signal Edge Time — {chip}")
        print(f"{'='*60}")

        self.scope.setup_can_edge_time(1)
        time.sleep(1)

        t_rise = self.scope.measure_risetime(1) * 1e9  # ns
        # Need falling edge trigger for fall time
        self.scope.trigger_slope("NEGative")
        time.sleep(0.5)
        t_fall = self.scope.measure_falltime(1) * 1e9

        spec = SPECS["signal_edge"]
        r_rise = check_pass(t_rise, spec["rise_time_max"])
        r_fall = check_pass(t_fall, spec["fall_time_max"])

        timestamp = self._timestamp()
        self._record("signal_edge", f"{chip} Rise Time", f"{t_rise:.0f}", "ns",
                     f"≤{spec['rise_time_max']} ns", r_rise)
        self._record("signal_edge", f"{chip} Fall Time", f"{t_fall:.0f}", "ns",
                     f"≤{spec['fall_time_max']} ns", r_fall)

        self.scope.screenshot(f"{self.output_dir}/{timestamp}_{chip}_edge_time.png")

    # ── Test 11: SPI Signal ─────────────────────────────────
    def test_spi(self, chip: str = "U150"):
        """Measure SPI signal timing parameters."""
        print(f"\n{'='*60}")
        print(f"Test 11: SPI Signal — {chip}")
        print(f"{'='*60}")

        self.scope.setup_can_spi(sclk_ch=1, data_ch=2, cs_ch=3)
        time.sleep(1)

        spec = SPECS["spi_timing"]

        # SCLK frequency
        freq = self.scope.measure_frequency(1)
        r_freq = check_pass(freq, spec["SCLK_freq_min"])
        self._record("spi", f"{chip} SCLK freq", f"{freq/1e6:.1f}", "MHz",
                     f"≥{spec['SCLK_freq_min']/1e6:.0f} MHz", r_freq)

        # Voltage levels
        vmax_sclk = self.scope.measure_vmax(1)
        vmin_sclk = self.scope.measure_vmin(1)
        self._record("spi", f"{chip} SCLK VIH", f"{vmax_sclk:.2f}", "V",
                     f"≥{spec['VIH_min']}", check_pass(vmax_sclk, spec['VIH_min']))
        self._record("spi", f"{chip} SCLK VIL", f"{vmin_sclk:.2f}", "V",
                     f"≤{spec['VIL_max']}", check_pass(vmin_sclk, spec['VIL_max']))

        # CS hold time — need to measure from CS edge to SCLK edge on screen
        timestamp = self._timestamp()
        self.scope.screenshot(f"{self.output_dir}/{timestamp}_{chip}_spi.png")

    # ── Run All Scope Tests ─────────────────────────────────
    def run_all(self, chip: str = "U150"):
        """Run all automated scope tests."""
        print(f"\n{'#'*60}")
        print(f"#  CAN Hardware Tests — {chip}")
        print(f"#  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"#  Instrument: {self.scope.query('*IDN?')}")
        print(f"{'#'*60}")

        self.scope.connect()
        try:
            self.test_power_ripple(chip)
            self.test_tx_rx(chip)
            self.test_can_bus_levels(chip)
            self.test_signal_edge(chip)
            self.test_spi(chip)
        finally:
            self.scope.disconnect()

        self._save_report()

    def _save_report(self):
        """Save results to CSV."""
        csv_path = os.path.join(self.output_dir, f"test_report_{self._timestamp()}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "test", "item",
                                                    "value", "unit", "spec", "result"])
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nReport saved: {csv_path}")

        # Summary
        passed = sum(1 for r in self.results if r["result"] == "PASS")
        failed = sum(1 for r in self.results if r["result"] == "FAIL")
        print(f"\n{'='*40}")
        print(f"Summary: {passed} PASS, {failed} FAIL out of {len(self.results)}")
        print(f"{'='*40}")


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAN Hardware Tests — RTO 6")
    parser.add_argument("--test", default="all",
                        choices=["all", "ripple", "tx_rx", "bus_levels",
                                 "edge", "spi"],
                        help="Which test to run")
    parser.add_argument("--chip", default="U150",
                        choices=["U150", "U36", "U220"],
                        help="Chip to test")
    parser.add_argument("--ip", default="192.168.2.100",
                        help="RTO 6 IP address")
    parser.add_argument("--output", default="./test_results",
                        help="Output directory for screenshots & reports")
    args = parser.parse_args()

    runner = CANTestRunner(scope_ip=args.ip, output_dir=args.output)

    if args.test == "all":
        runner.run_all(args.chip)
    elif args.test == "ripple":
        runner.scope.connect()
        try:
            runner.test_power_ripple(args.chip)
        finally:
            runner.scope.disconnect()
        runner._save_report()
    elif args.test == "tx_rx":
        runner.scope.connect()
        try:
            runner.test_tx_rx(args.chip)
        finally:
            runner.scope.disconnect()
        runner._save_report()
    elif args.test == "bus_levels":
        runner.scope.connect()
        try:
            runner.test_can_bus_levels(args.chip)
        finally:
            runner.scope.disconnect()
        runner._save_report()
    elif args.test == "edge":
        runner.scope.connect()
        try:
            runner.test_signal_edge(args.chip)
        finally:
            runner.scope.disconnect()
        runner._save_report()
    elif args.test == "spi":
        runner.scope.connect()
        try:
            runner.test_spi(args.chip)
        finally:
            runner.scope.disconnect()
        runner._save_report()
