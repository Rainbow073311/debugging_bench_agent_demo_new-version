"""
RTO 6 Series Oscilloscope — Remote Control Module
====================================================
Controls a Rohde & Schwarz RTO 6 oscilloscope via TCP/IP SCPI commands.

Usage:
    from rto6_scope import RTO6

    scope = RTO6("192.168.2.100")
    scope.connect()
    scope.autoscale()
    scope.screenshot("waveform.png")
    scope.disconnect()
"""

import socket
import time
import struct
from io import BytesIO
from contextlib import contextmanager


class RTO6:
    """Remote controller for R&S RTO 6 Series oscilloscope over TCP/IP."""

    # ── Constructor ──────────────────────────────────────────────
    def __init__(self, ip: str = "192.168.2.100", port: int = 5025):
        self._ip = ip
        self._port = port
        self._sock: socket.socket | None = None
        self._buf = b""
        self._timeout = 10.0

    # ── Connection ───────────────────────────────────────────────
    def connect(self) -> str:
        """Open TCP socket to the scope. Returns instrument ID."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        self._sock.connect((self._ip, self._port))
        idn = self.query("*IDN?").strip()
        print(f"Connected to: {idn}")
        return idn

    def disconnect(self):
        """Close the socket."""
        if self._sock:
            self._sock.close()
            self._sock = None
            print("Disconnected.")

    @contextmanager
    def session(self):
        """Context manager: auto-connect / auto-disconnect."""
        self.connect()
        try:
            yield self
        finally:
            self.disconnect()

    # ── Low-level I/O ────────────────────────────────────────────
    def write(self, cmd: str):
        """Send a SCPI command (no response expected)."""
        self._sock.sendall((cmd + "\n").encode())

    def query(self, cmd: str) -> str:
        """Send a SCPI command and return the response string."""
        self.write(cmd)
        return self._read()

    def query_binary(self, cmd: str) -> bytes:
        """Send a SCPI command that returns binary data (e.g. screenshot)."""
        self.write(cmd)
        return self._read_binary()

    def _read(self) -> str:
        """Read a line-terminated text response."""
        while b"\n" not in self._buf:
            chunk = self._sock.recv(8192)
            if not chunk:
                raise ConnectionError("Socket closed by instrument.")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("utf-8", errors="replace").strip()

    def _read_binary(self) -> bytes:
        """Read binary block data (IEEE 488.2 #<len><data> format)."""
        # Read the '#'
        while len(self._buf) < 2:
            self._buf += self._sock.recv(8192)
        if self._buf[0:1] == b"#":
            digit_count = int(self._buf[1:2])
            header_len = 2 + digit_count
            while len(self._buf) < header_len:
                self._buf += self._sock.recv(8192)
            data_len = int(self._buf[2:header_len])
            total_len = header_len + data_len
            while len(self._buf) < total_len:
                self._buf += self._sock.recv(max(8192, total_len - len(self._buf)))
            data = self._buf[header_len:total_len]
            self._buf = self._buf[total_len:]
            return data
        else:
            raise ValueError(f"Expected binary block, got: {self._buf[:20]}")

    def wait_opc(self):
        """Wait until all pending commands are done."""
        self.query("*OPC?")

    # ── Run Control ──────────────────────────────────────────────
    def run(self):
        """Start continuous acquisition."""
        self.write("RUN")

    def stop(self):
        """Stop acquisition."""
        self.write("STOP")

    def single(self):
        """Single-shot acquisition."""
        self.write("SINGle")

    # ── Channel Configuration ────────────────────────────────────
    def channel_on(self, ch: int = 1):
        self.write(f"CHANnel{ch}:STATe ON")

    def channel_off(self, ch: int = 1):
        self.write(f"CHANnel{ch}:STATe OFF")

    def channel_scale(self, ch: int, volts_per_div: float):
        """Set vertical scale in V/div. e.g. scope.channel_scale(1, 0.5)"""
        self.write(f"CHANnel{ch}:SCALe {volts_per_div}")

    def channel_offset(self, ch: int, offset: float):
        """Set vertical offset in volts."""
        self.write(f"CHANnel{ch}:OFFSet {offset}")

    def channel_coupling(self, ch: int, coupling: str):
        """Set coupling: 'DC', 'AC', or 'GND'."""
        self.write(f"CHANnel{ch}:COUPling {coupling}")

    def channel_bandwidth(self, ch: int, bw: str):
        """Set bandwidth limit. e.g. 'FULL', '20MHz', '200MHz'."""
        self.write(f"CHANnel{ch}:BANDwidth {bw}")

    def channel_probe(self, ch: int, attenuation: float):
        """Set probe attenuation. e.g. 1, 10, 100."""
        self.write(f"CHANnel{ch}:PROBe:SETup:ATTenuation {attenuation}")

    def channel_label(self, ch: int, label: str):
        """Set channel label."""
        self.write(f'CHANnel{ch}:LABel "{label}"')

    def channel_info(self, ch: int) -> dict:
        """Return key settings for a channel."""
        return {
            "state": self.query(f"CHANnel{ch}:STATe?"),
            "scale": float(self.query(f"CHANnel{ch}:SCALe?")),
            "offset": float(self.query(f"CHANnel{ch}:OFFSet?")),
            "coupling": self.query(f"CHANnel{ch}:COUPling?"),
        }

    # ── Timebase ─────────────────────────────────────────────────
    def timebase_scale(self, seconds: float):
        """Set horizontal scale in s/div."""
        self.write(f"TIMebase:SCALe {seconds}")

    def timebase_position(self, seconds: float):
        """Set horizontal position (delay)."""
        self.write(f"TIMebase:POSition {seconds}")

    def timebase_info(self) -> dict:
        """Return timebase settings."""
        return {
            "scale": float(self.query("TIMebase:SCALe?")),
            "position": float(self.query("TIMebase:POSition?")),
            "sample_rate": float(self.query("ACQuire:SRATe?")),
        }

    # ── Trigger ──────────────────────────────────────────────────
    def trigger_source(self, source: str):
        """Set trigger source. e.g. 'CH1', 'CH2', 'EXT'."""
        self.write(f"TRIGger:SOURce {source}")

    def trigger_level(self, level: float):
        """Set trigger level in volts."""
        self.write(f"TRIGger:LEVel {level}")

    def trigger_slope(self, slope: str):
        """Set trigger slope: 'POSitive' or 'NEGative'."""
        self.write(f"TRIGger:EDGE:SLOPe {slope}")

    def trigger_mode(self, mode: str):
        """Set trigger mode: 'AUTO' or 'NORMal'."""
        self.write(f"TRIGger:MODE {mode}")

    def trigger_info(self) -> dict:
        """Return trigger settings."""
        return {
            "source": self.query("TRIGger:SOURce?"),
            "level": float(self.query("TRIGger:LEVel?")),
            "slope": self.query("TRIGger:EDGE:SLOPe?"),
            "mode": self.query("TRIGger:MODE?"),
        }

    # ── Acquire ──────────────────────────────────────────────────
    def acquire_average(self, count: int | None = None):
        """Enable averaging. count=4,8,16,32,64,128,256,512,1024.
        Pass count=None or 0 to disable."""
        if count and count > 0:
            self.write(f"ACQuire:AVERage:COUNt {count}")
            self.write("ACQuire:AVERage:STATe ON")
        else:
            self.write("ACQuire:AVERage:STATe OFF")

    def acquire_mode(self, mode: str):
        """Set acquisition mode: 'SAMPle', 'PEAK', 'AVERage', 'HRESolution'."""
        self.write(f"ACQuire:TYPE {mode}")

    # ── Measurements ─────────────────────────────────────────────
    def measure(self, ch: int, func: str, retries: int = 5) -> float:
        """Run a measurement. func: 'VPP', 'VMAX', 'VMIN', 'VAVG', 'VRMS',
        'FREQuency', 'PERiod', 'PWIDth', 'RISEtime', 'FALLtime', etc.
        """
        self.write("MEASurement:ENABle ON")
        self.write(f"MEASurement1:SOURce CH{ch}")
        self.write(f"MEASurement1:MAIN {func}")
        self.write("MEASurement1:STATe ON")
        time.sleep(0.6)  # 等测量完成
        for attempt in range(retries):
            try:
                return float(self.query("MEASurement1:RESult?"))
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(1)

    def measure_vpp(self, ch: int = 1) -> float:
        """Measure peak-to-peak voltage."""
        return self.measure(ch, "VPP")

    def measure_vmax(self, ch: int = 1) -> float:
        """Measure maximum voltage."""
        return self.measure(ch, "VMAX")

    def measure_vmin(self, ch: int = 1) -> float:
        """Measure minimum voltage."""
        return self.measure(ch, "VMIN")

    def measure_vrms(self, ch: int = 1) -> float:
        """Measure RMS voltage."""
        return self.measure(ch, "VRMS")

    def measure_frequency(self, ch: int = 1) -> float:
        """Measure frequency in Hz."""
        return self.measure(ch, "FREQuency")

    def measure_risetime(self, ch: int = 1) -> float:
        """Measure rise time (10%-90%)."""
        return self.measure(ch, "RISEtime")

    def measure_falltime(self, ch: int = 1) -> float:
        """Measure fall time (90%-10%)."""
        return self.measure(ch, "FALLtime")

    def measure_pwidth(self, ch: int = 1) -> float:
        """Measure positive pulse width."""
        return self.measure(ch, "PWIDth")

    # ── Screenshot ───────────────────────────────────────────────
    def disconnect_scpi(self):
        """断开 SCPI 连接。"""
        if self._sock:
            self._sock.close()
            self._sock = None

    def reconnect_scpi(self):
        """重建 SCPI 连接。"""
        time.sleep(2.0)  # 等示波器 SCPI 服务释放旧连接
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        self._sock.connect((self._ip, self._port))
        self._buf = b""

    def screenshot(self, filepath: str = "screenshot.png") -> str:
        """Capture the scope's actual screen via Web UI API.

        断开 SCPI → 示波器恢复本地显示 → HTTP 截图 → 重连 SCPI。
        每次断连间隔足够长，避免 SCPI 服务崩溃。
        """
        import urllib.request
        import urllib.error
        import urllib.parse

        self.disconnect_scpi()
        time.sleep(2.0)

        base = f"http://{self._ip}/instrumentctrl/html/php"

        trigger_url = f"{base}/makescreenshot.php?metaDataOn=false"
        try:
            resp = urllib.request.urlopen(trigger_url, timeout=10)
            filename = resp.read().decode("utf-8", errors="replace").strip()
            if not filename.endswith(".jpg"):
                raise RuntimeError(f"Screenshot failed: {filename}")
        except urllib.error.URLError as e:
            self.reconnect_scpi()
            raise RuntimeError(f"Cannot reach Web UI: {e}")

        download_url = (
            f"{base}/makescreenshotdownload.php"
            f"?filename={urllib.parse.quote(filename)}&usage=display"
        )
        resp = urllib.request.urlopen(download_url, timeout=30)
        data = resp.read()

        with open(filepath, "wb") as f:
            f.write(data)
        print(f"Screenshot saved: {filepath}  ({len(data):,} bytes)")

        self.reconnect_scpi()
        return filepath

    def _drain_buffer(self):
        """Drain any pending data from the socket."""
        old_timeout = self._sock.gettimeout()
        self._sock.settimeout(0.2)
        try:
            while True:
                chunk = self._sock.recv(8192)
                if not chunk:
                    break
        except Exception:
            pass
        self._sock.settimeout(old_timeout)

    def _read_mmem_file(self, remote_path: str) -> bytes:
        """Read a file from the scope via MMEM:DATA? and return raw content."""
        self.write(f'MMEM:DATA? "{remote_path}"')
        time.sleep(0.5)
        return self._read_binary_mmem()

    def _read_binary_mmem(self) -> bytes:
        """Read binary block from MMEM:DATA? response."""
        # Read header
        while len(self._buf) < 2:
            chunk = self._sock.recv(8192)
            if not chunk:
                raise ConnectionError("Socket closed.")
            self._buf += chunk
        if self._buf[0:1] != b"#":
            raise ValueError(f"Expected #, got {self._buf[:20]}")
        digit_count = int(self._buf[1:2])
        header_len = 2 + digit_count
        while len(self._buf) < header_len:
            self._buf += self._sock.recv(8192)
        data_len = int(self._buf[2:header_len].decode())
        total_len = header_len + data_len
        # MMEM adds a newline after the header
        while len(self._buf) < total_len + 1:
            self._buf += self._sock.recv(max(8192, total_len - len(self._buf)))
        # Check for newline after header
        offset = header_len
        if self._buf[offset:offset + 1] == b"\n":
            offset += 1
            total_len += 1
        # Read more if needed
        while len(self._buf) < total_len:
            self._buf += self._sock.recv(8192)
        data = self._buf[offset:total_len]
        self._buf = self._buf[total_len:]
        return data

    # ── Web Interface ─────────────────────────────────────────────
    def open_web(self, zoom: float = 0.7):
        """Open the scope's Web Control interface in Chrome App Mode.

        Opens without browser toolbars, auto-zoomed to fit the window.
        Adjust ``zoom`` to match your screen (0.5 ~ 1.0).
        """
        import subprocess
        import shutil
        import os
        url = f"http://{self._ip}"

        # Try Chrome first (app mode = no tabs/toolbars/scrollbars)
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            shutil.which("chrome"),
            shutil.which("google-chrome"),
        ]

        chrome = None
        for p in chrome_paths:
            if p and os.path.exists(p):
                chrome = p
                break

        if chrome:
            # App mode + zoom + window size
            cmd = [
                chrome,
                f"--app={url}",
                f"--force-device-scale-factor={zoom}",
                "--window-size=1280,900",
                "--new-window",
            ]
            subprocess.Popen(cmd, shell=False)
            print(f"Chrome App Mode opened: {url}  (zoom={zoom}x)")
        else:
            # Fallback: default browser
            import webbrowser
            print(f"Chrome not found, using default browser: {url}")
            webbrowser.open(url)

    # ── Autoset ──────────────────────────────────────────────────
    def autoscale(self, timeout: float = 10.0):
        """Auto-set: scope automatically adjusts scale & trigger.
        Waits for completion before returning."""
        self.write("AUToscale")
        old = self._sock.gettimeout()
        self._sock.settimeout(timeout)
        try:
            self.query("*OPC?")
        except Exception:
            pass
        self._sock.settimeout(old)

    # ── Smart Auto Range ────────────────────────────────────────
    def smart_autoscale(self, ch: int = 1):
        """智能自适应：自动调整量程、位置、时基、触发，使波形居中清晰。

        1. AUToscale 找到信号
        2. 测 Vpp → 选最佳 V/div（信号占 4~6 格）
        3. 测 Vavg → 设垂直偏移使波形居中
        4. 设触发 = 信号中点
        5. 调时基
        """
        self.autoscale()
        time.sleep(2)

        try:
            vpp = self.measure_vpp(ch)
            vavg = self.measure(ch, 'VAVG')
        except Exception:
            vpp, vavg = 0, 0

        # 1. 垂直量程：信号占 4~6 格
        std_scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        if vpp > 0.002:
            target = vpp / 5  # 目标占 5 格
            best = min(std_scales, key=lambda s: abs(s - target))
            self.channel_scale(ch, best)
        else:
            best = 0.01
            self.channel_scale(ch, 0.01)

        # 2. 垂直位置：波形居中
        self.channel_offset(ch, vavg)

        # 3. 触发 = 信号中点
        self.trigger_level(vavg)
        self.trigger_source(f'CH{ch}')
        self.trigger_mode('AUTO')

        # 4. 时基
        self.timebase_scale(1e-3)  # 纹波默认 1ms/div
        time.sleep(0.5)

        print(f'[smart_autoscale] CH{ch}: Vpp={vpp*1000:.1f}mV, Vavg={vavg*1000:.1f}mV '
              f'→ {best*1000:.0f}mV/div ({vpp/best:.1f}divs), offset={vavg:.3f}V')

    # ── Quick Setup ──────────────────────────────────────────────
    def quick_setup(self, ch1_scale: float = 1.0,
                    time_scale: float = 1e-6,
                    trigger_level: float = 1.5):
        """Quickly set up a single-channel measurement."""
        self.stop()
        self.channel_on(1)
        self.channel_scale(1, ch1_scale)
        self.channel_coupling(1, "DC")
        self.timebase_scale(time_scale)
        self.trigger_source("CH1")
        self.trigger_level(trigger_level)
        self.trigger_mode("AUTO")
        self.run()

    # ── Differential Measurement ─────────────────────────────────
    def setup_diff(self, ch1_scale: float = 1.0, ch2_scale: float = 1.0,
                    time_scale: float = 1e-6, trig_src: str = "CH1",
                    trig_level: float = 2.0):
        """Quick setup for 2-channel differential measurement (e.g. CANH/CANL)."""
        self.stop()
        self.channel_on(1)
        self.channel_on(2)
        self.channel_scale(1, ch1_scale)
        self.channel_scale(2, ch2_scale)
        self.channel_coupling(1, "DC")
        self.channel_coupling(2, "DC")
        self.timebase_scale(time_scale)
        self.trigger_source(trig_src)
        self.trigger_level(trig_level)
        self.trigger_mode("AUTO")
        self.run()

    # ── CAN Bus Test Setups ──────────────────────────────────────
    def setup_can_power_ripple(self, ch: int = 1, rail: str = "VCC"):
        """Set up for power supply ripple measurement (Excel item 2).
        - AC coupling, small vertical scale, 20 MHz bandwidth limit."""
        self.stop()
        self.channel_on(ch)
        self.channel_coupling(ch, "AC")
        self.channel_scale(ch, 0.01)    # 10 mV/div
        self.channel_bandwidth(ch, "20MHz")
        self.channel_probe(ch, 1)
        self.channel_label(ch, f"{rail}_Ripple")
        self.timebase_scale(1e-6)       # 1 us/div
        self.trigger_source(f"CH{ch}")
        self.trigger_level(0.01)
        self.trigger_mode("AUTO")
        self.acquire_mode("PEAK")       # peak detect for transients
        self.run()
        print(f"[setup_can_power_ripple] CH{ch} AC-coupled, 10 mV/div, 20 MHz BW")
        print(f"  Measure VPP: {self.measure_vpp(ch):.4f} V")

    def setup_can_tx_rx(self, tx_ch: int = 1, rx_ch: int = 2):
        """Set up for CAN TX/RX signal test (Excel item 5).
        - TX and RX signals, measure voltage levels and bit width."""
        self.stop()
        for ch in [tx_ch, rx_ch]:
            self.channel_on(ch)
            self.channel_coupling(ch, "DC")
            self.channel_scale(ch, 1.0)
            self.channel_probe(ch, 1)
        self.channel_label(tx_ch, "CAN_TX")
        self.channel_label(rx_ch, "CAN_RX")
        self.timebase_scale(2e-6)       # 2 us/div
        self.trigger_source(f"CH{tx_ch}")
        self.trigger_level(1.5)
        self.trigger_mode("NORMal")
        self.run()
        print(f"[setup_can_tx_rx] CH{tx_ch}=TX, CH{rx_ch}=RX, 1 V/div, 2 us/div")

    def setup_can_bus_levels(self, canh_ch: int = 1, canl_ch: int = 2):
        """Set up for CAN bus level measurement (Excel item 7).
        - CANH and CANL DC-coupled, measure differential voltage."""
        self.stop()
        for ch in [canh_ch, canl_ch]:
            self.channel_on(ch)
            self.channel_coupling(ch, "DC")
            self.channel_scale(ch, 1.0)
            self.channel_probe(ch, 1)
        self.channel_label(canh_ch, "CAN_H")
        self.channel_label(canl_ch, "CAN_L")
        self.timebase_scale(1e-6)       # 1 us/div
        self.trigger_source(f"CH{canh_ch}")
        self.trigger_level(2.5)
        self.trigger_mode("AUTO")
        self.run()
        print(f"[setup_can_bus_levels] CH{canh_ch}=CANH, CH{canl_ch}=CANL")

    def setup_can_edge_time(self, ch: int = 1):
        """Set up for signal edge measurement (Excel item 8).
        - Measure rise/fall time on CAN bus."""
        self.stop()
        self.channel_on(ch)
        self.channel_coupling(ch, "DC")
        self.channel_scale(ch, 1.0)
        self.timebase_scale(50e-9)      # 50 ns/div — zoomed in for edges
        self.trigger_source(f"CH{ch}")
        self.trigger_level(2.0)
        self.trigger_slope("POSitive")
        self.trigger_mode("NORMal")
        self.run()
        print(f"[setup_can_edge_time] CH{ch}, 50 ns/div for edge measurement")

    def setup_can_spi(self, sclk_ch: int = 1, data_ch: int = 2, cs_ch: int = 3):
        """Set up for SPI signal measurement (Excel item 11)."""
        self.stop()
        for ch, label in [(sclk_ch, "SPI_SCLK"),
                          (data_ch, "SPI_MOSI"),
                          (cs_ch, "SPI_CS")]:
            self.channel_on(ch)
            self.channel_coupling(ch, "DC")
            self.channel_scale(ch, 1.0)
            self.channel_probe(ch, 1)
            self.channel_label(ch, label)
        self.timebase_scale(100e-6)     # 100 us/div for SPI frames
        self.trigger_source(f"CH{cs_ch}")
        self.trigger_level(2.0)
        self.trigger_slope("NEGative")
        self.trigger_mode("NORMal")
        self.run()
        print(f"[setup_can_spi] CH{sclk_ch}=SCLK, CH{data_ch}=MOSI, CH{cs_ch}=CS")


# ── Quick Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    scope = RTO6()
    with scope.session():
        print("\n=== Scope Info ===")
        print("Timebase:", scope.timebase_info())
        print("Trigger:", scope.trigger_info())
        print("CH1:", scope.channel_info(1))

        print("\n=== Quick measurement ===")
        vpp = scope.measure_vpp(1)
        freq = scope.measure_frequency(1)
        print(f"CH1 Vpp={vpp:.4f} V, Freq={freq:.2f} Hz")

        print("\n=== Screenshot ===")
        scope.screenshot("rto6_screenshot.png")
