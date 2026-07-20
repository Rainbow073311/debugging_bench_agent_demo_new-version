"""
RTO 6 自动测试 + 截屏返回客户端
================================
按测试流程自动配置示波器、测量、截图，生成 HTML 报告。

Usage:
    python auto_test.py                     # 运行全部 CAN 测试
    python auto_test.py --test ripple       # 只测电源纹波
    python auto_test.py --test tx_rx        # 只测 TX/RX 信号
    python auto_test.py --test bus_levels   # 只测 CAN 总线电平
    python auto_test.py --test edge         # 只测边沿时间
    python auto_test.py --test spi          # 只测 SPI 信号

输出:
    test_results/<timestamp>/
    ├── report.html         ← 浏览器打开，看所有截图 + 测量结果
    ├── 01_power_ripple.png
    ├── 02_tx_rx.png
    ├── 03_can_bus_levels.png
    ├── 04_signal_edge.png
    ├── 05_spi_signal.png
    └── results.csv         ← 原始数据
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from rto6_scope import RTO6


# ── 测试规格（来自 Excel）─────────────────────────────────────────
SPECS = {
    "U150_power_ripple": {"VCC_ripple": 25, "VIO_ripple": 50, "VBAT_ripple": 1500},
    "U36_power_ripple":  {"VCC_ripple": 50, "VIO_ripple": 100},
    "U220_power_ripple": {"VCC_ripple": 50, "VIO_ripple": 100},
    "tx_rx":  {"TX_H_min": 2.475, "TX_L_max": 0.825, "RX_H_min": 2.9, "RX_L_max": 0.4},
    "can_bus": {"CANH": (2.75, 4.5), "CANL": (0.5, 2.25), "CAN_diff": (1.5, 3.0)},
    "edge":    {"rise_max": 150, "fall_max": 300},  # ns
    "spi":     {"freq_min": 4e6, "VIH_min": 2.9, "VIL_max": 0.4},
}


def check(v, spec):
    """判断 PASS / FAIL。spec 可以是 (min,max) 或 max_value。"""
    if isinstance(spec, tuple):
        return "PASS" if spec[0] <= v <= spec[1] else "FAIL"
    return "PASS" if v <= spec else "FAIL"


class AutoTester:
    """自动测试引擎：配置 → 测量 → 截图 → 生成报告"""

    def __init__(self, ip="192.168.2.100"):
        self.scope = RTO6(ip)
        self.session_dir = ""
        self.screenshots = []     # (序号, 名称, 路径)
        self.results = []         # 测量结果
        self.scope.connect()

    def close(self):
        self.scope.disconnect()

    # ── 初始化目录 ───────────────────────────────────────────
    def init_session(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join("test_results", ts)
        os.makedirs(self.session_dir, exist_ok=True)
        return self.session_dir

    # ── 截图 + 记录 ──────────────────────────────────────────
    def snap(self, step: int, name: str) -> str:
        """截图并返回文件路径。自动等待示波器退出程控模式后截取。"""
        filename = f"{step:02d}_{name}.png"
        path = os.path.join(self.session_dir, filename)
        self.scope.screenshot(path)
        self.screenshots.append((step, name, path))
        return path

    # ── 测量 + 判断 + 记录 ───────────────────────────────────
    def log(self, step, item, value, unit, spec):
        result = check(value, spec)
        self.results.append({
            "step": step, "item": item,
            "value": f"{value:.2f}", "unit": unit,
            "spec": str(spec), "result": result,
        })
        icon = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"  [{step:02d}] {icon} {item}: {value:.2f} {unit}  (spec: {spec})")

    # ── 测试 1: 电源纹波 ─────────────────────────────────────
    def test_power_ripple(self, chip="U150", step_start=1):
        rails = SPECS[f"{chip}_power_ripple"]
        step = step_start
        print(f"\n{'─'*50}")
        print(f"[测试] 电源纹波 — {chip}")
        print(f"{'─'*50}")

        for i, (rail, max_mv) in enumerate(rails.items()):
            ch = i + 1
            self.scope.stop()
            self.scope.channel_on(ch)
            self.scope.channel_coupling(ch, "AC")
            self.scope.channel_bandwidth(ch, "20MHz")
            self.scope.channel_label(ch, rail.replace("_", " "))
            self.scope.run()
            self.scope.autoscale()
            time.sleep(1)

            vpp_mv = self.scope.measure_vpp(ch) * 1000
            self.log(step, f"{chip} {rail}", vpp_mv, "mV", max_mv)
            self.snap(step, f"{chip}_{rail}")
            step += 1

        return step

    # ── 测试 2: TX/RX 信号 ───────────────────────────────────
    def test_tx_rx(self, chip="U150", tx_ch=1, rx_ch=2, step_start=1):
        step = step_start
        print(f"\n{'─'*50}")
        print(f"[测试] TX/RX 信号 — {chip}")
        print(f"{'─'*50}")

        self.scope.stop()
        for ch, label in [(tx_ch, "CAN_TX"), (rx_ch, "CAN_RX")]:
            self.scope.channel_on(ch)
            self.scope.channel_coupling(ch, "DC")
            self.scope.channel_label(ch, label)
        self.scope.run()
        self.scope.autoscale()
        time.sleep(1)

        s = SPECS["tx_rx"]
        tx_max = self.scope.measure_vmax(tx_ch)
        tx_min = self.scope.measure_vmin(tx_ch)
        rx_max = self.scope.measure_vmax(rx_ch)
        rx_min = self.scope.measure_vmin(rx_ch)

        self.log(step, f"{chip} TX-H", tx_max, "V", (s["TX_H_min"], 999))
        step += 1
        self.log(step, f"{chip} TX-L", tx_min, "V", (-999, s["TX_L_max"]))
        step += 1
        self.log(step, f"{chip} RX-H", rx_max, "V", (s["RX_H_min"], 999))
        step += 1
        self.log(step, f"{chip} RX-L", rx_min, "V", (-999, s["RX_L_max"]))
        step += 1


        self.snap(step, f"{chip}_tx_rx")
        return step + 1

    # ── 测试 3: CAN 总线电平 ─────────────────────────────────
    def test_can_bus_levels(self, chip="U150", canh_ch=1, canl_ch=2, step_start=1):
        step = step_start
        print(f"\n{'─'*50}")
        print(f"[测试] CAN 总线电平 — {chip}")
        print(f"{'─'*50}")

        self.scope.stop()
        for ch, label in [(canh_ch, "CAN_H"), (canl_ch, "CAN_L")]:
            self.scope.channel_on(ch)
            self.scope.channel_coupling(ch, "DC")
            self.scope.channel_label(ch, label)
        self.scope.run()
        self.scope.autoscale()
        time.sleep(1)

        s = SPECS["can_bus"]
        canh = self.scope.measure(1, "VAVG")
        canl = self.scope.measure(2, "VAVG")
        diff = canh - canl

        self.log(step, f"{chip} CANH", canh, "V", s["CANH"])
        step += 1
        self.log(step, f"{chip} CANL", canl, "V", s["CANL"])
        step += 1
        self.log(step, f"{chip} CAN-diff", diff, "V", s["CAN_diff"])
        step += 1

        self.snap(step, f"{chip}_can_bus_levels")
        return step + 1

    # ── 测试 4: 边沿时间 ─────────────────────────────────────
    def test_signal_edge(self, chip="U150", ch=1, step_start=1):
        step = step_start
        print(f"\n{'─'*50}")
        print(f"[测试] 信号边沿时间 — {chip}")
        print(f"{'─'*50}")

        self.scope.stop()
        self.scope.channel_on(ch)
        self.scope.channel_coupling(ch, "DC")
        self.scope.run()
        self.scope.autoscale()
        time.sleep(1)

        t_rise = self.scope.measure_risetime(ch) * 1e9
        self.log(step, f"{chip} Rise Time", t_rise, "ns", SPECS["edge"]["rise_max"])
        step += 1

        self.scope.trigger_slope("NEGative")
        time.sleep(0.5)
        t_fall = self.scope.measure_falltime(ch) * 1e9
        self.log(step, f"{chip} Fall Time", t_fall, "ns", SPECS["edge"]["fall_max"])
        step += 1


        self.snap(step, f"{chip}_edge_time")
        return step + 1

    # ── 测试 5: SPI 信号 ─────────────────────────────────────
    def test_spi(self, chip="U150", sclk_ch=1, data_ch=2, cs_ch=3, step_start=1):
        step = step_start
        print(f"\n{'─'*50}")
        print(f"[测试] SPI 信号 — {chip}")
        print(f"{'─'*50}")

        self.scope.stop()
        for ch, label in [(sclk_ch, "SPI_SCLK"), (data_ch, "SPI_MOSI"), (cs_ch, "SPI_CS")]:
            self.scope.channel_on(ch)
            self.scope.channel_coupling(ch, "DC")
            self.scope.channel_label(ch, label)
        self.scope.run()
        self.scope.autoscale()
        time.sleep(1)

        s = SPECS["spi"]
        freq = self.scope.measure_frequency(sclk_ch)
        vmax = self.scope.measure_vmax(sclk_ch)
        vmin = self.scope.measure_vmin(sclk_ch)

        self.log(step, f"{chip} SCLK Freq", freq / 1e6, "MHz", (s["freq_min"] / 1e6, 999))
        step += 1
        self.log(step, f"{chip} SCLK VIH", vmax, "V", (s["VIH_min"], 999))
        step += 1
        self.log(step, f"{chip} SCLK VIL", vmin, "V", (-999, s["VIL_max"]))
        step += 1

        self.snap(step, f"{chip}_spi")
        return step + 1

    # ── 生成 HTML 报告 ───────────────────────────────────────
    def generate_report(self):
        """生成可在客户端浏览器查看的 HTML 报告。"""
        report_path = os.path.join(self.session_dir, "report.html")
        n_pass = sum(1 for r in self.results if r["result"] == "PASS")
        n_fail = sum(1 for r in self.results if r["result"] == "FAIL")

        # Build HTML
        rows_html = ""
        for r in self.results:
            cls = "pass" if r["result"] == "PASS" else "fail"
            rows_html += (
                f'<tr class="{cls}">'
                f"<td>{r['step']}</td><td>{r['item']}</td>"
                f"<td>{r['value']} {r['unit']}</td>"
                f"<td>{r['spec']}</td>"
                f'<td class="{cls}">{r["result"]}</td>'
                f"</tr>\n"
            )

        images_html = ""
        for step, name, path in self.screenshots:
            rel_path = os.path.basename(path)
            images_html += (
                f'<div class="img-card">'
                f'<h3>[{step:02d}] {name}</h3>'
                f'<img src="{rel_path}" alt="{name}" loading="lazy">'
                f"</div>\n"
            )

        # 生成 JSON 数据（方便程序读取）
        import json
        json_path = os.path.join(self.session_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump({"results": self.results, "screenshots": [
                {"step": s, "name": n, "path": os.path.basename(p)}
                for s, n, p in self.screenshots
            ]}, f, indent=2)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RTO 6 测试报告 — {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f0f; color: #ddd; padding: 20px; }}
.header {{ text-align: center; padding: 30px; background: #1a1a1a; border-radius: 12px; margin-bottom: 24px; }}
.header h1 {{ color: #fff; font-size: 24px; }}
.header p {{ color: #888; margin-top: 6px; }}
.summary {{ display: flex; gap: 16px; justify-content: center; margin: 20px 0; }}
.badge {{ padding: 10px 24px; border-radius: 8px; font-size: 16px; font-weight: 700; }}
.badge.pass {{ background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }}
.badge.fail {{ background: #3a1a1a; color: #f44336; border: 1px solid #f44336; }}
h2 {{ color: #fff; margin: 30px 0 16px 0; padding-bottom: 8px; border-bottom: 1px solid #333; }}
table {{ width: 100%; border-collapse: collapse; background: #1a1a1a; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid #2a2a2a; }}
th {{ background: #252525; color: #aaa; font-weight: 600; font-size: 13px; text-transform: uppercase; }}
tr:hover {{ background: #252525; }}
.pass {{ color: #4caf50; }}
.fail {{ color: #f44336; font-weight: 700; }}
.img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; }}
.img-card {{ background: #1a1a1a; border-radius: 10px; overflow: hidden; }}
.img-card h3 {{ padding: 12px 16px; font-size: 14px; color: #aaa; background: #252525; }}
.img-card img {{ width: 100%; display: block; }}
.footer {{ text-align: center; color: #555; padding: 40px 0 10px; font-size: 12px; }}
</style>
</head>
<body>
<div class="header">
  <h1>📊 RTO 6 CAN 硬件测试报告</h1>
  <p>仪器: Rohde & Schwarz RTO6 &nbsp;|&nbsp; 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <div class="summary">
    <span class="badge pass">✓ PASS: {n_pass}</span>
    <span class="badge fail">✗ FAIL: {n_fail}</span>
    <span class="badge" style="background:#1a1a2a;color:#64b5f6;border:1px solid #64b5f6;">总计: {len(self.results)}</span>
  </div>
</div>

<h2>📋 测量结果</h2>
<table>
<tr><th>#</th><th>测试项</th><th>测量值</th><th>规格</th><th>判定</th></tr>
{rows_html}
</table>

<h2>📸 波形截图</h2>
<div class="img-grid">
{images_html}
</div>

<div class="footer">RTO 6 Auto Test &mdash; Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body>
</html>"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        # 同时存 CSV
        csv_path = os.path.join(self.session_dir, "results.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "item", "value", "unit", "spec", "result"])
            w.writeheader()
            w.writerows(self.results)

        # 自动打开浏览器查看报告
        import webbrowser
        abs_path = os.path.abspath(report_path)
        webbrowser.open(f"file:///{abs_path.replace(chr(92), '/')}")

        print(f"\n{'='*50}")
        print(f"报告已生成: {abs_path}")
        print(f"  HTML:  {report_path}")
        print(f"  JSON:  {json_path}")
        print(f"  CSV:   {csv_path}")
        print(f"  PASS: {n_pass}  FAIL: {n_fail}")
        print(f"{'='*50}")

    # ── 运行全部测试 ─────────────────────────────────────────
    def run_all(self, chip="U150"):
        print(f"\n{'#'*55}")
        print(f"#  RTO 6 自动测试 — CAN 硬件 ({chip})")
        print(f"#  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*55}")

        self.init_session()
        step = 1  # 全局步骤编号

        step = self.test_power_ripple(chip, step_start=step)
        step = self.test_tx_rx(chip, step_start=step)
        step = self.test_can_bus_levels(chip, step_start=step)
        step = self.test_signal_edge(chip, step_start=step)
        step = self.test_spi(chip, step_start=step)

        self.generate_report()
        return self.session_dir


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTO 6 自动测试 + 截屏")
    parser.add_argument("--ip", default="192.168.2.100")
    parser.add_argument("--chip", default="U150",
                        choices=["U150", "U36", "U220"])
    parser.add_argument("--test", default="all",
                        choices=["all", "ripple", "tx_rx", "bus_levels",
                                 "edge", "spi"])
    args = parser.parse_args()

    tester = AutoTester(args.ip)
    try:
        tester.init_session()
        step = 1

        if args.test == "all":
            step = tester.test_power_ripple(args.chip, step_start=step)
            step = tester.test_tx_rx(args.chip, step_start=step)
            step = tester.test_can_bus_levels(args.chip, step_start=step)
            step = tester.test_signal_edge(args.chip, step_start=step)
            step = tester.test_spi(args.chip, step_start=step)
        elif args.test == "ripple":
            tester.test_power_ripple(args.chip, step_start=step)
        elif args.test == "tx_rx":
            tester.test_tx_rx(args.chip, step_start=step)
        elif args.test == "bus_levels":
            tester.test_can_bus_levels(args.chip, step_start=step)
        elif args.test == "edge":
            tester.test_signal_edge(args.chip, step_start=step)
        elif args.test == "spi":
            tester.test_spi(args.chip, step_start=step)

        tester.generate_report()
    finally:
        tester.close()
