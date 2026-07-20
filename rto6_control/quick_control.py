"""
RTO 6 Quick Control — 交互式 / 命令行控制脚本
==============================================
Usage:
    python quick_control.py screenshot          # 截图
    python quick_control.py info                 # 查看示波器状态
    python quick_control.py autoscale            # 自动设置
    python quick_control.py measure ch=1 func=VPP  # 单次测量
    python quick_control.py setup ripple          # 预设电源纹波测试
    python quick_control.py setup tx_rx           # 预设 TX/RX 信号测试
    python quick_control.py setup bus_levels      # 预设 CAN 总线电平测试
    python quick_control.py setup edge            # 预设边沿时间测试
    python quick_control.py setup spi             # 预设 SPI 测试
    python quick_control.py web                    # 打开 Web Control 界面
    python quick_control.py shell                 # 进入交互式命令行
"""

import argparse
import sys
from rto6_scope import RTO6


def cmd_screenshot(scope):
    path = "screenshot.png"
    scope.screenshot(path)
    print(f"截图已保存: {path}")


def cmd_info(scope):
    print("\n=== 时基 ===")
    tb = scope.timebase_info()
    print(f"  水平刻度: {tb['scale']*1e6:.2f} us/div")
    print(f"  水平位置: {tb['position']*1e6:.2f} us")
    print(f"  采样率:   {tb['sample_rate']/1e9:.2f} GS/s")

    for ch in range(1, 5):
        try:
            info = scope.channel_info(ch)
            if info["state"] == "1":
                print(f"\n=== CH{ch} ===")
                print(f"  状态:     ON")
                print(f"  垂直刻度: {info['scale']:.3f} V/div")
                print(f"  偏移:     {info['offset']:.3f} V")
                print(f"  耦合:     {info['coupling']}")
        except Exception:
            break

    print(f"\n=== 触发 ===")
    trig = scope.trigger_info()
    print(f"  源:       {trig['source']}")
    print(f"  电平:     {trig['level']:.2f} V")
    print(f"  斜率:     {trig['slope']}")
    print(f"  模式:     {trig['mode']}")


def cmd_measure(scope, ch, func):
    value = scope.measure(ch, func)
    print(f"CH{ch} {func} = {value:.6f}")
    return value


def cmd_autoscale(scope):
    print("正在自动设置...")
    scope.autoscale()
    print("完成。")


def cmd_shell(scope):
    """交互式命令行"""
    print("\n" + "="*50)
    print("RTO 6 交互式控制台")
    print("输入 help 查看可用命令，quit 退出")
    print("="*50)

    while True:
        try:
            cmd = input("\nRTO6> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        if action == "quit" or action == "exit":
            break
        elif action == "help":
            print("""
可用命令:
  screenshot [文件名]      — 截图
  info / status            — 查看状态
  web                      — 打开 Web Control 界面
  run / stop / single      — 运行控制
  autoscale                — 自动设置
  measure <ch> <func>      — 测量 (如: measure 1 VPP)
  ch_on <n> / ch_off <n>   — 开关通道
  scale <ch> <v/div>       — 设置垂直刻度
  timebase <s/div>         — 设置水平刻度
  trig_src <src>           — 触发源
  trig_level <v>           — 触发电平
  trig_mode <mode>         — 触发模式 (AUTO/NORM)
  coupling <ch> <DC/AC>    — 耦合方式
  average <n>              — 平均次数 (0=关闭)
  setup <preset>           — 预设配置
  quit / exit              — 退出
            """)
        elif action == "web":
            scope.open_web()
        elif action == "screenshot":
            path = parts[1] if len(parts) > 1 else "screenshot.png"
            scope.screenshot(path)
        elif action in ("info", "status"):
            cmd_info(scope)
        elif action == "run":
            scope.run()
            print("运行中...")
        elif action == "stop":
            scope.stop()
            print("已停止。")
        elif action == "single":
            scope.single()
            print("单次触发。")
        elif action == "autoscale":
            cmd_autoscale(scope)
        elif action == "measure":
            ch = int(parts[1]) if len(parts) > 1 else 1
            func = parts[2] if len(parts) > 2 else "VPP"
            cmd_measure(scope, ch, func)
        elif action == "ch_on":
            scope.channel_on(int(parts[1]))
        elif action == "ch_off":
            scope.channel_off(int(parts[1]))
        elif action == "scale":
            scope.channel_scale(int(parts[1]), float(parts[2]))
        elif action == "timebase":
            scope.timebase_scale(float(parts[1]))
        elif action == "trig_src":
            scope.trigger_source(parts[1])
        elif action == "trig_level":
            scope.trigger_level(float(parts[1]))
        elif action == "trig_mode":
            scope.trigger_mode(parts[1])
        elif action == "coupling":
            scope.channel_coupling(int(parts[1]), parts[2])
        elif action == "average":
            scope.acquire_average(int(parts[1]) if int(parts[1]) > 0 else None)
        elif action == "setup":
            preset = parts[1] if len(parts) > 1 else ""
            if preset == "ripple":
                scope.setup_can_power_ripple(1, "VCC")
            elif preset == "tx_rx":
                scope.setup_can_tx_rx(1, 2)
            elif preset == "bus_levels":
                scope.setup_can_bus_levels(1, 2)
            elif preset == "edge":
                scope.setup_can_edge_time(1)
            elif preset == "spi":
                scope.setup_can_spi(1, 2, 3)
            else:
                print(f"未知预设: {preset}")
                print("可选: ripple, tx_rx, bus_levels, edge, spi")
        else:
            print(f"未知命令: {action}")


def main():
    parser = argparse.ArgumentParser(description="RTO 6 Quick Control")
    parser.add_argument("command", nargs="?", default="shell",
                        choices=["screenshot", "info", "autoscale", "measure",
                                 "setup", "web", "shell"])
    parser.add_argument("--ip", default="192.168.2.100")
    parser.add_argument("--ch", type=int, default=1)
    parser.add_argument("--func", default="VPP")
    parser.add_argument("--preset", default="ripple",
                        choices=["ripple", "tx_rx", "bus_levels", "edge", "spi"])
    args = parser.parse_args()

    scope = RTO6(args.ip)

    try:
        idn = scope.connect()
        print(f"已连接: {idn}")

        if args.command == "screenshot":
            cmd_screenshot(scope)
        elif args.command == "info":
            cmd_info(scope)
        elif args.command == "autoscale":
            cmd_autoscale(scope)
        elif args.command == "web":
            scope.open_web()
        elif args.command == "measure":
            cmd_measure(scope, args.ch, args.func)
        elif args.command == "setup":
            setup_map = {
                "ripple": lambda: scope.setup_can_power_ripple(1, "VCC"),
                "tx_rx": lambda: scope.setup_can_tx_rx(1, 2),
                "bus_levels": lambda: scope.setup_can_bus_levels(1, 2),
                "edge": lambda: scope.setup_can_edge_time(1),
                "spi": lambda: scope.setup_can_spi(1, 2, 3),
            }
            setup_map[args.preset]()
        elif args.command == "shell":
            cmd_shell(scope)
    finally:
        scope.disconnect()


if __name__ == "__main__":
    main()
