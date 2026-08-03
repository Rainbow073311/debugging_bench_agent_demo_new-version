import json
import math
import socket
import sys
import time


DEFAULT_LIMITS = {
    "x": (-450.0, 450.0),
    "y": (-450.0, 450.0),
    "z": (-250.0, 250.0),
    "r": (-360.0, 360.0),
}

WORKSPACE_BOUNDARY = [
    (-250.0, 200.0, 360.0),
    (-100.0, 200.0, 380.0),
    (0.0, 205.0, 400.0),
    (10.391, 229.49, 328.019),
    (18.604, 228.507, 351.188),
    (26.816, 227.117, 365.915),
    (78.2, 209.433, 423.98),
    (100.0, 197.584, 438.7),
    (150.0, 133.358, 455.014),
    (317.2, 198.192, 423.032),
    (343.2, 197.277, 409.038),
    (396.371, 181.223, 348.0),
    (404.583, 188.86, 332.291),
    (412.796, 204.588, 315.923),
]
WORKSPACE_MARGIN_MM = 5.0
SEGMENT_SAMPLE_COUNT = 24

MODE_LABELS = {
    1: "INIT",
    2: "BRAKE_OPEN",
    3: "POWER_OFF",
    4: "DISABLED",
    5: "ENABLED_IDLE",
    6: "DRAG",
    7: "RUNNING",
    8: "RECORDING",
    9: "ERROR",
    10: "PAUSED",
    11: "JOGGING",
}


class Mg400Error(RuntimeError):
    pass


def friendly_socket_error(error, ip, port):
    if isinstance(error, socket.timeout):
        return f"连接 MG400 超时：{ip}:{port}。请检查本机以太网 IP、网线、机械臂 IP 和 TCP/IP 二次开发是否开启。"
    if isinstance(error, ConnectionRefusedError):
        return f"MG400 拒绝连接：{ip}:{port}。请检查端口是否开启，或 DobotStudio 是否占用连接。"
    if isinstance(error, OSError):
        return f"无法连接 MG400：{ip}:{port}，系统错误 {error}。"
    return str(error)


class Mg400:
    def __init__(self, config):
        self.config = config
        self.ip = config["ip"]
        self.dashboard_port = int(config.get("dashboardPort", 29999))
        self.motion_port = int(config.get("motionPort", 30003))
        self.timeout = float(config.get("timeoutMs", 5000)) / 1000.0
        self.dashboard = None
        self.motion = None

    def connect_dashboard(self):
        try:
            self.dashboard = socket.create_connection(
                (self.ip, self.dashboard_port), timeout=self.timeout
            )
        except OSError as error:
            raise Mg400Error(friendly_socket_error(error, self.ip, self.dashboard_port)) from error

    def connect_motion(self):
        try:
            self.motion = socket.create_connection(
                (self.ip, self.motion_port), timeout=max(self.timeout, 30.0)
            )
        except OSError as error:
            raise Mg400Error(friendly_socket_error(error, self.ip, self.motion_port)) from error

    def connect(self):
        self.connect_dashboard()
        self.connect_motion()

    def close(self):
        for sock in (self.dashboard, self.motion):
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass

    def _send(self, sock, command):
        sock.sendall((command + "\r\n").encode("utf-8"))
        response = sock.recv(2048).decode("utf-8", errors="replace").strip()
        try:
            error_id = int(response.split(",", 1)[0])
        except ValueError:
            error_id = -999
        return {"command": command, "response": response, "errorId": error_id, "ok": error_id == 0}

    def dash(self, command):
        if not self.dashboard:
            self.connect_dashboard()
        return self._send(self.dashboard, command)

    def move(self, command):
        if not self.motion:
            self.connect_motion()
        try:
            return self._send(self.motion, command)
        except socket.timeout:
            if command.strip().startswith("Sync"):
                return {
                    "command": command,
                    "response": "timeout while waiting for motion completion",
                    "errorId": -1,
                    "ok": False,
                }
            return {"command": command, "response": "timeout; command may still be accepted", "errorId": 0, "ok": True}

    def robot_mode(self):
        result = self.dash("RobotMode()")
        code = None
        label = None
        if result["ok"]:
            try:
                code = int(result["response"].split(",{", 1)[1].split("}", 1)[0])
                label = MODE_LABELS.get(code, f"UNKNOWN_{code}")
            except (IndexError, ValueError):
                label = "PARSE_FAILED"
        return {"code": code, "label": label, "raw": result}

    def get_pose(self):
        result = self.dash("GetPose()")
        pose = None
        if result["ok"]:
            try:
                values = result["response"].split(",{", 1)[1].split("}", 1)[0].split(",")
                pose = {
                    "x": float(values[0]),
                    "y": float(values[1]),
                    "z": float(values[2]),
                    "r": float(values[3]),
                }
            except (IndexError, ValueError):
                pose = None
        return {"pose": pose, "raw": result}


def read_payload():
    raw = sys.stdin.read().strip()
    return json.loads(raw) if raw else {}


def emit(payload, code=0):
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    sys.exit(code)


def require_ok(result):
    if not result["ok"]:
        raise Mg400Error(f"{result['command']} failed: {result['response']}")
    return result


def validate_pose(pose):
    clean = {}
    for key, (low, high) in DEFAULT_LIMITS.items():
        value = float(pose[key])
        if not low <= value <= high:
            raise Mg400Error(f"{key.upper()}={value} is outside safe range [{low}, {high}]")
        clean[key] = value
    return clean


def resolve_partial_pose(robot, pose):
    current = robot.get_pose()["pose"]
    if not current:
        raise Mg400Error("Current robot pose is unavailable; fill X/Y/Z/R explicitly.")
    merged = {}
    for key in ("x", "y", "z", "r"):
        value = pose.get(key)
        merged[key] = current[key] if value is None or value == "" else value
    return validate_pose(merged)


def format_number(value):
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def speed_value(config):
    return int(float(config.get("speed", 30)))


def bounded_speed(value, fallback, label):
    speed = int(round(float(fallback if value is None else value)))
    if speed < 1 or speed > 100:
        raise Mg400Error(f"{label} must be between 1 and 100")
    return speed


def interpolate_workspace(z):
    if z <= WORKSPACE_BOUNDARY[0][0]:
        return WORKSPACE_BOUNDARY[0][1], WORKSPACE_BOUNDARY[0][2]
    if z >= WORKSPACE_BOUNDARY[-1][0]:
        return WORKSPACE_BOUNDARY[-1][1], WORKSPACE_BOUNDARY[-1][2]
    for index in range(1, len(WORKSPACE_BOUNDARY)):
        prev = WORKSPACE_BOUNDARY[index - 1]
        nxt = WORKSPACE_BOUNDARY[index]
        if z <= nxt[0]:
            ratio = (z - prev[0]) / (nxt[0] - prev[0])
            return (
                prev[1] + ((nxt[1] - prev[1]) * ratio),
                prev[2] + ((nxt[2] - prev[2]) * ratio),
            )
    return WORKSPACE_BOUNDARY[-1][1], WORKSPACE_BOUNDARY[-1][2]


def validate_workspace_pose(pose, label):
    clean = validate_pose(pose)
    radius = (clean["x"] ** 2 + clean["y"] ** 2) ** 0.5
    theta = math.degrees(math.atan2(clean["y"], clean["x"]))
    z_low = WORKSPACE_BOUNDARY[0][0] + WORKSPACE_MARGIN_MM
    z_high = WORKSPACE_BOUNDARY[-1][0] - WORKSPACE_MARGIN_MM
    if clean["z"] < z_low or clean["z"] > z_high:
        raise Mg400Error(
            f"{label} Z={clean['z']} is outside sampled workspace [{z_low}, {z_high}]"
        )
    radial_low, radial_high = interpolate_workspace(clean["z"])
    radial_low += WORKSPACE_MARGIN_MM
    radial_high -= WORKSPACE_MARGIN_MM
    if abs(theta) > 160:
        raise Mg400Error(f"{label} base angle {theta:.1f} exceeds +/-160 degrees")
    if radius < radial_low or radius > radial_high:
        raise Mg400Error(
            f"{label} radius {radius:.1f} is outside [{radial_low:.1f}, {radial_high:.1f}] "
            f"at Z={clean['z']:.1f}"
        )
    return clean


def validate_linear_segment(start, end, label):
    for index in range(SEGMENT_SAMPLE_COUNT + 1):
        ratio = index / SEGMENT_SAMPLE_COUNT
        sample = {
            key: start[key] + ((end[key] - start[key]) * ratio)
            for key in ("x", "y", "z", "r")
        }
        validate_workspace_pose(sample, f"{label} sample {index}/{SEGMENT_SAMPLE_COUNT}")


def motion_command(command_name, pose, speed):
    speed_key = "SpeedL" if command_name == "MovL" else "SpeedJ"
    return (
        f"{command_name}({format_number(pose['x'])},{format_number(pose['y'])},"
        f"{format_number(pose['z'])},{format_number(pose['r'])},"
        f"{speed_key}={speed})"
    )


def build_safe_trajectory(current_pose, target_pose, options):
    start = validate_pose(current_pose)
    target = validate_pose(target_pose)
    safe_travel_z = float(options.get("safeTravelZ", 100))
    travel_speed = bounded_speed(options.get("travelSpeed"), 30, "Travel speed")
    descent_speed = bounded_speed(options.get("descentSpeed"), 10, "Descent speed")
    travel_z = max(start["z"], target["z"], safe_travel_z)
    staging_radius = options.get("stagingRadius")
    if staging_radius is None:
        lift = {**start, "z": travel_z}
        traverse = {
            "x": target["x"],
            "y": target["y"],
            "z": travel_z,
            "r": target["r"],
        }
        candidates = [
            ("lift", lift, travel_speed),
            ("traverse", traverse, travel_speed),
            ("descend", target, descent_speed),
        ]
    else:
        staging_radius = float(staging_radius)
        if not math.isfinite(staging_radius) or staging_radius <= 0:
            raise Mg400Error("stagingRadius must be a positive finite number")

        def radial_pose(source, z, r):
            theta = math.atan2(source["y"], source["x"])
            return {
                "x": staging_radius * math.cos(theta),
                "y": staging_radius * math.sin(theta),
                "z": z,
                "r": r,
            }

        staging_start = radial_pose(start, start["z"], start["r"])
        staging_start_high = {**staging_start, "z": travel_z}
        staging_target_high = radial_pose(target, travel_z, target["r"])
        staging_target = {**staging_target_high, "z": target["z"]}
        candidates = [
            ("retract_to_staging", staging_start, travel_speed),
            ("lift_staging", staging_start_high, travel_speed),
            ("traverse_staging", staging_target_high, travel_speed),
            ("descend_staging", staging_target, descent_speed),
            ("extend_from_staging", target, descent_speed),
        ]
    stages = []
    segment_start = start
    for name, stage_pose, stage_speed in candidates:
        if all(abs(segment_start[key] - stage_pose[key]) <= 0.001 for key in ("x", "y", "z", "r")):
            continue
        validate_linear_segment(segment_start, stage_pose, name)
        stages.append({
            "name": name,
            "targetPose": stage_pose,
            "speed": stage_speed,
            "command": motion_command("MovL", stage_pose, stage_speed),
        })
        segment_start = stage_pose
    return {
        "mode": "safe-lift-traverse-descend",
        "safeTravelZ": safe_travel_z,
        "travelZ": travel_z,
        "travelSpeed": travel_speed,
        "descentSpeed": descent_speed,
        "stagingRadius": staging_radius,
        "startPose": start,
        "targetPose": target,
        "stages": stages,
    }


def status(robot):
    mode = robot.robot_mode()
    pose = robot.get_pose()
    return {
        "mode": mode,
        "pose": pose["pose"],
        "poseRaw": pose["raw"],
    }


def prepare_for_motion(robot, config):
    responses = []
    # 清错
    responses.append(require_ok(robot.dash("ClearError()")))
    # 尝试继续, 失败则复位后重试 (碰撞限位后需要 ResetRobot)
    cont_result = robot.dash("Continue()")
    if not cont_result["ok"]:
        responses.append(require_ok(robot.dash("ResetRobot()")))
        time.sleep(0.5)
        responses.append(require_ok(robot.dash("ClearError()")))
        responses.append(require_ok(robot.dash("Continue()")))
    else:
        responses.append(cont_result)
    mode = robot.robot_mode()
    if mode["code"] == 4 and config.get("autoEnable", True):
        responses.append(enable_robot(robot, config))
        time.sleep(1)
        mode = robot.robot_mode()
    if mode["code"] not in (5, 7, 11):
        raise Mg400Error(f"Robot is not ready for motion: {mode['label']} ({mode['code']})")
    responses.append(require_ok(robot.dash(f"SpeedFactor({speed_value(config)})")))
    return responses


def enable_robot(robot, config):
    attempts = ["EnableRobot()"]
    load = config.get("load")
    if load is not None:
        attempts.append(f"EnableRobot({float(load)})")
        attempts.append(f"EnableRobot({float(load)},0,0,0)")

    results = []
    for command in attempts:
        result = robot.dash(command)
        results.append(result)
        if result["ok"]:
            return {"command": command, "response": result["response"], "errorId": result["errorId"], "ok": True, "attempts": results}

    last = results[-1]
    raise Mg400Error(f"EnableRobot failed after {len(results)} attempts: {last['response']}")


def action_test(payload):
    config = payload["config"]
    robot = Mg400(config)
    started = time.time()
    try:
        robot.connect()
        data = status(robot)
        return {
            "ok": True,
            "action": "test",
            "robot": data,
            "elapsedMs": round((time.time() - started) * 1000),
        }
    finally:
        robot.close()


def action_status(payload):
    config = payload["config"]
    robot = Mg400(config)
    try:
        robot.connect_dashboard()
        return {"ok": True, "action": "status", "robot": status(robot)}
    finally:
        robot.close()


def action_execute(payload):
    config = payload["config"]
    pose = validate_pose(payload["pose"])
    trajectory_options = payload.get("trajectory")
    command_name = config.get("motionCommand", "MovJ")
    command = motion_command(command_name, pose, speed_value(config))

    robot = Mg400(config)
    responses = []
    trajectory = None
    try:
        robot.connect()
        responses.extend(prepare_for_motion(robot, config))
        if (
            isinstance(trajectory_options, dict)
            and trajectory_options.get("mode") == "safe-lift-traverse-descend"
        ):
            trajectory_mode = robot.robot_mode()
            if trajectory_mode["code"] != 5:
                raise Mg400Error(
                    f"Safe trajectory requires ENABLED_IDLE (5), got "
                    f"{trajectory_mode['label']} ({trajectory_mode['code']})"
                )
            current_pose = robot.get_pose()["pose"]
            if not current_pose:
                raise Mg400Error("Current robot pose is unavailable for safe trajectory planning")
            trajectory = build_safe_trajectory(current_pose, pose, trajectory_options)
            for stage in trajectory["stages"]:
                responses.append(require_ok(robot.move(stage["command"])))
                responses.append(require_ok(robot.move("Sync()")))
            command = trajectory["stages"][-1]["command"] if trajectory["stages"] else None
        else:
            responses.append(require_ok(robot.move(command)))
            responses.append(require_ok(robot.move("Sync()")))
        if config.get("returnHome"):
            home = validate_pose(config["homePose"])
            home_command = (
                f"MovJ({format_number(home['x'])},{format_number(home['y'])},"
                f"{format_number(home['z'])},{format_number(home['r'])},SpeedJ={speed_value(config)})"
            )
            responses.append(require_ok(robot.move(home_command)))
            responses.append(require_ok(robot.move("Sync()")))
        return {
            "ok": True,
            "action": "execute",
            "command": command,
            "responses": responses,
            "trajectory": trajectory,
            "robot": status(robot),
        }
    finally:
        robot.close()


def action_command(payload):
    config = payload["config"]
    command = payload.get("command", {})
    name = command.get("name")
    robot = Mg400(config)
    try:
        if name == "pose":
            robot.connect_dashboard()
            return {"ok": True, "action": name, "robot": status(robot)}
        elif name == "clearError":
            robot.connect_dashboard()
            result = require_ok(robot.dash("ClearError()"))
        elif name == "enable":
            robot.connect_dashboard()
            result = enable_robot(robot, config)
        elif name == "disable":
            robot.connect_dashboard()
            result = require_ok(robot.dash("DisableRobot()"))
        elif name == "pause":
            robot.connect_dashboard()
            result = require_ok(robot.dash("Stop()"))  # MG400 uses Stop(), not Pause()
        elif name == "continue":
            robot.connect_dashboard()
            result = require_ok(robot.dash("Continue()"))
        elif name == "reset":
            robot.connect_dashboard()
            result = require_ok(robot.dash("ResetRobot()"))
        elif name == "jog":
            # 先通过 Dashboard 恢复 (清错+复位), 再通过 Motion 发 Jog
            robot.connect_dashboard()
            robot.dash("ClearError()")
            robot.dash("ResetRobot()")
            time.sleep(0.3)
            robot.dash("Continue()")
            mode = robot.robot_mode()
            if mode["code"] == 4:
                enable_robot(robot, config)
                time.sleep(0.5)
            robot.dash(f"SpeedFactor({speed_value(config)})")
            # 停止残留 jog
            robot.connect_motion()
            try:
                robot.move("MoveJog()")
            except Exception:
                pass
            time.sleep(0.05)
            # 发送新方向
            axis = str(command.get("axis", "")).upper()
            if axis not in {"X+", "X-", "Y+", "Y-", "Z+", "Z-", "R+", "R-", "J1+", "J1-", "J2+", "J2-", "J3+", "J3-", "J4+", "J4-"}:
                raise Mg400Error(f"Unsupported jog axis: {axis}")
            result = require_ok(robot.move(f"MoveJog({axis})"))
            return {"ok": True, "action": name, "result": result}
        elif name == "jogStop":
            robot.connect_motion()
            result = require_ok(robot.move("MoveJog()"))
        elif name == "probeStep":
            robot.connect()
            mode = robot.robot_mode()
            if mode["code"] != 5:
                raise Mg400Error(
                    f"Probe step requires ENABLED_IDLE (5), got "
                    f"{mode['label']} ({mode['code']})"
                )
            pose = validate_pose(command.get("pose", {}))
            speed_l = float(command.get("speedL", 1))
            acc_l = float(command.get("accL", 1))
            cp = float(command.get("cp", 0))
            if not 1 <= speed_l <= 100:
                raise Mg400Error("probeStep SpeedL must be between 1 and 100")
            if not 1 <= acc_l <= 100:
                raise Mg400Error("probeStep AccL must be between 1 and 100")
            if not 0 <= cp <= 100:
                raise Mg400Error("probeStep CP must be between 0 and 100")
            move_command = (
                f"MovL({format_number(pose['x'])},{format_number(pose['y'])},"
                f"{format_number(pose['z'])},{format_number(pose['r'])},"
                f"SpeedL={format_number(speed_l)},AccL={format_number(acc_l)},"
                f"CP={format_number(cp)})"
            )
            move_result = require_ok(robot.move(move_command))
            sync_result = require_ok(robot.move("Sync()"))
            result = {
                "command": move_command,
                "resolvedPose": pose,
                "responses": [move_result, sync_result],
            }
        elif name == "move":
            robot.connect()
            prepare = prepare_for_motion(robot, config)
            pose = resolve_partial_pose(robot, command.get("pose", {}))
            move_name = command.get("motionCommand") or config.get("motionCommand", "MovJ")
            speed_key = "SpeedL" if move_name == "MovL" else "SpeedJ"
            result = require_ok(robot.move(
                f"{move_name}({format_number(pose['x'])},{format_number(pose['y'])},"
                f"{format_number(pose['z'])},{format_number(pose['r'])},{speed_key}={speed_value(config)})"
            ))
            result = {"motionPrep": prepare, "resolvedPose": pose, **result}
        else:
            raise Mg400Error(f"Unsupported command: {name}")
        return {"ok": True, "action": name, "result": result, "robot": status(robot)}
    finally:
        robot.close()


HANDLERS = {
    "test": action_test,
    "status": action_status,
    "execute": action_execute,
    "command": action_command,
}


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action not in HANDLERS:
        emit({"ok": False, "error": f"Unknown action: {action}"}, 2)
    try:
        emit(HANDLERS[action](read_payload()))
    except Exception as error:
        emit({"ok": False, "error": str(error)}, 1)


if __name__ == "__main__":
    main()
