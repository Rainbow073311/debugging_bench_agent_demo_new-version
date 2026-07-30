import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


BRIDGE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mg400_bridge.py"
SPEC = importlib.util.spec_from_file_location("mg400_bridge_probe_test", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bridge)


class FakeRobot:
    mode_code = 5
    instances = []

    def __init__(self, config):
        self.commands = []
        self.dashboard_commands = []
        self.closed = False
        self.__class__.instances.append(self)

    def connect(self):
        return None

    def robot_mode(self):
        labels = {5: "ENABLED_IDLE", 7: "RUNNING"}
        return {
            "code": self.mode_code,
            "label": labels.get(self.mode_code, "UNKNOWN"),
            "raw": f"0,{{{self.mode_code}}}",
        }

    def move(self, command):
        self.commands.append(command)
        return {"ok": True, "errorId": 0, "response": "0,{}"}

    def dash(self, command):
        self.dashboard_commands.append(command)
        return {"ok": True, "errorId": 0, "response": "0,{}"}

    def get_pose(self):
        return {
            "pose": {
                "x": 304.666744,
                "y": -26.621413,
                "z": -119.69742,
                "r": 169.927811,
            },
            "raw": "0,{304.666744,-26.621413,-119.69742,169.927811}",
        }

    def close(self):
        self.closed = True


class Mg400ProbeStepTests(unittest.TestCase):
    def setUp(self):
        FakeRobot.instances.clear()
        FakeRobot.mode_code = 5

    def payload(self):
        return {
            "config": {"mode": "mg400"},
            "command": {
                "name": "probeStep",
                "pose": {
                    "x": 304.666744,
                    "y": -26.621413,
                    "z": -119.69742,
                    "r": 169.927811,
                },
                "speedL": 1,
                "accL": 1,
                "cp": 0,
            },
        }

    def test_probe_step_is_one_movl_followed_by_sync_without_recovery_commands(self):
        with patch.object(bridge, "Mg400", FakeRobot):
            result = bridge.action_command(self.payload())

        robot = FakeRobot.instances[-1]
        self.assertEqual(
            robot.commands,
            [
                "MovL(304.666744,-26.621413,-119.69742,169.927811,"
                "SpeedL=1,AccL=1,CP=0)",
                "Sync()",
            ],
        )
        self.assertEqual(robot.dashboard_commands, [])
        self.assertEqual(result["robot"]["mode"]["code"], 5)
        self.assertTrue(robot.closed)

    def test_probe_step_refuses_motion_unless_robot_is_enabled_idle(self):
        FakeRobot.mode_code = 7
        with patch.object(bridge, "Mg400", FakeRobot):
            with self.assertRaisesRegex(bridge.Mg400Error, "requires ENABLED_IDLE"):
                bridge.action_command(self.payload())

        robot = FakeRobot.instances[-1]
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.dashboard_commands, [])
        self.assertTrue(robot.closed)


if __name__ == "__main__":
    unittest.main()
