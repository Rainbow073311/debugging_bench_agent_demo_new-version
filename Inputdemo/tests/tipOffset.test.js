import assert from "node:assert/strict";
import test from "node:test";
import { tcpXyFromTip, tipOffsetXy, loadTipOffsetParams } from "../src/domain/tipOffset.js";

test("locked tip_offset params", () => {
  const p = loadTipOffsetParams();
  assert.equal(p.radius_xy_mm, 27);
  assert.equal(p.delta_deg, 6.15);
});

test("tcp_xy_from_tip is self-consistent", () => {
  const tip = { x: 350.228, y: 0.413 };
  const solved = tcpXyFromTip(tip);
  assert.ok(solved.errMm < 1e-6);
  const off = tipOffsetXy(solved.j1_deg);
  assert.ok(Math.hypot(solved.tcpXy.x + off.dx - tip.x, solved.tcpXy.y + off.dy - tip.y) < 1e-6);
});
