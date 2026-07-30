import { Dsox1204gEquipmentController } from "./dsox1204gEquipmentController.js";
import { Rto6EquipmentController } from "./rto6EquipmentController.js";

export function createEquipmentController({ env = process.env } = {}) {
  const driver = String(env.OSCILLOSCOPE_DRIVER || "rto6").trim().toLowerCase();
  if (["dsox1204g", "dsox", "keysight"].includes(driver)) {
    return new Dsox1204gEquipmentController();
  }
  if (driver === "rto6") {
    return new Rto6EquipmentController();
  }
  throw new Error(
    `Unsupported OSCILLOSCOPE_DRIVER=${driver}; expected rto6 or dsox1204g`
  );
}
