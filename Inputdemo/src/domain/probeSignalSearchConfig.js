export const PROBE_SIGNAL_SEARCH_CONFIG = Object.freeze({
  x: 307.660927,
  y: -12.981141,
  r: 169.927811,
  startZ: -128.251923,
  minimumZ: -132.251923,
  stepMm: 0.1,
  thresholdV: 3.0,
  speedL: 5,
  accL: 1,
  cp: 0,
  retractSpeedL: 10,
  settleMs: 50,
  confirmationSamples: 3,
  poseToleranceMm: 0.15
});

export function probeSignalSearchStartPose(config = PROBE_SIGNAL_SEARCH_CONFIG) {
  return {
    x: config.x,
    y: config.y,
    z: config.startZ,
    r: config.r
  };
}
