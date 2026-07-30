export class MockEquipmentController {
  async captureCurrentDisplayReport({ runId, caseId, targetPoints, robotPose }) {
    return {
      stepId: "rto6-current-display-capture",
      status: "COMPLETED",
      instrument: "RTO6",
      signal: "Current display: AMPL",
      measurementSlot: 1,
      measurement: "AMPL",
      source: "C1W1",
      value: 3.3,
      unit: "V",
      pass: null,
      capturedAt: "2026-07-27T00:00:00+0800",
      screenshotPath: `mock://${caseId}/${runId}/scope.jpg`,
      workbookPath: `mock://${caseId}/${runId}/measurement.xlsx`,
      workbookPreviewPath: `mock://${caseId}/${runId}/measurement.preview.png`,
      targetPoints,
      robotPose,
      durationMs: 280
    };
  }

  async readMeanVoltage() {
    return { value: 3.3, unit: "V", source: "CHANNEL2", durationMs: 1 };
  }

  async measure(step) {
    return {
      stepId: step.id,
      status: "COMPLETED",
      instrument: step.instrument,
      signal: step.signal,
      value: 4.98,
      unit: "V",
      rippleMvpp: 18.7,
      pass: true,
      durationMs: 280
    };
  }
}
