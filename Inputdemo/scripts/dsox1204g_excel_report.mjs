import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const rawInput = await readStdin();
const input = JSON.parse(rawInput || "{}");
const outputPath = path.resolve(input.outputPath);
const screenshotPath = path.resolve(input.screenshotPath);
const screenshotBytes = await fs.readFile(screenshotPath);
const screenshotDataUrl = `data:image/png;base64,${screenshotBytes.toString("base64")}`;
const pose = input.robotPose || {};
const poseValue = (value) => value ?? "";

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("DSOX1204G Measurement");
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(2);

const measurementLabel = String(input.measurement || "Current").toUpperCase();
sheet.getRange("A1:H1").merge();
sheet.getRange("A1").values = [[`DSOX1204G ${measurementLabel} Voltage Capture`]];
sheet.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  rowHeight: 30
};

sheet.getRange("A3:F6").values = [
  ["Case ID", input.caseId || "", "Run ID", input.runId || "", "Captured At", input.capturedAt || ""],
  ["Instrument", input.instrument || "", "VISA Address", input.scpiAddress || "", "Target Points", (input.targetPoints || []).join(", ")],
  ["Measurement Slot", input.measurementSlot ?? "", "Measurement", input.measurement || "", "Source", input.source || ""],
  ["Fixed MG400 Pose", pose.x == null && pose.y == null ? "" : `X=${poseValue(pose.x)}, Y=${poseValue(pose.y)}`, "Z / R", pose.z == null && pose.r == null ? "" : `Z=${poseValue(pose.z)}, R=${poseValue(pose.r)}`, "Status", input.status || "COMPLETED"]
];
sheet.getRange("A3:F6").format = {
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
    bottom: { style: "thin", color: "#B4C6E7" }
  }
};
for (const range of ["A3:A6", "C3:C6", "E3:E6"]) {
  sheet.getRange(range).format = {
    fill: "#D9EAF7",
    font: { bold: true, color: "#17365D" }
  };
}

sheet.getRange("A8:B8").merge();
sheet.getRange("A8").values = [[`${measurementLabel} Result (${input.unit || ""})`]];
sheet.getRange("A9:B10").merge();
sheet.getRange("A9").values = [[Number(input.value)]];
sheet.getRange("A8:B10").format = {
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { preset: "outside", style: "medium", color: "#4472C4" }
};
sheet.getRange("A8:B8").format = {
  fill: "#4472C4",
  font: { bold: true, color: "#FFFFFF" }
};
sheet.getRange("A9:B10").format = {
  fill: "#EAF2F8",
  font: { bold: true, color: "#17365D", size: 20 },
  numberFormat: `0.000000 "${String(input.unit || "").replace(/"/g, "")}"`
};

sheet.getRange("A12:H12").merge();
sheet.getRange("A12").values = [["DSOX1204G Screen Capture"]];
sheet.getRange("A12:H12").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  horizontalAlignment: "left"
};
sheet.images.add({
  dataUrl: screenshotDataUrl,
  anchor: {
    from: { row: 12, col: 0 },
    extent: { widthPx: 960, heightPx: 540 }
  }
});

sheet.getRange("A3:H40").format.font = { name: "Aptos", size: 10 };
sheet.getRange("A1:H1").format.font = {
  name: "Aptos Display",
  bold: true,
  color: "#FFFFFF",
  size: 16
};
sheet.getRange("A9:B10").format.font = {
  name: "Aptos Display",
  bold: true,
  color: "#17365D",
  size: 20
};
sheet.getRange("A:H").format.columnWidth = 18;
sheet.getRange("B:B").format.columnWidth = 32;
sheet.getRange("D:D").format.columnWidth = 32;
sheet.getRange("F:F").format.columnWidth = 24;
sheet.getRange("A3:H6").format.rowHeight = 28;
sheet.getRange("A4:H4").format.rowHeight = 46;
sheet.getRange("A13:H35").format.rowHeight = 20;

// ── Probe Descent Samples sheet (Z-axis search data) ──
const probeResult = input.probeSearchResult;
if (probeResult && Array.isArray(probeResult.samples) && probeResult.samples.length > 0) {
  const probeSheet = workbook.worksheets.add("Probe Descent Samples");
  probeSheet.showGridLines = true;
  probeSheet.freezePanes.freezeRows(1);

  probeSheet.getRange("A1:E1").merge();
  probeSheet.getRange("A1").values = [["Probe Signal Search — Z-Axis Descent Log"]];
  probeSheet.getRange("A1:E1").format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF", size: 14 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    rowHeight: 28
  };

  probeSheet.getRange("A3:E3").values = [
    ["Status", probeResult.status || "N/A",
     "Threshold (V)", probeResult.thresholdV ?? "N/A",
     "Signal Detected", String(probeResult.signalDetected ?? false)]
  ];
  probeSheet.getRange("A4:E4").values = [
    ["Stop Reason", probeResult.stopReason || "N/A",
     "Signal Confirmed", String(probeResult.signalConfirmed ?? false),
     "Samples", probeResult.samples.length]
  ];
  probeSheet.getRange("A3:E4").format = {
    verticalAlignment: "center",
    wrapText: true,
    borders: { insideHorizontal: { style: "thin", color: "#D9E2F3" } }
  };
  probeSheet.getRange("A3:A4").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
  probeSheet.getRange("C3:C4").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
  probeSheet.getRange("E3:E4").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };

  probeSheet.getRange("A6:C6").values = [["Step", "Z (mm)", "Voltage (V)"]];
  probeSheet.getRange("A6:C6").format = {
    fill: "#4472C4",
    font: { bold: true, color: "#FFFFFF", size: 11 },
    horizontalAlignment: "center",
    borders: { bottom: { style: "medium", color: "#17365D" } }
  };

  const sampleRows = probeResult.samples.map((s, i) => [
    s.index ?? i,
    Number(s.z)?.toFixed(4) ?? "",
    Number(s.value)?.toFixed(6) ?? ""
  ]);
  const dataStart = 7;
  const dataEnd = dataStart + sampleRows.length - 1;
  probeSheet.getRange(`A${dataStart}:C${dataEnd}`).values = sampleRows;
  probeSheet.getRange(`A${dataStart}:C${dataEnd}`).format = {
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: {
      insideHorizontal: { style: "thin", color: "#D9E2F3" },
      insideVertical: { style: "thin", color: "#D9E2F3" }
    },
    numberFormat: "0.000000"
  };

  // Highlight the row where signal was first detected
  if (probeResult.signalDetected && probeResult.samples.length > 1) {
    for (let i = 1; i < probeResult.samples.length; i++) {
      if (probeResult.samples[i].value >= (probeResult.thresholdV || 3.0)) {
        const row = dataStart + i;
        probeSheet.getRange(`A${row}:C${row}`).format = {
          fill: "#C6EFCE",
          font: { bold: true, color: "#006100" }
        };
        break;
      }
    }
  }

  // Confirmation readings section
  if (Array.isArray(probeResult.confirmationReadings) && probeResult.confirmationReadings.length > 0) {
    const confStart = dataEnd + 3;
    probeSheet.getRange(`A${confStart}:C${confStart}`).values = [["Confirmation Samples", "", ""]];
    probeSheet.getRange(`A${confStart}:C${confStart}`).merge();
    probeSheet.getRange(`A${confStart}:C${confStart}`).format = {
      fill: "#4472C4",
      font: { bold: true, color: "#FFFFFF", size: 11 },
      horizontalAlignment: "center"
    };
    const confRows = probeResult.confirmationReadings.map((r, i) => [
      `Confirm #${i + 1}`, "", Number(r.value)?.toFixed(6) ?? ""
    ]);
    probeSheet.getRange(`A${confStart + 1}:C${confStart + confRows.length}`).values = confRows;
  }

  probeSheet.getRange("A:E").format.columnWidth = 20;
  probeSheet.getRange("A:A").format.columnWidth = 14;
  probeSheet.getRange("B:B").format.columnWidth = 18;
  probeSheet.getRange("C:C").format.columnWidth = 22;
  probeSheet.getRange("A:C").format.font = { name: "Aptos", size: 10 };
}

const inspect = await workbook.inspect({
  kind: "table",
  range: "DSOX1204G Measurement!A1:F10",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 8
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan"
});
await fs.mkdir(path.dirname(outputPath), { recursive: true });
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

const imported = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const preview = await imported.render({
  sheetName: "DSOX1204G Measurement",
  autoCrop: "all",
  scale: 1,
  format: "png"
});
const previewPath = outputPath.replace(/\.xlsx$/i, ".preview.png");
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

process.stdout.write(JSON.stringify({
  ok: true,
  outputPath,
  previewPath,
  inspect: inspect.ndjson,
  errors: errors.ndjson
}));

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}
