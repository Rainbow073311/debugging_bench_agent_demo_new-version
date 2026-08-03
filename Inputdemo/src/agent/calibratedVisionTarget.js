function normalizePixel(pixel) {
  if (Array.isArray(pixel) && pixel.length === 2) {
    return { x: Number(pixel[0]), y: Number(pixel[1]) };
  }
  if (pixel && typeof pixel === "object") {
    return { x: Number(pixel.x), y: Number(pixel.y) };
  }
  return null;
}

export async function projectEyeInHandVlmPixel({ cameraExecution, cameraController, pixel }) {
  if (cameraExecution?.status !== "COMPLETED") return null;
  const cleanPixel = normalizePixel(pixel);
  if (!cleanPixel || !Number.isFinite(cleanPixel.x) || !Number.isFinite(cleanPixel.y)) return null;
  const robotPose = cameraExecution.selectedImage?.robotPose;
  if (!robotPose) throw new Error("Selected close image is missing its synchronized robot pose.");
  if (typeof cameraController?.pixelToBase !== "function") {
    throw new Error("Eye-in-hand pixel-to-base projection is unavailable.");
  }
  return cameraController.pixelToBase({
    calibrationFile: cameraExecution.calibrationFile,
    pixel: cleanPixel,
    robotPose
  });
}
