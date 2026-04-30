To re-run after code changes:
# Compile
kotlinc test_harness/MsiHarness.kt -include-runtime -d test_harness/MsiHarness.jar

# Run
java -cp test_harness/MsiHarness.jar MsiHarnessKt

Key details:
- Extracted all non-Android V3 decoder logic into a standalone file (no ImageProxy, no CameraX, no coroutines)
- Uses javax.imageio for PNG → GrayImage conversion
- Compares digits7 against the filename (which is the code value less the checksum)
- Decodes run ~2–22ms per image on this hardware
