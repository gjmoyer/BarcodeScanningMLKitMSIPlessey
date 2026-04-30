# AGENTS.md — ZXing Barcode Reader

## Project Overview

Android app using CameraX + Compose with custom MSI Plessey barcode decoding.
The primary decoder is `MsiPlesseyBarcodeDecoderV3` (multi-phase: rotation sweep, scale sweep, deskew recovery, sharpening recovery).

## Test Harness

A standalone Kotlin/JVM test harness that runs the V3 decoder logic against sample barcode images **without Android dependencies**. Lives outside the Android project since it reads PNGs directly via `javax.imageio`.

### Location

```
/Users/greg/AndroidStudioProjects/ZXing_BarcodeReader/MSI_samples/test_harness/
├── MsiHarness.kt             # standalone harness (all V3 non-Android logic extracted)
├── MsiHarness.jar            # compiled JAR
├── MsiHarnessProfiled.kt     # memory-profiling variant with allocation tracking
└── MsiHarnessProfiled.jar    # compiled profiled JAR
```

### Sample Images

```
/Users/greg/AndroidStudioProjects/ZXing_BarcodeReader/MSI_samples/*.png
```

9 sample images (356–541 × 83–140 px RGBA PNGs). Filename is the expected `digits7` value (code less the checksum).

### Compile & Run

```sh
# Main harness (quick validation)
kotlinc test_harness/MsiHarness.kt -include-runtime -d test_harness/MsiHarness.jar
java -cp test_harness/MsiHarness.jar MsiHarnessKt

# Memory-profiling harness (allocation counts, peak memory)
kotlinc test_harness/MsiHarnessProfiled.kt -include-runtime -d test_harness/MsiHarnessProfiled.jar
java -cp test_harness/MsiHarnessProfiled.jar MsiHarnessProfiledKt
```

Run from the `MSI_samples` directory (harness uses `../` relative path for samples).

### Kotlin Toolchain

```
# Installed via Homebrew
kotlin --version   # 2.3.21
java --version     # OpenJDK 25.0.2
```

### Key Differences from Android Code

The harness file (`MsiHarness.kt`) is a self-contained extraction of `MsiPlesseyBarcodeDecoderV3.kt` with:
- **Removed**: `ImageProxy`, `ImageFormat`, `androidx.camera` imports, cross-frame confidence accumulator (`frameLock`, `frameScores`, `reset`, `decode`, `decodeGray`)
- **Replace**: `extractGrayscale(ImageProxy)` → `loadGrayImage(path: String)` using `javax.imageio.ImageIO`
- **Replace**: `decodeBarcode` uses sequential calls instead of `runBlocking`/`async` (no coroutines)
- **Entry point**: `decodeSingleFrame(gray: GrayImage): DecodeResult?` is the main function

## Memory Optimizations (2026-04)

Three optimizations applied to `MsiPlesseyBarcodeDecoderV3.kt` (and both harness files):

| # | Optimization | Impact |
|---|---|---|
| 1 | **Inline RLE** — `rleFromRow` reads pixels directly, eliminating per-threshold `IntArray` allocations in `rowScan` | Eliminated 75% of allocations |
| 2 | **Reusable BooleanArray** — one `BooleanArray` per row in `zxingRowScan`, overwritten per threshold instead of allocating 13× per row | 92% reduction in `zxingscan_bin` |
| 3 | **Phase 3/4 guard** — skip sharpening sweeps on frames >400K pixels (`LARGE_FRAME_PIXELS`) to avoid OOM on camera frames | Saves ~100 MB worst-case |

Result: **86% total allocation reduction** (6927 KB → 945 KB across 9 test images). All 9 sample images still decode correctly with identical results.

## Android Project Structure

```
app/src/main/java/com/example/zxing_barcodereader/
├── MsiPlesseyBarcodeDecoder.kt     # V1 (original, 479 lines)
├── MsiPlesseyBarcodeDecoderV2.kt   # V2 (cross-frame confidence, 669 lines)
├── MsiPlesseyBarcodeDecoderV3.kt   # V3 (active, multi-phase, 1691 lines)
└── BarcodeScannerScreenMsi.kt      # Compose UI (uses V3 as ML Kit fallback)

app/src/test/java/com/example/zxing_barcodereader/
└── MsiDecoderTest.kt               # Unit tests for GrayImage rotation helpers
```
