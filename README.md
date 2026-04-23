# BarcodeScanner with MSI Plessey Decoder

Android barcode scanner built with Jetpack Compose and CameraX, featuring ML Kit for standard 1D barcodes and a custom MSI Plessey decoder for Publix shelf tags.

## Features

- **ML Kit Barcode Scanning** for Code 128, EAN-13/UPC-A, EAN-8, UPC-E, and more
- **Custom MSI Plessey decoder** running entirely in Kotlin on raw YUV data (no OpenCV required)
- Dual UI screens: ML Kit scanner and MSI Plessey scanner
- Real-time camera preview with CameraX Compose integration
- Torch toggle for low-light scanning

## Supported Barcode Types

| Symbology | Detection |
|-----------|-----------|
| Code 128 | ML Kit |
| EAN-13 / UPC-A | ML Kit |
| EAN-8 | ML Kit |
| UPC-E | ML Kit |
| Codabar | ML Kit |
| Code 39 | ML Kit |
| Code 93 | ML Kit |
| Data Matrix | ML Kit |
| PDF417 | ML Kit |
| ITF (Interleaved 2 of 5) | ML Kit |
| QR Code | ML Kit |
| Aztec | ML Kit |
| MSI Plessey | Custom decoder |

## Requirements

- Android 7.0+ (API 29+)
- Android Studio Hedgehog+
- Physical device with camera

## Architecture

```
app/src/main/java/com/example/zxing_barcodereader/
├── MainActivity.kt                 — App entry, OpenCV init, launches Compose UI
├── BarcodeScannerScreen.kt         — ML Kit barcode scanner (Compose + CameraX)
├── BarcodeScannerScreenMsi.kt      — MSI Plessey scanner (Compose + CameraX)
├── MsiPlesseyBarcodeDecoder.kt     — MSI Plessey decode engine (no native libs)
├── MSIPlesseyDecoder_BAD1.kt       — Earlier decoder attempt (archived)
├── getFormatName.kt                — ML Kit format constant → human-readable name
└── ui/theme/                       — Material3 theme
```

### MSI Plessey Decoder

The custom decoder (`MsiPlesseyBarcodeDecoder`) performs entirely in Kotlin/JVM using raw YUV pixel data from CameraX:

1. **GrayImage extraction** — reads the Y (luminance) plane from YUV_420_888
2. **Barcode band isolation** — scores rows by dark/light transition count, finds the tallest contiguous high-scoring region
3. **Row scan decoder** — binarizes each row at 13 thresholds, run-length encodes, splits narrow/wide modules via bimodal histogram, decodes MSI 4-bar-per-digit groups, votes across all (row, threshold) combinations
4. **Column greedy fallback** — builds a per-column darkness signal, sweeps grid of (N, W, offset) parameters, samples expected bit-pair windows
5. **Luhn check-digit validation** rejects false positives in both decoders

Performance: typically < 30 ms for a 720 × 1280 frame on mid-range hardware.

## Setup

1. Open `ZXing_BarcodeReader` in Android Studio
2. Sync Gradle (will download CameraX, ML Kit, OpenCV dependencies)
3. Build and run on a physical Android device

## Notes

- The MSI Plessey scanner (`BarcodeScannerScreenMsi.kt`) expects MSI-encoded shelf labels common in Publix stores
- OpenCV is included as a dependency but the MSI decoder does not require it (init is for optional future use)
- Rotation handling is available via `MsiPlesseyBarcodeDecoder.rotateCW90()` helper
