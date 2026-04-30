// ── BarcodeScannerScreenMsi ─────────────────────────────────────────────────────
//
// Compose UI for MSI Plessey barcode scanning using CameraX + ML Kit + custom V3 decoder.
//
// Image pipeline (per frame):
//   1. CameraX delivers a YUV_420_888 frame at 1280×960 (sensor-native landscape)
//   2. ML Kit runs first — if it finds a barcode, stop
//   3. If ML Kit finds nothing (expected for MSI barcodes, which ML Kit doesn't support):
//      a. extractGrayscale — reads the Y (luma) plane from the ImageProxy
//      b. correctOrientation — rotates from sensor landscape to display portrait using
//         rotateCW90 (lossless pixel transpose, no interpolation), so the barcode
//         bars run vertically (barcode runs horizontally)
//      c. Crop to barcode band — full width, middle 35% of height, shifted toward
//         the lower portion where shelf-tag barcodes appear in the viewfinder
//      d. decodeGray → cross-frame confidence → return result or null
//
// Performance optimizations in the V3 decoder (vs the original):
//   • decodeBarcode is sequential (removed runBlocking/async coroutine overhead)
//   • Deskew estimation skipped for camera frames (Sobel + morphological ops were
//     the single most expensive operation per orientation)
//   • Phase 2–4 (scale/sharpening sweeps) skipped for camera frames
//   • All-zeros results rejected in luhnNormalize (all-zeros is Luhn-valid)
//   • Weak results (col_greedy, short row_scan) rejected for live frames
//
// Debug frame saving (SAVE_DEBUG_FRAMES = true):
//   Saves up to 20 grayscale PNGs to external storage showing exactly what the
//   decoder receives (after crop + orientation correction).  Pull via adb.
// ────────────────────────────────────────────────────────────────────────────────
package com.example.zxing_barcodereader

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.common.InputImage
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

// Debug: save grayscale frames for visual inspection.  Set SAVE_DEBUG_FRAMES = true to re-enable.
// Pull via:
//   adb pull /sdcard/Android/data/com.example.zxing_barcodereader/files/Pictures/msi_frame_*.png
private const val SAVE_DEBUG_FRAMES = false
private var msiDebugFrameCount = 0
private const val MSI_DEBUG_MAX_FRAMES = 20

@OptIn(ExperimentalGetImage::class)
@Composable
fun BarcodeScannerScreenMsi() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var hasPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        hasPermission = granted
    }

    var detectedResult by remember { mutableStateOf<BarcodeResult?>(null) }
    var torchEnabled by remember { mutableStateOf(false) }
    var isScanning by remember { mutableStateOf(false) }
    var camera by remember { mutableStateOf<Camera?>(null) }

    val mlKitScanner = remember { BarcodeScanning.getClient() }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    LaunchedEffect(Unit) {
        if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    if (!hasPermission) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Camera permission required")
        }
        return
    }

    LaunchedEffect(torchEnabled) {
        camera?.cameraControl?.enableTorch(torchEnabled)
    }

    val surfaceRequest = remember { mutableStateOf<androidx.camera.core.SurfaceRequest?>(null) }

    LaunchedEffect(lifecycleOwner) {
        val cameraProvider = ProcessCameraProvider.awaitInstance(context)

        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(1280, 960),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val preview = Preview.Builder()
            .setResolutionSelector(resolutionSelector)
            .build().apply {
                setSurfaceProvider { surfaceRequest.value = it }
            }

        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    if (!isScanning) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    val mediaImage = imageProxy.image
                    if (mediaImage == null) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    // ML Kit gets the full frame (cropRect not used here — MSI crops after rotation)
                    val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

                    // 1. Always run ML Kit first
                    mlKitScanner.process(image)
                        .addOnSuccessListener { barcodes ->
                            val mlKitBarcode = barcodes.firstOrNull()

                            if (mlKitBarcode != null && !mlKitBarcode.rawValue.isNullOrBlank()) {
                                // ML Kit found something → show it and stop scanning
                                detectedResult = BarcodeResult(
                                    value = mlKitBarcode.rawValue!!,
                                    format = getFormatName(mlKitBarcode.format)
                                )
                                isScanning = false
                            } else {
                                // ML Kit found nothing → this is where we MUST try MSI
                                Log.d("BarcodeAnalyzer", "ML Kit returned no barcode → trying MSI Plessey fallback")

                                val gray = MsiPlesseyBarcodeDecoderV3.extractGrayscale(imageProxy)
                                if (gray != null) {
                                    // Step 1: rotate sensor-landscape → display-portrait (lossless transpose)
                                    val corrected = MsiPlesseyBarcodeDecoderV3.correctOrientation(
                                        gray, imageProxy.imageInfo.rotationDegrees
                                    )

                                    // Step 2: crop to barcode band in portrait coordinates.
                                    // Full width, ~35% of height, centered vertically.
                                    val bandHeight = (corrected.height * 0.35).toInt()
                                    val bandTop = (corrected.height - bandHeight) / 2
                                    val bandPixels = IntArray(corrected.width * bandHeight)
                                    for (y in 0 until bandHeight) {
                                        corrected.pixels.copyInto(
                                            bandPixels, y * corrected.width,
                                            (bandTop + y) * corrected.width, (bandTop + y + 1) * corrected.width
                                        )
                                    }
                                    val band = MsiPlesseyBarcodeDecoderV3.GrayImage(bandPixels, corrected.width, bandHeight)

                                    // Debug: save band frames for inspection (set SAVE_DEBUG_FRAMES = true to re-enable)
                                    if (SAVE_DEBUG_FRAMES && msiDebugFrameCount < MSI_DEBUG_MAX_FRAMES) {
                                        saveDebugFrame(context, band, msiDebugFrameCount)
                                        msiDebugFrameCount++
                                    }

                                    val msiResult = MsiPlesseyBarcodeDecoderV3.decodeGray(band)
                                    if (msiResult != null) {
                                        detectedResult = BarcodeResult(
                                            value = msiResult.fullDigits,
                                            format = "MSI Plessey1" //msiResult.method
                                        )
                                        isScanning = false
                                    }
                                }
//                                val msiResult2 = MsiPlesseyBarcodeDecoder.decode(imageProxy)
//                                if (msiResult2 != null) {
//                                    detectedResult = BarcodeResult(
//                                        value = msiResult2.fullDigits,
//                                        format = "MSI Plessey2" //msiResult.method
//                                    )
//                                    isScanning = false
//                                }
                            }
                        }
                        .addOnFailureListener { e ->
                            Log.e("BarcodeAnalyzer", "ML Kit failed", e)
                        }
                        .addOnCompleteListener {
                            imageProxy.close()          // Always close the proxy!
                        }
                }
            }

        cameraProvider.unbindAll()
        camera = cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalysis
        )
    }

    Box(modifier = Modifier.fillMaxSize()) {
        surfaceRequest.value?.let {
            CameraXViewfinder(surfaceRequest = it, modifier = Modifier.fillMaxSize())
        }

        // Centered viewfinder — uses Alignment.Center (not Column weight) so it never moves
        Box(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .height(130.dp)
                .border(3.dp, Color.White.copy(alpha = 0.8f))
                .align(Alignment.Center),
            contentAlignment = Alignment.TopEnd
        ) {
            IconButton(
                onClick = { torchEnabled = !torchEnabled },
                modifier = Modifier.padding(8.dp)
            ) {
                Icon(
                    imageVector = if (torchEnabled) Icons.Default.FlashOff else Icons.Default.FlashOn,
                    contentDescription = "Toggle Torch",
                    tint = Color.White
                )
            }
        }

        // Bottom content — fixed-height area so it never shifts the viewfinder
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (!isScanning) {
                Button(
                    onClick = {
                        detectedResult = null
                        isScanning = true
                    },
                    modifier = Modifier.padding(bottom = 8.dp)
                ) {
                    Text(if (detectedResult == null) "Start Scanning" else "Rescan")
                }
            } else {
                Text("Scanning...", color = Color.White, style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(16.dp))
            }

            detectedResult?.let { result ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Symbology: ${result.format}",
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(Modifier.height(8.dp))
                        Text(
                            text = "Value: ${result.value}",
                            style = MaterialTheme.typography.titleMedium
                        )
                    }
                }
            } ?: Text("Point camera at a barcode", color = Color.White)
        }
    }

    DisposableEffect(Unit) {
        onDispose { cameraExecutor.shutdown() }
    }
}

/** Save a grayscale image as PNG for debugging. */
private fun saveDebugFrame(context: android.content.Context, gray: MsiPlesseyBarcodeDecoderV3.GrayImage, index: Int) {
    try {
        val bitmap = Bitmap.createBitmap(gray.width, gray.height, Bitmap.Config.ARGB_8888)
        val argb = IntArray(gray.width * gray.height) { i ->
            val g = gray.pixels[i]
            (0xFF shl 24) or (g shl 16) or (g shl 8) or g
        }
        bitmap.setPixels(argb, 0, gray.width, 0, 0, gray.width, gray.height)
        val dir = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        dir?.mkdirs()
        val file = File(dir, "msi_frame_%03d_%dx%d.png".format(index, gray.width, gray.height))
        FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        Log.d("MSI_Debug", "Saved debug frame ${index + 1}/$MSI_DEBUG_MAX_FRAMES: ${file.absolutePath}")
        bitmap.recycle()
    } catch (e: Exception) {
        Log.e("MSI_Debug", "Failed to save debug frame $index", e)
    }
}
