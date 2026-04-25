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
import java.util.concurrent.Executors

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
                    Size(1920, 1080),
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

                    // Calculate crop rect to focus on the center (approx viewfinder area)
                    // The viewfinder is 280dp, which is roughly 25-50% of the screen width/height.
                    // We'll crop to the central 60% of the image to be safe and reduce noise.
                    val w = imageProxy.width
                    val h = imageProxy.height
                    val cropW = (w * 0.6).toInt()
                    val cropH = (h * 0.6).toInt()
                    val left = (w - cropW) / 2
                    val top = (h - cropH) / 2
                    imageProxy.setCropRect(android.graphics.Rect(left, top, left + cropW, top + cropH))

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
                                val msiResult = MsiPlesseyBarcodeDecoderV2.decode(imageProxy)
                                if (msiResult != null) {
                                    detectedResult = BarcodeResult(
                                        value = msiResult.fullDigits,
                                        format = "MSI Plessey1" //msiResult.method
                                    )
                                    isScanning = false
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

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(60.dp))

            Box(
                modifier = Modifier
                    .size(280.dp)
                    .border(3.dp, Color.White.copy(alpha = 0.8f)),
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

            Spacer(Modifier.height(40.dp))

            if (!isScanning) {
                Button(
                    onClick = {
                        detectedResult = null
                        isScanning = true
                    },
                    modifier = Modifier.padding(bottom = 16.dp)
                ) {
                    Text(if (detectedResult == null) "Start Scanning" else "Rescan")
                }
            } else {
                Text("Scanning...", color = Color.White, style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(16.dp))
            }

            detectedResult?.let { result ->
                Card(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
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
