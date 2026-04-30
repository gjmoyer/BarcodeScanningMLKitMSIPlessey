package com.example.zxing_barcodereader

import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import kotlin.math.PI

// ── Swift / iOS porting guide ─────────────────────────────────────────────────
//
// All image-processing logic in this file is pure math (no Android-specific
// algorithms) and ports to Swift with minimal changes.  The only platform-
// specific piece is the camera input (extractGrayscale), which you replace with
// a CVPixelBuffer reader.
//
// Platform / API mapping:
//   Kotlin `object`              → Swift `enum` (no cases) as a namespace, or a
//                                  class with `static let shared = MyDecoder()`.
//   `data class`                 → Swift `struct` (value type; == is synthesised).
//   `ImageProxy` (CameraX)       → `CVPixelBuffer` from AVFoundation, delivered by
//                                  `AVCaptureVideoDataOutput` or a Vision request.
//   YUV_420_888 luma plane       → plane 0 of `kCVPixelFormatType_420YpCbCr8BiPlanar-
//                                  VideoRange`; read with CVPixelBufferGetBaseAddress-
//                                  OfPlane(buffer, 0).  Stride = CVPixelBufferGetBytesPerRowOfPlane.
//   `synchronized(lock) { }`    → `NSLock` (lock()/unlock()) or a serial DispatchQueue
//                                  with `.sync {}`.
//   `mutableMapOf<K,V>()`        → `[K: V]()` Swift Dictionary literal.
//   `IntArray(n)` / `BooleanArray(n)` → `[Int](repeating:count:)` / `[Bool](repeating:count:)`.
//   `coerceAtLeast` / `coerceIn` → `Swift.max()` / `clamped(to:)`.
//   `shr` / `shl`                → `>>` / `<<` (Swift uses the same operators).
//   `xor` (Boolean)              → `!=` (Swift does not have infix `xor` for Bool).
//
// Key entry points to port in priority order:
//   1. extractGrayscale — swap for a CVPixelBuffer luma-plane reader (~15 lines).
//   2. decodeSingleFrame — no platform calls; ports verbatim.
//   3. decodeGray / reset — cross-frame accumulator; replicate the lock pattern.
//   4. correctOrientation — lossless 90° rotation for portrait phone (pure pixel remap).
//   5. decodeBarcode — sequential bimodal + ZXing row scan with Luhn validation.
// ─────────────────────────────────────────────────────────────────────────────
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt


/**
 * ── MSI Plessey barcode format (reference for porters) ───────────────────────
 *
 *   Each decimal digit is encoded as 4 bar/space pairs = 8 elements:
 *     Bit = 1  →  wide bar  (W, ≈2× narrow) + narrow space (N)
 *     Bit = 0  →  narrow bar (N)             + wide space  (W, ≈2× narrow)
 *
 *   A 9-character barcode (8 data digits + 1 Luhn check digit) has:
 *     Start guard : wide bar  + narrow space          (2 elements)
 *     9 digits    : 9 × 4 bit-pairs = 9 × 8 elements (72 elements)
 *     End guard   : narrow bar + wide space + narrow bar (3 elements)
 *     ─────────────────────────────────────────────────────────────
 *     Total       : ≈ 77 dark/light transitions per image row
 *
 *   The Luhn check digit (see luhnCheck / luhnNormalize) lets every candidate
 *   string be self-validated without an external look-up.
 *
 *   Publix shelf tags carry a 9-character code.  The meaningful consumer payload
 *   is the first 7 digits (DecodeResult.digits7), which uniquely identifies the
 *   item in the product catalogue.
 *
 * ── MSI Plessey barcode isolator and decoder for Android / CameraX ────────────
 *
 * Drop-in replacement for MsiPlesseyBarcodeDecoder that runs two independent
 * decode approaches on every frame and reconciles their results:
 *
 *   Approach A — Bimodal row scan (original)
 *     Collects ALL dark-bar widths in a row, finds the largest gap in the
 *     sorted distribution to establish the narrow/wide split, then decodes.
 *     The threshold adapts to the full row, making it robust to scale and
 *     contrast variation in live-camera frames.
 *
 *   Approach B — ZXing run-counter row scan (ported from MSIReader.cs)
 *     Establishes the narrow/wide pivot from the start-guard's 2 counters
 *     alone, then applies that single pivot to all subsequent characters.
 *     Uses the ZXing CHARACTER_ENCODINGS lookup table for digit matching.
 *
 *   Deskew recovery — barcode-mask geometry
 *     If the direct decode is weak, a barcode-like mask is built from strong
 *     horizontal gradients, the mask hull's dominant edge provides the skew
 *     angle, and the frame is retried around that estimate. A small fixed
 *     sweep of moderate rotations runs first so the decoder can still recover
 *     when the angle estimate is weak. The best candidate across all attempts
 *     wins.
 *
 *   Scale recovery — multi-scale barcode band decode
 *     If the full rotation sweep at native scale produces no strong result, the
 *     frame is retried at 2×, 0.5×, 1.5×, and 0.75× bilinear scale.  Upscaling
 *     (2×, 1.5×) helps when the barcode is small in frame (camera held far from
 *     the label); downscaling (0.5×, 0.75×) helps when the barcode fills most of
 *     the frame.  Each rescaled image runs the full deskew recovery path with the
 *     upright orientation only; the early-exit keeps cost bounded.
 *
 *   Sharpening recovery — unsharp-mask pre-processing (Phase 3 and 4)
 *     If both Phase 1 and Phase 2 fail, the original frame is sharpened before
 *     retrying.  Phase 3 applies moderate unsharp-mask (strength 1.5) and re-runs
 *     the full rotation sweep — effective for mildly blurred photographs where the
 *     bimodal split cannot distinguish narrow from wide bars.  Phase 4 applies
 *     aggressive sharpening (strength 10.0, which binarises edges) combined with
 *     up-scale (2×, 3×) — a last resort for heavily blurred thin-bar images.
 *
 *   Fallback — Column greedy (original)
 *     Builds a per-column 10th-percentile darkness signal and sweeps a
 *     parameter grid positionally.  Used only when both row-scan approaches
 *     fail.
 *
 * ── Result confidence ─────────────────────────────────────────────────────────
 *
 *   DecodeResult.method reports which path(s) succeeded:
 *     "combined"       — both A and B agreed (highest confidence)
 *     "row_scan"       — only approach A succeeded
 *     "zxing_row_scan" — only approach B succeeded
 *     "col_greedy"     — fallback path (lower confidence)
 *
 * ── Usage (identical to MsiPlesseyBarcodeDecoder) ─────────────────────────────
 *
 *   override fun analyze(imageProxy: ImageProxy) {
 *       val result = MsiPlesseyBarcodeDecoderV2.decode(imageProxy)
 *       imageProxy.close()
 *       result?.let { runOnUiThread { showResult(it.digits7) } }
 *   }
 *
 * ── Notes ─────────────────────────────────────────────────────────────────────
 *
 *   • The ZXing approach uses a private BitRow class. The `avgWidth` pivot is
 *     computed as a local value inside each row decode call, so both approaches
 *     are free of shared mutable state and safe to run in parallel.
 *   • All other notes from MsiPlesseyBarcodeDecoder apply here unchanged.
 *
 * ── Live-camera performance optimizations (2026-04) ────────────────────────────
 *
 *   decodeBarcode is sequential (rowScan then zxingRowScan) — removed the
 *   runBlocking/async coroutine overhead that was expensive on the single-thread
 *   camera analyzer executor.  Both row-scan approaches are independent and don't
 *   mutate shared state, so sequential execution is correct.
 *
 *   Camera frames (>100K pixels after crop) use a fast path in decodeSingleFrame:
 *     • Full 11-angle rotation sweep (same as offline) — correct orientation is
 *       critical for MSI decode accuracy
 *     • Deskew estimation SKIPPED — the Sobel gradient + morphological operations
 *       were the single most expensive per-orientation operation.  The rotation
 *       sweep already covers ±30° in 11 steps.
 *     • Phase 2 (scale sweep at 2×, 0.5×, 1.5×, 0.75×) SKIPPED — barcodes on
 *       live camera are already at a reasonable scale.
 *     • Phase 3/4 (sharpening sweeps) SKIPPED — adds latency without benefit for
 *       real-time scanning where the next frame comes quickly.
 *
 *   Weak results (decodeQuality < 35) are treated as no-decode for live frames so
 *   col_greedy fallbacks and very short row_scan results don't accumulate through
 *   cross-frame confidence and produce false positives.
 *
 *   luhnNormalize rejects all-zeros results — "00000000" is a legitimate Luhn-
 *   valid string that commonly arises as a decode artifact on blank/under-exposed
 *   frames; rejecting it prevents follow-through to the confidence accumulator.
 *
 *   correctOrientation applies a lossless 90°/180°/270° pixel-transpose rotation
 *   (via rotateCW90) to correct for the camera sensor's landscape orientation when
 *   the phone is held in portrait.  This is always applied in decode(imageProxy).
 *   extractGrayscale honours the cropRect if set on the ImageProxy, reading only
 *   the specified sub-region from the full YUV plane buffer.
 */
object MsiPlesseyBarcodeDecoderV3 {

    // ── Result ───────────────────────────────────────────────────────────────────

    data class DecodeResult(
        /** First 7 significant digits — the useful payload for Publix shelf tags. */
        val digits7: String,
        /** Full decoded string (typically 9 chars = 8 data + 1 Luhn check digit). */
        val fullDigits: String,
        /**
         * Which internal path succeeded:
         *   "combined"       — both row-scan approaches agreed
         *   "row_scan"       — bimodal approach only
         *   "zxing_row_scan" — ZXing approach only
         *   "col_greedy"     — column greedy fallback
         */
        val method: String
    )

    data class DeskewEstimate(
        /** Positive values mean the barcode appears rotated clockwise in image coordinates. */
        val degreesClockwise: Double,
        /** Fraction of the hull perimeter covered by the dominant edge. */
        val coherence: Double,
        /** Number of boundary pixels contributing to the hull. */
        val sampleCount: Int
    )

    // ── MSI Plessey bit-pattern ↔ digit mapping (Approach A) ────────────────────
    //
    // Each digit is represented by 4 bits; each bit describes one bar/space pair:
    //   bit = 1  →  wide bar  (W) + narrow space (N)   [dark run is LONG]
    //   bit = 0  →  narrow bar (N) + wide space  (W)   [dark run is SHORT]
    //
    // In the run-length stream, only the DARK (bar) run widths are compared to the
    // bimodal split threshold to determine each bit.  The space widths are consumed
    // by the RLE but not directly tested.
    //
    // Swift port: replace with a [([Int], Int)] array or a switch/dictionary.

    private val BITS_TO_DIGIT: Map<List<Int>, Int> = mapOf(
        listOf(0, 0, 0, 0) to 0, listOf(0, 0, 0, 1) to 1,
        listOf(0, 0, 1, 0) to 2, listOf(0, 0, 1, 1) to 3,
        listOf(0, 1, 0, 0) to 4, listOf(0, 1, 0, 1) to 5,
        listOf(0, 1, 1, 0) to 6, listOf(0, 1, 1, 1) to 7,
        listOf(1, 0, 0, 0) to 8, listOf(1, 0, 0, 1) to 9
    )

    // Fast digit lookup: index = b0*8 + b1*4 + b2*2 + b3.  -1 = invalid pattern.
    // Avoids Map + List allocation on every digit decode.
    private val BITS_LOOKUP = IntArray(16) { -1 }.also {
        it[0] = 0; it[1] = 1; it[2] = 2; it[3] = 3; it[4] = 4
        it[5] = 5; it[6] = 6; it[7] = 7; it[8] = 8; it[9] = 9
    }

    // ── ZXing CHARACTER_ENCODINGS (Approach B) ───────────────────────────────────
    //
    // Each entry is a compact bit-pattern for one digit, matching the ZXing.Net
    // MSIReader encoding table.  The pattern is built by zxingToPattern(): narrow
    // runs contribute 1 bit and wide runs contribute 2 identical bits, so each
    // 8-element character becomes a 12-bit integer.
    //
    // Decoding digits 0–9:
    //   index 0 → 0x924, index 1 → 0x926, … index 9 → 0xD26
    //
    // ZXING_START (0x06): wide bar + narrow space  — matches the MSI start guard.
    // ZXING_END   (0x09): narrow bar + wide space + narrow bar — MSI end guard.
    //
    // Swift port: use an [Int] array; the lookup in zxingPatternToChar is just a
    // linear scan, safe to keep as-is for 10 elements.

    private val ZXING_CHARACTER_ENCODINGS = intArrayOf(
        0x924, 0x926, 0x934, 0x936, 0x9A4, 0x9A6, 0x9B4, 0x9B6, 0xD24, 0xD26
    )
    private val ZXING_ALPHABET = "0123456789"
    private const val ZXING_START = 0x06   // wide bar  + narrow space (MSI start guard)
    private const val ZXING_END = 0x09   // narrow bar + wide space + narrow bar (MSI end guard)

    // ── Cross-frame confidence accumulator ───────────────────────────────────────
    //
    // Each call to decode()/decodeGray() scores the best single-frame result and
    // adds to a running total per candidate value.  A result is only returned to
    // the caller once one value's total reaches CONFIDENCE_THRESHOLD, ensuring
    // that a single noisy or borderline frame can never trigger a false positive.
    //
    // Points per method reflect single-frame decode strength:
    //   combined       → 3  (both A and B agreed on this frame)
    //   row_scan       → 2  (bimodal approach succeeded)
    //   zxing_row_scan → 2  (ZXing approach succeeded)
    //   col_greedy     → 1  (fallback; least reliable)
    //
    // CONFIDENCE_THRESHOLD = 5 requires at minimum two frames, for example:
    //   combined  + row_scan   = 3+2 = 5  ✓  (2 frames)
    //   row_scan  + row_scan   = 2+2 = 4  ✗  → 3rd frame needed (2+2+2 = 6 ✓)
    //   col_greedy × 5         = 5        ✓  (5 frames — slow, but eventually fires)
    //
    // Call reset() when the user navigates away or starts scanning a new item,
    // so stale votes from a previous label don't bleed into the next scan.

    private const val SCORE_COMBINED = 3   // both A and B agreed — highest single-frame evidence
    private const val SCORE_ROW_SCAN = 2   // bimodal approach succeeded
    private const val SCORE_ZXING_ROW_SCAN = 2   // ZXing approach succeeded
    private const val SCORE_COL_GREEDY = 1   // fallback; least reliable
    private const val CONFIDENCE_THRESHOLD = 5   // minimum cumulative score to fire a result
    // STRONG_DECODE_SCORE: when a single-frame result exceeds this quality threshold
    // the outer sweep loop exits immediately without trying further rotations or scales.
    // Breakdown: "combined" base=40 + fullDigits.length (≥9) = 49 → triggers early exit.
    private const val STRONG_DECODE_SCORE = 49
    // Minimum quality for a live-camera frame result to be accepted (>= "row_scan" + 5 digits).
    // Weaker results (col_greedy, very short row_scan) are treated as no-decode so they
    // don't accumulate garbage through cross-frame confidence.
    private const val MIN_QUALITY_LIVE = 35

    // Deskew mask pipeline tuning — see buildDeskewMask for how these are used.
    private const val DESKEW_MIN_ABS_DEGREES = 2.5   // ignore near-zero estimates (upright barcode)
    private const val DESKEW_MAX_ABS_DEGREES = 30.0  // cap; beyond 30° the rotation sweep covers it
    private const val DESKEW_MIN_EDGE_LENGTH = 30.0  // hull edge must be long enough to be reliable
    private const val DESKEW_MASK_CLOSE_W = 25    // morphological close kernel width (px)
    private const val DESKEW_MASK_CLOSE_H = 7     // morphological close kernel height (px)
    private const val DESKEW_MASK_ERODE_PASSES = 3   // erosion passes after close (noise removal)
    private const val DESKEW_MASK_DILATE_PASSES = 4   // dilation passes after erosion (re-expand)

    // Fixed rotation angles tried in Phase 1.  0° is always first so an upright
    // barcode exits without any rotation cost.  Angles are tried in ±pairs from
    // small to large so the sweep finds moderate tilts before extreme ones.
    private val ROTATION_SWEEP_DEGREES = doubleArrayOf(
        0.0, -30.0, -24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0, 30.0
    )
    // Tried in order only when the native-scale rotation sweep yields no strong result.
    // 2× / 1.5× help for barcodes that are small in frame (camera far from label).
    // 0.5× / 0.75× help for barcodes that nearly fill the frame (camera very close).
    private val SCALE_SWEEP_FACTORS = doubleArrayOf(2.0, 0.5, 1.5, 0.75)

    // Unsharp-mask strengths used in Phase 3 and Phase 4.
    // Moderate (1.5): enhances mild camera blur while preserving image structure;
    //   fixes barcodes that blur caused both primary row-scan approaches to fail.
    // Aggressive (10.0): drives bar/space edges toward 0/255, effectively binarising
    //   the image near edges — last resort for heavily blurred photographs.
    private const val SHARP_STRENGTH_MODERATE = 1.5f
    private const val SHARP_STRENGTH_AGGRESSIVE = 10.0f
    // Scale factors applied in Phase 4 (after aggressive sharpening).
    // Larger up-scales help when bars are so thin that sharpening alone is insufficient.
    private val SHARP_SCALE_FACTORS = doubleArrayOf(2.0, 3.0)
    // Frames with more than this many pixels skip Phase 3 and 4 (sharpening sweeps) to avoid
    // blowing the heap on high-resolution camera inputs.  400K pixels ≈ 1.6 MB per GrayImage.
    private const val LARGE_FRAME_PIXELS = 400_000L
    // Frames with more than this many pixels use the fast live-camera path:
    //   • deskew estimation is skipped (rotation sweep handles skew)
    //   • Phase 2–4 (scale/sharpening sweeps) are skipped
    private const val LIVE_FRAME_PIXELS = 100_000L

    private val frameLock = Any()
    private val frameScores = mutableMapOf<String, Int>()

    /** Clear all accumulated frame evidence.  Call when starting a new scan. */
    fun reset() = synchronized(frameLock) { frameScores.clear() }

    // ── Public API ───────────────────────────────────────────────────────────────

    /**
     * Main entry point.  Accepts a CameraX [ImageProxy] and returns a
     * [DecodeResult] once the accumulated cross-frame confidence for one
     * candidate value reaches [CONFIDENCE_THRESHOLD], or null if more frames
     * are needed.  Call [reset] when starting a new scan.
     *
     * Automatically corrects for the display/sensor rotation misalignment so the
     * barcode bars appear vertical (horizontal barcode) regardless of phone orientation.
     *
     * Does NOT close the [ImageProxy] — call [ImageProxy.close] yourself.
     */
    fun decode(imageProxy: ImageProxy): DecodeResult? {
        val gray = extractGrayscale(imageProxy) ?: return null
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val corrected = correctOrientation(gray, rotationDegrees)
        return decodeGray(corrected)
    }

    /**
     * Decode from an already-extracted [GrayImage].
     * Accumulates cross-frame confidence the same as [decode].
     */
    fun decodeGray(gray: GrayImage): DecodeResult? {
        val single = decodeSingleFrame(gray) ?: return null

        val points = when (single.method) {
            "combined" -> SCORE_COMBINED
            "row_scan" -> SCORE_ROW_SCAN
            "zxing_row_scan" -> SCORE_ZXING_ROW_SCAN
            else -> SCORE_COL_GREEDY
        }

        val total = synchronized(frameLock) {
            val t = (frameScores[single.digits7] ?: 0) + points
            frameScores[single.digits7] = t
            t
        }

        return if (total >= CONFIDENCE_THRESHOLD) {
            synchronized(frameLock) { frameScores.clear() }
            single
        } else {
            null
        }
    }

    /**
     * Decode a single frame without cross-frame confidence accumulation.
     *
     * Normal flow:
     *   Phase 1 — native scale, full rotation sweep
     *     1. try a small fixed sweep of moderate frame rotations
     *     2. at each orientation, isolate the barcode band and attempt the normal decode
     *     3. if that candidate is weak, estimate a local deskew angle and retry nearby
     *     4. return early if any orientation produces a strong result
     *
     *   Phase 2 — alternate scales, upright only (if Phase 1 yields no strong result)
     *     5. rescale the original frame at 2×, 0.5×, 1.5×, 0.75× using bilinear sampling
     *     6. for each rescaled image run the full deskew recovery path (upright orientation)
     *     7. return the strongest candidate across all phases
     *
     *   Phase 3 — moderate sharpening + full rotation sweep (if Phase 2 still weak)
     *     8. apply unsharp-mask (strength 1.5) to the original frame
     *     9. re-run the rotation sweep — helps images with mild camera blur
     *
     *   Phase 4 — aggressive sharpening + scale sweep (last resort)
     *    10. apply unsharp-mask (strength 10.0) which binarises edges
     *    11. try at 2× and 3× scale — recovers heavily blurred thin-bar photographs
     */
    fun decodeSingleFrame(gray: GrayImage): DecodeResult? {
        val pixelCount = gray.width.toLong() * gray.height.toLong()
        val isLive = pixelCount > LIVE_FRAME_PIXELS
        // Phase 1: native scale, full rotation sweep
        var best: DecodeResult? = null
        for (rotation in ROTATION_SWEEP_DEGREES) {
            val oriented = if (rotation == 0.0) gray else rotate(gray, rotation)
            best = chooseBetterDecode(best, decodeSingleOrientation(oriented, skipDeskew = isLive))
            if (decodeQuality(best) >= STRONG_DECODE_SCORE) return best
        }
        // Phase 2: alternate scales, upright only — deskew recovery handles residual rotation
        // Skip scale sweep for live camera frames (barcode is already at reasonable scale)
        if (!isLive) {
            for (factor in SCALE_SWEEP_FACTORS) {
                val scaled = scale(gray, factor)
                best = chooseBetterDecode(best, decodeSingleOrientation(scaled))
                if (decodeQuality(best) >= STRONG_DECODE_SCORE) return best
            }
        }
        // Phases 3/4 (sharpening sweeps) are expensive — skip on large frames (OOM risk)
        // and on live-camera frames (they add latency without benefit for real-time scanning).
        if (pixelCount <= LARGE_FRAME_PIXELS && !isLive) {
            // Phase 3: moderate sharpening + full rotation sweep (mild-blur recovery)
            val sharpMod = sharpen(gray, SHARP_STRENGTH_MODERATE)
            for (rotation in ROTATION_SWEEP_DEGREES) {
                val oriented = if (rotation == 0.0) sharpMod else rotate(sharpMod, rotation)
                best = chooseBetterDecode(best, decodeSingleOrientation(oriented))
                if (decodeQuality(best) >= STRONG_DECODE_SCORE) return best
            }
            // Phase 4: aggressive sharpening + scale sweep (heavy-blur / thin-bar last resort)
            val sharpAgg = sharpen(gray, SHARP_STRENGTH_AGGRESSIVE)
            for (factor in SHARP_SCALE_FACTORS) {
                val scaled = scale(sharpAgg, factor)
                best = chooseBetterDecode(best, decodeSingleOrientation(scaled))
                if (decodeQuality(best) >= STRONG_DECODE_SCORE) return best
            }
        }
        return best
    }

    private fun decodeSingleOrientation(gray: GrayImage, skipDeskew: Boolean = false): DecodeResult? {
        val directCrop = isolateBarcode(gray) ?: gray
        val direct = decodeBarcode(directCrop)
        if (decodeQuality(direct) >= STRONG_DECODE_SCORE) return direct

        // Deskew estimation is expensive (Sobel gradient + morphological operations on full image).
        // Skip it for live camera frames — the rotation sweep already covers common tilt angles.
        // Also reject weak results for live frames so garbage doesn't accumulate through cross-frame confidence.
        if (skipDeskew) return if (decodeQuality(direct) >= MIN_QUALITY_LIVE) direct else null

        var best = direct
        val deskewEstimate = estimateDeskew(gray)
        for (correctionDegrees in rotationCorrections(deskewEstimate)) {
            if (abs(correctionDegrees) < 1e-6) continue
            val rotated = rotate(gray, correctionDegrees)
            val retryCrop = isolateBarcode(rotated) ?: rotated
            best = chooseBetterDecode(best, decodeBarcode(retryCrop))
            if (decodeQuality(best) >= STRONG_DECODE_SCORE) return best
        }
        return best
    }

    // ── GrayImage ────────────────────────────────────────────────────────────────
    //
    // Flat row-major pixel store: pixel at (x, y) lives at index y*width + x.
    // Values are 0 (black) … 255 (white), matching the luma (Y) channel of YUV.
    //
    // Swift port: model as a struct with a [UInt8] or [Int] buffer and (width, height).
    // Using [UInt8] saves memory at the cost of one cast when doing arithmetic.

    class GrayImage(val pixels: IntArray, val width: Int, val height: Int) {
        fun pixel(x: Int, y: Int): Int = pixels[y * width + x]
    }

    /**
     * Convenience: rotate a GrayImage 90° clockwise (portrait → landscape).
     *
     * Use this before feeding a portrait camera frame to the decoder, which
     * expects barcodes to run horizontally (bars are vertical stripes).
     *
     * Mapping: dst[x, dstW-1-y] = src[x, y]  where dstW = src.height.
     * Swift: transpose src.height × src.width into a src.width × src.height buffer.
     */
    fun rotateCW90(src: GrayImage): GrayImage {
        val dstW = src.height
        val dstH = src.width
        val dst = IntArray(dstW * dstH)
        for (y in 0 until src.height)
            for (x in 0 until src.width)
                dst[x * dstW + (dstW - 1 - y)] = src.pixel(x, y)
        return GrayImage(dst, dstW, dstH)
    }

    /**
     * Correct for the display/sensor rotation misalignment so the barcode runs
     * horizontally (bars are vertical stripes).  [rotationDegrees] is the
     * value from [ImageProxy.imageInfo.rotationDegrees] — typically 0, 90, 180, or 270.
     *
     * Background: the camera sensor has a fixed landscape orientation.  When the phone
     * is held in portrait, the raw image is landscape but `rotationDegrees` = 90.
     * Applying that rotation brings the image to display orientation, making the
     * real-world-horizontal barcode actually horizontal in pixel coordinates.
     */
    fun correctOrientation(gray: GrayImage, rotationDegrees: Int): GrayImage {
        return when (rotationDegrees) {
            0 -> gray
            90 -> rotateCW90(gray)
            180 -> rotateCW90(rotateCW90(gray))
            270 -> rotateCW90(rotateCW90(rotateCW90(gray)))
            else -> gray // fallback; shouldn't happen for CameraX frames
        }
    }

    /**
     * Rotate a grayscale image by an arbitrary number of degrees.
     *
     * Positive values are clockwise in image coordinates (x right, y down).
     * The destination canvas is expanded to avoid clipping and is filled with
     * [background] (white = 255) outside the source image.
     *
     * Algorithm — inverse mapping with bilinear interpolation:
     *   For every destination pixel (x, y) compute the corresponding source
     *   coordinate by applying the INVERSE rotation (i.e. rotate backwards by
     *   the same angle).  Then sample the source at that sub-pixel location
     *   with bilinear interpolation.  Inverse mapping guarantees every output
     *   pixel is filled (forward mapping leaves holes).
     *
     *   Canvas sizing: a rotated rectangle of size W×H fits in a bounding box of
     *     newW = W·|cos θ| + H·|sin θ|
     *     newH = W·|sin θ| + H·|cos θ|
     *   Both source and destination are centered so rotation happens about the
     *   image center (srcCx/srcCy and dstCx/dstCy).
     *
     * Swift port: same math; use vImage_Rotate (Accelerate) for performance if
     * needed, but the plain loop is fast enough for the frame sizes here.
     */
    fun rotate(gray: GrayImage, degreesClockwise: Double, background: Int = 255): GrayImage {
        if (abs(degreesClockwise) < 1e-6) return gray

        val radians = degreesClockwise * PI / 180.0
        val cosA = cos(radians)
        val sinA = sin(radians)
        val absCos = abs(cosA)
        val absSin = abs(sinA)
        val dstW = ceil(gray.width * absCos + gray.height * absSin).toInt().coerceAtLeast(1)
        val dstH = ceil(gray.width * absSin + gray.height * absCos).toInt().coerceAtLeast(1)

        val srcCx = (gray.width - 1) / 2.0   // center of source image (fractional pixel)
        val srcCy = (gray.height - 1) / 2.0
        val dstCx = (dstW - 1) / 2.0         // center of destination canvas
        val dstCy = (dstH - 1) / 2.0
        val out = IntArray(dstW * dstH)

        // Inverse-map: for each dst pixel, find its source coordinate via un-rotation.
        // Incremental per-pixel: srcX advances by cosA, srcY advances by -sinA per x-step.
        // This replaces 2 multiplies per pixel with 2 additions.
        for (y in 0 until dstH) {
            val dy = y - dstCy
            var srcX = cosA * (-dstCx) + sinA * dy + srcCx
            var srcY = sinA * dstCx + cosA * dy + srcCy
            val rowStart = y * dstW
            for (x in 0 until dstW) {
                out[rowStart + x] = sampleBilinear(gray, srcX, srcY, background)
                srcX += cosA
                srcY -= sinA
            }
        }

        return GrayImage(out, dstW, dstH)
    }

    /**
     * Rescale a grayscale image by [factor] using bilinear interpolation.
     *
     * factor > 1.0  →  enlarge (upscale)   — helps when the barcode is small in frame
     * factor < 1.0  →  shrink (downscale)  — helps when the barcode fills the frame
     */
    fun scale(gray: GrayImage, factor: Double): GrayImage {
        if (abs(factor - 1.0) < 1e-6) return gray
        val newW = (gray.width * factor).roundToInt().coerceAtLeast(1)
        val newH = (gray.height * factor).roundToInt().coerceAtLeast(1)
        val out = IntArray(newW * newH)
        val invFactor = 1.0 / factor
        for (y in 0 until newH) {
            val srcY = y * invFactor
            var srcX = 0.0
            val rowStart = y * newW
            for (x in 0 until newW) {
                out[rowStart + x] = sampleBilinear(gray, srcX, srcY, 255)
                srcX += invFactor
            }
        }
        return GrayImage(out, newW, newH)
    }

    /**
     * Sharpen a grayscale image using a 4-neighbour unsharp mask.
     *
     * Each output pixel = input + [strength] × (input − mean_of_4_neighbours).
     * Values are clamped to [0, 255].
     *
     * [strength] ≈ 1.5  — moderate: enhances edge contrast for mildly blurred images
     * [strength] ≈ 10.0 — aggressive: drives bar edges toward 0/255, effectively
     *                     binarising the image near transitions; used as a last resort
     *                     for heavily blurred photographs where the bimodal split fails
     *
     * Swift port: same arithmetic; vImage_Sharpen (Accelerate) can be used for
     * performance, but the plain loop is fast enough for typical frame sizes.
     */
    fun sharpen(gray: GrayImage, strength: Float = SHARP_STRENGTH_MODERATE): GrayImage {
        val w = gray.width
        val h = gray.height
        val p = gray.pixels
        val out = IntArray(w * h)
        // Integer fixed-point: strength * 256, then divide by 1024 (= strength/4)
        val str256 = (strength * 256f).toInt()
        for (y in 0 until h) {
            val rowOff = y * w
            val rowAbove = if (y > 0) rowOff - w else rowOff
            val rowBelow = if (y < h - 1) rowOff + w else rowOff
            for (x in 0 until w) {
                val c = p[rowOff + x]
                val n = p[rowAbove + x]
                val s = p[rowBelow + x]
                val e = if (x < w - 1) p[rowOff + x + 1] else c
                val ww = if (x > 0) p[rowOff + x - 1] else c
                // c + strength * (c - (n+s+e+ww)/4)  =  c + str256 * (4c-n-s-e-ww) / 1024
                val result = c + (str256 * (4 * c - n - s - e - ww) + 512) / 1024
                out[rowOff + x] = if (result < 0) 0 else if (result > 255) 255 else result
            }
        }
        return GrayImage(out, w, h)
    }

    /**
     * Estimate the barcode's clockwise skew angle from a barcode-like edge mask.
     *
     * Pipeline:
     *   1. buildDeskewMask — Sobel horizontal gradient → blur → Otsu threshold →
     *      morphological close + erode + dilate.  Produces a binary blob that
     *      approximates the barcode region.
     *   2. collectBoundaryPoints — 4-connected edge pixels of the blob.
     *   3. convexHull — Andrew's monotone chain on the boundary points.
     *   4. Dominant edge — the longest hull edge gives the barcode's tilt angle
     *      (barcodes are rectangular, so the longest edge is the long axis or the
     *      short axis, whichever is more prominent).
     *   5. normalizeDeskewAngle — folds the angle into (−45°, +45°] so the caller
     *      only needs to correct by a small amount.
     *
     * Returns null if the image is too small, the blob is too sparse, or the
     * estimated angle is outside [DESKEW_MIN_ABS_DEGREES, DESKEW_MAX_ABS_DEGREES].
     * A null estimate causes rotationCorrections to fall back to a coarse blind sweep.
     */
    fun estimateDeskew(gray: GrayImage): DeskewEstimate? {
        val mask = buildDeskewMask(gray) ?: return null
        val boundary = collectBoundaryPoints(mask)
        if (boundary.size < 8) return null

        val hull = convexHull(boundary)
        if (hull.size < 2) return null

        var bestAngle = 0.0
        var bestLength = 0.0
        var perimeter = 0.0
        for (i in hull.indices) {
            val a = hull[i]
            val b = hull[(i + 1) % hull.size]
            val dx = (b.x - a.x).toDouble()
            val dy = (b.y - a.y).toDouble()
            val length = sqrt(dx * dx + dy * dy)
            if (length <= 0.0) continue
            perimeter += length
            if (length > bestLength) {
                bestLength = length
                bestAngle = normalizeDeskewAngle(-atan2(dy, dx) * 180.0 / PI)
            }
        }

        if (bestLength < DESKEW_MIN_EDGE_LENGTH || perimeter <= 0.0) return null
        val absDegrees = abs(bestAngle)
        // Reject near-zero angles (barcode is effectively upright — no correction needed)
        // and angles beyond 30° (those are covered by the fixed ROTATION_SWEEP_DEGREES).
        if (absDegrees < DESKEW_MIN_ABS_DEGREES || absDegrees > DESKEW_MAX_ABS_DEGREES) return null

        return DeskewEstimate(
            degreesClockwise = bestAngle,
            coherence = bestLength / perimeter,
            sampleCount = boundary.size
        )
    }

    /** Rotate the image back toward axis alignment using a previously estimated skew. */
    fun deskew(gray: GrayImage, estimate: DeskewEstimate? = estimateDeskew(gray)): GrayImage? {
        val resolved = estimate ?: return null
        return rotate(gray, degreesClockwise = -resolved.degreesClockwise)
    }

    // Heuristic quality score used to pick the winner when multiple decode attempts
    // are tried (different rotations, scales, or deskew corrections).  Higher is
    // better; STRONG_DECODE_SCORE (49) triggers an early exit from the sweep loops.
    // "combined" is worth more because two independent approaches agreed.
    // A full-length (9-digit) result is scored above shorter ones.
    // An all-zeros result is penalised — it is almost always a decode artifact.
    private fun decodeQuality(result: DecodeResult?): Int {
        if (result == null) return Int.MIN_VALUE
        var score = when (result.method) {
            "combined" -> 40
            "row_scan" -> 30
            "zxing_row_scan" -> 30
            else -> 10
        }
        score += min(result.fullDigits.length, 10)
        if (result.fullDigits.isNotEmpty() && result.fullDigits.all { it == '0' }) score -= 20
        return score
    }

    private fun chooseBetterDecode(
        primary: DecodeResult?, secondary: DecodeResult?
    ): DecodeResult? {
        if (primary == null) return secondary
        if (secondary == null) return primary
        return if (decodeQuality(secondary) > decodeQuality(primary)) secondary else primary
    }

    // Fold any angle into the range (−45°, +45°] by taking advantage of barcode
    // symmetry: a barcode rotated 91° looks the same as one rotated −89°, and a
    // 46° tilt is indistinguishable from a −44° tilt when the barcode is symmetric
    // about its long axis.  The two while-loops handle multi-wrap, then the
    // if/else clamps the final ±45° ambiguity.
    private fun normalizeDeskewAngle(degreesClockwise: Double): Double {
        var normalized = degreesClockwise
        while (normalized <= -90.0) normalized += 180.0
        while (normalized > 90.0) normalized -= 180.0
        if (normalized > 45.0) normalized -= 90.0
        else if (normalized < -45.0) normalized += 90.0
        return normalized
    }

    // Build the list of correction angles to try for a given deskew estimate.
    // If an estimate exists: try the exact correction plus ±6° and ±12° around it
    // (in case the hull edge gives a slightly biased angle).  Clamp to ±45° and
    // deduplicate candidates that are within 0.25° of each other.
    // If no estimate: fall back to a coarse blind sweep at ±12° and ±24°.
    private fun rotationCorrections(estimate: DeskewEstimate?): List<Double> {
        val corrections = mutableListOf<Double>()
        val base = estimate?.let { -it.degreesClockwise }
        val deltas = doubleArrayOf(0.0, -6.0, 6.0, -12.0, 12.0)

        if (base != null) {
            for (delta in deltas) {
                val candidate = (base + delta).coerceIn(-45.0, 45.0)
                if (corrections.none { abs(it - candidate) < 0.25 }) corrections.add(candidate)
            }
        } else {
            corrections.addAll(listOf(-24.0, -12.0, 12.0, 24.0))
        }
        return corrections
    }

    // Build a binary mask that approximates the barcode region using morphological
    // image processing.  Steps and why:
    //
    //   1. Sobel-X minus Sobel-Y gradient — prefer horizontal edges (bar edges)
    //      over vertical ones (tag borders).  gx is the 3×3 Sobel-X magnitude;
    //      gy is the 3×3 Sobel-Y magnitude; we take (gx − gy) clamped to ≥0.
    //   2. Box blur (radius 4) — smooth out the gradient response so nearby bars
    //      merge into a single high-energy band.
    //   3. Otsu threshold — automatically separate "barcode area" (high gradient)
    //      from "background" (low gradient) without a hard-coded threshold.
    //   4. Morphological close (dilate then erode with a wide flat kernel 25×7) —
    //      bridges horizontal gaps between individual bars so the whole barcode
    //      region becomes one connected blob.
    //   5. Erosion × 3 (3×3 kernel) — shrink the blob to remove thin noise spurs.
    //   6. Dilation × 4 (3×3 kernel) — re-expand slightly so the hull encompasses
    //      the full barcode extent.
    //
    // Returns null if the resulting mask is entirely empty (no barcode-like content).
    private fun buildDeskewMask(gray: GrayImage): BinaryImage? {
        if (gray.width < 3 || gray.height < 3) return null

        val gradient = IntArray(gray.width * gray.height)
        val gw = gray.width
        val p = gray.pixels
        for (y in 1 until gray.height - 1) {
            val rowAbove = (y - 1) * gw
            val rowCurr = y * gw
            val rowBelow = (y + 1) * gw
            for (x in 1 until gw - 1) {
                val gx = abs(
                    p[rowAbove + x + 1] + 2 * p[rowCurr + x + 1] + p[rowBelow + x + 1] -
                            p[rowAbove + x - 1] - 2 * p[rowCurr + x - 1] - p[rowBelow + x - 1]
                )
                val gy = abs(
                    p[rowBelow + x - 1] + 2 * p[rowBelow + x] + p[rowBelow + x + 1] -
                            p[rowAbove + x - 1] - 2 * p[rowAbove + x] + p[rowAbove + x + 1]
                )
                gradient[rowCurr + x] = (gx - gy).coerceAtLeast(0)
            }
        }

        val blurred = boxBlur(gradient, gray.width, gray.height, radius = 4)
        val threshold = otsuThreshold(blurred)
        var mask = BinaryImage(
            bits = BooleanArray(gray.width * gray.height) { i -> blurred[i] > threshold },
            width = gray.width,
            height = gray.height
        )

        mask = erodeRect(
            dilateRect(mask, DESKEW_MASK_CLOSE_W, DESKEW_MASK_CLOSE_H),
            DESKEW_MASK_CLOSE_W,
            DESKEW_MASK_CLOSE_H
        )
        repeat(DESKEW_MASK_ERODE_PASSES) { mask = erodeRect(mask, 3, 3) }
        repeat(DESKEW_MASK_DILATE_PASSES) { mask = dilateRect(mask, 3, 3) }
        return if (mask.bits.any { it }) mask else null
    }

    // Box blur using a 2-D integral (summed-area) image.
    // Building the integral image is O(w·h); each pixel's box average is then O(1)
    // regardless of radius.  Total cost O(w·h) vs O(w·h·r²) for a naive approach.
    //
    // Integral image layout: (width+1) × (height+1), with a zero-padded top row
    // and left column.  integral[(y+1)*(width+1) + (x+1)] = sum of src[0..y][0..x].
    // rectSum() retrieves any rectangular sum in O(1) via inclusion-exclusion.
    private fun boxBlur(src: IntArray, width: Int, height: Int, radius: Int): IntArray {
        if (radius <= 0) return src.copyOf()
        val integral = LongArray((width + 1) * (height + 1))
        for (y in 0 until height) {
            var rowSum = 0L
            val rowOffset = y * width
            val integralRow = (y + 1) * (width + 1)
            val prevIntegralRow = y * (width + 1)
            for (x in 0 until width) {
                rowSum += src[rowOffset + x].toLong()
                integral[integralRow + x + 1] = integral[prevIntegralRow + x + 1] + rowSum
            }
        }

        val out = IntArray(width * height)
        for (y in 0 until height) {
            val y0 = (y - radius).coerceAtLeast(0)
            val y1 = (y + radius).coerceAtMost(height - 1)
            for (x in 0 until width) {
                val x0 = (x - radius).coerceAtLeast(0)
                val x1 = (x + radius).coerceAtMost(width - 1)
                val sum = rectSum(integral, width, x0, y0, x1, y1)
                val area = (x1 - x0 + 1) * (y1 - y0 + 1)
                out[y * width + x] = (sum / area).toInt()
            }
        }
        return out
    }

    // Otsu's method: find the threshold that maximises between-class variance,
    // i.e. best separates "bright" (background/space) from "dark" (bar/foreground).
    // Runs in O(256) after building the 256-bin histogram — essentially free.
    // between-class variance = w_bg · w_fg · (μ_bg − μ_fg)²
    // Swift: identical algorithm; use a [Int](count: 256) histogram array.
    private fun otsuThreshold(values: IntArray): Int {
        val hist = IntArray(256)
        for (value in values) hist[value.coerceIn(0, 255)]++
        val total = values.size.toDouble()
        if (total == 0.0) return 0

        var sum = 0.0
        for (i in hist.indices) sum += i * hist[i]

        var sumBackground = 0.0
        var weightBackground = 0.0
        var bestThreshold = 0
        var bestVariance = -1.0

        for (i in hist.indices) {
            weightBackground += hist[i]
            if (weightBackground == 0.0) continue
            val weightForeground = total - weightBackground
            if (weightForeground == 0.0) break

            sumBackground += i * hist[i]
            val meanBackground = sumBackground / weightBackground
            val meanForeground = (sum - sumBackground) / weightForeground
            val variance =
                weightBackground * weightForeground * (meanBackground - meanForeground) * (meanBackground - meanForeground)
            if (variance > bestVariance) {
                bestVariance = variance
                bestThreshold = i
            }
        }
        return bestThreshold
    }

    // dilateRect: a pixel is set if ANY pixel in the kernel window is set.
    // erodeRect:  a pixel is set only if ALL pixels in the kernel window are set.
    // Both delegate to rectMorph, which uses the binary integral image for O(1)
    // per-pixel rectangle queries.
    private fun dilateRect(src: BinaryImage, kernelW: Int, kernelH: Int): BinaryImage =
        rectMorph(src, kernelW, kernelH, matchAny = true)

    private fun erodeRect(src: BinaryImage, kernelW: Int, kernelH: Int): BinaryImage =
        rectMorph(src, kernelW, kernelH, matchAny = false)

    // Generic rectangular morphological operation via binary integral image.
    // matchAny=true → dilation (set if sum > 0 in the window).
    // matchAny=false → erosion  (set if sum == full kernel area in the window).
    private fun rectMorph(
        src: BinaryImage, kernelW: Int, kernelH: Int, matchAny: Boolean
    ): BinaryImage {
        val integral = binaryIntegral(src)
        val out = BooleanArray(src.width * src.height)
        val radiusX = kernelW / 2
        val radiusY = kernelH / 2
        val fullArea = kernelW * kernelH

        for (y in 0 until src.height) {
            val y0 = (y - radiusY).coerceAtLeast(0)
            val y1 = (y + radiusY).coerceAtMost(src.height - 1)
            for (x in 0 until src.width) {
                val x0 = (x - radiusX).coerceAtLeast(0)
                val x1 = (x + radiusX).coerceAtMost(src.width - 1)
                val sum = rectSum(integral, src.width, x0, y0, x1, y1)
                out[y * src.width + x] = if (matchAny) sum > 0 else sum == fullArea.toLong()
            }
        }
        return BinaryImage(out, src.width, src.height)
    }

    // Prefix-sum (integral) image for a binary mask.
    // Layout is (width+1) × (height+1) with a sentinel zero row/column at top-left.
    // integral[row * (width+1) + col] = count of true pixels in src[0..row-1][0..col-1].
    private fun binaryIntegral(src: BinaryImage): LongArray {
        val integral = LongArray((src.width + 1) * (src.height + 1))
        for (y in 0 until src.height) {
            var rowSum = 0L
            val integralRow = (y + 1) * (src.width + 1)
            val prevIntegralRow = y * (src.width + 1)
            for (x in 0 until src.width) {
                if (src.bits[y * src.width + x]) rowSum++
                integral[integralRow + x + 1] = integral[prevIntegralRow + x + 1] + rowSum
            }
        }
        return integral
    }

    // Inclusion-exclusion sum of a rectangle [x0,x1] × [y0,y1] (all inclusive)
    // from a (width+1)-stride integral image.  O(1) per query.
    private fun rectSum(integral: LongArray, width: Int, x0: Int, y0: Int, x1: Int, y1: Int): Long {
        val stride = width + 1
        val left = x0
        val right = x1 + 1
        val top = y0
        val bottom = y1 + 1
        return integral[bottom * stride + right] -
                integral[top * stride + right] -
                integral[bottom * stride + left] +
                integral[top * stride + left]
    }

    // Collect the 4-connected boundary pixels of a binary mask (pixels that are
    // true and have at least one false 4-neighbour, or touch the image border).
    // These boundary pixels feed into convexHull to find the dominant edge angle.
    private fun collectBoundaryPoints(src: BinaryImage): List<IntPoint> {
        val points = ArrayList<IntPoint>()
        for (y in 0 until src.height) {
            for (x in 0 until src.width) {
                if (!src.pixel(x, y)) continue
                val boundary =
                    x == 0 || x == src.width - 1 ||
                            y == 0 || y == src.height - 1 ||
                            !src.pixel(x - 1, y) || !src.pixel(x + 1, y) ||
                            !src.pixel(x, y - 1) || !src.pixel(x, y + 1)
                if (boundary) points.add(IntPoint(x, y))
            }
        }
        return points
    }

    // Andrew's monotone chain convex hull — O(n log n).
    // Sorts points lexicographically (x then y), builds lower hull left-to-right
    // then upper hull right-to-left.  The cross product test (cross > 0) keeps only
    // left turns; ≤ 0 means the last point is redundant and is removed.
    // Result is the hull vertices in counter-clockwise order.
    private fun convexHull(points: List<IntPoint>): List<IntPoint> {
        val sorted = points.distinct().sortedWith(compareBy<IntPoint> { it.x }.thenBy { it.y })
        if (sorted.size <= 1) return sorted

        val lower = ArrayList<IntPoint>()
        for (point in sorted) {
            while (lower.size >= 2 && cross(
                    lower[lower.size - 2], lower[lower.size - 1], point
                ) <= 0L
            ) {
                lower.removeAt(lower.lastIndex)
            }
            lower.add(point)
        }

        val upper = ArrayList<IntPoint>()
        for (i in sorted.indices.reversed()) {
            val point = sorted[i]
            while (upper.size >= 2 && cross(
                    upper[upper.size - 2], upper[upper.size - 1], point
                ) <= 0L
            ) {
                upper.removeAt(upper.lastIndex)
            }
            upper.add(point)
        }

        lower.removeAt(lower.lastIndex)
        upper.removeAt(upper.lastIndex)
        return lower + upper
    }

    // 2-D cross product of vectors AB and AC.  Positive → left turn (CCW);
    // zero → collinear; negative → right turn (CW).  Used by convexHull to
    // decide whether the current chain makes a valid left turn.
    private fun cross(a: IntPoint, b: IntPoint, c: IntPoint): Long =
        (b.x - a.x).toLong() * (c.y - a.y).toLong() - (b.y - a.y).toLong() * (c.x - a.x).toLong()

    // Bilinear interpolation for sub-pixel source sampling.
    //
    // Given a fractional source coordinate (x, y), blends the four surrounding
    // integer-grid pixels using linear weights:
    //   top    = lerp(p00, p10, fx)
    //   bottom = lerp(p01, p11, fx)
    //   result = lerp(top, bottom, fy)
    // where fx = x − floor(x), fy = y − floor(y).
    //
    // Pixels outside the source bounds return [background] (white = 255).
    // This is called for every output pixel in both rotate() and scale(), so
    // keeping it branchless pays off; the bounds check is the only branch.
    // Bilinear interpolation using 10-bit fixed-point arithmetic.
    // Avoids all Double math in the inner loop — critical for mobile rotate/scale.
    // Max intermediate: 1024 * 1024 * 255 = 267,386,880 — fits in Int.
    private fun sampleBilinear(src: GrayImage, x: Double, y: Double, background: Int): Int {
        if (x < 0.0 || y < 0.0 || x > src.width - 1.0 || y > src.height - 1.0) return background

        val x0 = x.toInt()
        val y0 = y.toInt()
        val x1 = if (x0 + 1 < src.width) x0 + 1 else x0
        val y1 = if (y0 + 1 < src.height) y0 + 1 else y0

        // 10-bit fractional parts (0..1024)
        val fx = ((x - x0) * 1024.0).toInt()
        val fy = ((y - y0) * 1024.0).toInt()
        val ifx = 1024 - fx
        val ify = 1024 - fy

        val row0 = y0 * src.width
        val row1 = y1 * src.width
        val p00 = src.pixels[row0 + x0]
        val p10 = src.pixels[row0 + x1]
        val p01 = src.pixels[row1 + x0]
        val p11 = src.pixels[row1 + x1]

        // Weighted blend: (ify*(ifx*p00 + fx*p10) + fy*(ifx*p01 + fx*p11) + 524288) >> 20
        return (ify * (ifx * p00 + fx * p10) + fy * (ifx * p01 + fx * p11) + 524288) shr 20
    }

    // ── Step 1: ImageProxy → GrayImage ──────────────────────────────────────────
    //
    // Android CameraX delivers frames in YUV_420_888 format.  The luma (Y) plane
    // is plane index 0 and is a full-resolution grayscale image — exactly what the
    // decoder needs.  The U and V chroma planes (indices 1 and 2) are ignored.
    //
    // rowStride is the byte distance between the start of one row and the next
    // (may be larger than width due to padding).  pixelStride is the byte distance
    // between adjacent pixels in the same row (usually 1 for the Y plane).
    //
    // Swift / iOS equivalent:
    //   Lock the CVPixelBuffer, get plane 0 base address and bytesPerRow, then read
    //   each luma byte directly.  Typical format: kCVPixelFormatType_420YpCbCr8BiPlanar-
    //   VideoRange, where plane 0 is the full-resolution luma (Y) channel.
    //
    //     CVPixelBufferLockBaseAddress(buffer, .readOnly)
    //     let lumaBase   = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!
    //     let lumaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
    //     // read lumaBase[y * lumaStride + x] for each pixel
    //     CVPixelBufferUnlockBaseAddress(buffer, .readOnly)

    private data class BinaryImage(val bits: BooleanArray, val width: Int, val height: Int) {
        fun pixel(x: Int, y: Int): Boolean = bits[y * width + x]
    }

    private data class IntPoint(val x: Int, val y: Int)

    fun extractGrayscale(proxy: ImageProxy): GrayImage? {
        if (proxy.format != ImageFormat.YUV_420_888) return null
        val plane = proxy.planes[0]
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val buf = plane.buffer
        val crop = proxy.cropRect
        val w: Int
        val h: Int
        val x0: Int
        val y0: Int
        if (crop != null) {
            w = crop.width()
            h = crop.height()
            x0 = crop.left
            y0 = crop.top
        } else {
            w = proxy.width
            h = proxy.height
            x0 = 0
            y0 = 0
        }
        val pixels = IntArray(w * h)
        for (y in 0 until h)
            for (x in 0 until w)
                pixels[y * w + x] = buf[(y0 + y) * rowStride + (x0 + x) * pixelStride].toInt() and 0xFF
        return GrayImage(pixels, w, h)
    }

    // ── Step 2: Barcode band isolation ──────────────────────────────────────────
    //
    // Barcodes produce many dark/light transitions per row.  Score every row by
    // counting transitions (rowTransitions), then find the contiguous vertical band
    // where scores are consistently above 1/3 of the maximum.  That band is the
    // barcode strip.  Crop to it (with a 4-pixel margin) to give the decoder a
    // tight input; this reduces noise from the surrounding tag artwork.
    //
    // Returns null (caller falls back to full image) if:
    //   • the maximum score is below 10 (image has no meaningful transitions), or
    //   • the best band is fewer than 5 rows tall (too narrow to be a real barcode).

    private fun isolateBarcode(img: GrayImage): GrayImage? {
        val w = img.width
        val h = img.height
        val scores = IntArray(h) { y -> rowTransitions(img, y) }
        val maxScore = scores.maxOrNull() ?: return null
        if (maxScore < 10) return null
        val threshold = maxScore / 3

        var bestStart = 0
        var bestEnd = 0
        var bestSum = 0
        var bandStart = -1
        var bandSum = 0

        for (y in 0 until h) {
            if (scores[y] > threshold) {
                if (bandStart < 0) bandStart = y
                bandSum += scores[y]
            } else {
                if (bandStart >= 0 && y - bandStart >= 5 && bandSum > bestSum) {
                    bestSum = bandSum; bestStart = bandStart; bestEnd = y
                }
                bandStart = -1; bandSum = 0
            }
        }
        if (bandStart >= 0 && h - bandStart >= 5 && bandSum > bestSum) {
            bestStart = bandStart; bestEnd = h
        }
        if (bestEnd - bestStart < 5) return null

        val top = (bestStart - 4).coerceAtLeast(0)
        val bottom = (bestEnd + 4).coerceAtMost(h)
        val cropH = bottom - top
        val out = IntArray(w * cropH) { i -> img.pixel(i % w, top + i / w) }
        return GrayImage(out, w, cropH)
    }

    // Count dark↔light transitions in row y after binarising at the row's midpoint.
    // Rows with contrast below 30 grey levels (mx − mn < 30) score 0 and are
    // skipped by isolateBarcode — they contain no useful barcode information.
    // A 9-digit MSI barcode produces ≈77 transitions; real rows score 55–110.
    private fun rowTransitions(img: GrayImage, y: Int): Int {
        val w = img.width
        val p = img.pixels
        val rowOff = y * w
        var mn = 255
        var mx = 0
        for (x in 0 until w) {
            val v = p[rowOff + x]
            if (v < mn) mn = v
            if (v > mx) mx = v
        }
        if (mx - mn < 30) return 0
        val thr = (mn + mx) / 2
        var prev = p[rowOff] < thr
        var count = 0
        for (x in 1 until w) {
            val cur = p[rowOff + x] < thr
            if (cur != prev) {
                count++; prev = cur
            }
        }
        return count
    }

    // ── Step 3: Combined decode ──────────────────────────────────────────────────
    //
    // Approaches A (bimodal row scan) and B (ZXing run-counter) run sequentially,
    // then their results are reconciled:
    //   • Both agree on the first 7 digits → "combined"   (highest confidence)
    //   • Only A succeeds               → "row_scan"
    //   • Only B succeeds               → "zxing_row_scan"
    //   • Neither succeeds              → column greedy fallback → "col_greedy"
    //
    // Sequential execution avoids the coroutine overhead of runBlocking/async,
    // which is significant on the single-thread camera analyzer executor.

    private fun decodeBarcode(img: GrayImage): DecodeResult? {
        val bimodal = rowScan(img)?.let { luhnNormalize(it) }
        val zxing = zxingRowScan(img)?.let { luhnNormalize(it) }

        if (bimodal != null && zxing != null && bimodal.take(7) == zxing.take(7))
            return DecodeResult(bimodal.take(7), bimodal, "combined")

        if (bimodal != null) return DecodeResult(bimodal.take(7), bimodal, "row_scan")
        if (zxing != null) return DecodeResult(zxing.take(7), zxing, "zxing_row_scan")

        colGreedy(img)?.let { v -> return DecodeResult(v.take(7), v, "col_greedy") }
        return null
    }

    // ── Approach A: Bimodal row scan ─────────────────────────────────────────────
    //
    // For each image row and each binarisation threshold (20%–80% of the row's
    // contrast range in 5% steps):
    //   1. Binarise the row.
    //   2. Run-length encode to get (isDark, width) pairs.
    //   3. Find the bimodal split between narrow and wide dark bars.
    //   4. Decode the bit pattern using BITS_TO_DIGIT.
    //   5. Vote for the resulting digit string across all rows and thresholds.
    // The most-voted Luhn-valid string wins.  Requiring ≥2 votes prevents a single
    // noisy row from producing a false positive.

    // Run-length encode directly from GrayImage pixels, avoiding a per-threshold IntArray
    // allocation.  Reads pixels directly instead of creating a binarised intermediate array.
    private fun rleFromRow(img: GrayImage, y: Int, thr: Int): List<Pair<Boolean, Int>> {
        val w = img.width
        val p = img.pixels
        val rowOff = y * w
        val runs = mutableListOf<Pair<Boolean, Int>>()
        var isDark = p[rowOff] < thr
        var count = 1
        for (x in 1 until w) {
            val d = p[rowOff + x] < thr
            if (d == isDark) count++
            else { runs.add(isDark to count); isDark = d; count = 1 }
        }
        runs.add(isDark to count)
        return runs
    }

    // Find the threshold that best separates narrow bars from wide bars in [darkWidths].
    //
    // MSI Plessey bars are either narrow (N) or wide (W ≈ 2×N).  All dark-run widths
    // from a single row therefore cluster into two groups.  We find the largest gap
    // between consecutive distinct widths in the sorted list — that gap's midpoint is
    // the split.  The two outliers (widest bars) are dropped first to handle guard bars
    // or ink spread at the extremes.
    //
    // Returns null if there are fewer than 6 dark bars (not enough for a barcode) or
    // all bars have the same width (can't distinguish narrow from wide).
    private fun bimodalSplit(darkWidths: List<Int>): Double? {
        if (darkWidths.size < 6) return null
        val sorted = darkWidths.toIntArray()
        sorted.sort()
        // Drop last 2 (widest bars — guard bars or ink spread)
        val end = sorted.size - 2
        var bestGap = 0
        var bestSplit = 0.0
        var prev = sorted[0]
        for (i in 1 until end) {
            val v = sorted[i]
            if (v != prev) {
                val gap = v - prev
                if (gap > bestGap) {
                    bestGap = gap; bestSplit = (prev + v) / 2.0
                }
                prev = v
            }
        }
        return if (bestGap >= 1) bestSplit else null
    }

    // Decode the run-length list into a digit string using the bimodal [split].
    //
    // Start guard detection: scan for the first wide dark run (run.second > split).
    // That is the start guard's wide bar; the digit data begins 2 runs later
    // (pos = guardIndex + 2, skipping the narrow space after the guard).
    //
    // Digit loop: each digit occupies 8 runs (4 bar/space pairs at indices pos…pos+7).
    // Only the even-indexed runs (bars, isDark=true) are compared to split; the
    // odd-indexed runs (spaces) are consumed but not tested.  pos advances by 8 per
    // digit.  Loop ends when BITS_TO_DIGIT returns null (unknown pattern) or the
    // string runs out of runs.
    //
    // Returns null if the start guard is not found or fewer than 7 digits decoded.
    private fun decodeRuns(runs: List<Pair<Boolean, Int>>, split: Double): String? {
        var pos = -1
        for (i in runs.indices) {
            if (runs[i].first && runs[i].second > split) {
                pos = i + 2; break
            }
        }
        if (pos < 0) return null

        val sb = StringBuilder()
        while (pos + 6 < runs.size) {
            val bitIdx = (if (runs[pos].second > split) 8 else 0) or
                    (if (runs[pos + 2].second > split) 4 else 0) or
                    (if (runs[pos + 4].second > split) 2 else 0) or
                    (if (runs[pos + 6].second > split) 1 else 0)
            val digit = BITS_LOOKUP[bitIdx]
            if (digit < 0) break
            sb.append(('0'.code + digit).toChar())
            pos += 8
        }
        return if (sb.length >= 7) sb.toString() else null
    }

    // Approach A entry point.  Sweeps all rows × 13 threshold percentages.
    // validVotes accumulates Luhn-valid strings; rawVotes accumulates everything
    // else.  The winner must appear ≥2 times to be returned (single-row flukes
    // are rejected).  Luhn-valid strings are preferred over raw strings.
    private fun rowScan(img: GrayImage): String? {
        val w = img.width
        val h = img.height
        val p = img.pixels
        val validVotes = mutableMapOf<String, Int>()
        val rawVotes = mutableMapOf<String, Int>()
        var maxValidVotes = 0
        var rowsScanned = 0

        for (y in 0 until h) {
            val rowOff = y * w
            var mn = 255
            var mx = 0
            for (x in 0 until w) {
                val v = p[rowOff + x]
                if (v < mn) mn = v
                if (v > mx) mx = v
            }
            if (mx - mn < 30) continue
            rowsScanned++

            for (thrPct in 20..80 step 5) {
                val thr = mn + (mx - mn) * thrPct / 100
                val runs = rleFromRow(img, y, thr)
                if (runs.size !in 55..110) continue  // 9-digit MSI produces ≈77 transitions; allow ±30 slack
                val darkWidths = runs.filter { it.first }.map { it.second }
                val split = bimodalSplit(darkWidths) ?: continue
                val value = decodeRuns(runs, split) ?: continue
                if (value.length < 7) continue
                if (value.all { it == value[0] }) continue  // all-same-digit = decode artifact

                val normed = luhnNormalize(value)
                if (normed != null) {
                    val cnt = (validVotes[normed] ?: 0) + 1
                    validVotes[normed] = cnt
                    if (cnt > maxValidVotes) maxValidVotes = cnt
                } else {
                    rawVotes[value] = (rawVotes[value] ?: 0) + 1
                }
            }
            // Early exit: once enough high-contrast rows have been scanned and a
            // Luhn-valid candidate has strong consensus, stop early.  The minimum
            // row guard prevents a wrong value from winning before the correct one
            // has had a chance to accumulate votes.
            if (maxValidVotes >= 20 && rowsScanned >= 15) break
        }

        validVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        rawVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        return null
    }

    // ── Approach B: ZXing run-counter row scan
    //
    // Ported from ZXing.Net MSIReader.cs.  Unlike Approach A (which finds the
    // narrow/wide split from ALL dark bars in the row), Approach B establishes the
    // pivot from the start guard's two runs alone, then applies that fixed pivot to
    // every character.  This is more brittle in highly variable contrast but decodes
    // some frames that confuse the bimodal split.
    //
    // avgWidth is stored as a fixed-point integer (real_width × 256) so that the
    // narrow/wide classification in zxingToPattern avoids floating-point division:
    //   (counter << 8) < avgWidth  →  narrow;  ≥ avgWidth  →  wide.
    //
    // The same multi-threshold voting scheme as Approach A is used here.

    // Approach B entry point.  Same voting structure as rowScan.
    private fun zxingRowScan(img: GrayImage): String? {
        val w = img.width
        val h = img.height
        val p = img.pixels
        val validVotes = mutableMapOf<String, Int>()
        val rawVotes = mutableMapOf<String, Int>()
        var maxValidVotes = 0
        var rowsScanned = 0

        for (y in 0 until h) {
            val rowOff = y * w
            var mn = 255
            var mx = 0
            for (x in 0 until w) {
                val v = p[rowOff + x]
                if (v < mn) mn = v
                if (v > mx) mx = v
            }
            if (mx - mn < 30) continue
            rowsScanned++

            val bin = BooleanArray(w) // reusable across thresholds for this row
            for (thrPct in 20..80 step 5) {
                val thr = mn + (mx - mn) * thrPct / 100
                for (x in 0 until w) bin[x] = p[rowOff + x] < thr
                val bitRow = BitRow(bin)
                val decoded = zxingDecodeRow(bitRow) ?: continue
                if (decoded.all { it == decoded[0] }) continue  // all-same-digit = decode artifact

                val normed = luhnNormalize(decoded)
                if (normed != null) {
                    val cnt = (validVotes[normed] ?: 0) + 1
                    validVotes[normed] = cnt
                    if (cnt > maxValidVotes) maxValidVotes = cnt
                } else {
                    rawVotes[decoded] = (rawVotes[decoded] ?: 0) + 1
                }
            }
            if (maxValidVotes >= 20 && rowsScanned >= 15) break
        }

        validVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        rawVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        return null
    }

    // ZXing row decoder state machine.
    // 1. zxingFindStart locates the start guard and returns [startX, endX, avgWidth].
    // 2. Loop: zxingRecordPattern fills `counters` with run widths for the next 8 runs.
    //    zxingToPattern converts counters to a bit pattern; zxingPatternToChar maps it
    //    to a character.  If the pattern doesn't match any known character, try to
    //    match the end guard via zxingFindEnd and stop cleanly.
    // 3. After each character, advance nextStart past the 8 runs and skip whitespace.
    // Returns the decoded string if ≥3 characters were found (partial result still
    // useful to callers, which check for ≥7 or ≥8 via Luhn validation).
    private fun zxingDecodeRow(row: BitRow): String? {
        val counters = IntArray(8)
        val start = zxingFindStart(row, counters) ?: return null
        val avgWidth = start[2]
        var nextStart = row.getNextSet(start[1])
        val result = StringBuilder()

        while (true) {
            if (!zxingRecordPattern(row, nextStart, counters, 8)) {
                zxingFindEnd(row, nextStart, counters, avgWidth) ?: return null
                break
            }
            val pattern = zxingToPattern(counters, 8, avgWidth)
            val ch = zxingPatternToChar(pattern)
            if (ch == null) {
                zxingFindEnd(row, nextStart, counters, avgWidth) ?: return null
                break
            }
            result.append(ch)
            for (c in counters) nextStart += c
            nextStart = row.getNextSet(nextStart)
        }

        return if (result.length >= 3) result.toString() else null
    }

    /** Find start guard: wide bar + narrow space (0x06).
     *
     *  Scans left-to-right looking for two consecutive runs where bar/space ratio
     *  is in [1.5, 5.0] and the 2-counter pattern matches ZXING_START.  Also checks
     *  that a quiet zone (blank space) precedes the guard.
     *  Returns intArrayOf(startX, endX, avgWidth256) or null if not found.
     *  avgWidth256 is the fixed-point average of the two guard run widths × 256,
     *  used as the narrow/wide pivot for the rest of the row.
     */
    private fun zxingFindStart(row: BitRow, counters: IntArray): IntArray? {
        val width = row.size
        val rowOffset = row.getNextSet(0)
        var cp = 0
        var ps = rowOffset
        var isWhite = false

        counters[0] = 0; counters[1] = 0

        for (i in rowOffset until width) {
            if (row[i] xor isWhite) {
                counters[cp]++
            } else {
                if (cp == 1) {
                    if (counters[1] != 0) {
                        val factor = counters[0].toFloat() / counters[1].toFloat()
                        if (factor in 1.5f..5.0f) {
                            val avgWidth = zxingCalcAvgWidth(counters, 2)
                            if (zxingToPattern(counters, 2, avgWidth) == ZXING_START) {
                                val quietStart = max(0, ps - ((i - ps) shr 1))
                                if (row.isRange(quietStart, ps, false))
                                    return intArrayOf(ps, i, avgWidth)
                            }
                        }
                    }
                    ps += counters[0] + counters[1]
                    counters[0] = 0; counters[1] = 0
                    cp--
                } else {
                    cp++
                }
                counters[cp] = 1
                isWhite = !isWhite
            }
        }
        return null
    }

    /** Find end guard: narrow bar + wide space + narrow bar (0x09).
     *
     *  Matches 3 consecutive runs starting at rowOffset where the middle run
     *  (space) is 1.5–5× the width of the outer runs (bars), and the 3-counter
     *  pattern matches ZXING_END.  Also verifies a trailing quiet zone.
     *  Returns intArrayOf(startX, endX) or null.
     *  Note: returns null immediately if the pattern doesn't match — the end guard
     *  is the only legal termination so there's no point scanning further.
     */
    private fun zxingFindEnd(
        row: BitRow, rowOffset: Int, counters: IntArray, avgWidth: Int
    ): IntArray? {
        val width = row.size
        var cp = 0
        var ps = rowOffset
        var isWhite = false

        counters[0] = 0; counters[1] = 0; counters[2] = 0

        for (i in rowOffset until width) {
            if (row[i] xor isWhite) {
                counters[cp]++
            } else {
                if (cp == 2) {
                    if (counters[0] != 0) {
                        val factor = counters[1].toFloat() / counters[0].toFloat()
                        if (factor in 1.5f..5.0f && zxingToPattern(
                                counters, 3, avgWidth
                            ) == ZXING_END
                        ) {
                            val minEnd = min(row.size - 1, i + ((i - ps) shr 1))
                            if (row.isRange(i, minEnd, false))
                                return intArrayOf(ps, i)
                        }
                    }
                    return null
                }
                cp++
                counters[cp] = 1
                isWhite = !isWhite
            }
        }
        return null
    }

    // Record the widths of the next [n] alternating runs starting at [start] into
    // [counters].  Returns true if exactly n transitions were recorded (or if the
    // last run runs to the end of the row, which counts as complete).
    private fun zxingRecordPattern(row: BitRow, start: Int, counters: IntArray, n: Int): Boolean {
        for (i in 0 until n) counters[i] = 0
        if (start >= row.size) return false
        var isWhite = !row[start]
        var cp = 0
        var i = start
        while (i < row.size) {
            if (row[i] xor isWhite) {
                counters[cp]++
            } else {
                cp++
                if (cp == n) break
                counters[cp] = 1
                isWhite = !isWhite
            }
            i++
        }
        return cp == n || (cp == n - 1 && i == row.size)
    }

    // Compute the fixed-point average of min and max among the first [len] counters.
    // Result is ((max + min) / 2) × 256, stored as an integer to avoid float division
    // in the hot path of zxingToPattern.  Multiplied by 256 so the caller can test
    // (counter << 8) < avgWidth instead of (counter < avgWidth / 256.0).
    private fun zxingCalcAvgWidth(counters: IntArray, len: Int): Int {
        var mn = Int.MAX_VALUE
        var mx = 0
        for (i in 0 until len) {
            if (counters[i] < mn) mn = counters[i]
            if (counters[i] > mx) mx = counters[i]
        }
        return ((mx shl 8) + (mn shl 8)) / 2
    }

    // Encode [len] run-width counters into a compact bit pattern using [avgWidth256]
    // as the narrow/wide pivot.  Builds the pattern LSB-first alternating bar/space:
    //   narrow run → append `bit`       (1 bit)
    //   wide run   → append `doubleBit` (2 bits = 11 or 00, alternating)
    // `bit` toggles 0↔1 and `doubleBit` toggles 3↔0 on each iteration so that bars
    // and spaces contribute different codes.  The resulting integer is compared
    // against ZXING_CHARACTER_ENCODINGS (or ZXING_START / ZXING_END for guards).
    private fun zxingToPattern(counters: IntArray, len: Int, avgWidth: Int): Int {
        var pattern = 0
        var bit = 1
        var doubleBit = 3
        for (i in 0 until len) {
            if ((counters[i] shl 8) < avgWidth)
                pattern = (pattern shl 1) or bit
            else
                pattern = (pattern shl 2) or doubleBit
            bit = bit xor 1
            doubleBit = doubleBit xor 3
        }
        return pattern
    }

    // Linear scan of ZXING_CHARACTER_ENCODINGS (10 elements) — acceptable cost.
    private fun zxingPatternToChar(pattern: Int): Char? {
        for (i in ZXING_CHARACTER_ENCODINGS.indices)
            if (ZXING_CHARACTER_ENCODINGS[i] == pattern) return ZXING_ALPHABET[i]
        return null
    }

    // ── Fallback: Column greedy ──────────────────────────────────────────────────
    //
    // Used only when both row-scan approaches fail (e.g. the image has very low
    // per-row contrast or the barcode band is too narrow for reliable RLE).
    //
    // Instead of row-by-row analysis, compress the image to a 1-D column signal:
    // for each x, take the 10th percentile pixel value across all rows.  Dark bars
    // produce a low (near 0) column value; bright spaces produce a high (near 1)
    // value after normalization.  Then sweep a grid of narrow-bar width (N) and
    // wide-bar width (W = 2–3×N) and starting offset, trying to decode the signal
    // directly using average-energy thresholding per bar/space pair.

    // Compute the normalised 10th-percentile column darkness signal.
    // pctIdx ≈ 10% of the column height; sorting each column and reading that index
    // gives a stable "darkest typical pixel" for the column, ignoring outlier noise.
    // The result is normalised to [0, 1] where 1 = darkest (bar), 0 = lightest (space).
    // Returns null if the overall contrast is below 10 grey levels (flat image).
    private fun colSignal(img: GrayImage): DoubleArray? {
        val w = img.width
        val h = img.height
        val pctIdx = (h * 0.10).toInt().coerceAtLeast(0)
        val sig = DoubleArray(w) { x ->
            val col = DoubleArray(h) { y -> img.pixel(x, y).toDouble() }.also { it.sort() }
            col[pctIdx]
        }
        val mn = sig.minOrNull()!!
        val mx = sig.maxOrNull()!!
        if (mx - mn < 10.0) return null
        return DoubleArray(w) { x -> 1.0 - (sig[x] - mn) / (mx - mn) }
    }

    // Attempt to decode the column signal with fixed narrow-bar width N, wide-bar
    // width W, and start offset [offset] (in pixels from the left edge).
    // Each digit occupies 4 bar/space pairs = 4 × (N+W) pixels.  For each pair,
    // average the signal over the combined (N+W)-pixel window; if the average > 0.5
    // the pair is classified as "wide bar" (bit=1), else "narrow bar" (bit=0).
    // Returns null if fewer than 7 digits are decoded or the signal runs out.
    private fun greedy(sig: DoubleArray, N: Double, W: Double, offset: Int): String? {
        val nPx = N.roundToInt()
        val wPx = W.roundToInt()
        val pw = nPx + wPx
        var x = offset + wPx + nPx
        val sb = StringBuilder()
        while (x + pw * 4 <= sig.size && sb.length < 12) {
            var bitIdx = 0
            for (j in 0 until 4) {
                val s = x + j * pw
                val e = (s + pw).coerceAtMost(sig.size)
                var sum = 0.0; for (k in s until e) sum += sig[k]
                if (sum / (e - s) > 0.5) bitIdx = bitIdx or (8 shr j)
            }
            val digit = BITS_LOOKUP[bitIdx]
            if (digit < 0) break
            sb.append(('0'.code + digit).toChar())
            x += pw * 4
        }
        return if (sb.length >= 7) sb.toString() else null
    }

    // Sweep the parameter grid: N (narrow bar px) from 1.5 to 5.0, W/N ratio from
    // 2.0 to 3.0, and horizontal offset from 0 to min(imageWidth−minBarcode, 80).
    // Collect Luhn-valid votes; the most-voted string wins.  No minimum vote count
    // is required here (unlike row-scan) because the parameter sweep itself provides
    // redundancy — the same correct value will win across many (N, W, offset) combos.
    private fun colGreedy(img: GrayImage): String? {
        val sig = colSignal(img) ?: return null
        val w = sig.size
        val votes = mutableMapOf<String, Int>()

        var N = 1.5
        while (N <= 5.0) {
            for (ratioStep in 0..2) {
                val W = N * (2.0 + ratioStep * 0.5)
                val nPx = N.roundToInt()
                val wPx = W.roundToInt()
                val pw = nPx + wPx
                val minW = wPx + nPx + 7 * 4 * pw + nPx + wPx + nPx
                if (minW > w) continue
                val maxOff = min(w - minW, 80)
                for (off in 0..maxOff step 5) {
                    val r = greedy(sig, N, W, off) ?: continue
                    if (r.all { it == r[0] }) continue  // all-same-digit = decode artifact
                    luhnNormalize(r)?.let { votes[it] = (votes[it] ?: 0) + 1 }
                }
            }
            N += 0.5
        }
        return votes.maxByOrNull { it.value }?.key
    }

    // ── Luhn check digit ─────────────────────────────────────────────────────────
    //
    // The Luhn algorithm (ISO/IEC 7812) validates the last digit of a numeric string.
    // Working right-to-left from the second-to-last digit:
    //   • Digits at even positions (0-indexed from right) are doubled; if the result
    //     exceeds 9, subtract 9 (equivalent to summing the two decimal digits).
    //   • Digits at odd positions are taken as-is.
    // The check digit is chosen so that the total mod 10 == 0.
    //
    // luhnCheck(prefix) returns what the check digit SHOULD be for [prefix].
    // luhnNormalize(s) scans from the end of [s] to find a suffix that is a valid
    // check digit for the preceding digits, returning the trimmed string if found.

    // Compute the Luhn check digit for the string [digits] (which must NOT include
    // the check digit itself — it will be appended by the caller if needed).
    private fun luhnCheck(digits: String): Char {
        var total = 0
        digits.reversed().forEachIndexed { i, ch ->
            var d = ch.digitToInt()
            if (i % 2 == 0) {
                d *= 2; if (d > 9) d -= 9
            }
            total += d
        }
        return '0' + (10 - total % 10) % 10
    }

    // Try to find a Luhn-valid suffix in [s] by testing lengths 10, 9, and 8
    // (longest first).  MSI Plessey barcodes on Publix tags are 9 characters
    // (8 data + 1 check), so 9 is the expected hit.  Lengths 10 and 8 tolerate
    // minor decode over- or under-runs.  Returns null if none of the lengths match
    // or if the result is all zeros (a common decode artifact on blank/underexposed frames).
    private fun luhnNormalize(s: String): String? {
        for (n in min(s.length, 10) downTo 8) {
            if (luhnCheck(s.substring(0, n - 1)) == s[n - 1]) {
                val result = s.substring(0, n)
                if (result.all { it == '0' }) return null
                return result
            }
        }
        return null
    }
}

// ── BitRow ────────────────────────────────────────────────────────────────────
//
// A thin wrapper around a binarised image row, used exclusively by Approach B.
// Provides the three helper methods that ZXing's MSIReader.cs relies on:
//   get(i)          — is pixel i dark (true = dark bar)?
//   getNextSet(from) — index of the next dark pixel at or after [from].
//   isRange(start, end, value) — are all pixels in [start, end) equal to [value]?
//                                Used to verify quiet zones around guards.
//
// Swift port: model as a struct with a [Bool] array and the same three methods.

private class BitRow(val bits: BooleanArray) {
    val size: Int get() = bits.size
    operator fun get(i: Int): Boolean = bits[i]
    fun getNextSet(from: Int): Int {
        for (i in from until bits.size) if (bits[i]) return i
        return bits.size
    }
    fun isRange(start: Int, end: Int, value: Boolean): Boolean {
        if (start >= end) return true
        for (i in start until end) if (bits[i] != value) return false
        return true
    }
}
