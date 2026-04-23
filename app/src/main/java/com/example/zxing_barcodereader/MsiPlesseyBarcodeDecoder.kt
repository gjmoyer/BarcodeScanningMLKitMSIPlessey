package com.example.zxing_barcodereader

import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * MSI Plessey barcode isolator and decoder for Android / CameraX.
 *
 * Drop-in replacement for the Python pipeline (isolate_barcodes.py +
 * decode_barcodes.py).  No OpenCV or additional native libraries required —
 * runs entirely in Kotlin/JVM using the raw YUV pixel data from CameraX.
 *
 * ── Usage ──────────────────────────────────────────────────────────────────────
 *
 *   // Inside your ImageAnalysis.Analyzer (already on a background thread):
 *   override fun analyze(imageProxy: ImageProxy) {
 *       val result = MsiPlesseyBarcodeDecoder.decode(imageProxy)
 *       imageProxy.close()               // always close after decode()
 *       result?.let { runOnUiThread { showResult(it.digits7) } }
 *   }
 *
 * ── Algorithm ──────────────────────────────────────────────────────────────────
 *
 *   1. Extract the Y (luminance) plane of the YUV_420_888 ImageProxy → grayscale
 *   2. Score every row by its dark↔light transition count; isolate the barcode
 *      band as the tallest contiguous high-scoring region
 *   3. Primary decoder  — row scan: binarise every row at 13 thresholds, run-
 *      length encode, bimodal-split to separate narrow/wide modules, decode
 *      MSI Plessey digit groups, vote across all (row, threshold) combinations
 *   4. Fallback decoder — column greedy: build a per-column 10th-percentile
 *      darkness signal, sweep a grid of (N, W, offset) parameters and sample
 *      expected bit-pair windows positionally
 *   5. Luhn check-digit validation rejects false positives in both decoders
 *
 * ── Notes ──────────────────────────────────────────────────────────────────────
 *
 *   • Rotation: ImageProxy.imageInfo.rotationDegrees tells you the frame
 *     rotation.  For a landscape barcode held in portrait mode you may need to
 *     transpose the pixel grid before calling decode().  A helper is provided:
 *     MsiPlesseyBarcodeDecoder.rotateCW90(imageProxy) returns a pre-rotated
 *     GrayImage you can pass to decodeGray() directly.
 *
 *   • Performance: call from a background thread (ImageAnalysis.Analyzer
 *     already guarantees this).  On modern mid-range Android hardware the row
 *     scan typically completes in < 30 ms for a 720 × 1280 frame.
 *
 *   • Image format: ImageAnalysis defaults to YUV_420_888.  If you configured
 *     OUTPUT_IMAGE_FORMAT_RGBA_8888 instead, use decodeGray() after converting
 *     the bitmap to a GrayImage manually.
 */
object MsiPlesseyBarcodeDecoder {

    // ── Result ──────────────────────────────────────────────────────────────────

    data class DecodeResult(
        /** First 7 significant digits — the useful payload for Publix shelf tags. */
        val digits7: String,
        /** Full decoded string (typically 9 chars = 8 data + 1 Luhn check digit). */
        val fullDigits: String,
        /** Which internal method succeeded: "row_scan" or "col_greedy". */
        val method: String
    )

    // ── MSI Plessey bit-pattern ↔ digit mapping ─────────────────────────────────
    // Each digit is encoded as 4 bit-pairs (MSB first).
    //   bit = 1  →  wide bar  (W) + narrow space (N)
    //   bit = 0  →  narrow bar (N) + wide space  (W)

    private val BITS_TO_DIGIT: Map<List<Int>, Int> = mapOf(
        listOf(0, 0, 0, 0) to 0,  listOf(0, 0, 0, 1) to 1,
        listOf(0, 0, 1, 0) to 2,  listOf(0, 0, 1, 1) to 3,
        listOf(0, 1, 0, 0) to 4,  listOf(0, 1, 0, 1) to 5,
        listOf(0, 1, 1, 0) to 6,  listOf(0, 1, 1, 1) to 7,
        listOf(1, 0, 0, 0) to 8,  listOf(1, 0, 0, 1) to 9
    )

    // ── Public API ──────────────────────────────────────────────────────────────

    /**
     * Main entry point.  Accepts a CameraX [ImageProxy] and returns a
     * [DecodeResult] with the first 7 digits, or null if no valid barcode found.
     *
     * Automatically handles [ImageProxy.cropRect] and [ImageProxy.imageInfo.rotationDegrees].
     *
     * The [ImageProxy] is read but NOT closed — call [ImageProxy.close] yourself
     * after this returns (same as with ML Kit analyzers).
     */
    fun decode(imageProxy: ImageProxy): DecodeResult? {
        val rotation = imageProxy.imageInfo.rotationDegrees
        var gray = extractGrayscale(imageProxy) ?: return null

        gray = when (rotation) {
            90 -> rotateCW90(gray)
            180 -> rotateCW180(gray)
            270 -> rotateCW270(gray)
            else -> gray
        }

        return decodeGray(gray)
    }

    /**
     * Decode from an already-extracted [GrayImage].
     * Use this when you need to pre-rotate or pre-crop the frame yourself.
     */
    fun decodeGray(gray: GrayImage): DecodeResult? {
        val crop = isolateBarcode(gray) ?: gray   // fall back to full frame
        return decodeBarcode(crop)
    }

    // ── GrayImage (public so callers can build one from a Bitmap if needed) ─────

    class GrayImage(val pixels: IntArray, val width: Int, val height: Int) {
        fun pixel(x: Int, y: Int): Int = pixels[y * width + x]
    }

    /** Convenience: rotate a GrayImage 90° clockwise. */
    fun rotateCW90(src: GrayImage): GrayImage {
        val dstW = src.height; val dstH = src.width
        val dst  = IntArray(dstW * dstH)
        for (y in 0 until src.height) {
            for (x in 0 until src.width) {
                dst[x * dstW + (dstW - 1 - y)] = src.pixel(x, y)
            }
        }
        return GrayImage(dst, dstW, dstH)
    }

    /** Convenience: rotate a GrayImage 180°. */
    fun rotateCW180(src: GrayImage): GrayImage {
        val w = src.width; val h = src.height
        val dst = IntArray(w * h)
        for (i in 0 until (w * h)) {
            dst[i] = src.pixels[w * h - 1 - i]
        }
        return GrayImage(dst, w, h)
    }

    /** Convenience: rotate a GrayImage 270° clockwise (90° CCW). */
    fun rotateCW270(src: GrayImage): GrayImage {
        val dstW = src.height; val dstH = src.width
        val dst  = IntArray(dstW * dstH)
        for (y in 0 until src.height) {
            for (x in 0 until src.width) {
                dst[(dstH - 1 - x) * dstW + y] = src.pixel(x, y)
            }
        }
        return GrayImage(dst, dstW, dstH)
    }

    // ── Step 1: ImageProxy → GrayImage ─────────────────────────────────────────

    /**
     * Extract the Y (luminance) plane of a YUV_420_888 [ImageProxy].
     * Respects [ImageProxy.cropRect].
     * Returns null if the format is unsupported.
     */
    fun extractGrayscale(proxy: ImageProxy): GrayImage? {
        if (proxy.format != ImageFormat.YUV_420_888) return null
        
        val rect = proxy.cropRect
        val plane = proxy.planes[0]
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val buffer = plane.buffer
        
        val width = rect.width()
        val height = rect.height()
        val pixels = IntArray(width * height)
        
        val rowData = ByteArray(rowStride)
        for (y in 0 until height) {
            buffer.position((rect.top + y) * rowStride + rect.left * pixelStride)
            if (pixelStride == 1) {
                buffer.get(rowData, 0, width)
                for (x in 0 until width) {
                    pixels[y * width + x] = rowData[x].toInt() and 0xFF
                }
            } else {
                for (x in 0 until width) {
                    pixels[y * width + x] = buffer.get(buffer.position() + x * pixelStride).toInt() and 0xFF
                }
            }
        }
        return GrayImage(pixels, width, height)
    }

    // ── Step 2: Barcode band isolation ──────────────────────────────────────────

    /**
     * Find the horizontal band of the image containing the barcode.
     *
     * Scores each row by the number of dark↔light transitions after
     * mid-point binarisation.  Barcode rows score high (many bar edges);
     * plain background rows score near zero.  The tallest contiguous
     * high-scoring band is returned as a cropped [GrayImage].
     *
     * Mirrors the transitions-per-row heuristic from isolate_barcodes.py.
     */
    private fun isolateBarcode(img: GrayImage): GrayImage? {
        val w = img.width
        val h = img.height

        val scores    = IntArray(h) { y -> rowTransitions(img, y) }
        val maxScore  = scores.maxOrNull() ?: return null
        if (maxScore < 10) return null
        val threshold = maxScore / 3

        var bestStart = 0; var bestEnd = 0; var bestSum = 0
        var bandStart = -1; var bandSum  = 0

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
        // Band may extend to the last row
        if (bandStart >= 0 && h - bandStart >= 5 && bandSum > bestSum) {
            bestStart = bandStart; bestEnd = h
        }
        if (bestEnd - bestStart < 5) return null

        val top    = (bestStart - 4).coerceAtLeast(0)
        val bottom = (bestEnd   + 4).coerceAtMost(h)
        val cropH  = bottom - top
        val out    = IntArray(w * cropH) { i -> img.pixel(i % w, top + i / w) }
        return GrayImage(out, w, cropH)
    }

    /** Number of dark↔light transitions in row [y] (binarised at row midpoint). */
    private fun rowTransitions(img: GrayImage, y: Int): Int {
        val w   = img.width
        val mn  = (0 until w).minOf { img.pixel(it, y) }
        val mx  = (0 until w).maxOf { img.pixel(it, y) }
        if (mx - mn < 30) return 0
        val thr  = (mn + mx) / 2
        var prev = if (img.pixel(0, y) < thr) 1 else 0
        var count = 0
        for (x in 1 until w) {
            val cur = if (img.pixel(x, y) < thr) 1 else 0
            if (cur != prev) { count++; prev = cur }
        }
        return count
    }

    // ── Step 3: MSI Plessey decode ──────────────────────────────────────────────

    private fun decodeBarcode(img: GrayImage): DecodeResult? {
        rowScan(img)?.let { v ->
            luhnNormalize(v)?.let { norm ->
                return DecodeResult(norm.take(7), norm, "row_scan")
            }
        }
        colGreedy(img)?.let { v ->
            return DecodeResult(v.take(7), v, "col_greedy")
        }
        return null
    }

    // ── Run-length encoding ──────────────────────────────────────────────────────

    /** Encode a binarised row as a list of (isDark, runLength) pairs. */
    private fun rle(row: IntArray): List<Pair<Boolean, Int>> {
        if (row.isEmpty()) return emptyList()
        val runs   = mutableListOf<Pair<Boolean, Int>>()
        var isDark = row[0] != 0
        var count  = 1
        for (i in 1 until row.size) {
            val d = row[i] != 0
            if (d == isDark) { count++ }
            else { runs.add(isDark to count); isDark = d; count = 1 }
        }
        runs.add(isDark to count)
        return runs
    }

    /** Binarise one row: pixel < [thr] → dark (1), else light (0). */
    private fun binariseRow(img: GrayImage, y: Int, thr: Int): IntArray =
        IntArray(img.width) { x -> if (img.pixel(x, y) < thr) 1 else 0 }

    // ── Bimodal split ────────────────────────────────────────────────────────────

    /**
     * Find the midpoint of the largest gap between consecutive unique dark-run
     * widths (after trimming the top-2 outliers, typically quiet-zone bleeds).
     * Returns null when no clear narrow/wide separation exists.
     */
    private fun bimodalSplit(darkWidths: List<Int>): Double? {
        if (darkWidths.size < 6) return null
        val trimmed = darkWidths.sorted().dropLast(2)
        val vals    = trimmed.toSortedSet().toList()
        if (vals.size < 2) return null
        var bestGap   = 0
        var bestSplit = 0.0
        for (i in 0 until vals.size - 1) {
            val gap = vals[i + 1] - vals[i]
            if (gap > bestGap) { bestGap = gap; bestSplit = (vals[i] + vals[i + 1]) / 2.0 }
        }
        return if (bestGap >= 1) bestSplit else null
    }

    // ── Run-list → digit string ──────────────────────────────────────────────────

    /**
     * Decode an MSI Plessey barcode from its run-length representation.
     *
     * Finds the start guard (first dark run wider than [split]), then reads
     * groups of 8 runs (4 bar+space pairs per digit) and maps each to a digit.
     * Returns a digit string of length ≥ 7, or null on failure.
     */
    private fun decodeRuns(runs: List<Pair<Boolean, Int>>, split: Double): String? {
        // Locate start guard: first dark run with width > split
        var pos = -1
        for (i in runs.indices) {
            if (runs[i].first && runs[i].second > split) { pos = i + 2; break }
        }
        if (pos < 0) return null

        val sb = StringBuilder()
        while (pos + 6 < runs.size) {
            // Read widths of 4 dark bars (at even offsets from pos)
            val bits = List(4) { j -> if (runs[pos + j * 2].second > split) 1 else 0 }
            val digit = BITS_TO_DIGIT[bits] ?: break
            sb.append(digit)
            pos += 8   // advance past 4 bar+space pairs
        }
        return if (sb.length >= 7) sb.toString() else null
    }

    // ── Method 1: Row scan ───────────────────────────────────────────────────────

    /**
     * Primary decoder.  Scans every row at 13 binarisation thresholds (20%–80%
     * of each row's dynamic range in 5% steps).  Votes are cast for each decoded
     * result; Luhn-valid results are strongly preferred over unvalidated ones.
     *
     * Returns the most-voted Luhn-valid candidate (≥ 2 votes), or the most-voted
     * raw candidate as a last resort.
     */
    private fun rowScan(img: GrayImage): String? {
        val validVotes = mutableMapOf<String, Int>()
        val rawVotes   = mutableMapOf<String, Int>()

        for (y in 0 until img.height) {
            val mn = (0 until img.width).minOf { img.pixel(it, y) }
            val mx = (0 until img.width).maxOf { img.pixel(it, y) }
            if (mx - mn < 30) continue                  // flat row — skip

            for (thrPct in 20..80 step 5) {
                val thr  = mn + (mx - mn) * thrPct / 100
                val bin  = binariseRow(img, y, thr)
                val runs = rle(bin)
                // A 7–12 digit MSI barcode produces roughly 61–103 runs (incl. quiet zones)
                if (runs.size !in 55..110) continue
                val darkWidths = runs.filter { it.first }.map { it.second }
                val split = bimodalSplit(darkWidths) ?: continue
                val value = decodeRuns(runs, split)   ?: continue
                if (value.length < 7) continue

                val normed = luhnNormalize(value)
                if (normed != null) validVotes[normed] = (validVotes[normed] ?: 0) + 1
                else                rawVotes  [value]  = (rawVotes  [value]  ?: 0) + 1
            }
        }

        validVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        rawVotes  .maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        return null
    }

    // ── Method 2: Column greedy ──────────────────────────────────────────────────

    /**
     * Fallback decoder for images with horizontal white stripes (scanner
     * reflection, label damage) that break individual row scans.
     *
     * Builds a per-column darkness signal using the 10th-percentile luminance
     * across all rows — so at least ~10 % of rows must be dark in a column for
     * it to register as a bar.  Then sweeps a grid of (N, W-ratio, offset)
     * parameters and positionally samples each expected bit-pair window.
     */
    private fun colSignal(img: GrayImage): DoubleArray? {
        val w      = img.width
        val h      = img.height
        val pctIdx = (h * 0.10).toInt().coerceAtLeast(0)
        val sig    = DoubleArray(w) { x ->
            val col = DoubleArray(h) { y -> img.pixel(x, y).toDouble() }.also { it.sort() }
            col[pctIdx]
        }
        val mn = sig.minOrNull()!!; val mx = sig.maxOrNull()!!
        if (mx - mn < 10.0) return null
        return DoubleArray(w) { x -> 1.0 - (sig[x] - mn) / (mx - mn) }
    }

    /**
     * Positionally sample the column signal with given module widths and start
     * offset.  avg(window) > 0.5 → bit=1 (wide bar), else bit=0 (wide space).
     */
    private fun greedy(sig: DoubleArray, N: Double, W: Double, offset: Int): String? {
        val nPx = N.roundToInt(); val wPx = W.roundToInt(); val pw = nPx + wPx
        var x   = offset + wPx + nPx    // skip start guard BW + SN
        val sb  = StringBuilder()
        while (x + pw * 4 <= sig.size && sb.length < 12) {
            val bits = List(4) { j ->
                val s = x + j * pw; val e = (s + pw).coerceAtMost(sig.size)
                var sum = 0.0; for (k in s until e) sum += sig[k]
                if (sum / (e - s) > 0.5) 1 else 0
            }
            val digit = BITS_TO_DIGIT[bits] ?: break
            sb.append(digit)
            x += pw * 4
        }
        return if (sb.length >= 7) sb.toString() else null
    }

    /** Sweep parameter grid; return most-voted Luhn-valid decode. */
    private fun colGreedy(img: GrayImage): String? {
        val sig   = colSignal(img) ?: return null
        val w     = sig.size
        val votes = mutableMapOf<String, Int>()

        var N = 1.5
        while (N <= 5.0) {
            for (ratioStep in 0..2) {
                val W    = N * (2.0 + ratioStep * 0.5)
                val nPx  = N.roundToInt(); val wPx = W.roundToInt(); val pw = nPx + wPx
                val minW = wPx + nPx + 7 * 4 * pw + nPx + wPx + nPx
                if (minW > w) continue
                val maxOff = min(w - minW, 80)
                for (off in 0..maxOff step 5) {
                    val r = greedy(sig, N, W, off) ?: continue
                    luhnNormalize(r)?.let { votes[it] = (votes[it] ?: 0) + 1 }
                }
            }
            N += 0.5
        }
        return votes.maxByOrNull { it.value }?.key
    }

    // ── Luhn check digit ─────────────────────────────────────────────────────────

    /**
     * Compute the Luhn (Mod-10) check digit for [digits].
     * [digits] must NOT include the check digit itself.
     */
    private fun luhnCheck(digits: String): Char {
        var total = 0
        digits.reversed().forEachIndexed { i, ch ->
            var d = ch.digitToInt()
            if (i % 2 == 0) { d *= 2; if (d > 9) d -= 9 }
            total += d
        }
        return '0' + (10 - total % 10) % 10
    }

    /**
     * Try prefix lengths 10 → 8 to find one whose last character is a valid
     * Luhn check digit for the preceding characters.  Returns that prefix
     * (the canonical decoded value), or null if none match.
     *
     * Handles clean 9-char decodes ("082814799") and 10-char over-runs where
     * the stop guard is decoded as an extra digit ("0828147998" → "082814799").
     * Also rejects spurious decodes that fail Luhn at every candidate length.
     */
    private fun luhnNormalize(s: String): String? {
        for (n in min(s.length, 10) downTo 8) {
            if (luhnCheck(s.substring(0, n - 1)) == s[n - 1]) return s.substring(0, n)
        }
        return null
    }
}
