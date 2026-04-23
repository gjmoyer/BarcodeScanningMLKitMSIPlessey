package com.example.zxing_barcodereader

import android.util.Log
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.roundToInt

data class BarcodeResult(val value: String, val format: String)

class MSIPlesseyDecoder {

    private val TAG = "MSI_DBG"

    private val digitPatterns = mapOf(
        "100100100100" to '0', "100100100110" to '1',
        "100100110100" to '2', "100100110110" to '3',
        "100110100100" to '4', "100110100110" to '5',
        "100110110100" to '6', "100110110110" to '7',
        "110100100100" to '8', "110100100110" to '9'
    )

    private val START_PATTERN = listOf(1, 1, 0)

    // ─────────────────────────────────────────────────────────────────────
    // Entry point
    // ─────────────────────────────────────────────────────────────────────

    fun decode(imageProxy: ImageProxy): BarcodeResult? {
        Log.d(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        Log.d(TAG, "STAGE 0  ${imageProxy.width}x${imageProxy.height}")

        val bitmap = try { imageProxy.toBitmap() }
        catch (e: Exception) { Log.e(TAG, "STAGE 1 FAIL: ${e.message}"); return null }

        val gray = Mat()
        Utils.bitmapToMat(bitmap, gray)
        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGBA2GRAY)

        val binary  = preprocess(gray)
        val nonZero = Core.countNonZero(binary)
        val pct     = 100 * nonZero / (binary.rows() * binary.cols())
        Log.d(TAG, "STAGE 3  nonZero=$nonZero ($pct%)")

        val roi = findBarcodeROI(binary, gray.cols(), gray.rows())
        Log.d(TAG, "STAGE 4  ROI=$roi")

        // Try ROI first, then full image
        val result = if (roi != null) {
            val cropped = Mat(binary, roi)
            tryMultiRowDecode(cropped, "ROI").also { cropped.release() }
        } else null

        val final = result ?: tryMultiRowDecode(binary, "full")

        gray.release(); binary.release()
        if (final == null) Log.w(TAG, "FAIL") else Log.i(TAG, "OK '${final.value}'")
        return final
    }

    // ─────────────────────────────────────────────────────────────────────
    // Preprocessing
    // ─────────────────────────────────────────────────────────────────────

    private fun preprocess(gray: Mat): Mat {
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(3.0, 3.0), 0.0)
        val binary = Mat()
        Imgproc.adaptiveThreshold(
            blurred, binary, 255.0,
            Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY_INV, 15, 10.0
        )
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 1.0))
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel)
        kernel.release(); blurred.release()
        return binary
    }

    // ─────────────────────────────────────────────────────────────────────
    // ROI
    // ─────────────────────────────────────────────────────────────────────

    private fun findBarcodeROI(binary: Mat, imgW: Int, imgH: Int): Rect? {
        val contours = mutableListOf<MatOfPoint>()
        val hier = Mat()
        Imgproc.findContours(binary.clone(), contours, hier,
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hier.release()

        var best: Rect? = null; var bestScore = 0.0
        for (c in contours) {
            val r = Imgproc.boundingRect(c)
            val aspect = r.width.toDouble() / r.height.coerceAtLeast(1)
            if (r.width >= imgW * 0.15 && r.height >= 20 && aspect > 2.5) {
                val score = r.width.toDouble() * r.height + aspect
                if (score > bestScore) { best = r; bestScore = score }
            }
        }
        Log.d(TAG, "  ROI $best score=$bestScore")
        return best
    }

    // ─────────────────────────────────────────────────────────────────────
    // Multi-row scan
    // ─────────────────────────────────────────────────────────────────────

    private fun tryMultiRowDecode(binary: Mat, src: String): BarcodeResult? {
        val candidates = mutableListOf<Pair<String, Int>>()
        val fractions  = listOf(0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)

        for (frac in fractions) {
            val row    = (binary.rows() * frac).toInt().coerceIn(0, binary.rows() - 1)
            val bits   = extractRowBits(binary, row)
            val runs   = extractRuns(bits)
            val label  = "$src/r${(frac * 100).toInt()}"
            Log.d(TAG, "  $label  runs=${runs.size}  first8=${runs.take(8)}")
            if (runs.size < 10) continue
            decodeRunsWithXSweep(runs, label)?.let { candidates.add(it) }
        }

        // Vertical projection fallback
        val proj = extractRuns(projectTo1D(binary))
        Log.d(TAG, "  $src/proj  runs=${proj.size}")
        if (proj.size >= 10) decodeRunsWithXSweep(proj, "$src/proj")?.let { candidates.add(it) }

        val best = candidates.maxByOrNull { it.second }
        Log.d(TAG, "  [$src] best=$best")
        return if (best != null && best.second >= 40) BarcodeResult(best.first, "MSI Plessey") else null
    }

    // ─────────────────────────────────────────────────────────────────────
    // X-sweep: try multiple candidate narrow-module widths
    //
    // Why: k-means / median estimation reliably gives us a ballpark X but
    // the ratio ends up 2.5-4.5 instead of 2.0, meaning the actual X is
    // smaller than estimated.  We sweep X = estimate * {0.4, 0.5, 0.6, 0.7,
    // 0.8, 1.0} and keep whichever produces the highest-scoring decode.
    // This costs almost nothing (decode is O(n) per X) and is robust to
    // estimation error.
    // ─────────────────────────────────────────────────────────────────────

    private fun decodeRunsWithXSweep(runs: List<Int>, label: String): Pair<String, Int>? {
        // Estimate a base X from the smallest runs (after stripping outliers)
        val sorted   = runs.filter { it > 1 }.sorted()
        if (sorted.size < 4) return null
        val medianX  = sorted[sorted.size / 4].toFloat()   // 25th percentile as base

        // Sweep: try X = baseX × multiplier for several multipliers
        val multipliers = listOf(0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 1.0f, 1.2f)
        var best: Pair<String, Int>? = null

        for (mult in multipliers) {
            val x = medianX * mult
            if (x < 1.5f) continue

            val modules = quantiseRuns(runs, x) ?: continue
            val result  = decodeAllVariants(modules, "$label/x${"%.1f".format(x)}")
            if (result != null && (best == null || result.second > best!!.second)) {
                best = result
            }
        }

        if (best != null) Log.d(TAG, "  [$label] sweep best=$best")
        return best
    }

    // ─────────────────────────────────────────────────────────────────────
    // Quantise runs → module list using a given X
    // ─────────────────────────────────────────────────────────────────────

    private fun quantiseRuns(runs: List<Int>, x: Float): List<Int>? {
        // Strip quiet-zone outliers: anything > 6× the assumed narrow width
        val quietZone = x * 6f
        val modules   = mutableListOf<Int>()
        var polarity  = 1  // first real run is a bar

        for (i in 1 until runs.size - 1) {
            val r = runs[i]
            if (r <= 1) continue
            if (r > quietZone) { polarity = 1; continue }

            val count = (r / x).roundToInt().coerceIn(1, 4)
            repeat(count) { modules.add(polarity) }
            polarity = 1 - polarity
        }

        return if (modules.size >= 15) modules else null
    }

    // ─────────────────────────────────────────────────────────────────────
    // Bit / run helpers
    // ─────────────────────────────────────────────────────────────────────

    private fun extractRowBits(binary: Mat, row: Int): List<Int> {
        val r = ArrayList<Int>(binary.cols())
        for (x in 0 until binary.cols()) r.add(if (binary.get(row, x)[0] > 128.0) 1 else 0)
        return r
    }

    private fun projectTo1D(mat: Mat): List<Int> {
        val thr = mat.rows() * 0.35
        return IntArray(mat.cols()) { x ->
            var s = 0
            for (y in 0 until mat.rows()) if (mat.get(y, x)[0] > 128.0) s++
            if (s > thr) 1 else 0
        }.toList()
    }

    private fun extractRuns(bits: List<Int>): List<Int> {
        if (bits.isEmpty()) return emptyList()
        val runs = mutableListOf<Int>(); var count = 1
        for (i in 1 until bits.size) {
            if (bits[i] == bits[i - 1]) count++ else { runs.add(count); count = 1 }
        }
        runs.add(count); return runs
    }

    // ─────────────────────────────────────────────────────────────────────
    // Decode variants (forward / reverse / inverted / inv+rev)
    // ─────────────────────────────────────────────────────────────────────

    private fun decodeAllVariants(modules: List<Int>, label: String): Pair<String, Int>? {
        val names    = listOf("fwd", "rev", "inv", "i+r")
        val variants = listOf(
            modules,
            modules.reversed(),
            modules.map { 1 - it },
            modules.map { 1 - it }.reversed()
        )
        var best: Pair<String, Int>? = null
        for ((i, v) in variants.withIndex()) {
            decodeModules(v, "$label/${names[i]}")?.let {
                if (best == null || it.second > best!!.second) best = it
            }
        }
        return best
    }

    private fun decodeModules(modules: List<Int>, label: String): Pair<String, Int>? {
        var best: Pair<String, Int>? = null

        for (start in 0 until modules.size - 15) {
            if (!matchesAt(modules, start, START_PATTERN)) continue

            val dataStart = start + START_PATTERN.size
            var dataLen   = 12

            while (dataStart + dataLen + 4 <= modules.size) {
                val dataEnd = dataStart + dataLen
                val slice   = modules.subList(dataStart, dataEnd)

                val digits = mutableListOf<Char>(); var errors = 0
                for (i in 0 until slice.size step 12) {
                    val chunk = slice.subList(i, i + 12).joinToString("")
                    digitPatterns[chunk]?.let { digits.add(it) } ?: errors++
                }

                if (digits.size >= 2) {   // need at least payload + check digit
                    val raw     = digits.joinToString("")
                    val hasStop = dataEnd + 4 <= modules.size &&
                            modules[dataEnd]==1 && modules[dataEnd+1]==0 &&
                            modules[dataEnd+2]==0 && modules[dataEnd+3]==1
                    val payload = raw.dropLast(1)
                    val ckOk    = validateLuhn(payload, raw.last())
                    val score   = (if (ckOk) 50 else 0) +
                                  (if (hasStop) 20 else 0) +
                                  (digits.size * 5) - (errors * 15)

                    Log.d(TAG, "  [$label] s=$start l=$dataLen raw=$raw err=$errors ck=$ckOk stop=$hasStop score=$score")

                    if (best == null || score > best!!.second) best = raw to score
                }
                dataLen += 12
            }
        }
        return best
    }

    private fun matchesAt(m: List<Int>, start: Int, p: List<Int>): Boolean {
        if (start + p.size > m.size) return false
        return p.indices.all { m[start + it] == p[it] }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Checksum — Luhn on payload (all digits except the last)
    // ─────────────────────────────────────────────────────────────────────

    private fun validateLuhn(payload: String, checkDigit: Char): Boolean {
        if (payload.isEmpty() || !payload.all { it.isDigit() } || !checkDigit.isDigit()) return false
        var sum = 0
        payload.reversed().forEachIndexed { i, c ->
            var d = c.digitToInt()
            if (i % 2 == 0) { d *= 2; if (d > 9) d -= 9 }
            sum += d
        }
        val expected = (10 - (sum % 10)) % 10
        return expected == checkDigit.digitToInt()
    }
}
