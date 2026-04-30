package com.example.zxing_barcodereader

import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import com.example.zxing_barcodereader.MsiPlesseyBarcodeDecoder.GrayImage
import com.example.zxing_barcodereader.MsiPlesseyBarcodeDecoder.rotateCW180
import com.example.zxing_barcodereader.MsiPlesseyBarcodeDecoder.rotateCW270
import com.example.zxing_barcodereader.MsiPlesseyBarcodeDecoder.rotateCW90
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking

/**
 * MSI Plessey barcode isolator and decoder for Android / CameraX.
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
 */
object MsiPlesseyBarcodeDecoderV2 {

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

    // ── MSI Plessey bit-pattern ↔ digit mapping (Approach A) ────────────────────
    //   bit = 1  →  wide bar  (W) + narrow space (N)
    //   bit = 0  →  narrow bar (N) + wide space  (W)

    private val BITS_TO_DIGIT: Map<List<Int>, Int> = mapOf(
        listOf(0, 0, 0, 0) to 0, listOf(0, 0, 0, 1) to 1,
        listOf(0, 0, 1, 0) to 2, listOf(0, 0, 1, 1) to 3,
        listOf(0, 1, 0, 0) to 4, listOf(0, 1, 0, 1) to 5,
        listOf(0, 1, 1, 0) to 6, listOf(0, 1, 1, 1) to 7,
        listOf(1, 0, 0, 0) to 8, listOf(1, 0, 0, 1) to 9
    )

    // ── ZXing CHARACTER_ENCODINGS (Approach B) ───────────────────────────────────
    //   Digit encodings as wide/narrow bit patterns; index i = digit i (0–9).
    //   Narrow unit → 1 bit; wide unit → 2 identical bits.

    private val ZXING_CHARACTER_ENCODINGS = intArrayOf(
        0x924, 0x926, 0x934, 0x936, 0x9A4, 0x9A6, 0x9B4, 0x9B6, 0xD24, 0xD26
    )
    private const val ZXING_ALPHABET = "0123456789"
    private const val ZXING_START = 0x06   // wide bar  + narrow space
    private const val ZXING_END = 0x09   // narrow bar + wide space + narrow bar

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

    private const val SCORE_COMBINED = 3
    private const val SCORE_ROW_SCAN = 2
    private const val SCORE_ZXING_ROW_SCAN = 2
    private const val SCORE_COL_GREEDY = 1
    private const val CONFIDENCE_THRESHOLD = 5

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
     * Does NOT close the [ImageProxy] — call [ImageProxy.close] yourself.
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
     * Accumulates cross-frame confidence the same as [decode].
     */
    private fun decodeGray(gray: GrayImage): DecodeResult? {
        val crop = isolateBarcode(gray) ?: gray
        val single = decodeBarcode(crop) ?: return null

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

    // ── GrayImage ────────────────────────────────────────────────────────────────

    // Using GrayImage and rotation helpers from MsiPlesseyBarcodeDecoder

    // ── Step 1: ImageProxy → GrayImage ──────────────────────────────────────────

    private fun extractGrayscale(proxy: ImageProxy): GrayImage? {
        if (proxy.format != ImageFormat.YUV_420_888) return null
        val plane = proxy.planes[0]
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val w = proxy.width;
        val h = proxy.height
        val buf = plane.buffer
        val pixels = IntArray(w * h)
        for (y in 0 until h)
            for (x in 0 until w)
                pixels[y * w + x] = buf[y * rowStride + x * pixelStride].toInt() and 0xFF
        return GrayImage(pixels, w, h)
    }

    // ── Step 2: Barcode band isolation ──────────────────────────────────────────

    private fun isolateBarcode(img: GrayImage): GrayImage? {
        val w = img.width;
        val h = img.height
        val scores = IntArray(h) { y -> rowTransitions(img, y) }
        val maxScore = scores.maxOrNull() ?: return null
        if (maxScore < 10) return null
        val threshold = maxScore / 3

        var bestStart = 0;
        var bestEnd = 0;
        var bestSum = 0
        var bandStart = -1;
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

    private fun rowTransitions(img: GrayImage, y: Int): Int {
        val w = img.width
        val mn = (0 until w).minOf { img.pixel(it, y) }
        val mx = (0 until w).maxOf { img.pixel(it, y) }
        if (mx - mn < 30) return 0
        val thr = (mn + mx) / 2
        var prev = if (img.pixel(0, y) < thr) 1 else 0
        var count = 0
        for (x in 1 until w) {
            val cur = if (img.pixel(x, y) < thr) 1 else 0
            if (cur != prev) {
                count++; prev = cur
            }
        }
        return count
    }

    // ── Step 3: Combined decode ──────────────────────────────────────────────────
    //
    // Both row-scan approaches run in parallel (Dispatchers.Default), then results
    // are reconciled:
    //   • Both agree  → "combined"   (highest confidence)
    //   • A only      → "row_scan"
    //   • B only      → "zxing_row_scan"
    //   • Neither     → column greedy fallback → "col_greedy"

    private fun decodeBarcode(img: GrayImage): DecodeResult? = runBlocking {
        val bimodalJob = async(Dispatchers.Default) { rowScan(img)?.let { luhnNormalize(it) } }
        val zxingJob = async(Dispatchers.Default) { zxingRowScan(img)?.let { luhnNormalize(it) } }
        val bimodal = bimodalJob.await()
        val zxing = zxingJob.await()

        if (bimodal != null && zxing != null && bimodal.take(7) == zxing.take(7))
            return@runBlocking DecodeResult(bimodal.take(7), bimodal, "combined")

        if (bimodal != null) return@runBlocking DecodeResult(bimodal.take(7), bimodal, "row_scan")
        if (zxing != null) return@runBlocking DecodeResult(zxing.take(7), zxing, "zxing_row_scan")

        colGreedy(img)?.let { v -> return@runBlocking DecodeResult(v.take(7), v, "col_greedy") }
        null
    }

    // ── Approach A: Bimodal row scan ─────────────────────────────────────────────

    private fun rle(row: IntArray): List<Pair<Boolean, Int>> {
        if (row.isEmpty()) return emptyList()
        val runs = mutableListOf<Pair<Boolean, Int>>()
        var isDark = row[0] != 0
        var count = 1
        for (i in 1 until row.size) {
            val d = row[i] != 0
            if (d == isDark) {
                count++
            } else {
                runs.add(isDark to count); isDark = d; count = 1
            }
        }
        runs.add(isDark to count)
        return runs
    }

    private fun binariseRow(img: GrayImage, y: Int, thr: Int): IntArray =
        IntArray(img.width) { x -> if (img.pixel(x, y) < thr) 1 else 0 }

    private fun bimodalSplit(darkWidths: List<Int>): Double? {
        if (darkWidths.size < 6) return null
        val trimmed = darkWidths.sorted().dropLast(2)
        val vals = trimmed.toSortedSet().toList()
        if (vals.size < 2) return null
        var bestGap = 0
        var bestSplit = 0.0
        for (i in 0 until vals.size - 1) {
            val gap = vals[i + 1] - vals[i]
            if (gap > bestGap) {
                bestGap = gap; bestSplit = (vals[i] + vals[i + 1]) / 2.0
            }
        }
        return if (bestGap >= 1) bestSplit else null
    }

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
            val bits = List(4) { j -> if (runs[pos + j * 2].second > split) 1 else 0 }
            val digit = BITS_TO_DIGIT[bits] ?: break
            sb.append(digit)
            pos += 8
        }
        return if (sb.length >= 7) sb.toString() else null
    }

    private fun rowScan(img: GrayImage): String? {
        val validVotes = mutableMapOf<String, Int>()
        val rawVotes = mutableMapOf<String, Int>()

        for (y in 0 until img.height) {
            val mn = (0 until img.width).minOf { img.pixel(it, y) }
            val mx = (0 until img.width).maxOf { img.pixel(it, y) }
            if (mx - mn < 30) continue

            for (thrPct in 20..80 step 5) {
                val thr = mn + (mx - mn) * thrPct / 100
                val bin = binariseRow(img, y, thr)
                val runs = rle(bin)
                if (runs.size !in 55..110) continue
                val darkWidths = runs.filter { it.first }.map { it.second }
                val split = bimodalSplit(darkWidths) ?: continue
                val value = decodeRuns(runs, split) ?: continue
                if (value.length < 7) continue
                if (value.all { it == value[0] }) continue  // all-same-digit = decode artifact

                val normed = luhnNormalize(value)
                if (normed != null) validVotes[normed] = (validVotes[normed] ?: 0) + 1
                else rawVotes[value] = (rawVotes[value] ?: 0) + 1
            }
        }

        validVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        rawVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        return null
    }

    // ── Approach B: ZXing run-counter row scan ───────────────────────────────────
    //
    // Ported from ZXing.Net MSIReader.cs.  The narrow/wide pivot is established
    // from only the start guard's 2 counters (min+max)/2 × 256 fixed-point.
    // A BitRow wraps each binarised row and provides the ZXing helper API.

    private fun zxingRowScan(img: GrayImage): String? {
        val validVotes = mutableMapOf<String, Int>()
        val rawVotes = mutableMapOf<String, Int>()

        for (y in 0 until img.height) {
            val mn = (0 until img.width).minOf { img.pixel(it, y) }
            val mx = (0 until img.width).maxOf { img.pixel(it, y) }
            if (mx - mn < 30) continue

            for (thrPct in 20..80 step 5) {
                val thr = mn + (mx - mn) * thrPct / 100
                val bitRow = BitRow(BooleanArray(img.width) { x -> img.pixel(x, y) < thr })
                val decoded = zxingDecodeRow(bitRow) ?: continue
                if (decoded.all { it == decoded[0] }) continue  // all-same-digit = decode artifact

                val normed = luhnNormalize(decoded)
                if (normed != null) validVotes[normed] = (validVotes[normed] ?: 0) + 1
                else rawVotes[decoded] = (rawVotes[decoded] ?: 0) + 1
            }
        }

        validVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        rawVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
        return null
    }

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
     *  Returns intArrayOf(startX, endX, avgWidth) or null. */
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

    /** Find end guard: narrow bar + wide space + narrow bar (0x09). */
    private fun zxingFindEnd(
        row: BitRow,
        rowOffset: Int,
        counters: IntArray,
        avgWidth: Int
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
                                counters,
                                3,
                                avgWidth
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

    private fun zxingRecordPattern(row: BitRow, start: Int, counters: IntArray, n: Int): Boolean {
        for (i in 0 until n) counters[i] = 0
        if (start >= row.size) return false
        var isWhite = !row[start]
        var cp = 0;
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

    private fun zxingCalcAvgWidth(counters: IntArray, len: Int): Int {
        var mn = Int.MAX_VALUE;
        var mx = 0
        for (i in 0 until len) {
            if (counters[i] < mn) mn = counters[i]
            if (counters[i] > mx) mx = counters[i]
        }
        return ((mx shl 8) + (mn shl 8)) / 2
    }

    private fun zxingToPattern(counters: IntArray, len: Int, avgWidth: Int): Int {
        var pattern = 0;
        var bit = 1;
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

    private fun zxingPatternToChar(pattern: Int): Char? {
        for (i in ZXING_CHARACTER_ENCODINGS.indices)
            if (ZXING_CHARACTER_ENCODINGS[i] == pattern) return ZXING_ALPHABET[i]
        return null
    }

    // ── Fallback: Column greedy ──────────────────────────────────────────────────

    private fun colSignal(img: GrayImage): DoubleArray? {
        val w = img.width;
        val h = img.height
        val pctIdx = (h * 0.10).toInt().coerceAtLeast(0)
        val sig = DoubleArray(w) { x ->
            val col = DoubleArray(h) { y -> img.pixel(x, y).toDouble() }.also { it.sort() }
            col[pctIdx]
        }
        val mn = sig.minOrNull()!!;
        val mx = sig.maxOrNull()!!
        if (mx - mn < 10.0) return null
        return DoubleArray(w) { x -> 1.0 - (sig[x] - mn) / (mx - mn) }
    }

    private fun greedy(sig: DoubleArray, N: Double, W: Double, offset: Int): String? {
        val nPx = N.roundToInt();
        val wPx = W.roundToInt();
        val pw = nPx + wPx
        var x = offset + wPx + nPx
        val sb = StringBuilder()
        while (x + pw * 4 <= sig.size && sb.length < 12) {
            val bits = List(4) { j ->
                val s = x + j * pw;
                val e = (s + pw).coerceAtMost(sig.size)
                var sum = 0.0; for (k in s until e) sum += sig[k]
                if (sum / (e - s) > 0.5) 1 else 0
            }
            val digit = BITS_TO_DIGIT[bits] ?: break
            sb.append(digit)
            x += pw * 4
        }
        return if (sb.length >= 7) sb.toString() else null
    }

    private fun colGreedy(img: GrayImage): String? {
        val sig = colSignal(img) ?: return null
        val w = sig.size
        val votes = mutableMapOf<String, Int>()

        var N = 1.5
        while (N <= 5.0) {
            for (ratioStep in 0..2) {
                val W = N * (2.0 + ratioStep * 0.5)
                val nPx = N.roundToInt();
                val wPx = W.roundToInt();
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

    private fun luhnNormalize(s: String): String? {
        for (n in min(s.length, 10) downTo 8) {
            if (luhnCheck(s.substring(0, n - 1)) == s[n - 1]) return s.substring(0, n)
        }
        return null
    }


// ── BitRow ────────────────────────────────────────────────────────────────────
// Private to this file; wraps a binarised image row for the ZXing approach.

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
}