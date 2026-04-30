import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

// ── Result types ───────────────────────────────────────────────────────────────

data class DecodeResult(
    val digits7: String,
    val fullDigits: String,
    val method: String
)

data class DeskewEstimate(
    val degreesClockwise: Double,
    val coherence: Double,
    val sampleCount: Int
)

// ── Image types ────────────────────────────────────────────────────────────────

data class GrayImage(val pixels: IntArray, val width: Int, val height: Int) {
    fun pixel(x: Int, y: Int): Int = pixels[y * width + x]
}

private data class BinaryImage(val bits: BooleanArray, val width: Int, val height: Int) {
    fun pixel(x: Int, y: Int): Boolean = bits[y * width + x]
}

private data class IntPoint(val x: Int, val y: Int)

// ── BitRow ─────────────────────────────────────────────────────────────────────

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

// ── Constants ──────────────────────────────────────────────────────────────────

private val BITS_LOOKUP = IntArray(16) { -1 }.also {
    it[0] = 0; it[1] = 1; it[2] = 2; it[3] = 3; it[4] = 4
    it[5] = 5; it[6] = 6; it[7] = 7; it[8] = 8; it[9] = 9
}

private val ZXING_CHARACTER_ENCODINGS = intArrayOf(
    0x924, 0x926, 0x934, 0x936, 0x9A4, 0x9A6, 0x9B4, 0x9B6, 0xD24, 0xD26
)
private const val ZXING_ALPHABET = "0123456789"
private const val ZXING_START = 0x06
private const val ZXING_END = 0x09

private const val STRONG_DECODE_SCORE = 49

private const val DESKEW_MIN_ABS_DEGREES = 2.5
private const val DESKEW_MAX_ABS_DEGREES = 30.0
private const val DESKEW_MIN_EDGE_LENGTH = 30.0
private const val DESKEW_MASK_CLOSE_W = 25
private const val DESKEW_MASK_CLOSE_H = 7
private const val DESKEW_MASK_ERODE_PASSES = 3
private const val DESKEW_MASK_DILATE_PASSES = 4

private val ROTATION_SWEEP_DEGREES = doubleArrayOf(
    0.0, -30.0, -24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0, 30.0
)
private val SCALE_SWEEP_FACTORS = doubleArrayOf(2.0, 0.5, 1.5, 0.75)

private const val SHARP_STRENGTH_MODERATE = 1.5f
private const val SHARP_STRENGTH_AGGRESSIVE = 10.0f
private val SHARP_SCALE_FACTORS = doubleArrayOf(2.0, 3.0)
private const val LARGE_FRAME_PIXELS = 400_000L

// ── Image loading ──────────────────────────────────────────────────────────────

fun loadGrayImage(path: String): GrayImage {
    val img: BufferedImage = ImageIO.read(File(path))
    val w = img.width
    val h = img.height
    val pixels = IntArray(w * h)
    for (y in 0 until h) {
        for (x in 0 until w) {
            val rgb = img.getRGB(x, y)
            // Standard luminance conversion: 0.299R + 0.587G + 0.114B
            val r = (rgb shr 16) and 0xFF
            val g = (rgb shr 8) and 0xFF
            val b = rgb and 0xFF
            pixels[y * w + x] = (0.299 * r + 0.587 * g + 0.114 * b).toInt().coerceIn(0, 255)
        }
    }
    return GrayImage(pixels, w, h)
}

// ── Public entry point (single-frame decode, no cross-frame accumulation) ──────

fun decodeSingleFrame(gray: GrayImage): DecodeResult? {
    val pixelCount = gray.width.toLong() * gray.height.toLong()
    var best: DecodeResult? = null
    for (rotation in ROTATION_SWEEP_DEGREES) {
        val oriented = if (rotation == 0.0) gray else rotate(gray, rotation)
        best = chooseBetterDecode(best, decodeSingleOrientation(oriented))
        if (decodeQuality(best) >= STRONG_DECODE_SCORE) return best
    }
    for (factor in SCALE_SWEEP_FACTORS) {
        val scaled = scale(gray, factor)
        best = chooseBetterDecode(best, decodeSingleOrientation(scaled))
        if (decodeQuality(best) >= STRONG_DECODE_SCORE) return best
    }
    // Phases 3/4 (sharpening) are expensive — skip on large camera frames to avoid OOM.
    if (pixelCount <= LARGE_FRAME_PIXELS) {
        val sharpMod = sharpen(gray, SHARP_STRENGTH_MODERATE)
        for (rotation in ROTATION_SWEEP_DEGREES) {
            val oriented = if (rotation == 0.0) sharpMod else rotate(sharpMod, rotation)
            best = chooseBetterDecode(best, decodeSingleOrientation(oriented))
            if (decodeQuality(best) >= STRONG_DECODE_SCORE) return best
        }
        val sharpAgg = sharpen(gray, SHARP_STRENGTH_AGGRESSIVE)
        for (factor in SHARP_SCALE_FACTORS) {
            val scaled = scale(sharpAgg, factor)
            best = chooseBetterDecode(best, decodeSingleOrientation(scaled))
            if (decodeQuality(best) >= STRONG_DECODE_SCORE) return best
        }
    }
    return best
}

private fun decodeSingleOrientation(gray: GrayImage): DecodeResult? {
    val directCrop = isolateBarcode(gray) ?: gray
    val direct = decodeBarcode(directCrop)
    if (decodeQuality(direct) >= STRONG_DECODE_SCORE) return direct

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

// ── Quality / selection helpers ────────────────────────────────────────────────

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

private fun chooseBetterDecode(primary: DecodeResult?, secondary: DecodeResult?): DecodeResult? {
    if (primary == null) return secondary
    if (secondary == null) return primary
    return if (decodeQuality(secondary) > decodeQuality(primary)) secondary else primary
}

// ── Rotation ───────────────────────────────────────────────────────────────────

fun rotate(gray: GrayImage, degreesClockwise: Double, background: Int = 255): GrayImage {
    if (abs(degreesClockwise) < 1e-6) return gray

    val radians = degreesClockwise * PI / 180.0
    val cosA = cos(radians)
    val sinA = sin(radians)
    val absCos = abs(cosA)
    val absSin = abs(sinA)
    val dstW = ceil(gray.width * absCos + gray.height * absSin).toInt().coerceAtLeast(1)
    val dstH = ceil(gray.width * absSin + gray.height * absCos).toInt().coerceAtLeast(1)

    val srcCx = (gray.width - 1) / 2.0
    val srcCy = (gray.height - 1) / 2.0
    val dstCx = (dstW - 1) / 2.0
    val dstCy = (dstH - 1) / 2.0
    val out = IntArray(dstW * dstH)

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

// ── Scale ──────────────────────────────────────────────────────────────────────

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

// ── Sharpen ────────────────────────────────────────────────────────────────────

fun sharpen(gray: GrayImage, strength: Float = SHARP_STRENGTH_MODERATE): GrayImage {
    val w = gray.width
    val h = gray.height
    val p = gray.pixels
    val out = IntArray(w * h)
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
            val result = c + (str256 * (4 * c - n - s - e - ww) + 512) / 1024
            out[rowOff + x] = if (result < 0) 0 else if (result > 255) 255 else result
        }
    }
    return GrayImage(out, w, h)
}

// ── Bilinear sample ────────────────────────────────────────────────────────────

private fun sampleBilinear(src: GrayImage, x: Double, y: Double, background: Int): Int {
    if (x < 0.0 || y < 0.0 || x > src.width - 1.0 || y > src.height - 1.0) return background

    val x0 = x.toInt()
    val y0 = y.toInt()
    val x1 = if (x0 + 1 < src.width) x0 + 1 else x0
    val y1 = if (y0 + 1 < src.height) y0 + 1 else y0

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

    return (ify * (ifx * p00 + fx * p10) + fy * (ifx * p01 + fx * p11) + 524288) shr 20
}

// ── Deskew recovery ────────────────────────────────────────────────────────────

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
    if (absDegrees < DESKEW_MIN_ABS_DEGREES || absDegrees > DESKEW_MAX_ABS_DEGREES) return null

    return DeskewEstimate(
        degreesClockwise = bestAngle,
        coherence = bestLength / perimeter,
        sampleCount = boundary.size
    )
}

private fun normalizeDeskewAngle(degreesClockwise: Double): Double {
    var normalized = degreesClockwise
    while (normalized <= -90.0) normalized += 180.0
    while (normalized > 90.0) normalized -= 180.0
    if (normalized > 45.0) normalized -= 90.0
    else if (normalized < -45.0) normalized += 90.0
    return normalized
}

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

private fun dilateRect(src: BinaryImage, kernelW: Int, kernelH: Int): BinaryImage =
    rectMorph(src, kernelW, kernelH, matchAny = true)

private fun erodeRect(src: BinaryImage, kernelW: Int, kernelH: Int): BinaryImage =
    rectMorph(src, kernelW, kernelH, matchAny = false)

private fun rectMorph(src: BinaryImage, kernelW: Int, kernelH: Int, matchAny: Boolean): BinaryImage {
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

private fun convexHull(points: List<IntPoint>): List<IntPoint> {
    val sorted = points.distinct().sortedWith(compareBy<IntPoint> { it.x }.thenBy { it.y })
    if (sorted.size <= 1) return sorted

    val lower = ArrayList<IntPoint>()
    for (point in sorted) {
        while (lower.size >= 2 && cross(lower[lower.size - 2], lower[lower.size - 1], point) <= 0L) {
            lower.removeAt(lower.lastIndex)
        }
        lower.add(point)
    }

    val upper = ArrayList<IntPoint>()
    for (i in sorted.indices.reversed()) {
        val point = sorted[i]
        while (upper.size >= 2 && cross(upper[upper.size - 2], upper[upper.size - 1], point) <= 0L) {
            upper.removeAt(upper.lastIndex)
        }
        upper.add(point)
    }

    lower.removeAt(lower.lastIndex)
    upper.removeAt(upper.lastIndex)
    return lower + upper
}

private fun cross(a: IntPoint, b: IntPoint, c: IntPoint): Long =
    (b.x - a.x).toLong() * (c.y - a.y).toLong() - (b.y - a.y).toLong() * (c.x - a.x).toLong()

// ── Band isolation ─────────────────────────────────────────────────────────────

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

// ── Combined decode (sequential, no coroutines) ────────────────────────────────

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

// ── Approach A: Bimodal row scan ───────────────────────────────────────────────

// Run-length encode directly from GrayImage pixels, avoiding a per-threshold IntArray allocation.
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

private fun bimodalSplit(darkWidths: List<Int>): Double? {
    if (darkWidths.size < 6) return null
    val sorted = darkWidths.toIntArray()
    sorted.sort()
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
            if (runs.size !in 55..110) continue
            val darkWidths = runs.filter { it.first }.map { it.second }
            val split = bimodalSplit(darkWidths) ?: continue
            val value = decodeRuns(runs, split) ?: continue
            if (value.length < 7) continue
            if (value.all { it == value[0] }) continue

            val normed = luhnNormalize(value)
            if (normed != null) {
                val cnt = (validVotes[normed] ?: 0) + 1
                validVotes[normed] = cnt
                if (cnt > maxValidVotes) maxValidVotes = cnt
            } else {
                rawVotes[value] = (rawVotes[value] ?: 0) + 1
            }
        }
        if (maxValidVotes >= 20 && rowsScanned >= 15) break
    }

    validVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
    rawVotes.maxByOrNull { it.value }?.takeIf { it.value >= 2 }?.let { return it.key }
    return null
}

// ── Approach B: ZXing run-counter row scan ─────────────────────────────────────

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
            if (decoded.all { it == decoded[0] }) continue

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

private fun zxingFindEnd(row: BitRow, rowOffset: Int, counters: IntArray, avgWidth: Int): IntArray? {
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
                    if (factor in 1.5f..5.0f && zxingToPattern(counters, 3, avgWidth) == ZXING_END) {
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

private fun zxingCalcAvgWidth(counters: IntArray, len: Int): Int {
    var mn = Int.MAX_VALUE
    var mx = 0
    for (i in 0 until len) {
        if (counters[i] < mn) mn = counters[i]
        if (counters[i] > mx) mx = counters[i]
    }
    return ((mx shl 8) + (mn shl 8)) / 2
}

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

private fun zxingPatternToChar(pattern: Int): Char? {
    for (i in ZXING_CHARACTER_ENCODINGS.indices)
        if (ZXING_CHARACTER_ENCODINGS[i] == pattern) return ZXING_ALPHABET[i]
    return null
}

// ── Fallback: Column greedy ────────────────────────────────────────────────────

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
                if (r.all { it == r[0] }) continue
                luhnNormalize(r)?.let { votes[it] = (votes[it] ?: 0) + 1 }
            }
        }
        N += 0.5
    }
    return votes.maxByOrNull { it.value }?.key
}

// ── Luhn check ────────────────────────────────────────────────────────────────

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
        if (luhnCheck(s.substring(0, n - 1)) == s[n - 1]) {
            val result = s.substring(0, n)
            if (result.all { it == '0' }) return null
            return result
        }
    }
    return null
}

// ── Main test harness ──────────────────────────────────────────────────────────

fun main() {
    val samplesDir = File("../")
    val pngFiles = samplesDir.listFiles { _, name -> name.endsWith(".png") }
        ?.sortedBy { it.name }
        ?: emptyList()

    if (pngFiles.isEmpty()) {
        println("No PNG files found in ${samplesDir.absolutePath}")
        return
    }

    // Rotation angles to test (degrees clockwise applied BEFORE decodeSingleFrame).
    // decodeSingleFrame's internal sweep should correct these back to upright.
    val testAngles = doubleArrayOf(0.0, -12.0, 12.0, -24.0, 24.0, -30.0, 30.0)

    println("MSI Plessey V3 Decoder — Test Harness")
    println("======================================")
    println("Found ${pngFiles.size} sample images, testing at ${testAngles.size} rotations each\n")

    var passed = 0
    var failed = 0
    var errors = 0
    var totalMs = 0L

    for (file in pngFiles) {
        val expected = file.nameWithoutExtension

        print("${file.name}: ")
        try {
            val gray = loadGrayImage(file.absolutePath)
            print("${gray.width}x${gray.height}")

            var filePassed = 0
            var fileFailed = 0
            val angleResults = mutableListOf<String>()

            for (angle in testAngles) {
                val testImage = if (angle == 0.0) gray else rotate(gray, angle)

                val startTime = System.currentTimeMillis()
                val result = decodeSingleFrame(testImage)
                val elapsed = System.currentTimeMillis() - startTime
                totalMs += elapsed

                val angleLabel = if (angle == 0.0) "  0°" else if (angle > 0) "+%2d°".format(angle.toInt()) else "%3d°".format(angle.toInt())

                if (result == null) {
                    angleResults.add("$angleLabel=FAIL(no result,${elapsed}ms)")
                    fileFailed++
                    failed++
                } else {
                    val match = result.digits7 == expected
                    if (match) {
                        angleResults.add("$angleLabel=PASS(${elapsed}ms)")
                        filePassed++
                        passed++
                    } else {
                        angleResults.add("$angleLabel=FAIL(got ${result.digits7},${elapsed}ms)")
                        fileFailed++
                        failed++
                    }
                }
            }

            println(" → ${filePassed}/${testAngles.size} passed" + if (fileFailed > 0) " (${fileFailed} failed)" else "")
            for (r in angleResults) println("         $r")
        } catch (e: Exception) {
            println(" → ERROR: ${e.message}")
            errors++
        }
    }

    println()
    println("======================================")
    println("Results: $passed passed, $failed failed, $errors errors")
    println("Total decode time: ${totalMs}ms across ${passed + failed} tests")
    println("Average per test: ${if (passed + failed > 0) totalMs / (passed + failed) else 0}ms")

    if (failed > 0 || errors > 0) {
        kotlin.system.exitProcess(1)
    }
}
