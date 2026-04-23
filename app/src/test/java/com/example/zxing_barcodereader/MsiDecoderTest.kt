package com.example.zxing_barcodereader

import org.junit.Assert.assertEquals
import org.junit.Test

class MsiDecoderTest {

    @Test
    fun testRotationCW90() {
        // 2x3 image
        // 1 2
        // 3 4
        // 5 6
        val pixels = intArrayOf(1, 2, 3, 4, 5, 6)
        val src = MsiPlesseyBarcodeDecoder.GrayImage(pixels, 2, 3)
        val dst = MsiPlesseyBarcodeDecoder.rotateCW90(src)
        
        // Expected 3x2
        // 5 3 1
        // 6 4 2
        assertEquals(3, dst.width)
        assertEquals(2, dst.height)
        assertEquals(5, dst.pixel(0, 0))
        assertEquals(3, dst.pixel(1, 0))
        assertEquals(1, dst.pixel(2, 0))
        assertEquals(6, dst.pixel(0, 1))
        assertEquals(4, dst.pixel(1, 1))
        assertEquals(2, dst.pixel(2, 1))
    }

    @Test
    fun testRotationCW180() {
        val pixels = intArrayOf(1, 2, 3, 4, 5, 6)
        val src = MsiPlesseyBarcodeDecoder.GrayImage(pixels, 2, 3)
        val dst = MsiPlesseyBarcodeDecoder.rotateCW180(src)
        
        // Expected 2x3
        // 6 5
        // 4 3
        // 2 1
        assertEquals(2, dst.width)
        assertEquals(3, dst.height)
        assertEquals(6, dst.pixel(0, 0))
        assertEquals(5, dst.pixel(1, 0))
        assertEquals(4, dst.pixel(0, 1))
        assertEquals(3, dst.pixel(1, 1))
        assertEquals(2, dst.pixel(0, 2))
        assertEquals(1, dst.pixel(1, 2))
    }

    @Test
    fun testRotationCW270() {
        val pixels = intArrayOf(1, 2, 3, 4, 5, 6)
        val src = MsiPlesseyBarcodeDecoder.GrayImage(pixels, 2, 3)
        val dst = MsiPlesseyBarcodeDecoder.rotateCW270(src)
        
        // Expected 3x2 (90 CCW)
        // 2 4 6
        // 1 3 5
        assertEquals(3, dst.width)
        assertEquals(2, dst.height)
        assertEquals(2, dst.pixel(0, 0))
        assertEquals(4, dst.pixel(1, 0))
        assertEquals(6, dst.pixel(2, 0))
        assertEquals(1, dst.pixel(0, 1))
        assertEquals(3, dst.pixel(1, 1))
        assertEquals(5, dst.pixel(2, 1))
    }
}
