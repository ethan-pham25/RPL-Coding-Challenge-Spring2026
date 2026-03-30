from __future__ import annotations

import unittest

import numpy as np

from tests.support import SharedRingBuffer, cleanup_ring, make_name, mark_reader_alive, release_mem_views


class SharedRingBufferCoreTests(unittest.TestCase):
    def _make_ring(self, *, size: int = 32, num_readers: int = 1, reader: int = 0):
        ring = SharedRingBuffer(
            name=make_name("core"),
            create=True,
            size=size,
            num_readers=num_readers,
            reader=reader,
        )
        self.addCleanup(cleanup_ring, ring)
        return ring

    def test_initial_header_and_sizes(self):
        ring = self._make_ring(size=64, num_readers=2, reader=0)

        self.assertEqual(ring.header_size, 8 * (6 + 2 * 3))
        self.assertEqual(ring.shared_mem_size, ring.header_size + 64)
        self.assertEqual(int(ring.header[0]), 64)
        self.assertEqual(int(ring.header[ring.num_readers_index]), 2)
        self.assertEqual(len(ring.ring_buffer), 64)

    def test_header_size_cache_aligned_rounds_up(self):
        ring = SharedRingBuffer(
            name=make_name("core"),
            create=True,
            size=64,
            num_readers=2,
            reader=0,
            cache_align=True,
            cache_size=64,
        )
        self.addCleanup(cleanup_ring, ring)

        self.assertEqual(ring.header_size, 128)
        self.assertEqual(ring.header_size % 64, 0)
        self.assertEqual(len(ring.ring_buffer), 64)

    def test_header_size_cache_aligned_already_aligned(self):
        ring = SharedRingBuffer(
            name=make_name("core"),
            create=True,
            size=64,
            num_readers=2,
            reader=0,
            cache_align=True,
            cache_size=32,
        )
        self.addCleanup(cleanup_ring, ring)

        self.assertEqual(ring.header_size, 96)
        self.assertEqual(ring.header_size % 32, 0)
        self.assertEqual(len(ring.ring_buffer), 64)

    def test_cache_aligned_invalid_cache_size_raises(self):
        with self.assertRaises(ValueError):
            SharedRingBuffer(
                name=make_name("core"),
                create=True,
                size=64,
                num_readers=1,
                reader=0,
                cache_align=True,
                cache_size=48,
            )
        with self.assertRaises(ValueError):
            SharedRingBuffer(
                name=make_name("core"),
                create=True,
                size=64,
                num_readers=1,
                reader=0,
                cache_align=True,
                cache_size=0,
            )

    def test_update_and_increment_positions(self):
        ring = self._make_ring(size=64, num_readers=1, reader=0)

        ring.update_write_pos(10)
        ring.update_reader_pos(4)
        self.assertEqual(int(ring.get_write_pos()), 10)
        self.assertEqual(int(ring.header[ring.reader_pos_index]), 4)

        ring.inc_writer_pos(7)
        ring.inc_reader_pos(3)
        self.assertEqual(int(ring.header[ring.write_pos_index]), 17)
        self.assertEqual(int(ring.header[ring.reader_pos_index]), 7)

    def test_compute_max_amount_writable_single_reader(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)

        ring.update_write_pos(20)
        ring.update_reader_pos(5)
        mark_reader_alive(ring)
        writable = ring.compute_max_amount_writable()
        self.assertEqual(writable, 17)
        self.assertEqual(int(ring.header[ring.max_amount_writable_index]), 17)

    def test_set_reader_active_controls_liveness_slot(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)

        ring.set_reader_active(True)
        self.assertTrue(ring.is_reader_active())
        self.assertEqual(int(ring.header[ring.reader_pos_index + 1]), 1)

        ring.set_reader_active(False)
        self.assertFalse(ring.is_reader_active())
        self.assertEqual(int(ring.header[ring.reader_pos_index + 1]), 0)

    def test_expose_writer_mem_view_contiguous(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)

        ring.update_write_pos(3)
        ring.update_reader_pos(0)
        ring.compute_max_amount_writable()

        mv1, mv2, n, wrap = ring.expose_writer_mem_view(8)
        self.assertEqual(n, 8)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 8)
        self.assertIsNone(mv2)
        release_mem_views(mv1, mv2)

    def test_expose_writer_mem_view_wraparound(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)

        ring.update_write_pos(14)
        ring.update_reader_pos(6)
        ring.compute_max_amount_writable()

        mv1, mv2, n, wrap = ring.expose_writer_mem_view(6)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)
        self.assertEqual(len(mv2), 4)
        release_mem_views(mv1, mv2)

    def test_expose_reader_mem_view_contiguous(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)

        ring.update_reader_pos(4)
        ring.update_write_pos(15)

        mv1, mv2, n, wrap = ring.expose_reader_mem_view(8)
        self.assertEqual(n, 8)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 8)
        self.assertIsNone(mv2)
        release_mem_views(mv1, mv2)

    def test_expose_reader_mem_view_wraparound_shape(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)

        ring.update_reader_pos(14)
        ring.update_write_pos(20)

        mv1, mv2, n, wrap = ring.expose_reader_mem_view(6)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)
        self.assertEqual(len(mv2), 4)
        release_mem_views(mv1, mv2)

    def test_simple_write_contiguous(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes([0] * 16)
        ring.update_write_pos(3)
        ring.update_reader_pos(0)
        writer_mv = ring.expose_writer_mem_view(5)

        src = np.array([10, 11, 12, 13, 14], dtype=np.uint8)
        ring.simple_write(writer_mv, src)

        self.assertEqual(bytes(ring.ring_buffer[3:8]), b"\x0a\x0b\x0c\x0d\x0e")
        release_mem_views(writer_mv[0], writer_mv[1])

    def test_simple_write_wrap(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes([0] * 16)
        ring.update_write_pos(14)
        ring.update_reader_pos(6)
        writer_mv = ring.expose_writer_mem_view(6)

        ring.simple_write(writer_mv, b"\x01\x02\x03\x04\x05\x06")
        self.assertEqual(bytes(ring.ring_buffer[14:16]), b"\x01\x02")
        self.assertEqual(bytes(ring.ring_buffer[0:4]), b"\x03\x04\x05\x06")
        release_mem_views(writer_mv[0], writer_mv[1])
