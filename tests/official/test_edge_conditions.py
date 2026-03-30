from __future__ import annotations

import unittest

from tests.support import SharedRingBuffer, cleanup_ring, make_name, mark_reader_alive, release_mem_views, set_reader_state


class SharedRingBufferEdgeConditionTests(unittest.TestCase):
    def _make_ring(self, *, size: int = 16, num_readers: int = 1, reader: int = 0):
        ring = SharedRingBuffer(
            name=make_name("edge"),
            create=True,
            size=size,
            num_readers=num_readers,
            reader=reader,
        )
        self.addCleanup(cleanup_ring, ring)
        return ring

    def test_int_to_pos_wraps_large_values(self):
        ring = self._make_ring(size=16)
        self.assertEqual(ring.int_to_pos(0), 0)
        self.assertEqual(ring.int_to_pos(16), 0)
        self.assertEqual(ring.int_to_pos(17), 1)
        self.assertEqual(ring.int_to_pos((16 * 100) + 7), 7)

    def test_expose_writer_zero_request_returns_empty(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(5)
        ring.update_reader_pos(0)
        mv1, mv2, n, wrap = ring.expose_writer_mem_view(0)
        self.assertEqual(n, 0)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 0)
        self.assertIsNone(mv2)
        release_mem_views(mv1, mv2)

    def test_expose_reader_zero_request_returns_empty(self):
        ring = self._make_ring(size=16)
        ring.update_reader_pos(3)
        ring.update_write_pos(8)
        mv1, mv2, n, wrap = ring.expose_reader_mem_view(0)
        self.assertEqual(n, 0)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 0)
        self.assertIsNone(mv2)
        release_mem_views(mv1, mv2)

    def test_expose_writer_clamps_when_request_exceeds_writable(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(12)
        ring.update_reader_pos(0)
        mark_reader_alive(ring)
        mv1, mv2, n, wrap = ring.expose_writer_mem_view(100)
        self.assertEqual(n, 4)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 4)
        self.assertEqual(len(mv2) if mv2 is not None else 0, 0)
        release_mem_views(mv1, mv2)

    def test_expose_reader_clamps_when_request_exceeds_readable_wrap(self):
        ring = self._make_ring(size=16)
        ring.update_reader_pos(14)
        ring.update_write_pos(20)
        mv1, mv2, n, wrap = ring.expose_reader_mem_view(100)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)
        self.assertEqual(len(mv2) if mv2 is not None else 0, 4)
        release_mem_views(mv1, mv2)

    def test_expose_writer_many_wraps_positions(self):
        ring = self._make_ring(size=16)
        write_pos = (16 * 100) + 14
        read_pos = write_pos - 8
        ring.update_write_pos(write_pos)
        ring.update_reader_pos(read_pos)
        mv1, mv2, n, wrap = ring.expose_writer_mem_view(6)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)
        self.assertEqual(len(mv2) if mv2 is not None else 0, 4)
        release_mem_views(mv1, mv2)

    def test_expose_reader_many_wraps_positions(self):
        ring = self._make_ring(size=16)
        read_pos = (16 * 100) + 14
        write_pos = read_pos + 6
        ring.update_reader_pos(read_pos)
        ring.update_write_pos(write_pos)
        mv1, mv2, n, wrap = ring.expose_reader_mem_view(6)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)
        self.assertEqual(len(mv2) if mv2 is not None else 0, 4)
        release_mem_views(mv1, mv2)

    def test_simple_read_leaves_destination_tail_unchanged_when_larger(self):
        ring = self._make_ring(size=16)
        ring.ring_buffer[:] = bytes(range(16))
        ring.update_reader_pos(14)
        ring.update_write_pos(20)
        reader_mv = ring.expose_reader_mem_view(6)

        dst = bytearray([0xAA] * 10)
        ring.simple_read(reader_mv, dst)
        self.assertEqual(bytes(dst[:6]), b"\x0e\x0f\x00\x01\x02\x03")
        self.assertEqual(bytes(dst[6:]), b"\xaa\xaa\xaa\xaa")
        release_mem_views(reader_mv[0], reader_mv[1])

    def test_simple_write_leaves_ring_tail_unchanged_when_src_shorter_wrap(self):
        ring = self._make_ring(size=16)
        ring.ring_buffer[:] = bytes([0xEE] * 16)
        ring.update_write_pos(14)
        ring.update_reader_pos(6)
        writer_mv = ring.expose_writer_mem_view(6)

        ring.simple_write(writer_mv, b"\x01\x02\x03")
        self.assertEqual(bytes(ring.ring_buffer[14:16]), b"\x01\x02")
        self.assertEqual(bytes(ring.ring_buffer[0:1]), b"\x03")
        self.assertEqual(bytes(ring.ring_buffer[1:4]), b"\xee\xee\xee")
        release_mem_views(writer_mv[0], writer_mv[1])

    def test_compute_max_force_rescan_overrides_stale_cache(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(40)
        ring.update_reader_pos(30)
        mark_reader_alive(ring)
        ring._min_reader_pos_cache = 0
        ring._reader_positions_dirty = False
        ring._writes_since_min_scan = 0
        writable = ring.compute_max_amount_writable(force_rescan=True)
        self.assertEqual(writable, 6)

    def test_compute_max_recovers_when_cached_used_exceeds_size(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(100)
        ring.update_reader_pos(95)
        mark_reader_alive(ring)
        ring._min_reader_pos_cache = 80
        ring._reader_positions_dirty = False
        ring._writes_since_min_scan = 0
        writable = ring.compute_max_amount_writable()
        self.assertEqual(writable, 11)

    def test_compute_max_periodic_rescan_picks_external_reader_update(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(20)
        ring.update_reader_pos(10)
        mark_reader_alive(ring)
        self.assertEqual(ring.compute_max_amount_writable(), 6)

        ring.header[ring.reader_pos_index] = 19
        ring._reader_positions_dirty = False
        ring._writes_since_min_scan = ring._min_reader_pos_refresh_interval

        self.assertEqual(ring.compute_max_amount_writable(), 15)

    def test_compute_max_multi_reader_uses_global_min(self):
        ring = self._make_ring(size=16, num_readers=3, reader=0)
        ring.update_write_pos(105)
        set_reader_state(ring, 0, pos=100, alive=1)
        set_reader_state(ring, 1, pos=90, alive=1)
        set_reader_state(ring, 2, pos=95, alive=1)
        ring._reader_positions_dirty = True
        writable = ring.compute_max_amount_writable()
        self.assertEqual(writable, 1)

    def test_compute_max_negative_used_asserts(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(5)
        ring._min_reader_pos_cache = 10
        ring._reader_positions_dirty = False
        ring._writes_since_min_scan = 0
        with self.assertRaises(AssertionError):
            ring.compute_max_amount_writable()
