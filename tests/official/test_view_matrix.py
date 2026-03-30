from __future__ import annotations

import unittest

from tests.support import SharedRingBuffer, cleanup_ring, make_name, release_mem_views, set_reader_state


class SharedRingBufferViewMatrixTests(unittest.TestCase):
    def _make_ring(self, *, size: int = 16, num_readers: int = 1, reader: int = 0):
        ring = SharedRingBuffer(
            name=make_name("matrix"),
            create=True,
            size=size,
            num_readers=num_readers,
            reader=reader,
        )
        self.addCleanup(cleanup_ring, ring)
        ring.ring_buffer[:] = bytes(i % 256 for i in range(size))
        return ring

    @staticmethod
    def _expected_writer(ring_size: int, write_pos: int, min_reader_pos: int, request: int) -> tuple[int, bool, int, int]:
        used = write_pos - min_reader_pos
        writable = ring_size - used
        n = request if request <= writable else writable
        idx = write_pos % ring_size
        if idx + n <= ring_size:
            return (n, False, n, 0)
        mv1 = ring_size - idx
        mv2 = n - mv1
        return (n, True, mv1, mv2)

    @staticmethod
    def _expected_reader(ring_size: int, write_pos: int, read_pos: int, request: int) -> tuple[int, bool, int, int]:
        readable = write_pos - read_pos
        n = request if request <= readable else readable
        idx = read_pos % ring_size
        if idx + n <= ring_size:
            return (n, False, n, 0)
        mv1 = ring_size - idx
        mv2 = n - mv1
        return (n, True, mv1, mv2)

    def test_writer_view_cases(self):
        cases = [
            ("writer-zero", 16, 3, 0, 0),
            ("writer-contiguous-small", 16, 3, 0, 5),
            ("writer-exact-end-boundary", 16, 12, 0, 4),
            ("writer-wrap", 16, 14, 6, 6),
            ("writer-clamp-to-writable", 16, 12, 0, 8),
            ("writer-clamp-small-free", 16, 14, 13, 8),
            ("writer-full", 16, 16, 0, 5),
        ]

        for label, size, write_pos, read_pos, request in cases:
            with self.subTest(case=label):
                ring = self._make_ring(size=size)
                ring.update_write_pos(write_pos)
                set_reader_state(ring, 0, pos=read_pos, alive=1)
                ring.compute_max_amount_writable(force_rescan=True)

                got = ring.expose_writer_mem_view(request)
                expected = self._expected_writer(size, write_pos, read_pos, request)

                mv1, mv2, n, wrap = got
                exp_n, exp_wrap, exp_mv1_len, exp_mv2_len = expected
                self.assertEqual(n, exp_n, label)
                self.assertEqual(wrap, exp_wrap, label)
                self.assertEqual(len(mv1), exp_mv1_len, label)
                self.assertEqual(len(mv2) if mv2 is not None else 0, exp_mv2_len, label)
                release_mem_views(mv1, mv2)

    def test_reader_view_cases(self):
        cases = [
            ("reader-zero", 16, 8, 4, 0),
            ("reader-empty", 16, 8, 8, 5),
            ("reader-contiguous-small", 16, 12, 4, 5),
            ("reader-clamp-to-readable", 16, 12, 4, 20),
            ("reader-exact-end-boundary", 16, 20, 12, 4),
            ("reader-wrap", 16, 20, 14, 6),
            ("reader-wrap-small-request", 16, 21, 14, 3),
        ]

        for label, size, write_pos, read_pos, request in cases:
            with self.subTest(case=label):
                ring = self._make_ring(size=size)
                ring.update_write_pos(write_pos)
                ring.update_reader_pos(read_pos)

                got = ring.expose_reader_mem_view(request)
                expected = self._expected_reader(size, write_pos, read_pos, request)

                mv1, mv2, n, wrap = got
                exp_n, exp_wrap, exp_mv1_len, exp_mv2_len = expected
                self.assertEqual(n, exp_n, label)
                self.assertEqual(wrap, exp_wrap, label)
                self.assertEqual(len(mv1), exp_mv1_len, label)
                self.assertEqual(len(mv2) if mv2 is not None else 0, exp_mv2_len, label)
                release_mem_views(mv1, mv2)

    def test_compute_max_amount_writable_multi_reader_uses_min_reader(self):
        ring = self._make_ring(size=32, num_readers=3, reader=0)
        ring.update_write_pos(40)
        set_reader_state(ring, 0, pos=35, alive=1)
        set_reader_state(ring, 1, pos=22, alive=1)
        set_reader_state(ring, 2, pos=30, alive=1)

        writable = ring.compute_max_amount_writable(force_rescan=True)
        self.assertEqual(writable, 14)
        self.assertEqual(int(ring.header[ring.max_amount_writable_index]), 14)

    def test_reader_invariant_cases_raise(self):
        ring1 = self._make_ring(size=16)
        ring1.update_write_pos(5)
        ring1.update_reader_pos(9)
        with self.assertRaises(AssertionError):
            ring1.expose_reader_mem_view(1)

        ring2 = self._make_ring(size=16)
        ring2.update_write_pos(33)
        ring2.update_reader_pos(0)
        with self.assertRaises(AssertionError):
            ring2.expose_reader_mem_view(1)
