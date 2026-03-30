from __future__ import annotations

import unittest

from tests.support import SharedRingBuffer, cleanup_ring, make_name, release_mem_views


class MySharedRingBufferTests(unittest.TestCase):
    def _make_ring(self, *, size: int = 16, num_readers: int = 1, reader: int = 0):
        ring = SharedRingBuffer(
            name=make_name("student"),
            create=True,
            size=size,
            num_readers=num_readers,
            reader=reader,
        )
        self.addCleanup(cleanup_ring, ring)
        return ring

    def test_example_roundtrip(self):
        ring = self._make_ring(size=16)
        ring.update_reader_pos(0)
        ring.update_write_pos(0)

        writer_view = ring.expose_writer_mem_view(4)
        ring.simple_write(writer_view, b"UCI!")
        ring.inc_writer_pos(writer_view[2])
        release_mem_views(writer_view[0], writer_view[1])

        reader_view = ring.expose_reader_mem_view(4)
        dst = bytearray(4)
        ring.simple_read(reader_view, dst)
        release_mem_views(reader_view[0], reader_view[1])

        self.assertEqual(bytes(dst), b"UCI!")
