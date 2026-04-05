from __future__ import annotations

import unittest

from tests.support import SharedBuffer, cleanup_buffer, make_name, release_mem_views


class ApplicantSharedBufferContractTests(unittest.TestCase):
    def _make_buffer(self, *, size: int = 16, num_readers: int = 1, reader: int = 0):
        buffer_obj = SharedBuffer(
            name=make_name("applicant"),
            create=True,
            size=size,
            num_readers=num_readers,
            reader=reader,
        )
        self.addCleanup(cleanup_buffer, buffer_obj)
        return buffer_obj

    def test_constructor_rejects_negative_buffer_size(self) -> None:
        with self.assertRaises(ValueError):
            self._make_buffer(size=-1)

    def test_constructor_rejects_negative_reader_count(self) -> None:
        with self.assertRaises(ValueError):
            self._make_buffer(num_readers=-1)

    def test_reader_only_instances_reject_writer_methods(self) -> None:
        ring = self._make_buffer()
        for method_name, args in (
            ("update_write_pos", (1,)),
            ("inc_writer_pos", (1,)),
            ("expose_writer_mem_view", (1,)),
        ):
            with self.subTest(method=method_name):
                with self.assertRaises(RuntimeError):
                    getattr(ring, method_name)(*args)
