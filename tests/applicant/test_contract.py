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

    # This is not part of the challenge contract but this file is still a good place
    def test_slowest_reader_position_is_none_initially(self):
        ring = self._make_buffer()
        self.assertEqual(None, ring.get_slowest_reader_position())

    def test_slowest_reader_position_is_zero_after_activating_reader(self):
        ring = self._make_buffer()
        ring.set_reader_active(True)
        self.assertEqual(0, ring.get_slowest_reader_position())

    def test_slowest_reader_position_is_min_of_active_readers(self):
        with self._make_buffer(num_readers = 3) as reader0:
            reader0.set_reader_active(True)
            reader0.update_reader_pos(100)

            reader1 = SharedBuffer(
                name = reader0.name,
                create = False,
                size = 16,
                num_readers = 3,
                reader = 1,
            )
            self.addCleanup(cleanup_buffer, reader1)
            reader1.update_reader_pos(50) # Inactive so shouldn't count

            reader2 = SharedBuffer(
                name = reader0.name,
                create = False,
                size = 16,
                num_readers = 3,
                reader = 2,
            )
            self.addCleanup(cleanup_buffer, reader2, unlink = True)
            reader2.set_reader_active(True)
            reader2.update_reader_pos(1000)

            self.assertEqual(100, reader0.get_slowest_reader_position())