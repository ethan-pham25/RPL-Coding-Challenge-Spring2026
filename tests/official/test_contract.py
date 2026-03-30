from __future__ import annotations

from dataclasses import dataclass
import time
import unittest
import weakref
from multiprocessing import shared_memory
from unittest.mock import patch

from tests.support import (
    NO_READER,
    RingSpec,
    SharedRingBuffer,
    cleanup_ring,
    drop_local_views,
    make_name,
    reader_slot,
    release_mem_views,
    set_reader_state,
)


@dataclass
class RingPair:
    creator: SharedRingBuffer
    peer: SharedRingBuffer
    probe: shared_memory.SharedMemory


class SharedRingBufferContractTests(unittest.TestCase):
    def _make_pair(
        self,
        *,
        size: int = 16,
        num_readers: int = 1,
        creator_reader: int = NO_READER,
        peer_reader: int = 0,
        cache_align: bool = False,
        cache_size: int = 64,
    ) -> RingPair:
        name = make_name("pair")
        creator = SharedRingBuffer(
            name=name,
            create=True,
            size=size,
            num_readers=num_readers,
            reader=creator_reader,
            cache_align=cache_align,
            cache_size=cache_size,
        )
        peer = SharedRingBuffer(
            name=name,
            create=False,
            size=size,
            num_readers=num_readers,
            reader=peer_reader,
            cache_align=cache_align,
            cache_size=cache_size,
        )
        probe = shared_memory.SharedMemory(name=name, create=False)

        self.addCleanup(cleanup_ring, creator, unlink=True)
        self.addCleanup(cleanup_ring, peer, unlink=False)
        self.addCleanup(probe.close)
        return RingPair(creator=creator, peer=peer, probe=probe)

    def _collect_until_gone(self, ref: weakref.ReferenceType[SharedRingBuffer], attempts: int = 8) -> None:
        for _ in range(attempts):
            if ref() is None:
                return
            import gc

            gc.collect()
            time.sleep(0.01)
        self.assertIsNone(ref())

    def test_construction_reader_and_writer_slots(self) -> None:
        pair = self._make_pair(num_readers=3, peer_reader=2)
        self.assertIsNone(pair.creator.reader_pos_index)
        self.assertEqual(pair.peer.reader_pos_index, reader_slot(2))
        self.assertEqual(pair.probe.name, pair.creator.name)

    def test_cache_alignment_accepts_valid_sizes(self) -> None:
        for cache_size, expected_header_size in ((32, 96), (64, 128)):
            with self.subTest(cache_size=cache_size):
                pair = self._make_pair(size=64, num_readers=2, cache_align=True, cache_size=cache_size)
                self.assertEqual(pair.creator.header_size, expected_header_size)
                self.assertEqual(pair.creator.header_size % cache_size, 0)

    def test_cache_alignment_rejects_invalid_sizes(self) -> None:
        for cache_size in (0, 48):
            with self.subTest(cache_size=cache_size):
                with self.assertRaises(ValueError):
                    SharedRingBuffer(
                        name=make_name("pair"),
                        create=True,
                        size=32,
                        num_readers=1,
                        reader=NO_READER,
                        cache_align=True,
                        cache_size=cache_size,
                    )

    def test_ring_spec_rejects_invalid_size_and_reader_count(self) -> None:
        for size, num_readers in ((0, 1), (-1, 1), (16, 0), (16, -1)):
            with self.subTest(size=size, num_readers=num_readers):
                with self.assertRaises(ValueError):
                    RingSpec(name="spec", size=size, num_readers=num_readers)

    def test_reader_only_methods_raise_on_writer_instances(self) -> None:
        pair = self._make_pair()
        cases = [
            ("update_reader_pos", (1,)),
            ("inc_reader_pos", (1,)),
            ("expose_reader_mem_view", (1,)),
            ("jump_to_writer", ()),
        ]
        for method_name, args in cases:
            with self.subTest(method=method_name):
                with self.assertRaises(RuntimeError):
                    getattr(pair.creator, method_name)(*args)

    def test_reader_only_methods_work_on_reader_instances(self) -> None:
        pair = self._make_pair()
        pair.creator.update_write_pos(5)

        pair.peer.update_reader_pos(1)
        pair.peer.inc_reader_pos(1)
        self.assertEqual(int(pair.peer.header[pair.peer.reader_pos_index]), 2)

        mv1, mv2, size_readable, wrap_around = pair.peer.expose_reader_mem_view(2)
        self.assertEqual(size_readable, 2)
        self.assertFalse(wrap_around)
        release_mem_views(mv1, mv2)

        pair.peer.jump_to_writer()
        self.assertEqual(int(pair.peer.header[pair.peer.reader_pos_index]), 5)

    def test_reader_context_manager_updates_alive_flag(self) -> None:
        pair = self._make_pair()
        alive_index = pair.peer.reader_pos_index + 1

        self.assertEqual(int(pair.creator.header[alive_index]), 0)
        pair.peer.__enter__()
        self.assertEqual(int(pair.creator.header[alive_index]), 1)
        pair.peer.__exit__(None, None, None)
        self.assertEqual(int(pair.creator.header[alive_index]), 0)

    def test_writer_context_manager_does_not_touch_alive_slots(self) -> None:
        pair = self._make_pair()
        alive_index = reader_slot(0) + 1
        pair.creator.header[alive_index] = 7

        pair.creator.__enter__()
        pair.creator.__exit__(None, None, None)

        self.assertEqual(int(pair.peer.header[alive_index]), 7)

    def test_compute_max_amount_writable_rescans_when_cache_is_stale(self) -> None:
        pair = self._make_pair(size=16)
        pair.creator.update_write_pos(20)
        set_reader_state(pair.creator, 0, pos=10, alive=1)
        pair.creator._min_reader_pos_cache = 0
        pair.creator._reader_positions_dirty = False
        pair.creator._writes_since_min_scan = pair.creator._min_reader_pos_refresh_interval

        with patch.object(pair.creator, "_scan_min_reader_pos", wraps=pair.creator._scan_min_reader_pos) as scan:
            writable = pair.creator.compute_max_amount_writable()

        self.assertEqual(writable, 6)
        self.assertGreaterEqual(scan.call_count, 1)

    def test_compute_max_amount_writable_rescans_when_cached_used_exceeds_ring_size(self) -> None:
        pair = self._make_pair(size=16)
        pair.creator.update_write_pos(100)
        set_reader_state(pair.creator, 0, pos=95, alive=1)
        pair.creator._min_reader_pos_cache = 80
        pair.creator._reader_positions_dirty = False
        pair.creator._writes_since_min_scan = 0
        pair.creator._last_min_scan_t = time.perf_counter()

        with patch.object(pair.creator, "_scan_min_reader_pos", wraps=pair.creator._scan_min_reader_pos) as scan:
            writable = pair.creator.compute_max_amount_writable()

        self.assertEqual(writable, 11)
        self.assertGreaterEqual(scan.call_count, 1)

    def test_compute_max_amount_writable_asserts_on_impossible_state(self) -> None:
        negative_used = self._make_pair(size=16)
        negative_used.creator.update_write_pos(5)
        negative_used.creator._min_reader_pos_cache = 10
        negative_used.creator._reader_positions_dirty = False
        negative_used.creator._writes_since_min_scan = 0
        negative_used.creator._last_min_scan_t = time.perf_counter()
        with self.assertRaisesRegex(AssertionError, "used < 0"):
            negative_used.creator.compute_max_amount_writable()

        too_much_used = self._make_pair(size=16)
        too_much_used.creator.update_write_pos(40)
        set_reader_state(too_much_used.creator, 0, pos=0, alive=1)
        too_much_used.creator._min_reader_pos_cache = 0
        too_much_used.creator._reader_positions_dirty = False
        too_much_used.creator._writes_since_min_scan = 0
        too_much_used.creator._last_min_scan_t = time.perf_counter()
        with self.assertRaisesRegex(AssertionError, "used > ring_buffer_size"):
            too_much_used.creator.compute_max_amount_writable()

    def test_simple_write_and_read_roundtrip(self) -> None:
        for label, start_pos, payload in (("contiguous", 0, b"hello"), ("wrapping", 14, b"ABCDEF")):
            with self.subTest(case=label):
                pair = self._make_pair(size=16)
                pair.creator.ring_buffer[:] = b"\x00" * pair.creator.ring_buffer_size
                pair.creator.update_write_pos(start_pos)
                pair.peer.update_reader_pos(start_pos)
                set_reader_state(pair.creator, 0, pos=start_pos, alive=1)

                writer_mem_view = pair.creator.expose_writer_mem_view(len(payload))
                pair.creator.simple_write(writer_mem_view, payload)
                pair.creator.inc_writer_pos(writer_mem_view[2])
                release_mem_views(writer_mem_view[0], writer_mem_view[1])

                reader_mem_view = pair.peer.expose_reader_mem_view(len(payload))
                dst = bytearray(len(payload))
                pair.peer.simple_read(reader_mem_view, dst)
                self.assertEqual(bytes(dst), payload)
                release_mem_views(reader_mem_view[0], reader_mem_view[1])

    def test_calculate_pressure_updates_header(self) -> None:
        empty_pair = self._make_pair(size=16)
        self.assertEqual(empty_pair.creator.calculate_pressure(), 0)
        self.assertEqual(int(empty_pair.creator.header[empty_pair.creator.pressure_index]), 0)

        full_pair = self._make_pair(size=16)
        full_pair.creator.update_write_pos(full_pair.creator.ring_buffer_size)
        set_reader_state(full_pair.creator, 0, pos=0, alive=1)
        self.assertEqual(full_pair.creator.calculate_pressure(), 100)
        self.assertEqual(int(full_pair.creator.header[full_pair.creator.pressure_index]), 100)

    def test_jump_to_writer_discards_unread_data(self) -> None:
        pair = self._make_pair(size=16)
        pair.creator.update_write_pos(9)
        pair.peer.update_reader_pos(2)

        pair.peer.jump_to_writer()
        self.assertEqual(int(pair.peer.header[pair.peer.reader_pos_index]), 9)

        mv1, mv2, size_readable, wrap_around = pair.peer.expose_reader_mem_view(4)
        self.assertEqual(size_readable, 0)
        self.assertFalse(wrap_around)
        self.assertEqual(len(mv1), 0)
        self.assertIsNone(mv2)
        release_mem_views(mv1, mv2)

    def test_finalizer_cleanup_unlinks_only_for_creator(self) -> None:
        creator_calls = []
        creator_name = make_name("finalizer")

        def creator_recorder(name: str, is_creator: bool) -> None:
            creator_calls.append((name, is_creator))
            creator_cleanup(name, is_creator)

        creator_cleanup = SharedRingBuffer._finalizer_cleanup
        with patch.object(SharedRingBuffer, "_finalizer_cleanup", new=staticmethod(creator_recorder)):
            creator = SharedRingBuffer(
                name=creator_name,
                create=True,
                size=16,
                num_readers=1,
                reader=NO_READER,
            )
            creator_ref = weakref.ref(creator)
            drop_local_views(creator)
            del creator
            self._collect_until_gone(creator_ref)

        self.assertEqual(creator_calls, [(creator_name, True)])
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=creator_name, create=False)
