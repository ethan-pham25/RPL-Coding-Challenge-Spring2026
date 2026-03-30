from __future__ import annotations

import gc
import importlib
import os
import uuid


SUBMISSION_MODULE_NAME = os.environ.get("RING_BUFFER_MODULE", "solution")
submission_module = importlib.import_module(SUBMISSION_MODULE_NAME)
SharedRingBuffer = getattr(submission_module, "SharedRingBuffer")
RingSpec = getattr(submission_module, "RingSpec", None)
NO_READER = getattr(SharedRingBuffer, "_NO_READER", -1)


def make_name(prefix: str = "rb") -> str:
    return f"{prefix}{uuid.uuid4().hex[:20]}"


def reader_slot(reader: int) -> int:
    return 6 + (reader * 3)


def release_mem_views(*views: memoryview | None) -> None:
    for view in views:
        if view is None:
            continue
        try:
            view.release()
        except Exception:
            pass


def drop_local_views(ring) -> None:
    if ring is None:
        return
    try:
        ring.ring_buffer.release()
    except Exception:
        pass
    try:
        del ring.ring_buffer
    except Exception:
        pass
    try:
        ring.header = None
    except Exception:
        pass
    try:
        del ring.header
    except Exception:
        pass


def cleanup_ring(ring, *, unlink: bool = True) -> None:
    if ring is None:
        return
    drop_local_views(ring)
    gc.collect()
    try:
        ring.close()
    except Exception:
        pass
    if unlink:
        try:
            ring.unlink()
        except FileNotFoundError:
            pass


def set_reader_state(ring, reader: int, *, pos: int | None = None, alive: int | None = None) -> None:
    slot = reader_slot(reader)
    if pos is not None:
        ring.header[slot] = pos
    if alive is not None:
        ring.header[slot + 1] = alive
    if hasattr(ring, "_reader_positions_dirty"):
        ring._reader_positions_dirty = True


def mark_reader_alive(ring, reader_index: int | None = None) -> None:
    if reader_index is None:
        reader_index = ring.reader
    set_reader_state(ring, reader_index, alive=1)
