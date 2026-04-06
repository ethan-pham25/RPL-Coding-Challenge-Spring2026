from __future__ import annotations

import math
import struct
from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np


__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]

# Buffer structure: [header, data]
# Header layout:
# WRITER:
# 0 write_position
# 1 data_view_size
# 2 num_readers
# 3 max_bytes_writable
# 4 pressure
#TODO idk why this 5 dropped_size

# READER:
# 0 read_position
# 1 active
# 2 max_amount_readable
HEADER_SLOT_SIZE = 8 # 8 bytes for 64-bit integer
SLOTS_PER_WRITER = 6
SLOTS_PER_READER = 3


class SharedBuffer(shared_memory.SharedMemory):
    """
    Applicant template.

    Replace every method body with your own implementation while preserving the
    public API used by the official tests.

    The intended contract is:
    - one writer and one or more readers
    - shared state visible across processes
    - bounded storage with reusable space after readers advance
    - reads and writes report how many bytes are actually available
    """

    _NO_READER = -1

    def __init__(
        self,
        name: str,
        create: bool,
        size: int,
        num_readers: int,
        reader: int,
        cache_align: bool = False,
        cache_size: int = 64,
    ):
        """
        Open or create the shared buffer.

        Expected behavior:
        - validate constructor arguments
        - allocate or attach to shared memory
        - initialize any shared metadata needed to track writer and reader state
        - set up local views/fields used by the rest of the methods

        Parameters:
        - `name`: shared memory block name
        - `create`: `True` for the creator/owner, `False` to attach to an existing block
        - `size`: logical payload capacity in bytes
        - `num_readers`: number of reader slots to support
        - `reader`: reader index for this instance, or `_NO_READER` for the writer instance
        - `cache_align` / `cache_size`: optional metadata-layout knobs; you may ignore
          them internally as long as validation and behavior remain correct
        """

        # Calculate the total buffer size, which is the usable size plus the header size
        self.header_size = (SLOTS_PER_WRITER + (num_readers * SLOTS_PER_READER)) * HEADER_SLOT_SIZE
        total_size = self.header_size + size
        self.buffer_size = size

        # Init SharedMemory
        # Note that track = create because only the writer should track the SharedBuffer
        # in standalone Python processes (see https://docs.python.org/3/library/multiprocessing.shared_memory.html
        # , doesn't matter on Windows)
        super().__init__(name, create, total_size, track = create)

        if size < 0:
            raise ValueError(f'Size of buffer in bytes must be positive, was: {size}')

        if num_readers < 0:
            raise ValueError(f'Number of readers must be positive, was: {num_readers}')

        # Reject out of range reader index (0-indexed)
        if reader < -1:
            raise ValueError(f'Reader index must be positive or _NO_READER (-1)'
                             f'for writer, was: {reader}')
        elif reader >= num_readers:
            raise ValueError(f'Reader index must be less than the number of readers, was: {reader}')

        # When cache is aligned, the cache size must be a power of 2
        if cache_align and not self._is_power_of_two(cache_size):
            raise ValueError(f'Cache size must be a power of 2 when aligned, was: {cache_size}')

        # Init attributes
        self._name = name
        self._buffer_size_power_of_two = self._is_power_of_two(self.buffer_size)
        self.num_readers = num_readers
        self._reader = reader

        # Partition the buffer into metadata and data
        self._metadata_view = self.buf[:self.header_size]
        self._data_view = self.buf[self.header_size:]

        # Init shared state/metadata view between SharedBuffer instances
        # On creation only, zero out the header
        # Importantly, this makes all the reader and writer positions 0 cheaply,
        # and also makes readers inactive to begin with!
        if create:
            self._metadata_view[:self.header_size] = b'\x00' * self.header_size

        if self._is_reader():
            # Readers are initially inactive
            self._active = False

    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        """
        Checks whether an input number is a power of two.

        Args:
            n: Input value to check.

        Returns:
            bool: Whether n is a power of two.

        """

        # Since powers of 2 are formatted like 8 = 1000 in binary, a fast way to check
        # is to use bitwise & with n - 1, since it will have the form 7 = 0111
        return n > 0 and not (n & (n - 1))

    def close(self) -> None:
        """
        Release local views and close this process's handle to the shared memory.

        This should not destroy the buffer for other attached processes.
        """
        try:
            super().close()
            # Release local views
            self._metadata_view.release()
            self._data_view.release()
        except Exception:
            pass

    def __enter__(self) -> "SharedBuffer":
        """
        Enter the context manager.

        Reader instances are expected to mark themselves active while inside the
        context. Writer-only instances can simply return `self`.
        """
        # If reader, mark self active
        if self._is_reader():
            self.set_reader_active(True)

        return self

    def __exit__(self, *_):
        """
        Exit the context manager.

        Reader instances are expected to mark themselves inactive on exit, then
        close local resources.
        """
        # If reader, mark self inactive
        if self._is_reader():
            self.set_reader_active(False)

        self.close()

    def calculate_pressure(self) -> int:
        """
        Return current writer pressure as an integer percentage.

        Pressure is based on how much of the bounded storage is currently in use
        relative to the slowest active reader.
        """
        #TODO can optimize by storing pressure in header

        # Storage in use relative to the slowest active reader is simply the number of bytes
        # (difference in position) between the writer and the slowest active reader
        slowest_reader_position = self.get_slowest_reader_position()
        if slowest_reader_position is None:
            stored_bytes = 0
        else:
            stored_bytes = self.get_write_pos() - self.get_slowest_reader_position()

        # Then the pressure is just the stored_bytes as a percentage of total buffer size
        return (100 * stored_bytes) // self.buffer_size

    def _get_reader_header_offset(self, reader_num: int) -> int:
        """
        Gets the starting index of the specified reader's metadata header area.

        Params:
            reader_num: The reader to get the offset of.

        Returns:
            int: The header offset, where this instance's metadata starts.

        """
        return ((SLOTS_PER_WRITER + reader_num * SLOTS_PER_READER)
                    * HEADER_SLOT_SIZE)

    def int_to_pos(self, value: int) -> int:
        """
        Convert an absolute position counter into a position inside the bounded payload area.

        If your design does not use modulo arithmetic internally, you may still
        keep this helper as the mapping from logical positions to buffer offsets.

        Raises:
            ValueError: If value is less than 0.
        """
        if value < 0:
            raise ValueError(f'Absolute position value must be non-negative, was: {value}')
        # If the buffer size is a power of two, use bitwise AND optimization (see power of two func)
        if self._buffer_size_power_of_two:
            return value & (self.buffer_size - 1)
        else:
            # Else just use modulo
            return value % self.buffer_size

    def update_reader_pos(self, new_reader_pos: int) -> None:
        """
        Store this reader's absolute read position in shared state.

        This must fail clearly when called on a writer-only instance.
        """
        self._validate_is_reader()
        # Update shared reader pos
        offset = self._get_reader_header_offset(self._reader)
        struct.pack_into('Q', self._metadata_view, offset, new_reader_pos)

    def set_reader_active(self, active: bool) -> None:
        """
        Mark this reader as active or inactive in shared state.

        Active readers apply backpressure. Inactive readers should not reduce
        writer capacity.
        """
        self._validate_is_reader()
        self._active = active

        # Update shared activity state in header
        offset = self._get_reader_header_offset(self._reader) + HEADER_SLOT_SIZE
        value = 1 if active else 0
        struct.pack_into('Q', self._metadata_view, offset, value)

    def is_reader_active(self) -> bool:
        """
        Return whether this reader is currently marked active.

        This must fail clearly when called on a writer-only instance.
        """
        self._validate_is_reader()
        return self._active

    def update_write_pos(self, new_writer_pos: int) -> None:
        """
        Store the writer's absolute write position in shared state.

        The write position is what makes newly written bytes visible to readers.
        """
        self._validate_is_writer()
        # Update shared writer pos
        # Writer pos has 0 offset in header
        struct.pack_into('Q', self._metadata_view, 0, new_writer_pos)

    def inc_writer_pos(self, inc_amount: int) -> None:
        """
        Advance the writer's absolute position by `inc_amount` bytes.

        This is how a writer publishes bytes after copying them into the buffer.
        """
        self._validate_is_writer()
        # Increment shared writer pos
        new_position = self.get_write_pos() + inc_amount
        self.update_write_pos(new_position)

    def inc_reader_pos(self, inc_amount: int) -> None:
        """
        Advance this reader's absolute position by `inc_amount` bytes.

        This is how a reader consumes bytes after reading them.
        """
        self._validate_is_reader()
        new_position = self.get_reader_pos(self._reader) + inc_amount
        self.update_reader_pos(new_position)

    def get_write_pos(self) -> int:
        """
        Return the current absolute writer position.

        Readers can use this to resynchronize or compute how much data is available.
        """
        # Note that lack of writer validation is intentional; see above
        # Get write pos stored in shared header
        return struct.unpack_from('Q', self._metadata_view, 0)[0]

    def get_reader_pos(self, reader_num: int) -> int:
        """
        Return the absolute reader position for the given reader num.

        Params:
            reader_num: The reader to fetch the position for.

        Returns:
            int: The reader's absolute position.

        """

        # Stored in shared header
        offset = self._get_reader_header_offset(reader_num)
        return struct.unpack_from('Q', self._metadata_view, offset)[0]

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        """
        Return how many bytes the writer can safely expose right now.

        This should take active readers into account. `force_rescan=True` is used
        by the tests to ensure externally updated reader positions are observed.
        """
        # Lack of writer validation is intentional, as all buffers can report this
        # If there are no active readers, the writer can write the whole buffer
        slowest_reader_pos = self.get_slowest_reader_position()
        if slowest_reader_pos is None:
            return self.buffer_size
        else:
            # Otherwise we can write up to the slowest active reader
            # If write pos equals slowest reader, we are actually empty not full
            # And we will keep the writer 1 byte away from the slowest reader when not empty
            write_pos = self.get_write_pos()
            if write_pos == slowest_reader_pos: # Empty
                return self.buffer_size
            else:
                unread_bytes = self.get_write_pos() - slowest_reader_pos
                return self.buffer_size - unread_bytes

    def jump_to_writer(self) -> None:
        """
        Move this reader directly to the current writer position.

        Use this when a reader has fallen too far behind and old unread data is
        no longer retained.
        """
        self._validate_is_reader()
        self.update_reader_pos(self.get_write_pos())

    def _make_memory_views(self, size: int, read: bool):
        # Guard clause to save time
        if size <= 0:
            return memoryview(bytearray()), None, 0, False

        # Get the current position and usably bytes
        if read:
            current_pos = self.get_reader_pos(self._reader)
            usable_bytes = self.get_write_pos() - current_pos
        else:
            current_pos = self.get_write_pos()
            usable_bytes = self.compute_max_amount_writable()

        # Clamp actual size
        actual_size = min(size, usable_bytes)

        # Map absolute position to buffer offset
        start_idx = self.int_to_pos(current_pos)
        # If buffer would overflow, split and wrap around
        if start_idx + actual_size > self.buffer_size:
            split = True
            # First part of memory view is at the end of the buffer
            # Clamp to logical buffer boundary (buffer_size), not physical allocation size
            # If we don't do this, we end up going too far due to the page allocation size
            # of SharedMemory
            end_idx = self.buffer_size
            mv1 = self._data_view[start_idx : end_idx]

            # Second part of memory view (wrapped around) is at beginning
            remaining_bytes = actual_size - (end_idx - start_idx)
            mv2 = self._data_view[0:remaining_bytes]
        else:
            split = False
            mv1 = self._data_view[start_idx : start_idx + actual_size]
            mv2 = None

        return mv1, mv2, actual_size, split

    def expose_writer_mem_view(self, size: int) -> RingView:
        """
        Return a writable view tuple for up to `size` bytes.

        The return shape is:
        - `mv1`: first writable view
        - `mv2`: optional second writable view if the exposed region is split
        - `actual_size`: how many bytes are actually writable right now
        - `split`: whether the writable region is split across two views

        If less than `size` bytes are currently writable, clamp to the amount
        available rather than raising.
        """
        self._validate_is_writer()
        return self._make_memory_views(size, False)

    def expose_reader_mem_view(self, size: int) -> RingView:
        """
        Return a readable view tuple for up to `size` bytes.

        The shape matches `expose_writer_mem_view()`. If less than `size` bytes
        are currently readable, clamp to the amount available rather than raising.
        """
        self._validate_is_reader()
        mv1, mv2, actual_size, split = self._make_memory_views(size, True)
        mv1 = mv1.toreadonly()
        if mv2 is not None:
            mv2 = mv2.toreadonly()

        return mv1, mv2, actual_size, split

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        """
        Copy bytes from `src` into the exposed writer view(s).

        If `src` is larger than the destination region, copy only the prefix that fits.
        This helper should not publish data by itself; publishing happens when the
        writer position is advanced.
        """
        self._validate_is_writer()
        mv1, mv2, total_size, split = writer_mem_view

        # 1. If src fits into mv1 completely, copy it all
        # 2. Else, fill up mv1 then put the rest in mv2 if it exists
        src_len = len(src)
        write_len = min(src_len, total_size)
        if write_len <= 0:
            return

        if not split or write_len <= mv1.nbytes:
            mv1[:write_len] = src[:write_len] # Fits entirely
        else:
            # Make sure to use slice assignment and not normal assignment to copy data in
            mv1[:] = src[:mv1.nbytes] # Fill up mv1
            remaining_bytes = write_len - mv1.nbytes
            # If mv2 exists, fill it up until its full or remaining bytes are all written
            if mv2 is not None:
                mv2[:remaining_bytes] = src[mv1.nbytes:write_len]

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        """
        Copy bytes from the exposed reader view(s) into `dst`.

        If `dst` is smaller than the readable region, copy only the prefix that fits.
        This helper should not consume data by itself; consumption happens when the
        reader position is advanced.
        """
        self._validate_is_reader()
        mv1, mv2, total_size, split = reader_mem_view

        # 1. If mv1 fits entirely into dst, copy the whole thing
        # 2. Otherwise, start spilling over into mv2
        dst_len = len(dst)
        read_len = min(dst_len, total_size)
        if read_len <= 0:
            return

        if not split or read_len <= mv1.nbytes:
            dst[:read_len] = mv1[:read_len]
        else:
            dst[:mv1.nbytes] = mv1[:]
            if mv2 is not None:
                # We either copy mv2 entirely if it can fit in dst, otherwise just the remaining space
                remaining_bytes = read_len - mv1.nbytes
                dst[mv1.nbytes : read_len] = mv2[:remaining_bytes]

    def write_array(self, arr: np.ndarray) -> int:
        """
        Write a NumPy array's raw bytes into the shared buffer.

        Return the number of bytes written. If the full array does not fit, the
        contract used by the tests expects this method to return `0`.
        """
        self._validate_is_writer()
        # First, get memoryviews
        mv1, mv2, total_writable_bytes, split = self.expose_writer_mem_view(arr.nbytes)

        # Then, if there are not enough writable bytes, write nothing and return that we wrote 0
        # bytes
        if arr.nbytes > total_writable_bytes:
            return 0

        try:
            # Otherwise, write into the views
            # If only mv1 exists, we know we can just write the whole array
            input_bytes = memoryview(arr).cast('B')
            if not split:
                mv1[:arr.nbytes] = input_bytes[:arr.nbytes]
                return arr.nbytes
            else:
                # If they're not contiguous we have to write into mv1 first then mv2
                mv1[:] = input_bytes[:mv1.nbytes]
                if mv2 is not None:
                    remaining_bytes = arr.nbytes - mv1.nbytes
                    mv2[:remaining_bytes] = input_bytes[mv1.nbytes:]

                return arr.nbytes
        finally:
            # Release views that we just generated
            mv1.release()
            if mv2 is not None:
                mv2.release()

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """
        Read `nbytes` from the shared buffer and interpret them as `dtype`.

        Return a NumPy array view/copy of the requested bytes when enough data is
        available. If there are not enough readable bytes, return an empty array
        with the requested dtype.
        """
        self._validate_is_reader()
        # First, get the data into memoryviews
        mv1, mv2, total_readable_bytes, split = self.expose_reader_mem_view(nbytes)

        # Then, if there are not enough readable bytes, return an empty array
        if nbytes > total_readable_bytes:
            return np.empty(0, dtype = dtype)

        try:
            # Otherwise, make an array view of the bytes, starting with mv1, then mv2 if it exists
            if not split:
                return np.frombuffer(mv1, dtype = dtype)
            else:
                # If they're not contiguous we have to copy unfortunately into a contiguous
                # bytearray
                combined = bytearray(total_readable_bytes)
                combined[:mv1.nbytes] = mv1
                combined[mv1.nbytes:] = mv2
                return np.frombuffer(combined, dtype = dtype).copy()
        finally:
            # Release views that we just generated
            mv1.release()
            if mv2 is not None:
                mv2.release()

    def get_slowest_reader_position(self) -> int | None:
        """
        Gets the position, in bytes, of the slowest active reader.

        Returns:
            int: The position of the slowest active reader.
            None: If there are no active readers. This means that the
                writer is free to advance.

        """
        # The slowest active reader is the active reader with the minimum absolute position
        active_positions = []

        #TODO move num_readers, data size, etc. into header
        for i in range(self.num_readers):
            offset = self._get_reader_header_offset(i)
            # Active is 1 slot over
            active = struct.unpack_from('Q', self._metadata_view, offset +
                                        HEADER_SLOT_SIZE)[0]

            if active == 1:
                # Don't call get reader pos as this would require recalculating offset
                read_pos = struct.unpack_from('Q', self._metadata_view, offset)[0]
                active_positions.append(read_pos)

        # If there are no active readers, return None
        return min(active_positions) if active_positions else None

    def _is_reader(self) -> bool:
        """
        Returns whether self is a reader.

        Returns:
            bool: Whether self is a reader.

        """
        return self._reader != self._NO_READER

    def _is_writer(self) -> bool:
        """
        Returns whether self is a writer.

        Returns:
            bool: Whether self is a writer.

        """
        return self._reader == self._NO_READER

    def _validate_is_reader(self) -> None:
        """
        Raises an exception if the SharedBuffer is not a reader.

        Raises:
            ValueError: If the SharedBuffer is not a reader (is a writer).

        """

        if self._is_writer():
            raise RuntimeError('Cannot call reader-only method on a SharedBuffer that is a writer!')

    def _validate_is_writer(self) -> None:
        """
        Raises an exception if the SharedBuffer is not a writer.

        Raises:
            ValueError: If the SharedBuffer is not a writer (is a reader).

        """

        if self._is_reader():
            raise RuntimeError('Cannot call writer-only method on a SharedBuffer that is a reader!')
