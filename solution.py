from __future__ import annotations

import math
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
# 5 dropped_size/reserved

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
        
        # Create numpy view of header for efficient int64 access (eliminates struct overhead)
        self._header_array = np.frombuffer(self._metadata_view, dtype=np.uint64)

        # Init shared state/metadata view between SharedBuffer instances
        # On creation only, zero out the header
        # Importantly, this makes all the reader and writer positions 0 cheaply,
        # and also makes readers inactive to begin with!
        if create:
            self._metadata_view[:self.header_size] = b'\x00' * self.header_size
        
        # Precompute reader header indices for fast access (eliminates repeated offset calculations)
        self._reader_header_indices = {}
        for i in range(num_readers):
            offset = ((SLOTS_PER_WRITER + i * SLOTS_PER_READER) * HEADER_SLOT_SIZE)
            # Store the index directly into the numpy array (divide by 8 since each slot is 8 bytes)
            self._reader_header_indices[i] = offset // HEADER_SLOT_SIZE

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
            # Release local views on SharedMemory at end of lifecycle
            self._release_memory_views(self._metadata_view, self._data_view)
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

        # Storage in use relative to the slowest active reader is simply the number of bytes
        # (difference in position) between the writer and the slowest active reader
        slowest_reader_position = self.get_slowest_reader_position()
        if slowest_reader_position is None:
            stored_bytes = 0
        else:
            stored_bytes = self.get_write_pos() - self.get_slowest_reader_position()

        # Then the pressure is just the stored_bytes as a percentage of total buffer size
        return (100 * stored_bytes) // self.buffer_size

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
        # Update shared reader pos using numpy array (faster than struct.pack_into)
        idx = self._reader_header_indices[self._reader]
        self._header_array[idx] = new_reader_pos

    def set_reader_active(self, active: bool) -> None:
        """
        Mark this reader as active or inactive in shared state.

        Active readers apply backpressure. Inactive readers should not reduce
        writer capacity.
        """
        self._validate_is_reader()
        # Update shared activity state in header using numpy array (eliminate local state)
        idx = self._reader_header_indices[self._reader] + 1
        self._header_array[idx] = 1 if active else 0

    def is_reader_active(self) -> bool:
        """
        Return whether this reader is currently marked active.

        This must fail clearly when called on a writer-only instance.
        """
        self._validate_is_reader()
        # Always read from shared state (single source of truth)
        idx = self._reader_header_indices[self._reader] + 1
        return bool(self._header_array[idx])

    def update_write_pos(self, new_writer_pos: int) -> None:
        """
        Store the writer's absolute write position in shared state.

        The write position is what makes newly written bytes visible to readers.
        """
        self._validate_is_writer()
        # Update shared writer pos using numpy array (faster than struct.pack_into)
        self._header_array[0] = new_writer_pos

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
        # Get write pos using numpy array (faster than struct.unpack_from)
        return int(self._header_array[0])

    def get_reader_pos(self, reader_num: int) -> int:
        """
        Return the absolute reader position for the given reader num.

        Params:
            reader_num: The reader to fetch the position for.

        Returns:
            int: The reader's absolute position.

        """
        # Stored in shared header using numpy array (faster than struct.unpack_from)
        idx = self._reader_header_indices[reader_num]
        return int(self._header_array[idx])

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

    def _make_memory_views(self, size: int, read: bool) -> RingView:
        """
        Makes a RingView (two memory views and associated metadata),
        given the number of bytes to view and whether the views should be
        writable.

        Args:
            size: The number of bytes to include in the views.
            read: Whether the views should be view-only.

        Returns:
            RingView: The memory views, the number of accessible bytes, and whether
                the views were split (wrapped around the buffer).

        """

        # Guard clause to save time
        if size <= 0:
            return memoryview(bytearray()), None, 0, False

        # Get the current position and usably bytes
        if read:
            current_pos = self.get_reader_pos(self._reader)
            # If the writer is more than self.buffer size ahead of the reader,
            # we can't read any bytes right now; return 0 and resync to writer
            usable_bytes = self.get_write_pos() - current_pos
            if usable_bytes > self.buffer_size:
                self.jump_to_writer()
                return memoryview(bytearray()), None, 0, False
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
        input_bytes = arr.nbytes
        max_writable = self.compute_max_amount_writable()
        if max_writable < input_bytes:
            return 0 # Cannot fit full array, so contract says to not write anything

        # Get memory view to write into; once input array is converted to bytes,
        # it can just be written in a simple write (I had some duplicated code before)
        view = self.expose_writer_mem_view(input_bytes)
        self.simple_write(view, memoryview(arr).cast('B'))
        # Since simple_write() does not increment the writer position,
        # we do it here as it's expected for write_array()
        self.inc_writer_pos(input_bytes)

        # Release views to free up memory
        self._release_memory_views(view[0], view[1])
        return input_bytes

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """
        Read `nbytes` from the shared buffer and interpret them as `dtype`.

        Return a NumPy array view/copy of the requested bytes when enough data is
        available. If there are not enough readable bytes, return an empty array
        with the requested dtype.
        """
        self._validate_is_reader()
        # First, get the data into memoryviews
        view = self.expose_reader_mem_view(nbytes)

        # Then, if there are not enough readable bytes, return an empty array
        if view[2] < nbytes:
            return np.empty(0, dtype = dtype)

        # Now, simply read into a bytearray then convert to numpy array for return value
        dst = bytearray(nbytes) # Empty bytearray to read into
        self.simple_read(view, dst)
        # See above; increment reader_pos since simple_read() doesn't do it
        self.inc_reader_pos(nbytes)

        # Again, make sure to release views before returning
        self._release_memory_views(view[0], view[1])
        return np.frombuffer(dst, dtype = dtype)

    def get_slowest_reader_position(self) -> int | None:
        """
        Gets the position, in bytes, of the slowest active reader.

        Returns:
            int: The position of the slowest active reader.
            None: If there are no active readers. This means that the
                writer is free to advance.

        """
        # Use vectorized numpy operations to find slowest active reader
        # Build active mask and positions using precomputed indices
        min_active_pos = None
        
        for i in range(self.num_readers):
            # Use precomputed indices for fast access (no offset calculations)
            pos_idx = self._reader_header_indices[i]
            active_idx = pos_idx + 1
            
            # Direct array access (no struct unpacking)
            is_active = self._header_array[active_idx] == 1
            if is_active:
                read_pos = int(self._header_array[pos_idx])
                if min_active_pos is None or read_pos < min_active_pos:
                    min_active_pos = read_pos
        
        return min_active_pos

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

    @staticmethod
    def _release_memory_views(*views: memoryview | None) -> None:
        """
        See tests/support.py. Releases memory views to free up space,
        if they exist.

        Args:
            *views: Arbitrary amount of views to release.

        """

        for view in views:
            if view is None:
                continue
            try:
                view.release()
            except Exception:
                pass
