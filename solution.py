from __future__ import annotations

import math
from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np


__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]


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
    # Shared class attributes
    readers_active: dict[int, bool] = {}
    reader_positions: dict[int, int] = {}
    writer_position = 0

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
        # Init SharedMemory
        # Note that track = create because only the create should track the SharedBuffer
        # in standalone Python processes (see https://docs.python.org/3/library/multiprocessing.shared_memory.html
        # , doesn't matter on Windows)
        super().__init__(name, create, size, track = create)

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
        self.buffer_size = size
        self._buffer_size_power_of_two = self._is_power_of_two(self.buffer_size)
        self.num_readers = num_readers
        self._reader = reader
        self._position = 0 # Read position if reader, write position if writer

        #TODO finish track reader and writer state with metadata

        # Init shared state/metadata class attributes between SharedBuffer instances
        if self._is_reader():
            # Readers are initially inactive
            self._active = False
            SharedBuffer.readers_active[self._reader] = False
            # Reader position is 0 initially
            SharedBuffer.reader_positions[self._reader] = 0
        else:
            SharedBuffer.writer_position = 0

        #TODO setup local views/fields used by rest of methods

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
            #TODO release local views
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
            self._active = True

        return self

    def __exit__(self, *_):
        """
        Exit the context manager.

        Reader instances are expected to mark themselves inactive on exit, then
        close local resources.
        """
        # If reader, mark self inactive
        if self._is_reader():
            self._active = False

        self.close()

    def calculate_pressure(self) -> int:
        """
        Return current writer pressure as an integer percentage.

        Pressure is based on how much of the bounded storage is currently in use
        relative to the slowest active reader.
        """
        # Storage in use relative to the slowest active reader is simply the number of bytes
        # (difference in position) between the writer and the slowest active reader
        stored_bytes = SharedBuffer.writer_position - self.get_slowest_reader_position()

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
        # Update both local and shared reader pos
        self._position = new_reader_pos
        SharedBuffer.reader_positions[self._reader] = new_reader_pos

    def set_reader_active(self, active: bool) -> None:
        """
        Mark this reader as active or inactive in shared state.

        Active readers apply backpressure. Inactive readers should not reduce
        writer capacity.
        """
        self._validate_is_reader()
        self._active = active
        # Update shared activity state
        SharedBuffer.readers_active[self._reader] = active

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
        # Update both local and shared writer pos
        self._position = new_writer_pos
        SharedBuffer.writer_position = new_writer_pos

    def inc_writer_pos(self, inc_amount: int) -> None:
        """
        Advance the writer's absolute position by `inc_amount` bytes.

        This is how a writer publishes bytes after copying them into the buffer.
        """
        self._validate_is_writer()
        # Increment both local and shared writer pos
        self._position += inc_amount
        SharedBuffer.writer_position = self._position

    def inc_reader_pos(self, inc_amount: int) -> None:
        """
        Advance this reader's absolute position by `inc_amount` bytes.

        This is how a reader consumes bytes after reading them.
        """
        self._validate_is_reader()
        self._position += inc_amount
        SharedBuffer.reader_positions[self._reader] = self._position

    def get_write_pos(self) -> int:
        """
        Return the current absolute writer position.

        Readers can use this to resynchronize or compute how much data is available.
        """
        # Note that lack of writer validation is intentional; see above
        return SharedBuffer.writer_position

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
            return SharedBuffer.writer_position - slowest_reader_pos

    def jump_to_writer(self) -> None:
        """
        Move this reader directly to the current writer position.

        Use this when a reader has fallen too far behind and old unread data is
        no longer retained.
        """
        self._validate_is_reader()
        raise NotImplementedError("TODO: implement SharedBuffer.jump_to_writer")

    def _make_memory_views(self, size: int, read: bool):
        # Guard clause to save time
        if size <= 0:
            return memoryview(bytearray()), None, 0, False

        if read:
            usable_bytes = SharedBuffer.writer_position - self._position
        else:
            usable_bytes = self.buffer_size

        # Clamp actual size
        actual_size = min(size, usable_bytes)

        # If buffer would overflow, split and wrap around
        if self._position + actual_size > self.buffer_size:
            split = True
            # First part of memory view is at the end of the buffer
            mv1 = self.buf[self._position:]

            # Second part of memory view (wrapped around) is at beginning
            # Subtract 1 since position is 0-indexed
            remaining_bytes = actual_size - (self.buffer_size - self._position - 1)
            mv2 = self.buf[0:remaining_bytes]
        else:
            split = False
            mv1 = self.buf[self._position:self._position + actual_size]
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
        # 2. Elif writer_mem_view is split and src fits into the combined storage of mv1 and mv2,
        # copy into mv1 and mv2
        # 3. Otherwise, src doesn't fit into writer_mem_view, so
        if src.len < mv1.nbytes:
            mv1[:src.len] = src
        elif split and src.len < total_size:
            mv1 = src[:mv1.nbytes]
            remaining_bytes = src.len - mv1.nbytes
            mv2[:remaining_bytes] = src[mv1.nbytes:]

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        """
        Copy bytes from the exposed reader view(s) into `dst`.

        If `dst` is smaller than the readable region, copy only the prefix that fits.
        This helper should not consume data by itself; consumption happens when the
        reader position is advanced.
        """
        self._validate_is_reader()
        raise NotImplementedError("TODO: implement SharedBuffer.simple_read")

    def write_array(self, arr: np.ndarray) -> int:
        """
        Write a NumPy array's raw bytes into the shared buffer.

        Return the number of bytes written. If the full array does not fit, the
        contract used by the tests expects this method to return `0`.
        """
        self._validate_is_writer()
        raise NotImplementedError("TODO: implement SharedBuffer.write_array")

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """
        Read `nbytes` from the shared buffer and interpret them as `dtype`.

        Return a NumPy array view/copy of the requested bytes when enough data is
        available. If there are not enough readable bytes, return an empty array
        with the requested dtype.
        """
        self._validate_is_reader()
        raise NotImplementedError("TODO: implement SharedBuffer.read_array")

    def get_slowest_reader_position(self) -> int | None:
        """
        Gets the position, in bytes, of the slowest active reader.

        Returns:
            int: The position of the slowest active reader.
            None: If there are no active readers. This means that the
                writer is free to advance.

        """
        # The slowest active reader is the active reader with the minimum absolute position
        # If there are no active readers, return None
        try:
            return min(SharedBuffer.reader_positions[i]
                       for i in range(self.num_readers) if SharedBuffer.readers_active[i])
        except ValueError:
            return None

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
