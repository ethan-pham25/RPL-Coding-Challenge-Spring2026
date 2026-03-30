from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]

__all__ = ["RingSpec", "SharedRingBuffer"]


def _todo(method: str) -> NotImplementedError:
    return NotImplementedError(
        f"{method} is still a TODO. Right now this file is a ring buffer in the same way "
        f"a cardboard tube is a liquid engine."
    )


@dataclass(frozen=True)
class RingSpec:
    name: str
    size: int
    num_readers: int
    reader: int = -1
    cache_align: bool = False
    cache_size: int = 64

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.num_readers < 1:
            raise ValueError("num_readers must be >= 1")
        if self.cache_align:
            if self.cache_size <= 0:
                raise ValueError("cache_size must be > 0 when cache_align is True")
            if self.cache_size & (self.cache_size - 1):
                raise ValueError("cache_size must be a power of two when cache_align is True")

    def to_kwargs(self, *, create: bool, reader: int) -> dict:
        return {
            "name": self.name,
            "create": create,
            "size": self.size,
            "num_readers": self.num_readers,
            "reader": reader,
            "cache_align": self.cache_align,
            "cache_size": self.cache_size,
        }


class SharedRingBuffer:
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
        raise _todo("__init__")

    def close(self) -> None:
        raise _todo("close")

    def unlink(self) -> None:
        raise _todo("unlink")

    def __enter__(self) -> "SharedRingBuffer":
        raise _todo("__enter__")

    def __exit__(self, *_):
        raise _todo("__exit__")

    def calculate_pressure(self) -> int:
        raise _todo("calculate_pressure")

    def int_to_pos(self, value: int) -> int:
        raise _todo("int_to_pos")

    def update_reader_pos(self, new_reader_pos: int) -> None:
        raise _todo("update_reader_pos")

    def set_reader_active(self, active: bool) -> None:
        raise _todo("set_reader_active")

    def is_reader_active(self) -> bool:
        raise _todo("is_reader_active")

    def update_write_pos(self, new_writer_pos: int) -> None:
        raise _todo("update_write_pos")

    def inc_writer_pos(self, inc_amount: int) -> None:
        raise _todo("inc_writer_pos")

    def inc_reader_pos(self, inc_amount: int) -> None:
        raise _todo("inc_reader_pos")

    def get_write_pos(self):
        raise _todo("get_write_pos")

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        raise _todo("compute_max_amount_writable")

    def jump_to_writer(self) -> None:
        raise _todo("jump_to_writer")

    def expose_writer_mem_view(self, size: int) -> RingView:
        raise _todo("expose_writer_mem_view")

    def expose_reader_mem_view(self, size: int) -> RingView:
        raise _todo("expose_reader_mem_view")

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        raise _todo("simple_write")

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        raise _todo("simple_read")

    def write_array(self, arr) -> int:
        raise _todo("write_array")

    def read_array(self, nbytes: int, dtype):
        raise _todo("read_array")
