from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol, TextIO


class TaskProgress(Protocol):
    def update(self, advance: int = 1, description: str | None = None) -> None: ...

    def close(self) -> None: ...


class RunProgress(Protocol):
    def stage(self, description: str) -> None: ...

    def task(self, description: str, total: int) -> TaskProgress: ...

    def close(self) -> None: ...


class NullTaskProgress:
    def update(self, advance: int = 1, description: str | None = None) -> None:
        return None

    def close(self) -> None:
        return None


class NullRunProgress:
    def stage(self, description: str) -> None:
        return None

    def task(self, description: str, total: int) -> TaskProgress:
        return NullTaskProgress()

    def close(self) -> None:
        return None


@dataclass
class _BarState:
    total: int
    current: int = 0
    description: str = ""


class ConsoleTaskProgress:
    def __init__(self, description: str, total: int, stream: TextIO, width: int = 28) -> None:
        self.stream = stream
        self.width = width
        self.state = _BarState(total=total, description=description)
        self._render()

    def update(self, advance: int = 1, description: str | None = None) -> None:
        self.state.current = min(self.state.total, self.state.current + advance)
        if description is not None:
            self.state.description = description
        self._render()

    def close(self) -> None:
        self.state.current = self.state.total
        self._render(final=True)

    def _render(self, final: bool = False) -> None:
        total = max(1, self.state.total)
        filled = int(self.width * self.state.current / total)
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"    [{bar}] {self.state.current:>3}/{total:<3} {self.state.description}"
        end = "\n" if final else "\r"
        print(line, file=self.stream, end=end, flush=True)


class ConsoleRunProgress:
    def __init__(self, total_stages: int, stream: TextIO | None = None, width: int = 24) -> None:
        self.stream = stream or sys.stderr
        self.width = width
        self.state = _BarState(total=total_stages)

    def stage(self, description: str) -> None:
        self.state.current = min(self.state.total, self.state.current + 1)
        self.state.description = description
        self._render()

    def task(self, description: str, total: int) -> TaskProgress:
        print(f"  {description}", file=self.stream, flush=True)
        return ConsoleTaskProgress(description=description, total=total, stream=self.stream)

    def close(self) -> None:
        self.state.current = self.state.total
        self._render(final=True)

    def _render(self, final: bool = False) -> None:
        total = max(1, self.state.total)
        filled = int(self.width * self.state.current / total)
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"[{bar}] {self.state.current:>2}/{total:<2} {self.state.description}"
        end = "\n" if final else "\n"
        print(line, file=self.stream, end=end, flush=True)
