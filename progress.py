"""Progress tracking for CLI runs."""

import time


class ProgressRenderer:
    """Minimal in-terminal progress bar for CLI runs."""

    def __init__(self, enable: bool = True, width: int = 40):
        self.enable = enable
        self.width = width
        self.reset(0)

    def reset(self, total: int) -> None:
        self.total = max(total, 0)
        self.good = 0
        self.no_good = 0
        self.with_bright = 0
        self.start = time.time()
        self.last_line = ""

    def update(self, current: int, *, has_bright: bool = False, is_no_good: bool = False) -> None:
        if not self.enable or self.total <= 0:
            return
        if has_bright:
            self.with_bright += 1
        if is_no_good:
            self.no_good += 1
        else:
            self.good += 1

        pct = current / self.total if self.total else 0
        filled = int(self.width * pct)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        rate = current / elapsed if elapsed > 0 else 0
        remaining = self.total - current
        eta = remaining / rate if rate > 0 else None

        line = (
            f"[{bar}] {current}/{self.total} "
            f"good:{self.good} no_good:{self.no_good} bright:{self.with_bright} "
            f"elapsed:{elapsed:.1f}s"
        )
        if eta is not None:
            line += f" ETA:{eta:.1f}s"

        # Minimize flicker by only rewriting when content changes
        if line != self.last_line:
            print("\r" + line, end="", flush=True)
            self.last_line = line

        if current >= self.total:
            print()

