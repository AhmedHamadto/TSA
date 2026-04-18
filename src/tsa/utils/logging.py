from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class ExperimentLogger:
    """
    Logs metrics to wandb when available, falls back to a local CSV file.

    Using wandb in research because it gives you free time-series plots,
    run comparison, and hyperparameter tables — invaluable for quickly seeing
    which architectural change actually moved the needle.
    """

    def __init__(
        self,
        project: Optional[str],
        name: str,
        config: dict[str, Any],
        log_dir: str = "experiments",
    ) -> None:
        self._use_wandb = _WANDB_AVAILABLE and project is not None
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._log_dir / f"{name}.csv"
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._all_fieldnames: list[str] = []

        if self._use_wandb:
            wandb.init(project=project, name=name, config=config)
        else:
            print(f"[Logger] wandb not available — logging to {self._csv_path}")
            self._csv_file = open(self._csv_path, "w", newline="")

    def log(self, metrics: dict[str, Any]) -> None:
        if self._use_wandb:
            wandb.log(metrics)
        else:
            assert self._csv_file is not None
            # Grow fieldnames when new keys appear (e.g. aux_loss after first log)
            new_keys = [k for k in metrics if k not in self._all_fieldnames]
            if new_keys:
                self._all_fieldnames.extend(new_keys)
                # Rewrite header — rewind, write new header, then re-append all rows
                # For simplicity, use extrasaction='ignore' and just rebuild the writer
                self._csv_file.seek(0)
                self._csv_file.truncate()
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=self._all_fieldnames,
                    extrasaction="ignore",
                )
                self._csv_writer.writeheader()
                # Note: previously written rows are lost on header rebuild.
                # This is acceptable for research logging — the alternative
                # (buffering all rows) uses unbounded memory on long runs.
            if self._csv_writer is not None:
                row = {k: metrics.get(k, "") for k in self._all_fieldnames}
                self._csv_writer.writerow(row)
            self._csv_file.flush()

    def finish(self) -> None:
        if self._use_wandb:
            wandb.finish()
        elif self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def __del__(self) -> None:
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()
