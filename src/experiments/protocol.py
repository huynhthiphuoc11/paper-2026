from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def protocol_payload_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_immutable_run_directory(root: str | Path, protocol_sha256: str) -> Path:
    if len(protocol_sha256) != 64:
        raise ValueError("Protocol SHA-256 must contain 64 hexadecimal characters")

    root = Path(root)
    target = root / protocol_sha256
    if target.exists():
        run_number = 2
        while (root / f"{protocol_sha256}-run-{run_number:03d}").exists():
            run_number += 1
        target = root / f"{protocol_sha256}-run-{run_number:03d}"

    target.mkdir(parents=True, exist_ok=False)
    return target


@dataclass
class ExperimentProtocol:
    """Durable guard that makes Gold-test a one-time post-lock operation."""

    sentinel_path: Path | None = None
    protocol_sha256: str | None = None
    selected_threshold: float | None = None
    selected_formulation: str | None = None
    lock_metadata: dict = field(default_factory=dict)
    locked: bool = False

    def __post_init__(self) -> None:
        if self.sentinel_path is not None:
            self.sentinel_path = Path(self.sentinel_path)

    @property
    def test_opened(self) -> bool:
        return bool(self.sentinel_path and self.sentinel_path.exists())

    def lock(self, threshold: float, formulation: str, metadata: dict) -> dict:
        if self.locked:
            raise RuntimeError("Protocol is already locked")
        if formulation not in {"pointwise", "pairwise", "listwise"}:
            raise ValueError("Unknown selected formulation")
        self.selected_threshold = float(threshold)
        self.selected_formulation = formulation
        self.lock_metadata = dict(metadata)
        self.locked = True
        return self.manifest()

    def open_gold_test_once(self) -> None:
        if not self.locked:
            raise RuntimeError("Gold-test cannot be opened before protocol lock")
        if self.sentinel_path is None or self.protocol_sha256 is None:
            raise RuntimeError("Durable Gold-test sentinel is not configured")
        self.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol_sha256": self.protocol_sha256,
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": "gold-test-opened",
        }
        try:
            with self.sentinel_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except FileExistsError as error:
            raise RuntimeError(
                "Gold-test has already been opened for this protocol"
            ) from error

    def manifest(self) -> dict:
        return {
            "locked": self.locked,
            "protocol_sha256": self.protocol_sha256,
            "selected_threshold": self.selected_threshold,
            "selected_formulation": self.selected_formulation,
            "metadata": self.lock_metadata,
            "test_opened": self.test_opened,
            "gold_test_sentinel": (
                str(self.sentinel_path) if self.sentinel_path is not None else None
            ),
        }
