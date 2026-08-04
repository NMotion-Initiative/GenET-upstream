"""Order-independent reference-target selection."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Generic, Iterable, Mapping, Protocol, Sequence, TypeVar

from .schema import EpisodeRef


class NoReferenceCandidateError(LookupError):
    """Raised when an embodiment has no episode other than the target episode."""


class _ReferenceItem(Protocol):
    episode_id: str
    embodiment: str


T = TypeVar("T", bound=_ReferenceItem)


def stable_index(key: str, size: int, *, seed: int = 0) -> int:
    """Map a key to ``[0, size)`` without Python's process-randomized ``hash``."""

    if size <= 0:
        raise ValueError("size must be positive")
    digest = hashlib.blake2b(
        f"{seed}\0{key}".encode("utf-8"), digest_size=8, person=b"GenETRef"
    ).digest()
    return int.from_bytes(digest, "little", signed=False) % size


class StatelessReferencePool(Generic[T]):
    """A sorted, immutable pool grouped by target embodiment.

    Selection depends only on seed/sample key and candidate content.  It is
    therefore reproducible across workers, ranks, and Python processes.
    """

    def __init__(self, candidates: Iterable[T], *, seed: int = 0) -> None:
        grouped: dict[str, list[T]] = defaultdict(list)
        seen: set[tuple[str, str, str]] = set()
        for candidate in candidates:
            discriminator = str(getattr(candidate, "pool_key", repr(candidate)))
            key = (candidate.embodiment, candidate.episode_id, discriminator)
            if key not in seen:
                grouped[candidate.embodiment].append(candidate)
                seen.add(key)
        self._by_embodiment: Mapping[str, tuple[T, ...]] = {
            embodiment: tuple(
                sorted(
                    items,
                    key=lambda item: (
                        item.episode_id,
                        str(getattr(item, "pool_key", repr(item))),
                    ),
                )
            )
            for embodiment, items in grouped.items()
        }
        self.seed = int(seed)

    @property
    def embodiments(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_embodiment))

    def candidates(
        self, embodiment: str, *, exclude_episode_id: str
    ) -> tuple[T, ...]:
        return tuple(
            candidate
            for candidate in self._by_embodiment.get(embodiment, ())
            if candidate.episode_id != exclude_episode_id
        )

    def choose(
        self,
        *,
        embodiment: str,
        exclude_episode_id: str,
        sample_key: str,
        salt: str = "",
    ) -> T:
        candidates = self.candidates(
            embodiment, exclude_episode_id=exclude_episode_id
        )
        if not candidates:
            raise NoReferenceCandidateError(
                f"no reference for embodiment {embodiment!r} after excluding "
                f"episode {exclude_episode_id!r}"
            )
        index = stable_index(
            f"{sample_key}\0{embodiment}\0{exclude_episode_id}\0{salt}",
            len(candidates),
            seed=self.seed,
        )
        return candidates[index]


def episode_reference_pool(
    candidates: Sequence[EpisodeRef], *, seed: int = 0
) -> StatelessReferencePool[EpisodeRef]:
    return StatelessReferencePool(candidates, seed=seed)
