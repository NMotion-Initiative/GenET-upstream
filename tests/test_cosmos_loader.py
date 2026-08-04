import pytest

from genet.integrations.cosmos_loader import DeterministicEpochSampler


def test_epoch_sampler_is_permutation_and_changes_epoch() -> None:
    sampler = DeterministicEpochSampler(range(7), seed=42)
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert sorted(first) == list(range(7))
    assert sorted(second) == list(range(7))
    assert first != second


@pytest.mark.parametrize("consumed", [0, 1, 6, 7, 8, 20])
def test_epoch_sampler_exact_resume(consumed: int) -> None:
    uninterrupted = DeterministicEpochSampler(range(7), seed=123)
    stream: list[int] = []
    while len(stream) < consumed + 7:
        stream.extend(iter(uninterrupted))

    resumed = DeterministicEpochSampler(range(7), seed=123)
    resumed.set_start_iteration(consumed)
    actual: list[int] = []
    while len(actual) < 7:
        actual.extend(iter(resumed))

    assert actual[:7] == stream[consumed : consumed + 7]


def test_epoch_sampler_rejects_empty_and_negative_resume() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DeterministicEpochSampler([], seed=0)
    sampler = DeterministicEpochSampler(range(1), seed=0)
    with pytest.raises(ValueError, match="non-negative"):
        sampler.set_start_iteration(-1)


def test_epoch_sampler_can_tag_indices_for_epoch_rotating_dataset() -> None:
    sampler = DeterministicEpochSampler(range(3), seed=9, emit_epoch=True)
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert {item[0] for item in first} == {0}
    assert {item[0] for item in second} == {1}
    assert sorted(item[1] for item in first) == [0, 1, 2]


def test_epoch_tagged_sampler_exact_resume() -> None:
    uninterrupted = DeterministicEpochSampler(range(5), seed=17, emit_epoch=True)
    stream: list[int | tuple[int, int]] = []
    while len(stream) < 14:
        stream.extend(iter(uninterrupted))

    resumed = DeterministicEpochSampler(range(5), seed=17, emit_epoch=True)
    resumed.set_start_iteration(7)
    actual: list[int | tuple[int, int]] = []
    while len(actual) < 7:
        actual.extend(iter(resumed))

    assert actual[:7] == stream[7:14]
