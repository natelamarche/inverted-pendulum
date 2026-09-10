import numpy as np
import pytest

from pendulum.rl.buffer import RolloutBuffer


def make_buffer(buffer_size: int = 4) -> RolloutBuffer:
    return RolloutBuffer(
        buffer_size=buffer_size,
        observation_dim=2,
        action_dim=1,
        gamma=0.9,
        gae_lambda=0.8,
    )


def add_transition(
    buffer: RolloutBuffer,
    index: int,
    *,
    reward: float = 1.0,
    value: float = 0.5,
    next_value: float = 0.6,
    terminated: bool = False,
    truncated: bool = False,
) -> None:
    buffer.add(
        observation=np.array([index, index + 0.5], dtype=np.float32),
        action=np.array([index + 1], dtype=np.float32),
        raw_action=np.array([index + 2], dtype=np.float32),
        reward=reward,
        value=value,
        next_value=next_value,
        log_prob=-float(index),
        terminated=terminated,
        truncated=truncated,
    )


def test_add_stores_transition_and_advances_position() -> None:
    buffer = make_buffer()

    add_transition(
        buffer,
        3,
        reward=2.5,
        value=1.5,
        next_value=1.75,
        terminated=True,
    )

    assert buffer.pos == 1
    np.testing.assert_array_equal(buffer.observations[0], [3.0, 3.5])
    np.testing.assert_array_equal(buffer.actions[0], [4.0])
    np.testing.assert_array_equal(buffer.raw_actions[0], [5.0])
    assert buffer.rewards[0] == 2.5
    assert buffer.values[0] == 1.5
    assert buffer.next_values[0] == 1.75
    assert buffer.log_probs[0] == -3.0
    assert buffer.terminates[0]
    assert not buffer.truncates[0]


def test_add_rejects_transition_when_buffer_is_full() -> None:
    buffer = make_buffer(buffer_size=1)
    add_transition(buffer, 0)

    with pytest.raises(AssertionError):
        add_transition(buffer, 1)


def test_reset_clears_stored_and_computed_values() -> None:
    buffer = make_buffer(buffer_size=1)
    add_transition(buffer, 1, reward=2.0, terminated=True, truncated=True)
    buffer.compute_returns_and_advantages()

    buffer.reset()

    assert buffer.pos == 0
    arrays = (
        buffer.observations,
        buffer.actions,
        buffer.raw_actions,
        buffer.rewards,
        buffer.values,
        buffer.next_values,
        buffer.log_probs,
        buffer.terminates,
        buffer.truncates,
        buffer.advantages,
        buffer.returns,
    )
    assert all(not np.any(array) for array in arrays)


def test_compute_returns_and_advantages_handles_episode_boundaries() -> None:
    buffer = make_buffer()
    transitions = (
        # Normal transition: its GAE includes the following terminal transition.
        (1.0, 0.5, 0.6, False, False),
        # Termination: do not bootstrap or continue GAE into the next episode.
        (2.0, 0.6, 999.0, True, False),
        # First transition of a new episode.
        (3.0, 0.7, 0.8, False, False),
        # Truncation: bootstrap its value, but do not continue GAE further.
        (4.0, 0.8, 0.9, False, True),
    )
    for index, transition in enumerate(transitions):
        reward, value, next_value, terminated, truncated = transition
        add_transition(
            buffer,
            index,
            reward=reward,
            value=value,
            next_value=next_value,
            terminated=terminated,
            truncated=truncated,
        )

    buffer.compute_returns_and_advantages()

    raw_advantages = np.array([2.048, 1.4, 5.9072, 4.01])
    expected_advantages = (raw_advantages - raw_advantages.mean()) / (
        raw_advantages.std() + 1e-8
    )
    expected_returns = np.array([2.548, 2.0, 6.6072, 4.81])
    np.testing.assert_allclose(buffer.advantages, expected_advantages, rtol=1e-6)
    np.testing.assert_allclose(buffer.returns, expected_returns, rtol=1e-6)


def test_compute_returns_and_advantages_includes_final_transition() -> None:
    buffer = make_buffer(buffer_size=1)
    add_transition(buffer, 0, reward=2.0, value=0.5, next_value=1.0)

    buffer.compute_returns_and_advantages()

    # A single advantage normalizes to zero; its return retains the raw GAE.
    np.testing.assert_allclose(buffer.advantages, [0.0])
    np.testing.assert_allclose(buffer.returns, [2.9])


def test_advantage_normalization_uses_only_populated_entries() -> None:
    buffer = make_buffer(buffer_size=4)
    for index, reward in enumerate([1.5, 3.5]):
        add_transition(buffer, index, reward=reward, value=0.5, terminated=True)

    buffer.compute_returns_and_advantages()

    # Raw advantages [1, 3] normalize independently of the two unused slots.
    np.testing.assert_allclose(buffer.advantages, [-1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(buffer.returns, [1.5, 3.5, 0.0, 0.0])


def test_get_batches_returns_each_populated_transition_once() -> None:
    buffer = make_buffer(buffer_size=5)
    for index in range(3):
        add_transition(buffer, index)
    buffer.compute_returns_and_advantages()

    batches = list(buffer.get_batches(batch_size=2))

    assert [len(batch[0]) for batch in batches] == [2, 1]
    observations = np.concatenate([batch[0] for batch in batches])
    raw_actions = np.concatenate([batch[1] for batch in batches])
    log_probs = np.concatenate([batch[2] for batch in batches])

    order = np.argsort(observations[:, 0])
    np.testing.assert_array_equal(observations[order, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(raw_actions[order, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(log_probs[order], [0.0, -1.0, -2.0])


def test_get_batches_on_empty_buffer_yields_no_batches() -> None:
    buffer = make_buffer()

    assert list(buffer.get_batches(batch_size=2)) == []
