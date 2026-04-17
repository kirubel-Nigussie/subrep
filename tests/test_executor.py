"""
Skill Executor Validation Tests

Verifies that the rollout loop returns stable shapes/types, respects
termination controls, and supports payoff scalarization override.
"""

import numpy as np
from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor

def test_executor_output_shapes_and_types():
    #Smoke test with real MO-LunarLander wrapper for output contract.
    env = SubRepEnv(seed=42)
    policy = lambda obs: env.env.action_space.sample()
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=0.99, max_steps=20)

    total_payoff, motive_deltas, terminated = executor.run_episode()
    env.close()

    assert np.isscalar(total_payoff), "total_payoff should be scalar-like"
    assert isinstance(motive_deltas, np.ndarray), "motive_deltas should be np.ndarray"
    assert motive_deltas.shape == (2,), "motive_deltas should have shape (2,)"
    assert np.isfinite(total_payoff), "total_payoff must be finite"
    assert np.isfinite(motive_deltas).all(), "motive_deltas must be finite"
    assert isinstance(terminated, bool), "terminated should be bool"

def test_executor_max_steps_and_last_run_info():
    #Check max-step cap and diagnostics dictionary population.
    env = SubRepEnv(seed=7)
    policy = lambda obs: env.env.action_space.sample()
    max_steps = 5
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=0.99, max_steps=max_steps)

    _, motive_deltas, _ = executor.run_episode()
    env.close()

    assert motive_deltas.shape == (2,)
    assert executor.last_run_info is not None
    assert executor.last_run_info["steps"] <= max_steps
    assert executor.last_run_info["stop_reason"] in {"terminated", "truncated", "max_steps"}
    assert isinstance(executor.last_run_info["final_reward"], np.ndarray)
    assert executor.last_run_info["final_reward"].shape == (2,)

class _DeterministicEnv:
    #Small deterministic env stub for exact discounting assertions.

    def __init__(self):
        self._count = 0

    def reset(self):
        self._count = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self._count += 1
        obs = np.zeros(8, dtype=np.float32)
        reward = np.array([1.0, 2.0], dtype=np.float32)
        terminated = self._count >= 3
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

def test_executor_uses_custom_payoff_fn():
    #Verify custom scalar payoff function changes payoff only as expected.
    env = _DeterministicEnv()
    gamma = 0.5
    policy = lambda obs: 0
    payoff_fn = lambda r: float(r[0] - r[1])  # -1.0 each step
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=gamma, payoff_fn=payoff_fn)

    total_payoff, motive_deltas, terminated = executor.run_episode()

    # Discount factors for 3 steps: 1, 0.5, 0.25
    discount_sum = 1.0 + 0.5 + 0.25
    expected_payoff = -1.0 * discount_sum
    expected_motives = np.array([1.0, 2.0], dtype=np.float32) * discount_sum

    assert np.isclose(total_payoff, expected_payoff)
    assert np.allclose(motive_deltas, expected_motives)
    assert terminated is True

def test_executor_max_steps_zero_runs_no_steps():
    #`max_steps=0` should exit before taking any environment step.
    env = _DeterministicEnv()
    policy = lambda obs: 0
    executor = SkillExecutor(env=env, policy_fn=policy, max_steps=0)

    total_payoff, motive_deltas, terminated = executor.run_episode()

    assert total_payoff == 0.0
    assert np.allclose(motive_deltas, np.zeros(2, dtype=np.float32))
    assert terminated is False
    assert executor.last_run_info["steps"] == 0
    assert executor.last_run_info["stop_reason"] == "max_steps"

def test_executor_rejects_invalid_gamma():
    #Gamma outside [0, 1] should fail fast, without even showing summary
    env = _DeterministicEnv()
    policy = lambda obs: 0

    try:
        SkillExecutor(env=env, policy_fn=policy, gamma=1.1)
    except ValueError as exc:
        assert "gamma must be in [0, 1]" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid gamma")