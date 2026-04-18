# Environment Wrapper 

**Purpose:** Wraps MO-LunarLander (MO-Gymnasium) to standardize vector reward output for SubRep certification.  

## Goal
Provide a stable interface that returns **observation vectors** and **multi-objective reward vectors** (Safety, Fuel) for every step.


## Key Files
- `lunar_lander_wrapper.py`: Wraps `mo-gymnasium` to enforce reward shape.
- `config.py`: Environment constants (max steps, seed, etc.). * Note: Load system env variables from utils/config.py, not here.
## Validation
Run `python tests/test_env.py` to verify:
- Observation shape is `(8,)`.
- Reward shape is `(2,)`.
- Episode terminates correctly on crash/landing.


## Skill Execution Loop

**Purpose:** Runs a policy (random or custom) in the wrapped MO-LunarLander environment and collects discounted rollout outcomes for later CDS/PDS certification.

### File
- `skill_executor.py`: Executes one rollout and returns:
  - `total_payoff` (discounted scalar payoff),
  - `motive_deltas` (discounted vector `[Safety, Fuel]`),
  - `terminated` (true terminal flag from environment).

### Behavior
- Accepts any callable policy with signature `policy(obs) -> action`.
- Supports full episode execution (`max_steps=None`) or fixed-step execution.
- Stops on first of: `terminated`, `truncated`, or `max_steps` reached.
- Uses discount factor `gamma` (default `0.99`) for both payoff and motives.
- Prints episode summary: steps, total payoff, motive deltas, final reward, end reason.

### Validation
Run:
- `python -m pytest tests/test_executor.py -v`

To see printed episode summaries during tests:
- `python -m pytest tests/test_executor.py -v -s`