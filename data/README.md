# SubRep Rollout Dataset

## File Structure
data/
  raw/
    episode_001.npz
    episode_002.npz
    ...

## Schema (per .npz file)
| Key         | Shape | Type    | Description                          |
|-------------|-------|---------|--------------------------------------|
| obs         | (8,)  | float32 | Initial state observation            |
| payoff      | ()    | float32 | Discounted cumulative payoff          |
| motives     | (2,)  | float32 | [Safety_delta, Fuel_delta]           |
| skill_id    | ()    | str     | Policy identifier                    |
| terminated  | ()    | bool    | True if episode ended naturally      |

## Usage Example
```python
import numpy as np
data = np.load('data/raw/episode_001.npz', allow_pickle=True)
obs      = data['obs']       # shape (8,)
payoff   = float(data['payoff'])
motives  = data['motives']   # shape (2,)
skill_id = str(data['skill_id'])
terminated = bool(data['terminated'])
```

## Notes
- ALL episodes are collected (certified and uncertified) for unbiased training
- Use seed parameter in DataCollector for reproducibility
