# 12. Action Manager

The **Action Manager** is a critical component for successful real-world robot deployment in `eval_real.py`. It solves the fundamental mismatch between the *inference rate* of the policy and the *control rate* of the robot.

## The Problem: Rate Mismatch

*   **Policy Inference**: A typical vision-based policy is computationally expensive. It might run at a low frequency, like **5-10 Hz**, and it often produces a whole "chunk" of future actions (e.g., 16-64 steps) at once.
*   **Robot Control**: Most robot hardware requires a smooth, continuous stream of commands at a high frequency, like **50-100 Hz**, to avoid jerky movements.

Directly sending the policy's chunky, low-frequency actions to the robot would result in poor and unsafe performance.

## The Solution: A Smart Buffer

The Action Manager, located in `deploy/action_manager.py`, acts as a thread-safe, smart buffer and interpolator between the inference thread and the main robot control loop.

```
+--------------------------+
|    Inference Thread      |
| (Low Frequency, ~10 Hz)  |
+------------+-------------+
             |
             | Produces a chunk of 16 future actions
             | e.g., [a_t, a_t+1, ..., a_t+15]
             v
+------------+-------------+
|      Action Manager      |
| (Thread-safe Queue/Buffer)|
+------------+-------------+
             ^
             |
             | Queries for an action at the current timestamp
             | e.g., "Give me the action for t+0.02s"
             |
+------------+-------------+
| Main Robot Control Loop  |
| (High Frequency, ~50 Hz) |
+--------------------------+
```

1.  The **Inference Thread** calls `action_manager.put(action_chunk, timestamp)`.
2.  The **Main Control Loop** calls `action_manager.get(current_time)` at a much higher rate.
3.  The Action Manager's job is to look at its buffer of future action chunks and intelligently return the best possible action for the requested `current_time`. This might involve interpolation between two waypoints in the action chunk.

## Available Managers

IL-Studio provides several Action Manager strategies. You can find them in `deploy/action_manager.py`.

*   `OlderFirstManager`: A simple baseline manager that maintains a queue of actions and returns the oldest one available. It does not perform interpolation.

## Configuration

You select and configure the Action Manager via command-line arguments in `eval_real.py`.

```bash
python eval_real.py \
    --action_manager OlderFirstManager \
    --manager_coef 1.0 # An optional coefficient for the manager
```

## Customization

You can create your own Action Manager by:
1.  Creating a new class in `deploy/action_manager.py`.
2.  Implementing the `put(self, action_chunk, timestamp)` and `get(self, timestamp)` methods.
3.  Adding your class to the `load_action_manager` factory function in the same file.

This allows you to experiment with different interpolation strategies (e.g., linear, spline) or buffering techniques to achieve the smoothest possible robot motion.
