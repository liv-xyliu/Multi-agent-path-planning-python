import time
import numpy as np
from environment import MultiAgentPathfindingEnv
from agent import MultiTaskSchedulingAgent


def run_single(seed: int):
    env = MultiAgentPathfindingEnv(
        num_robots=30,
        seed=seed,
        max_timesteps=200,
        grid_size=(40, 40),
        obstacle_density=0.3
    )

    agent = MultiTaskSchedulingAgent(env)

    start = time.time()
    completed_tasks = agent.run(iterations=1, display=False)
    runtime = time.time() - start

    return {
        "seed": seed,
        "completed_tasks": completed_tasks,
        "runtime_seconds": runtime,
        "tasks_per_second": completed_tasks / runtime if runtime > 0 else 0
    }


if __name__ == "__main__":
    results = [run_single(seed) for seed in range(10)]

    completed = [r["completed_tasks"] for r in results]
    runtime = [r["runtime_seconds"] for r in results]

    print("Experiment Results")
    print("------------------")
    print(f"Average completed tasks: {np.mean(completed):.2f}")
    print(f"Best completed tasks: {np.max(completed)}")
    print(f"Average runtime: {np.mean(runtime):.2f}s")
