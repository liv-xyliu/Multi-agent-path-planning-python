import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from environment import MultiAgentPathfindingEnv
from agent import MultiTaskSchedulingAgent


def run_until_snapshot(seed=7, steps=80):
    env = MultiAgentPathfindingEnv(
        num_robots=30,
        seed=seed,
        max_timesteps=200,
        grid_size=(40, 40),
        obstacle_density=0.3,
        show_all_tasks=True
    )

    agent = MultiTaskSchedulingAgent(env)
    obs = env._get_observation()

    for _ in range(steps):
        action = agent.choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    return env


def plot_environment(env, output_path="assets/simulation_snapshot.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 0 = free cell, 1 = obstacle
    grid = env.grid.copy()

    plt.figure(figsize=(10, 10))

    cmap = ListedColormap(["white", "black"])
    plt.imshow(grid, cmap=cmap, origin="upper")

    # Plot task targets
    task_x = []
    task_y = []

    for task in env.tasks:
        if task.revealed and not task.is_closed():
            errand = task.get_current_errand()
            if errand is not None:
                task_x.append(errand.target_x)
                task_y.append(errand.target_y)

    plt.scatter(task_x, task_y, marker="*", s=90, label="Task targets")

    # Plot robots
    robot_x = [robot.x for robot in env.robots]
    robot_y = [robot.y for robot in env.robots]

    plt.scatter(robot_x, robot_y, marker="o", s=45, label="Robots")

    # Add robot IDs
    for robot in env.robots:
        plt.text(robot.x, robot.y, str(robot.id), fontsize=7, ha="center", va="center")

    plt.title(
        f"Multi-Agent Path Planning Snapshot\n"
        f"Timestep: {env.timestep}, Completed Tasks: {env.completed_tasks}"
    )

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend(loc="upper right")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    env = run_until_snapshot(seed=7, steps=80)
    plot_environment(env)
