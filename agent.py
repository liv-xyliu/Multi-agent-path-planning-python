import numpy as np
from queue import PriorityQueue
from typing import List, Tuple, Dict, Optional, Set, Any, Callable
from collections import deque
from environment import MultiAgentPathfindingEnv, Robot, Task, Errand, Action, Direction
import random
from dataclasses import dataclass, field
import copy
from itertools import permutations

def manhattan_heuristic(state: Dict[str, Any], goal: Tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm using Manhattan distance.
    
    Args:
        state: Current state of the environment
        goal: Goal position (x, y)
    
    Returns:
        Estimated cost from current state to goal
    """
    robot_position = state['robots'][0][1:3]
    return abs(robot_position[0] - goal[0]) + abs(robot_position[1] - goal[1])

class BaseAgent:    
    """
    Base class for multi-agent pathfinding agents.
    """
    def __init__(self, env: MultiAgentPathfindingEnv):
        self.env = env
        self.grid_width = env.grid_width
        self.grid_height = env.grid_height

    def run(self, iterations: int = 10, display: bool = False):
        for i in range(iterations):
            if i > 0:
                obs, info = self.env.reset()
            else:
                obs = self.env._get_observation()
           
            terminated = False
            truncated = False
            while not terminated and not truncated:
                if display:
                    self.env.render()

                action = self.choose_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
        
        return self.env.completed_tasks

    def choose_action(self, obs: np.ndarray) -> Any:
        raise NotImplementedError("This method should be overridden by subclasses.")


class MultiTaskSchedulingAgent(BaseAgent):
    def choose_action(self, obs: np.ndarray) -> Any:
        """
        Choose an action based on the current observation.
        """
        state = self.env.get_state()
        robots = self.env.robots
        tasks = self.env.tasks

        moves = [Action.WAIT] * self.env.num_robots
        assignments = [-1] * self.env.num_robots

        free_tasks = self.env.get_unassigned_tasks()

        # assign nearest free task to idle robots
        for robot in robots:
            if robot.task_id is None:
                best_task = -1
                best_dist = 10**9

                for task_id in free_tasks:
                    task = tasks[task_id]

                    if not task.revealed or task.is_closed():
                        continue

                    errand = task.get_current_errand()
                    if errand is None:
                        continue

                    dist = abs(robot.x - errand.target_x) + abs(robot.y - errand.target_y)

                    if dist < best_dist:
                        best_dist = dist
                        best_task = task_id

                if best_task != -1:
                    assignments[robot.id] = best_task
                    free_tasks.remove(best_task)

        occupied = set()
        for r in robots:
            occupied.add((r.x, r.y))

        reserved_next = set()

        def astar_next_step(start, goal, blocked):
            """
            Use A* search to find the next cell from start to goal.
            """
            if start == goal:
                return None

            def heuristic(pos):
                return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

            open_set = PriorityQueue()
            open_set.put((heuristic(start), 0, start))

            parent = {start: None}
            g_score = {start: 0}

            dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]

            while not open_set.empty():
                _, current_cost, current = open_set.get()

                if current == goal:
                    break

                x, y = current

                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    next_cell = (nx, ny)

                    if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
                        continue

                    if self.env.grid[ny, nx] == 1:
                        continue

                    if next_cell in blocked and next_cell != goal:
                        continue

                    new_cost = g_score[current] + 1

                    if next_cell not in g_score or new_cost < g_score[next_cell]:
                        g_score[next_cell] = new_cost
                        priority = new_cost + heuristic(next_cell)
                        open_set.put((priority, new_cost, next_cell))
                        parent[next_cell] = current

            if goal not in parent:
                return None

            cur = goal
            while parent[cur] is not None and parent[cur] != start:
                cur = parent[cur]

            if parent[cur] is None:
                return None

            return cur

        for robot in robots:
            task_id = robot.task_id

            if task_id is None and assignments[robot.id] != -1:
                task_id = assignments[robot.id]

            if task_id is None:
                moves[robot.id] = Action.WAIT
                continue

            task = tasks[task_id]

            if task.is_closed():
                moves[robot.id] = Action.WAIT
                continue

            errand = task.get_current_errand()
            if errand is None:
                moves[robot.id] = Action.WAIT
                continue

            start = (robot.x, robot.y)
            goal = (errand.target_x, errand.target_y)

            if start == goal:
                moves[robot.id] = Action.WAIT
                continue

            blocked = set(occupied)
            blocked.discard(start)
            blocked |= reserved_next

            next_cell = astar_next_step(start, goal, blocked)

            if next_cell is None:
                moves[robot.id] = Action.WAIT
                continue

            nx, ny = next_cell

            if nx == robot.x and ny == robot.y - 1:
                target_dir = Direction.NORTH
            elif nx == robot.x + 1 and ny == robot.y:
                target_dir = Direction.EAST
            elif nx == robot.x and ny == robot.y + 1:
                target_dir = Direction.SOUTH
            else:
                target_dir = Direction.WEST

            if robot.direction == target_dir:
                fx, fy = robot.get_forward_position()
                if (fx, fy) == (nx, ny):
                    moves[robot.id] = Action.MOVE_FORWARD
                    reserved_next.add((nx, ny))
                else:
                    moves[robot.id] = Action.WAIT
            else:
                diff = (target_dir - robot.direction) % 4
                if diff == 1:
                    moves[robot.id] = Action.ROTATE_CW
                elif diff == 3:
                    moves[robot.id] = Action.ROTATE_CCW
                else:
                    moves[robot.id] = Action.ROTATE_CW

        return (moves, assignments)


if __name__ == "__main__":
    env = MultiAgentPathfindingEnv(
        num_robots=30,
        seed=7,
        max_timesteps=200,
        grid_size=(40, 40),
        obstacle_density=0.3
    )
    agent = MultiTaskSchedulingAgent(env)
    agent.run(iterations=1, display=True)
