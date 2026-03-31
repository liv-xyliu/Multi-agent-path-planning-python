import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import IntEnum
from dataclasses import dataclass
import random
from collections import defaultdict


class Action(IntEnum):
    """Available actions for robots"""
    MOVE_FORWARD = 0
    ROTATE_CW = 1
    ROTATE_CCW = 2
    WAIT = 3


class Direction(IntEnum):
    """Robot orientations"""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


@dataclass
class Robot:
    """Represents a robot with position and orientation"""
    id: int
    x: int
    y: int
    direction: Direction
    task_id: Optional[int] = None
    
    def get_forward_position(self) -> Tuple[int, int]:
        """Get the position the robot would move to if moving forward"""
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]
        return self.x + dx, self.y + dy


@dataclass
class Errand:
    """An errand is a request to visit a specific location"""
    target_x: int
    target_y: int
    
    def is_completed(self, robot: Robot) -> bool:
        return robot.x == self.target_x and robot.y == self.target_y


@dataclass
class Task:
    """A task is an ordered sequence of errands"""
    id: int
    errands: List[Errand]
    assigned_robot: Optional[int] = None
    current_errand_idx: int = 0
    revealed: bool = False
    
    def is_open(self) -> bool:
        """Task is open if at least one errand has been completed"""
        return self.assigned_robot is not None and self.current_errand_idx > 0
    
    def is_closed(self) -> bool:
        """Task is closed if all errands are completed"""
        return self.current_errand_idx >= len(self.errands)
    
    def get_current_errand(self) -> Optional[Errand]:
        if self.is_closed():
            return None
        return self.errands[self.current_errand_idx]


class MultiAgentPathfindingEnv(gym.Env):
    """
    Multi-Agent Pathfinding Environment
    
    This environment simulates the competition problem where robots must complete
    tasks (sequences of errands) while avoiding collisions.
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (10, 10),
                 num_robots: int = 3,
                 tasks_per_robot: int = 2,
                 max_errands_per_task: int = 3,
                 obstacle_density: float = 0.1,
                 max_timesteps: int = 100,
                 task_reveal_rate: float = 1.1,
                 seed: Optional[int] = None,
                 grid_layout: Optional[np.ndarray] = None,
                 show_all_tasks: bool = False,
                 ):
        """
        Initialize the environment
        
        Args:
            grid_size: (width, height) of the grid
            num_robots: Number of robots
            tasks_per_robot: Initial tasks per robot
            max_errands_per_task: Maximum errands per task
            obstacle_density: Fraction of grid cells that are obstacles
            max_timesteps: Maximum timesteps before episode ends
            task_reveal_rate: Probability of revealing new tasks each timestep
        """
        super().__init__()
        
        self.grid_width, self.grid_height = grid_size
        self.num_robots = num_robots
        self.tasks_per_robot = tasks_per_robot
        self.max_errands_per_task = max_errands_per_task
        self.obstacle_density = obstacle_density
        self.max_timesteps = max_timesteps
        self.task_reveal_rate = task_reveal_rate
        self.grid_layout = grid_layout
        self.show_all_tasks = show_all_tasks
        
        # Initialize grid and robots
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        self.robots = []
        self.tasks = []
        self.completed_tasks = 0
        self.revealed_tasks = 0
        self.timestep = 0
        
        # Action and observation spaces
        # Action space: (moves, task_assignments)
        # moves: 4 actions per robot (MOVE_FORWARD, ROTATE_CW, ROTATE_CCW, WAIT)
        # task_assignments: positive integers for task IDs per robot
        self.action_space = spaces.Tuple([
            spaces.MultiDiscrete([4] * num_robots),  # moves
            spaces.MultiDiscrete([10000] * num_robots)  # task assignments (0-999, where 0 means no assignment)
        ])
        
        # Observation space: grid + robot states + task info
        obs_size = (self.grid_height * self.grid_width +  # grid
                   self.num_robots * 4 +  # robot states (x, y, dir, task_id)
                   100)  # task information (simplified)
        self.observation_space = spaces.Box(
            low=0, high=max(self.grid_height, self.grid_width, self.num_robots),
            shape=(obs_size,), dtype=np.int32
        )
        
        self.reset(seed)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize grid with obstacles
        if self.grid_layout is not None:
            self.grid = self.grid_layout.copy()
        else:
            self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
            obstacle_count = int(self.grid_height * self.grid_width * self.obstacle_density)
            
            for _ in range(obstacle_count):
                x, y = random.randint(0, self.grid_width-1), random.randint(0, self.grid_height-1)
                self.grid[y, x] = 1  # 1 represents obstacle
        
        # Initialize robots at random free positions
        self.robots = []
        occupied_positions = set()
        
        for i in range(self.num_robots):
            while True:
                x = random.randint(0, self.grid_width-1)
                y = random.randint(0, self.grid_height-1)
                if self.grid[y, x] == 0 and (x, y) not in occupied_positions:
                    direction = Direction(random.randint(0, 3))
                    self.robots.append(Robot(i, x, y, direction))
                    occupied_positions.add((x, y))
                    break
        
        # Initialize tasks
        self.tasks = []
        self.completed_tasks = 0
        self.timestep = 0
        self.revealed_tasks = 0
        
        # Generate initial revealed tasks
        self._generate_initial_tasks()
        
        return self._get_observation(), self._get_info()
    
    def step(self, actions: Tuple[List[int],List[int]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep"""
        if len(actions[0]) != self.num_robots or len(actions[1]) != self.num_robots:
            raise ValueError(f"Expected {self.num_robots} actions, got {len(actions[0])} and {len(actions[1])}")
        
        # Validate and execute actions
        valid_actions = self._validate_actions(actions)
        self._execute_actions(valid_actions)
        
        # Update task progress
        prior_tasks_completed = self.completed_tasks
        self._update_task_progress()
        tasks_completed = self.completed_tasks - prior_tasks_completed
        
        # Reveal new tasks
        self._maybe_reveal_tasks(tasks_completed)
        
        # Calculate reward
        reward = self._calculate_reward(tasks_completed)
        
        # Check if episode is done
        self.timestep += 1
        done = self.timestep >= self.max_timesteps
        
        return self._get_observation(), reward, done, False, self._get_info()
    
    def _validate_actions(self, actions: Tuple[List[int],List[int]]) -> Tuple[List[int],List[int]]:
        moves, assignments = actions
        valid_moves = self._validate_moves(moves)
        valid_assignments = self._validate_assignments(assignments)
        return valid_moves, valid_assignments

    def _validate_moves(self, actions: List[int]) -> List[int]:
        """Validate actions to prevent collisions"""
        valid_actions = actions[:]
        
        # Check for collisions
        next_positions = {}
        edges = {}
        
        for i, action in enumerate(actions):
            robot = self.robots[i]
            
            if action == Action.MOVE_FORWARD:
                next_x, next_y = robot.get_forward_position()
                
                # Check bounds and obstacles
                if (next_x < 0 or next_x >= self.grid_width or 
                    next_y < 0 or next_y >= self.grid_height or
                    self.grid[next_y, next_x] == 1):
                    valid_actions[i] = Action.WAIT
                    continue
                
                # Check vertex collision
                if (next_x, next_y) in next_positions:
                    valid_actions[i] = Action.WAIT
                    valid_actions[next_positions[(next_x, next_y)]] = Action.WAIT
                    continue
                
                # Check edge collision
                edge = tuple(sorted([(robot.x, robot.y), (next_x, next_y)]))
                if edge in edges:
                    valid_actions[i] = Action.WAIT
                    valid_actions[edges[edge]] = Action.WAIT
                    continue
                
                next_positions[(next_x, next_y)] = i
                edges[edge] = i
            else:
                # If the current robot position clashes with a prior one, the prior one must WAIT
                # This might mean that it's position clashes, with a prior one, etc, so we must resolve all of these.
                if (robot.x, robot.y) in next_positions:
                    blocking_robot = robot
                    while (blocking_robot.x, blocking_robot.y) in next_positions:
                        blocked_robot = self.robots[next_positions[(blocking_robot.x, blocking_robot.y)]]
                        valid_actions[blocked_robot.id] = Action.WAIT
                        blocking_robot = blocked_robot


                next_positions[(robot.x, robot.y)] = i
        
        return valid_actions
    
    def _validate_assignments(self, actions: List[int]) -> List[int]:
        """Validate task assignments"""
        valid_assignments = actions[:]
        
        for i, task_id in enumerate(actions):
            robot = self.robots[i]
            # Check if task IDs are valid
            if task_id < 0 or task_id >= len(self.tasks) or not self.tasks[task_id].revealed:
                valid_assignments[i] = -1
            # Check if robot already has an open task
            elif robot.task_id is not None and self.tasks[robot.task_id].is_open():
                valid_assignments[i] = -1
            # Check if task is already assigned and opened
            elif self.tasks[task_id].assigned_robot is not None and self.tasks[task_id].is_open():
                valid_assignments[i] = -1

        return valid_assignments

    def _execute_actions(self, actions: Tuple[List[int],List[int]]):
        moves, assignments = actions
        """Execute validated moves"""
        for i, action in enumerate(moves):
            robot = self.robots[i]
            
            if action == Action.MOVE_FORWARD:
                robot.x, robot.y = robot.get_forward_position()
            elif action == Action.ROTATE_CW:
                robot.direction = Direction((robot.direction + 1) % 4)
            elif action == Action.ROTATE_CCW:
                robot.direction = Direction((robot.direction - 1) % 4)
            # WAIT does nothing

        """Execute task assignments"""
        for i, task_id in enumerate(assignments):
            if task_id > -1:
                self._assign_task(i, task_id)

    def _update_task_progress(self):
        """Update task progress based on robot positions"""
        for robot_id, robot in enumerate(self.robots):
            if robot.task_id is None:
                continue
            
            task = self.tasks[robot.task_id]
            current_errand = task.get_current_errand()
            
            if current_errand and current_errand.is_completed(robot):
                task.current_errand_idx += 1
                
                if task.is_closed():
                    self.completed_tasks += 1
                    robot.task_id = None
    
    def _maybe_reveal_tasks(self, tasks_completed: int = 0):
        """Reveal new tasks probabilistically"""
        for _ in range(tasks_completed):
            task_reveal_rate = self.task_reveal_rate
            while task_reveal_rate > 0:
                if random.random() < task_reveal_rate:
                    self._reveal_task()
                task_reveal_rate -= 1
    
    def _reveal_initial_tasks(self):
        """Generate initial set of revealed tasks"""
        for _ in range(self.num_robots * self.tasks_per_robot):
            self._reveal_task()
    
    def _generate_initial_tasks(self):
        """Generate initial set of revealed tasks"""
        for _ in range(self.num_robots * self.max_timesteps//30):
            self._generate_task()
        self._reveal_initial_tasks()

    def _reveal_task(self):
        if self.revealed_tasks >= len(self.tasks):
            self._generate_task()
        self.tasks[self.revealed_tasks].revealed = True
        self.revealed_tasks += 1


    def _generate_task(self):
        """Generate a new task"""
        num_errands = random.randint(1, self.max_errands_per_task)
        errands = []
        id = len(self.tasks)  # Unique ID for the task
        for _ in range(num_errands):
            # Generate random target position (not on obstacle)
            while True:
                x = random.randint(0, self.grid_width-1)
                y = random.randint(0, self.grid_height-1)
                if self.grid[y, x] == 0:
                    errands.append(Errand(x, y))
                    break
        
        task = Task(id=id,errands=errands, revealed=False)
        self.tasks.append(task)
    
    def _calculate_reward(self, tasks_completed) -> float:
        """Calculate reward for the current timestep"""
        # Reward for completing tasks
        reward = 0.0
        
        reward += tasks_completed * 1.0  # Reward for task completion
        
        # Small penalty for time passing to encourage efficiency
        #reward -= 0.1
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Grid state (flattened)
        grid_with_robots = self.grid.copy()
        for i, robot in enumerate(self.robots):
            grid_with_robots[robot.y, robot.x] = i + 2  # robots are 2+
        obs.extend(grid_with_robots.flatten())
        
        # Robot states
        for robot in self.robots:
            obs.extend([robot.x, robot.y, robot.direction, robot.task_id or -1])
        
        # Task information (simplified)
        task_info = []
        for i, task in enumerate(self.tasks[:100]):  # Limit to first 100 tasks
            task_info.extend([
                len(task.errands),
                task.assigned_robot or -1,
                task.current_errand_idx,
                int(task.revealed)
            ])
        
        # Pad to fixed size
        while len(task_info) < 400:
            task_info.append(0)
        
        obs.extend(task_info)
        
        return np.array(obs, dtype=np.int32)
    
    def _get_info(self) -> Dict:
        """Get info dictionary"""
        return {
            'completed_tasks': self.completed_tasks,
            'timestep': self.timestep,
            'num_active_tasks': len([t for t in self.tasks if t.assigned_robot is not None]),
            'num_revealed_tasks': len([t for t in self.tasks if t.revealed and not t.is_closed()])
        }
    
    # Model-based search interface methods
    def get_state(self) -> Dict[str, Any]:
        """Get complete state for model-based search"""
        return {
            'robots': [(r.id, r.x, r.y, r.direction, r.task_id) for r in self.robots],
            'tasks': [(t.id, t.errands, t.assigned_robot, t.current_errand_idx, t.revealed) for t in self.tasks],
            'completed_tasks': self.completed_tasks,
            'revealed_tasks': self.revealed_tasks,
            'timestep': self.timestep,
            'grid': self.grid.copy()
        }
    
    def _set_state(self, state: Dict[str, Any]):
        """Set complete state for model-based search"""
        self.robots = [Robot(id, x, y, Direction(d), tid) 
                      for id, x, y, d, tid in state['robots']]
        
        self.tasks = []
        for id, errands, assigned_robot, current_errand_idx, revealed in state['tasks']:
            task = Task(id=id, errands=errands, assigned_robot=assigned_robot, current_errand_idx=current_errand_idx, revealed=revealed)
            self.tasks.append(task)
        
        self.completed_tasks = state['completed_tasks']
        self.revealed_tasks = state['revealed_tasks']
        self.timestep = state['timestep']
        self.grid = state['grid'].copy()
    
    def get_valid_actions(self, state: Optional[Dict[str, Any]] = None, robot_id: Optional[int] = None) -> List[List[int]]:
        if state is not None:
            """Get valid actions based on a given state"""
            saved_state = self.get_state()
            self._set_state(state)
            actions = self._get_valid_actions(robot_id)
            self._set_state(saved_state)
            return actions
        else:
            """Get valid actions based on the current environment state"""
            return self._get_valid_actions(robot_id)

    def _get_valid_actions(self, robot_id: Optional[int] = None) -> Tuple[List[List[int]],List[List[int]]]:
        """Get valid actions for planning.
        Note: This function returns a tuple of two lists: valid actions and valid assignments.
        The lists are *factored* for each robot, meaning each robot has its own set of valid actions and assignments.
        To get a full list of actions, you must take the cartesian product of all valid actions and assignments.
        """
        if robot_id is not None:
            # Get valid actions for specific robot
            valid_actions = []
            for action in Action:
                if self._is_action_valid(robot_id, action):
                    valid_actions.append(action)

            current_task = self.robots[robot_id].task_id
            if current_task is None or not self.tasks[current_task].is_open():
                valid_assignments = self.get_unassigned_tasks()
            else:
                valid_assignments = []
            
            return [valid_actions], [valid_assignments]
        else:
            valid_moves = []
            valid_assignments = []
            available_tasks = self.get_unassigned_tasks()
            for robot_id, robot in enumerate(self.robots):
                robot_moves = []
                for action in Action:
                    if self._is_action_valid(robot_id, action):
                        robot_moves.append(action)
                valid_moves.append( robot_moves)
                # Get valid task assignments
                if robot.task_id is None or not self.tasks[robot.task_id].is_open():
                    valid_assignments.append(available_tasks.copy())
                else:
                    valid_assignments.append([])                
            
            return valid_moves, valid_assignments
            
    
    def _is_action_valid(self, robot_id: int, action: int) -> bool:
        """Check if action is valid for robot"""
        robot = self.robots[robot_id]
        
        if action == Action.MOVE_FORWARD:
            next_x, next_y = robot.get_forward_position()
            
            # Check bounds and obstacles
            if (next_x < 0 or next_x >= self.grid_width or 
                next_y < 0 or next_y >= self.grid_height or
                self.grid[next_y, next_x] == 1):
                return False
            
            # Check if another robot is there
            for other_robot in self.robots:
                if other_robot != robot and other_robot.x == next_x and other_robot.y == next_y:
                    return False
        
        return True

    def get_successor(self, state: Dict[str, Any], action: Tuple[List[int], List[int]]) -> Dict[str, Any]:
        old_state = self.get_state()
        self._set_state(state)
        self.step(action)
        new_state = self.get_state()
        self._set_state(old_state)
        return new_state

    def _assign_task(self, robot_id: int, task_id: int) -> bool:
        """Assign a task to a robot (for task scheduling)"""
        if (robot_id >= len(self.robots) or 
            task_id >= len(self.tasks) or
            not self.tasks[task_id].revealed or
            task_id is None):
            return False
        
        # Check if task is already assigned to another robot
        if self.tasks[task_id].assigned_robot is not None:
            self.robots[self.tasks[task_id].assigned_robot].task_id = None
        
        self.robots[robot_id].task_id = task_id
        self.tasks[task_id].assigned_robot = robot_id
        return True
    
    def get_unassigned_tasks(self) -> List[int]:
        """Get list of revealed but unassigned task IDs"""
        return [i for i, task in enumerate(self.tasks) 
                if task.revealed and task.assigned_robot is None and not task.is_closed()]
    
    def get_available_robots(self) -> List[int]:
        """Get list of robots without assigned tasks"""
        return [robot.id for robot in self.robots if robot.task_id is None]
    
    def render(self, mode='human'):
        """Render the environment"""
        print(f"Timestep: {self.timestep}, Completed Tasks: {self.completed_tasks}")
        
        # Create display grid
        display = np.full((self.grid_height, self.grid_width), '.', dtype=str)
        
        # Add obstacles
        display[self.grid == 1] = '#'
        
        # Add robots
        for i, robot in enumerate(self.robots):
            direction_chars = ['^', '>', 'v', '<']
            display[robot.y, robot.x] = direction_chars[robot.direction]
        
        # Add task targets
        for task in self.tasks:
            if (task.assigned_robot is not None or self.show_all_tasks) and task.revealed and not task.is_closed():
                errand = task.get_current_errand()
                if errand:
                    display[errand.target_y, errand.target_x] = 'T'
        
        # Print grid
        for row in display:
            print(''.join(row))
        print()
