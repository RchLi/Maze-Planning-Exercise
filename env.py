# Envrionment: maze task as suggested in 
# https://www.sciencedirect.com/science/article/pii/S0896627306002728

import numpy as np
import torch
import math
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode

class Name:
    down = 0
    up = 1
    right = 2
    left = 3

    blank = 1
    current = 2
    goal  = 3
    wall = -1



class MazeEnv:
    def __init__(self, length=9, resize=18, walls_count = 2):
        self.length = length
        self.maze = torch.ones(length, length)
        self.walls_count = walls_count
        self.get_task(walls_count)
        self.start, self.goal = self.get_postion()
        self.current = self.start
        self.solution = None
        self.transform = Resize((resize, resize), interpolation=InterpolationMode.NEAREST)

    
    # input: action, scalar
    # output: reward, scalar
    # -1 for not hitting the goal, -2 for hitting the wall, 0 for hitting the goal
    def step(self, action):
        xc, yc = self.current
        if action == Name.down:
            loc_next = (min(xc + 1, self.length - 1), yc)
        elif action == Name.up:
            loc_next = (max(xc - 1, 0), yc)
        elif action == Name.right:
            loc_next = (xc, min(yc + 1, self.length - 1))
        elif action == Name.left:
            loc_next = (xc, max(yc - 1, 0))

        if self.maze[loc_next] == Name.wall:
            return -2
        elif self.maze[loc_next] == Name.goal:
            self.maze[self.current] = Name.blank
            self.maze[loc_next] = Name.current
            self.current = loc_next
            return 0
        else:
            self.maze[self.current] = Name.blank
            self.maze[loc_next] = Name.current
            self.current = loc_next
            return -1

    # generate a random trajectory for pretraining
    # return np array of size (size, 1)
    def random_walk(self, num_step=50, state=True):
        walks = []
        num_voc = self.length**2 + 5
        x, y = self.current
        start = x * self.length + y
        walks.append(start)
        for _ in range(num_step):
            action = np.random.randint(0, 4)
            walks.append(action + self.length**2)
            self.step(action)
            if state:
                x, y = self.current
                walks.append(x * self.length + y)
        if not state:
            x, y = self.current
            walks.append(x * self.length + y)

        return np.array(walks)

    # return img of the current state, tensor [3, length, length]
    def img(self):
        # mz = self.maze.clone()
        # return torch.stack([mz, mz, mz]) / 4 + .25
        wall = torch.zeros(self.length, self.length)
        wall[self.maze == -1] = .5
        goal = torch.zeros(self.length, self.length)
        goal[self.goal] = .5
        cur = torch.zeros(self.length, self.length)
        cur[self.current] = .5

        return torch.stack([wall, goal, cur])

    # reset the maze to start position
    def reset(self):
        self.maze[self.current] = Name.blank
        self.maze[self.start] = Name.current
        self.maze[self.goal] = Name.goal
        self.current = self.start

    # change a starting location
    def random_start(self):
        self.maze[self.current] = Name.blank
        xs, ys = torch.randint(0, self.length, (1,)).item(), torch.randint(0, self.length, (1,)).item()
        while self.maze[xs, ys] == Name.goal:
            xs, ys = torch.randint(0, self.length, (1,)).item(), torch.randint(0, self.length, (1,)).item()
        self.maze[xs, ys] =  Name.current
        self.start = (xs, ys)
        self.current = (xs, ys)

    def set_start(self, loc):
        self.maze[self.current] = Name.blank
        self.maze[loc] = Name.current
        self.start = loc
        self.current = loc


    def set_state(self, maze):
        self.maze = maze.clone()
        for i in range(self.length):
            for j in range(self.length):
                if maze[i][j] == 2:
                    self.start = (i, j)
                    self.current = (i, j)
                elif maze[i][j] == 3:
                    self.goal = (i, j)

    # simple greedy algorithm for finding the optimal path for complex task simply return none
    # output: tensor of scalar representation
    def get_solution(self, state=False):
        # if self.solution:
        #     return self.solution.copy()
        xg, yg = self.goal
        xc, yc = self.start
        solution = []
        if xc == xg:
            direct_x = 0
        else:
            direct_x = 1 if xg > xc else -1
        if yc == yg:
            direct_y = 0
        else:
            direct_y = 1 if yg > yc else -1
        while (xg, yg) != (xc, yc):
            if xc != xg and self.maze[xc + direct_x, yc] != -1:
                xc += direct_x
                act = 0 if direct_x == 1 else 1
                solution.append(act )
            elif yc != yg and self.maze[xc, yc + direct_y] != -1:
                yc += direct_y
                act = 2 if direct_y == 1 else 3
                solution.append(act)
            else:
                return None
            if state:
                solution.append(xc * self.length + yc)
        self.solution = solution.copy()
        return torch.tensor(solution)


    # generate walls and goal
    # walls are specified by (start location, direction, wall length)
    def get_task(self, walls_count): 
        for i in range(walls_count):
            x, y = torch.randint(1, self.length-1, (1,)).item(), torch.randint(1, self.length-1, (1,)).item()
            self.maze[x,y] = -1
            # wall direction, 0,1,2,3 corresponds to south, north, east, west
            direction = torch.randint(0, 4, (1,)).item()
            wall_len = torch.randint(math.floor(self.length/2)-2, math.floor(self.length/2), (1,)).item()
            if direction == 0:
                for i in range(x, min(self.length, x + wall_len)):
                    self.maze[i, y] = -1
            elif direction == 1:
                for i in range(max(0, x - wall_len), x):
                    self.maze[i, y] = -1
            elif direction == 2:
                for j in range(y, min(self.length, y + wall_len)):
                    self.maze[x, j] = -1
            elif direction == 3:
                for j in range(max(0, y - wall_len), y):
                    self.maze[x, j] = -1

    # generate start and goal
    def get_postion(self):
        xs, ys = torch.randint(0, self.length, (1,)).item(), torch.randint(0, self.length, (1,)).item()
        while self.maze[xs, ys] != 1:
            xs, ys = torch.randint(0, self.length, (1,)).item(), torch.randint(0, self.length, (1,)).item()
        self.maze[xs, ys] = 2

        xg, yg = torch.randint(0, self.length, (1,)).item(), torch.randint(0, self.length, (1,)).item()
        while self.maze[xg, yg] != 1:
            xg, yg = torch.randint(0, self.length, (1,)).item(), torch.randint(0, self.length, (1,)).item()
        self.maze[xg, yg] = 3

        return (xs, ys), (xg, yg)

    # convert number to action
    def convert(self, num):
        if num == 0:
            return 'down'
        elif num == 1:
            return 'up'
        elif num == 2:
            return 'right'
        elif num == 3:
            return 'left'
        else:
            return num



    # def state(self, cur_only=False):
    #     mat_cur = torch.zeros(self.length, self.length)
    #     mat_cur[self.current] = 1
    #     mat_goal = torch.zeros(self.length, self.length)
    #     mat_goal[self.goal] = 1
    #     mat_obs = torch.zeros(self.length, self.length)
    #     mat_obs[self.maze == -1] = 1
        
    #     if cur_only:
    #         return mat_cur.view(-1)
        
    #     else:
    #         return torch.stack([mat_obs, mat_goal, mat_cur])

    # def initial(self):
    #     mat_start = torch.zeros(self.length, self.length)
    #     mat_start[self.start] = 1
    #     mat_goal = torch.zeros(self.length, self.length)
    #     mat_goal[self.goal] = 1
    #     mat_obs = torch.zeros(self.length, self.length)
    #     mat_obs[self.maze == -1] = 1
        
    #     return torch.stack([mat_start, mat_goal, mat_obs])