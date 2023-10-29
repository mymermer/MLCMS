import numpy as np
import math
from queue import PriorityQueue


# cost_matrix and
class Dijkstra_algorithm:
    def __init__(self, height, width, target_locations, check_for_obstacle):
        self.cost_matrix = np.full((height, width),
                                   np.inf)  # to make it modular,grid is considered to be size of the current scenario.
        self.width = width
        self.height = height
        self.target_locations = target_locations
        self.check_for_obstacle = check_for_obstacle

    def execute(self):

        queue = PriorityQueue()
        sqrt2 = round(math.sqrt(2), 2)

        for target in self.target_locations:
            x, y = target[0], target[1]
            self.cost_matrix[x][y] = 0
            queue.put((0, (x, y)))

        while not queue.empty():
            cost, (x, y) = queue.get()
            for i in [-1, 0, 1]:  # will be used for x
                for j in [-1, 0, 1]:  # will be used for y
                    if i == 0 and j == 0:
                        continue  # skip the current point

                    if 0 <= x + i < self.height and 0 <= y + j < self.width and not \
                            self.check_for_obstacle[x + i][y + j]:

                        if (i * j != 0):  # if true, then diagonal
                            if self.cost_matrix[x + i][y + j] > sqrt2 + cost:
                                self.cost_matrix[x + i][y + j] = sqrt2 + cost  # diagonal moves are sqrt(2) long.
                                queue.put((sqrt2 + cost, (x + i, y + j)))
                        else:
                            if self.cost_matrix[x + i][y + j] > 1 + cost:
                                self.cost_matrix[x + i][y + j] = 1 + cost
                                queue.put((1 + cost, (x + i, y + j)))

        self.cost_matrix = self.cost_matrix.T

    def estimate_cost(self, x, y):
        if x % 1 == 0 and y % 1 == 0:  # Cell coordinates are integer
            return self.cost_matrix[int(x)][int(y)]
        else:
            ceil_x, ceil_y = math.ceil(x), math.ceil(y)
            floor_x, floor_y = math.floor(x), math.floor(y)

            if x % 1 == 0:  # Position lies on y-axis only
                distance1 = abs(ceil_y - y)
                distance2 = abs(floor_y - y)
                effect1 = self.cost_matrix[int(x)][ceil_y] / distance1
                effect2 = self.cost_matrix[int(x)][floor_y] / distance2
                return (effect1 + effect2) / (1 / distance1 + 1 / distance2)

            elif y % 1 == 0:  # Position lies on x-axis only
                distance1 = abs(ceil_x - x)
                distance2 = abs(floor_x - x)
                effect1 = self.cost_matrix[ceil_x][int(y)] / distance1
                effect2 = self.cost_matrix[floor_x][int(y)] / distance2
                return (effect1 + effect2) / (1 / distance1 + 1 / distance2)

            else:  # Position is a diagonal from integer coordinates
                distance1 = math.sqrt((ceil_x - x) ** 2 + (ceil_y - y) ** 2)
                distance2 = math.sqrt((floor_x - x) ** 2 + (floor_y - y) ** 2)
                distance3 = math.sqrt((ceil_x - x) ** 2 + (floor_y - y) ** 2)
                distance4 = math.sqrt((floor_x - x) ** 2 + (ceil_y - y) ** 2)

                effect1 = self.cost_matrix[ceil_x][ceil_y] / distance1
                effect2 = self.cost_matrix[floor_x][floor_y] / distance2
                effect3 = self.cost_matrix[ceil_x][floor_y] / distance3
                effect4 = self.cost_matrix[floor_x][ceil_y] / distance4

                return (effect1 + effect2 + effect3 + effect4) / (
                        1 / distance1 + 1 / distance2 + 1 / distance3 + 1 / distance4)
