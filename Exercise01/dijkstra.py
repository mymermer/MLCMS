import numpy as np
import math
from queue import PriorityQueue



#used for visiulization of cost matrices you can uncomment if you want to visualize cost_matrix:
# import matplotlib.pyplot as plt




class Dijkstra_algorithm:

    """ This class implements the Dijkstra's algorithm to find the shortest paths from the targets to other points on the grid. """

    def __init__(self, height, width, target_locations, check_for_obstacle):

        """
    :param height: The height of the grid.
    :param width: The width of the grid.
    :param target_locations: the list of target locations.
    :param check_for_obstacle: the boolean matrix indicating which locations are occupied by obstacles.
        """
        
        self.cost_matrix = np.full((height, width),
                                   np.inf)  # to make it modular, grid is considered to be the size of the current scenario.
        self.width = width
        self.height = height
        self.target_locations = target_locations
        self.check_for_obstacle = check_for_obstacle

    def execute(self):

        """

        This method calculates the cost matrix for the given scenario using Dijkstra s algorithm. 
        The cost matrix is a 2D matrix that contains the cost of given point calulated from Dijkstra's Algorithm.

        Parameters:
        None

        Returns:
            None


        """
        #used to calculate min cost point which will be used to calculate cost of neigbours
        queue = PriorityQueue()
         
        #gradient movment is more costlier, therefore we increased its cost.
        sqrt2 = round(math.sqrt(2), 2)

        # A value of 0 is assigned to each target on the grid in the cost matrix, these values directly pushed into priority queue
        for target in self.target_locations:
            x, y = target[0], target[1]
            self.cost_matrix[x][y] = 0
            queue.put((0, (x, y)))

        # Costs are updated for all neighbors of cells in the queue. 
        
        
        #iter=0 # explained below 20 25 lines
        while not queue.empty():
            cost, (x, y) = queue.get()
            for i in [-1, 0, 1]:  # will be used for x
                for j in [-1, 0, 1]:  # will be used for y
                    if i == 0 and j == 0:
                        continue  # skip the current point

                    if 0 <= x + i < self.height and 0 <= y + j < self.width and not \
                            self.check_for_obstacle[x + i][y + j]:

                        if (i * j != 0):  # diagonal
                            if self.cost_matrix[x + i][y + j] > sqrt2 + cost:
                                self.cost_matrix[x + i][y + j] = sqrt2 + cost  # diagonal moves are sqrt(2) long.
                                queue.put((sqrt2 + cost, (x + i, y + j)))
                        else:             # straight-line 
                            if self.cost_matrix[x + i][y + j] > 1 + cost:
                                self.cost_matrix[x + i][y + j] = 1 + cost
                                queue.put((1 + cost, (x + i, y + j)))
                    
            #to visulize any time in expension of cost matrix, you can chanage iter variable in here
            # begin:                    
            # if iter==0:
            #     temporal_cost_matrix=self.cost_matrix.copy()

            # iter=iter+1
            #:end

        #to copy a matrix use .copy(), if not, they will point to same matrix
        
        #vuslizing cost matrix
        #start:
        # plt.imshow(temporal_cost_matrix, cmap='coolwarm_r', interpolation='nearest')
        # plt.colorbar(label='Cost')

        # # Annotate the values
        # interval = 10 #avoid too much numbers at screen
        # for i in range(temporal_cost_matrix.shape[0]):
        #     for j in range(temporal_cost_matrix.shape[1]):
        #         if i % interval == 0 and j % interval == 0:
        #             plt.text(j, i, f'{temporal_cost_matrix[i, j]:.2f}', ha='center', va='center', color='black')


        # plt.title('Cost Matrix Visualization')
        # plt.show()
        #:end

        self.cost_matrix = self.cost_matrix.T #The cost matrix is transposed to adapt it to the grid representation.

    def estimate_cost(self, x, y):

        """
        Calculate the cost of given point using Dijkstra Algorithm
        :param x: x cordinate of desired point
        :param y: y cordinate of desired point

        Returns:
            None
        """

 

        if x % 1 == 0 and y % 1 == 0:  # if coordinates are integer, their value is already stored in cost_matrix
            return self.cost_matrix[int(x)][int(y)]
        
        else: 
             #One of the cordinate locations (x, y) is not integer.
            #In this case, we will use nearest points' costs (inderectly proportional to distance) in order to find aproximate cost of the point.
        

            ceil_x, ceil_y = math.ceil(x), math.ceil(y) # Coordinates of the cell below and to the right of (x, y)
            floor_x, floor_y = math.floor(x), math.floor(y) # Coordinates of the cell above and to the left of (x, y)

            if x % 1 == 0:  # Position lies on y-axis only (between two grid cells on the vertical line with coordinates x)
                distance1 = abs(ceil_y - y) #relative distance between the position and the cell below it
                distance2 = abs(floor_y - y) #relative distance between the position and the cell above it
                effect1 = self.cost_matrix[int(x)][ceil_y] / distance1
                effect2 = self.cost_matrix[int(x)][floor_y] / distance2
                return (effect1 + effect2) / (1 / distance1 + 1 / distance2) 

            elif y % 1 == 0:  # Position lies on x-axis only (between two grid cells on the horizontal line with coordinates x)
                distance1 = abs(ceil_x - x) #relative distance between the position and the cell to the right of it
                distance2 = abs(floor_x - x) #relative distance between the position and the cell to the left of it
                effect1 = self.cost_matrix[ceil_x][int(y)] / distance1 #inversely proportional to distance
                effect2 = self.cost_matrix[floor_x][int(y)] / distance2 #inversely proportional to distance
                return (effect1 + effect2) / (1 / distance1 + 1 / distance2)

            else:  # Position is not on the any integer valued x-y line therefore, we used nearest 4 known points
                distance1 = math.sqrt((ceil_x - x) ** 2 + (ceil_y - y) ** 2) 
                distance2 = math.sqrt((floor_x - x) ** 2 + (floor_y - y) ** 2)
                distance3 = math.sqrt((ceil_x - x) ** 2 + (floor_y - y) ** 2)
                distance4 = math.sqrt((floor_x - x) ** 2 + (ceil_y - y) ** 2)

                effect1 = self.cost_matrix[ceil_x][ceil_y] / distance1
                effect2 = self.cost_matrix[floor_x][floor_y] / distance2
                effect3 = self.cost_matrix[ceil_x][floor_y] / distance3
                effect4 = self.cost_matrix[floor_x][ceil_y] / distance4


                return (effect1 + effect2 + effect3 + effect4) / (
                        1 / distance1 + 1 / distance2 + 1 / distance3 + 1 / distance4) #finding average using inverse proportion

