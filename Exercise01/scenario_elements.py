import scipy.spatial.distance
from PIL import Image, ImageTk
import numpy as np
import math
import json
from dijkstra import Dijkstra_algorithm  # new file


class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed
        self._starting_position = position
        self.waiting = False
        self.total_time = 0
        self.waiting_time = 0
        self.distance_covered = 0

    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbours(self, scenario):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        def condition(x,y):
            return 0 <= x + self._position[0] < scenario.width and 0 <= y + self._position[1] < scenario.height and np.abs(
                x) + np.abs(y) > 0 

        neighbours_list=[]

        environment=[(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]


        for x,y in environment:
            if condition(x,y): neighbours_list.append((x + self._position[0] , y + self._position[1] ))

        return neighbours_list

    def update_euclidean_move(self, neighbours, scenario):
        """
        updates the move of the pedestrians when Euclidean distance is selected.
        """
        next_pos = self._position
        next_cell_distance= np.inf
        for (n_x, n_y) in neighbours:
            is_next_position_occupied = any(
                pedestrian.position == (n_x, n_y) for pedestrian in scenario.pedestrians)
            if not is_next_position_occupied and not bool(
                    scenario.grid[n_x, n_y] == scenario.NAME2ID['OBSTACLE']):
                if next_cell_distance > scenario.euclidean_update_target_grid()[n_x, n_y]:
                    next_pos = (n_x, n_y)
                    next_cell_distance = scenario.euclidean_update_target_grid()[n_x, n_y]

        return next_pos

    def update_djikstra_move(self, neighbours, scenario, next_cell_distance):
        """
        updates the move of the pedestrians when Euclidean distance is selected.
        """
        next_pos = self._position
        for (n_x, n_y) in neighbours:
            is_next_position_occupied = any(
                pedestrian.position == (n_x, n_y) for pedestrian in scenario.pedestrians)
            if not is_next_position_occupied and next_cell_distance > scenario.dijkstra.estimate_cost(n_x, n_y):
                next_pos = (n_x, n_y)
                next_cell_distance = scenario.dijkstra.estimate_cost(n_x, n_y)

        return next_pos

    def check_diagonal(self, next_pos):
        movement_x = abs(next_pos[0] - self._position[0])
        movement_y = abs(next_pos[1] - self._position[1])
        if movement_x >= 1 and movement_y >= 1:
            return True
        else:
            return False


    def update_step(self, scenario: "Scenario", algorithm_choice):
        """
        Moves to the cell by cost.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.

        :param scenario: The current scenario instance.
        """
        neighbors = self.get_neighbours(scenario)
        next_pos = self._position
        if algorithm_choice == "Dijkstra Algorithm":
            next_cell_distance = scenario.dijkstra.estimate_cost(self._position[0], self._position[1])
            if not self.waiting:
                self.waiting = True
                next_pos = self.update_djikstra_move(neighbors, scenario, next_cell_distance)
                self._position = next_pos
                self.waiting_time = math.sqrt(2) / self._desired_speed if self.check_diagonal(
                    next_pos) else 1 / self._desired_speed
            else:
                self.waiting_time -= 0.1
                self.total_time += 0.1
                if round(self.waiting_time, 2) <= 0:
                    self.waiting = False

        else:
            neighbors = self.get_neighbours(scenario)
            next_cell_distance = scenario.euclidean_update_target_grid()[self._position[0]][self._position[1]]
            if not self.waiting:
                self.waiting = True
                next_pos = self.update_euclidean_move(neighbors, scenario)
                # if diagonal increase waiting time to simulate real-life scenario as diagonal moves are longer
                self.waiting_time = math.sqrt(2) / self._desired_speed if self.check_diagonal(next_pos) else 1 / self._desired_speed
                self._position = next_pos
                self.distance_covered +=self.waiting_time*self._desired_speed
            else:
                self.waiting_time -= 0.1
                self.total_time += 0.1
                if round(self.waiting_time, 2) <= 0:
                    self.waiting = False

    def reset_step(self):
        self._position = self._starting_position
    
    def checkoloc(self,scenario: "Scenario"):
        if (225,18)==(self._position[0],self._position[1]):
            scenario.speeds225_18.append(self.desired_speed)
        elif (250,17)==(self._position[0],self._position[1]):
            scenario.speeds250_17.append(self.desired_speed)
        elif (250,18)==(self._position[0],self._position[1]):
            scenario.speeds250_18.append(self.desired_speed)

class Scenario:
    """
    A scenario for a cellular automaton.
    """
    GRID_SIZE = (500, 500)
    ID2NAME = {
        0: 'EMPTY',
        1: 'TARGET',
        2: 'OBSTACLE',
        3: 'PEDESTRIAN'
    }
    NAME2COLOR = {
        'EMPTY': (255, 255, 255),
        'PEDESTRIAN': (255, 0, 0),
        'TARGET': (0, 0, 255),
        'OBSTACLE': (255, 0, 255)
    }
    NAME2ID = {
        ID2NAME[0]: 0,
        ID2NAME[1]: 1,
        ID2NAME[2]: 2,
        ID2NAME[3]: 3
    }

    def __init__(self, width, height):
        if width < 1 or width > 1024:
            raise ValueError(f"Width {width} must be in [1, 1024].")
        if height < 1 or height > 1024:
            raise ValueError(f"Height {height} must be in [1, 1024].")

        self.width = width
        self.height = height
        self.grid_image = None
        self.grid = np.zeros((width, height))
        self.pedestrians = []
        self.speeds = []

        # there might be problem in deciding height, width in my coding, however, if grid is square it won't be problem.
        self.dijkstra = None

        self.total_speeds225_18=[]
        self.total_speeds250_17=[]
        self.total_speeds250_18=[]

    def update_cost(self):
        """
        Uses dijkstra algorthim to calculate cost of the plain.
        This does not take obstacles into account.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append([y, x])  # y and x are flipped because they are in image space.
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        check_for_obstacle = np.zeros((self.width, self.height))

        for x in range(self.width):
            for y in range(self.height):
                check_for_obstacle[y][x] = bool(self.grid[x, y] == Scenario.NAME2ID['OBSTACLE'])

        self.dijkstra = Dijkstra_algorithm(self.height, self.width, targets,
                                           check_for_obstacle)  # cost regarding to Dijkstra_algorithm
        self.dijkstra.execute()

    def update_step(self, algorithm_choice):
        """
        Updates the position of all pedestrians.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        for pedestrian in self.pedestrians:
            pedestrian.update_step(self, algorithm_choice)
            # removing the pedestrians if it reaches the target
            if self.grid[pedestrian._position[0], pedestrian._position[1]] == Scenario.NAME2ID['TARGET']:
                # Since target is absorbing,we need to add one more step as the final step is not considered while adding time
                pedestrian.total_time += 1/pedestrian._desired_speed
                self.speeds.append({"Totaltime":pedestrian.total_time, "DesiredSpeed":pedestrian._desired_speed, "Starting_Position":pedestrian._starting_position,"Total_distance":pedestrian.distance_covered, "ActualSpeed":round(pedestrian.distance_covered/pedestrian.total_time, 3)})
                print("Pedestrian at {} reached in time: {}".format(pedestrian._starting_position,round(pedestrian.total_time, 2)))
                print("Total distance: {}".format(pedestrian.distance_covered))
                print("Speed of pedestrian: {} m/s".format(round(pedestrian.distance_covered/pedestrian.total_time, 2)))
                self.pedestrians.remove(pedestrian)

        #saving the speed statistics to a json file
        if len(self.pedestrians) == 0:
            with open('./Exercise01/statistics.json', 'w') as file:
                json.dump(self.speeds, file, indent=2)

    @staticmethod
    def cell_to_color(_id):
        return Scenario.NAME2COLOR[Scenario.ID2NAME[_id]]

    def to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterwards, separately.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """

        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                pix[x, y] = self.cell_to_color(self.grid[x, y])
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            pix[x, y] = Scenario.NAME2COLOR['PEDESTRIAN']
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)

    def euclidean_update_target_grid(self):
        """
        Uses dijkstra algorthim to calculate cost of the plain.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append([y, x])  # y and x are flipped because they are in image space.
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        targets = np.row_stack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # Compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)
        distances = distances.reshape((self.width, self.height))

        return distances
    


    def control_points(self): 

        self.speeds225_18=[]
        self.speeds250_17=[]
        self.speeds250_18=[]





        for pedestrian in self.pedestrians:
            pedestrian.checkoloc(self)


        if len(self.speeds225_18)!=0: self.total_speeds225_18.append(sum(self.speeds225_18) / len(self.speeds225_18))
        if len(self.speeds250_17)!=0:self.total_speeds250_17.append(sum(self.speeds250_17) / len(self.speeds250_17))
        if len(self.speeds250_18)!=0:self.total_speeds250_18.append(sum(self.speeds250_18) / len(self.speeds250_18))


        if len(self.total_speeds250_18)==60:
            if len(self.total_speeds225_18)!=0:print("avarage speed in point (225,3): ",sum(self.total_speeds225_18) / len(self.speeds225_18)) 
            if len(self.total_speeds250_17)!=0:print("avarage speed in point (250,2):",sum(self.total_speeds250_17) / len(self.speeds250_17))
            if len(self.total_speeds250_18)!=0:print("avarage speed in point (250,3): ",sum(self.total_speeds250_18) / len(self.speeds250_18))