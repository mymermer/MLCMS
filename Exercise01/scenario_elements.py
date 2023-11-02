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
        """
        Initialize a Pedestrian.

        Args:
            position (tuple): Initial position (x, y).
            desired_speed (float): Desired speed.

        Attributes:
            _position (tuple): Current position.
            _desired_speed (float): Desired speed.
            _starting_position (tuple): Initial position.
            waiting (bool): Waiting state.
            total_time (float): Total time elapsed.
            waiting_time (float): Remaining wait time.
            distance_covered (float): Covered distance.
        """
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
        Get neighboring positions of the pedestrian within the scenario.


        This function calculates and returns a list of neighboring positions around the pedestrian's
        current position. Neighboring positions are constrained by the scenario's dimensions and exclude
        the pedestrian's own position.


        Parameters:
            scenario (Scenario): The scenario object that contains information about the environment.

        Returns:
            list of tuples: A list of neighboring positions as tuples.
        """

        def condition(x, y):
            return 0 <= x + self._position[0] < scenario.width and 0 <= y + self._position[
                1] < scenario.height and np.abs(
                x) + np.abs(y) > 0

        neighbours_list = []

        environment = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        for x, y in environment:
            if condition(x, y): neighbours_list.append((x + self._position[0], y + self._position[1]))

        return neighbours_list

    def update_euclidean_move(self, neighbours, scenario):
        """
        Update the pedestrian's position using the Euclidean method for movement.

        This function evaluates neighboring positions, checks for occupation by other pedestrians and
        obstacles, and selects the next position based on the Euclidean method. The next position aims to
        minimize the Euclidean distance towards the goal in the scenario.

        Parameters:
            neighbours (list of tuples): List of neighboring positions to consider.
            scenario (Scenario): The scenario object that contains information about the environment.

        Returns:
            tuple: The updated position based on the Euclidean method.
        """
        next_pos = self._position

        next_cell_distance = np.inf
        for (n_x, n_y) in neighbours:
            is_next_position_occupied = any(
                pedestrian.position == (n_x, n_y) for pedestrian in scenario.pedestrians)
            
            # Check if the next position is unoccupied and not an obstacle
            if not is_next_position_occupied and not bool(
                    scenario.grid[n_x, n_y] == scenario.NAME2ID['OBSTACLE']):
                
                # If the next cell's distance is shorter, update the position
                if next_cell_distance > scenario.euclidean_update_target_grid()[n_x, n_y]:
                    next_pos = (n_x, n_y)
                    next_cell_distance = scenario.euclidean_update_target_grid()[n_x, n_y]

        return next_pos

    def update_djikstra_move(self, neighbours, scenario, next_cell_distance):
        """
        Update the pedestrian's position using Dijkstra's algorithm for movement.

        This function evaluates neighboring positions, checks for occupation by other pedestrians, and
        selects the next position based on Dijkstra's algorithm. The next position minimizes the cost
        associated with moving towards the goal in the scenario.

        Parameters:
            neighbours (list of tuples): List of neighboring positions to consider.
            scenario (Scenario): The scenario object that contains information about the environment.
            next_cell_distance (float): The current estimated cost to reach the next cell.

        Returns:
            tuple: The updated position based on Dijkstra's algorithm.
        """
        next_pos = self._position
        for (n_x, n_y) in neighbours:
            is_next_position_occupied = any(
                pedestrian.position == (n_x, n_y) for pedestrian in scenario.pedestrians)
            
            # Check if the next position is unoccupied and has a lower cost according to Dijkstra's estimate
            if not is_next_position_occupied and next_cell_distance > scenario.dijkstra.estimate_cost(n_x, n_y):
                next_pos = (n_x, n_y)
                next_cell_distance = scenario.dijkstra.estimate_cost(n_x, n_y)

        return next_pos

    def check_diagonal(self, next_pos):
        """
        Check if the movement between the current position and the next position is diagonal.

        This function determines whether the movement from the current position to the next position
        is diagonal by comparing the absolute differences in the X and Y coordinates.

        Parameters:
            next_pos (tuple): The next position to which the pedestrian is moving.

        Returns:
            bool: True if the movement is diagonal, False otherwise.
        """
        movement_x = abs(next_pos[0] - self._position[0])
        movement_y = abs(next_pos[1] - self._position[1])

        # If the absolute differences in both X and Y coordinates are greater than or equal to 1, it's diagonal.
        if movement_x >= 1 and movement_y >= 1:
            return True
        else:
            return False

    def update_step(self, scenario: "Scenario", algorithm_choice):
        """
        
        Update the pedestrian's position and behavior within the scenario.

        This method updates the pedestrian's position and behavior based on the chosen algorithm.
        It takes into account the pedestrian's neighbors, desired speed, and algorithm choice.

        Parameters:
            scenario (Scenario): The scenario object where the pedestrian is placed.
            algorithm_choice (str): The choice of algorithm to determine the pedestrian's movement.

        Returns:
            None
        """
        neighbors = self.get_neighbours(scenario)
        if algorithm_choice == "Dijkstra Algorithm":
            # Calculate the estimated cost using Dijkstra's algorithm
            next_cell_distance = scenario.dijkstra.estimate_cost(self._position[0], self._position[1])
            
            if not self.waiting:
                # If not waiting, update the position based on Dijkstra's algorithm
                self.waiting = True
                next_pos = self.update_djikstra_move(neighbors, scenario, next_cell_distance)
                self._position = next_pos
                self.waiting_time = math.sqrt(2) / self._desired_speed if self.check_diagonal(
                    next_pos) else 1 / self._desired_speed
                self.distance_covered += self.waiting_time * self._desired_speed
            else:
                # Decrease waiting time and update total time
                self.waiting_time -= 0.1
                self.total_time += 0.1
                if round(self.waiting_time, 2) <= 0:
                    self.waiting = False

        else:
            # Calculate the distance using the Euclidean method
            neighbors = self.get_neighbours(scenario)
            if not self.waiting:
                # If not waiting, update the position based on Euclidean method
                self.waiting = True
                next_pos = self.update_euclidean_move(neighbors, scenario)
                # if diagonal increase waiting time to simulate real-life scenario as diagonal moves are longer
                self.waiting_time = math.sqrt(2) / self._desired_speed if self.check_diagonal(
                    next_pos) else 1 / self._desired_speed
                self._position = next_pos
                self.distance_covered += self.waiting_time * self._desired_speed
            else:
                # Decrease waiting time and update total time
                self.waiting_time -= 0.1
                self.total_time += 0.1
                if round(self.waiting_time, 2) <= 0:
                    self.waiting = False


    def reset_step(self):
        """
        Reset the pedestrian's position to its starting position.

        This method resets the pedestrian's position to the starting position, effectively
        taking the pedestrian back to its initial location within a scenario.

        Returns:
            None
        """
        self._position = self._starting_position

    
    def checkoloc(self,scenario: "Scenario"):
        """
        Check the position of the pedestrian and update speeds recordings accordingly.

        This method checks the current position of the pedestrian and updates the speed values
        in the provided 'scenario' object based on the pedestrian's desired speed.

        Parameters:
            scenario (Scenario): The scenario object where speed values should be updated.

        Note:
            The method checks the pedestrian's position and compares it to predefined positions
            (225, 18), (250, 17), and (250, 18). If the position matches any of these coordinates,
            it appends the pedestrian's desired speed to the respective speed list in the 'scenario'
            object.

        Returns:
            None
        """

        
        if (225,18)==(self._position[0],self._position[1]):
            scenario.speeds225_18.append(self.desired_speed)
        elif (250, 17) == (self._position[0], self._position[1]):
            scenario.speeds250_17.append(self.desired_speed)
        elif (250, 18) == (self._position[0], self._position[1]):
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

    def __init__(self, width, height, cellSize):
        """
        Initialize a Scenario object with a grid of the specified dimensions.

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
        """
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
        self.cell_size = cellSize

        # there might be problem in deciding height, width in my coding, however, if grid is square it won't be problem.
        self.dijkstra = None

        self.total_speeds225_18 = []
        self.total_speeds250_17 = []
        self.total_speeds250_18 = []

    def update_cost(self):
        """
        Uses dijkstra algorithm to calculate cost of the plain.
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

        Update the simulation for one time step based on the chosen algorithm.

        This function iterates through each pedestrian, updating their position for one time step
        based on the specified algorithm choice. If a pedestrian reaches the target, their statistics
        are recorded and removed from the simulation.

        Args:
            self: The instance of the class where this method is called.
            algorithm_choice: The chosen algorithm for updating pedestrian positions.

        Returns:
            None
        """

        for pedestrian in self.pedestrians:
            pedestrian.update_step(self, algorithm_choice)

            
            # Check if a pedestrian reaches the target
            if self.grid[pedestrian._position[0], pedestrian._position[1]] == Scenario.NAME2ID['TARGET']:

                # Since target is absorbing,we need to add one more step as the final step is not considered while adding time
                pedestrian.total_time += 1 / pedestrian._desired_speed
                speed = round(pedestrian.distance_covered / pedestrian.total_time * self.cell_size, 3)
                total_distance = pedestrian.distance_covered * self.cell_size
                self.speeds.append({"Totaltime": pedestrian.total_time, "DesiredSpeed": pedestrian._desired_speed,
                                    "Starting_Position": pedestrian._starting_position,
                                    "Total_distance": total_distance,
                                    "ActualSpeed": speed})
                print("Pedestrian at {} reached in time: {}".format(pedestrian._starting_position,
                                                                    round(pedestrian.total_time, 2)))
                print("Total distance: {}".format(total_distance))
                print(
                    "Speed of pedestrian: {} m/s".format(speed))
                self.pedestrians.remove(pedestrian)

        # saving the statistics to a json file
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
        Compute the Euclidean distance from each grid cell to the nearest target.

        This function calculates the Euclidean distance from each grid cell to the nearest target
        on the grid. It returns a 2D array where each cell contains the distance to the closest target.

        If there are no targets on the grid, it returns an array filled with zeros.

        Returns:
            numpy.ndarray: A 2D array representing the distances to the nearest target for each grid cell.
        """
        
        # Find the positions of all targets on the grid.

        targets = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append([y, x])  # y and x are flipped because they are in image space.
        if len(targets) == 0:
            return np.zeros((self.width, self.height))


        # Convert the target positions to a numpy array.
        targets = np.row_stack(targets)

        # Create arrays for x and y coordinates.
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)

        # Create mesh grids for x and y coordinates.
        xx, yy = np.meshgrid(x_space, y_space)

        # Combine x and y coordinates to create positions.
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Calculate the pair-wise distances between targets and all grid cell positions using scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # Compute the minimum distance over all targets for each grid cell.
        distances = np.min(distances, axis=0)

        # Reshape the distances to match the grid dimensions.
        distances = distances.reshape((self.width, self.height))

        return distances

    


    def control_points(self): 
        """
        Calculate and display average speeds for specific points.

        This function calculates and displays average speeds for predefined points based on
        pedestrian speeds. It checks pedestrian locations and updates the averages. If any
        of the points (225,3), (250,2), or (250,3) reach 60 average speed value updates, it prints
        the results.

        Args:
            self: The instance of the class where this method is called.

        Returns:
            None
        """

        # Initialize speed lists for different points
        self.speeds225_18=[]
        self.speeds250_17=[]
        self.speeds250_18=[]

        self.speeds225_18 = []
        self.speeds250_17 = []
        self.speeds250_18 = []

        # Iterate over pedestrians and check their locations
        for pedestrian in self.pedestrians:
            pedestrian.checkoloc(self)

        # Calculate and update average speeds for specific points
        if len(self.speeds225_18)!=0: self.total_speeds225_18.append(sum(self.speeds225_18) / len(self.speeds225_18))
        if len(self.speeds250_17)!=0:self.total_speeds250_17.append(sum(self.speeds250_17) / len(self.speeds250_17))
        if len(self.speeds250_18)!=0:self.total_speeds250_18.append(sum(self.speeds250_18) / len(self.speeds250_18))

        # Check if any of the points reached 60 updates and print the results
        if len(self.total_speeds250_18)==60 or (self.total_speeds225_18)==60 or (self.total_speeds250_17)==60:
            if len(self.total_speeds225_18)!=0:print("avarage speed in point (225,3): ",sum(self.total_speeds225_18) / len(self.total_speeds225_18)) 
            if len(self.total_speeds250_17)!=0:print("avarage speed in point (250,2):",sum(self.total_speeds250_17) / len(self.total_speeds250_17))
            if len(self.total_speeds250_18)!=0:print("avarage speed in point (250,3): ",sum(self.total_speeds250_18) / len(self.total_speeds250_18))

