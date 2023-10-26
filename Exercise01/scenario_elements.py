import scipy.spatial.distance
from PIL import Image, ImageTk
import numpy as np


class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed

    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (int(x + self._position[0]), int(y + self._position[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + self._position[0] < scenario.width and 0 <= y + self._position[1] < scenario.height and np.abs(x) + np.abs(y) > 0
        ]

    def update_step(self, scenario):
        """
        Moves to the cell with the lowest distance while avoiding obstacles.
        :param scenario: The current scenario instance.
        """
        neighbors = self.get_neighbors(scenario)
        current_cell_distance = scenario.target_distance_grids[self._position[0]][self._position[1]]
        next_pos = self._position

        # Find the neighboring cell with the lowest distance while avoiding obstacles
        for (n_x, n_y) in neighbors:
            neighbor_distance = scenario.target_distance_grids[n_x][n_y]
            if neighbor_distance < current_cell_distance:
                if scenario.grid[n_x][n_y] != Scenario.NAME2ID['OBSTACLE']:
                    next_pos = (n_x, n_y)
                    current_cell_distance = neighbor_distance

        # Check if the next cell is not an obstacle before updating position
        if scenario.grid[next_pos[0]][next_pos[1]] != Scenario.NAME2ID['OBSTACLE']:
            self._position = next_pos


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
        self.target_distance_grids = self.recompute_target_distances()

    def recompute_target_distances(self):
        self.target_distance_grids = self.update_target_grid()
        return self.target_distance_grids

    def update_target_grid(self):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        Obstacles are given a very high cost.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = []
        obstacles = []
        for x in range(self.width):
            for y in range(self.height):
                cell_value = self.grid[x, y]
                if cell_value == Scenario.NAME2ID['TARGET']:
                    targets.append([y, x])  # y and x are flipped because they are in image space.
                elif cell_value == Scenario.NAME2ID['OBSTACLE']:
                    obstacles.append([y, x])

        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        targets = np.row_stack(targets)
        obstacles = np.row_stack(obstacles) if obstacles else None

        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])


        # Calculate pair-wise distances
        distances_to_targets = scipy.spatial.distance.cdist(targets, positions)
        if obstacles is not None:
            distances_to_obstacles = scipy.spatial.distance.cdist(obstacles, positions)

            # Setting a high cost for obstacles
            max_obstacle_distance = 1000
            obstacle_costs = np.max(distances_to_obstacles, axis=0)
            distances = np.where(obstacle_costs == 0, max_obstacle_distance, distances_to_targets)
        else:
            distances = distances_to_targets

        # Compute the minimum distance over all distances to all targets.
        distances = np.min(distances, axis=0)

        return distances.reshape((self.width, self.height))

    def update_step(self):
        """
        Updates the position of all pedestrians.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        for pedestrian in self.pedestrians:
            pedestrian.update_step(self)

    @staticmethod
    def cell_to_color(_id):
        return Scenario.NAME2COLOR[Scenario.ID2NAME[_id]]

    def target_grid_to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the distance to the target stored in
        self.target_distance_gids.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                target_distance = self.target_distance_grids[x][y]
                pix[x, y] = (max(0, min(255, int(10 * target_distance) - 0 * 255)),
                             max(0, min(255, int(10 * target_distance) - 1 * 255)),
                             max(0, min(255, int(10 * target_distance) - 2 * 255)))
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)

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
