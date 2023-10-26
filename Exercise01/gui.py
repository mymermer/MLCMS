import sys
import tkinter
from tkinter import Button, Canvas, Menu, filedialog
from scenario_elements import Scenario, Pedestrian
import json

class MainGUI():

    def __init__(self):
        self.win = None
        self.canvas = None
        self.canvas_image = None
        self.scenario = None

    def create_scenario(self, scenario_file):
        with open(scenario_file, 'r') as file:
            data = json.load(file)

        self.scenario = Scenario(data['width'], data['height'])
        print(self.scenario)

        for target in data['targets']:
            self.scenario.grid[target[0], target[1]] = Scenario.NAME2ID['TARGET']

        self.scenario.pedestrians = []

        for pedestrian in data['pedestrians']:
            position = tuple(pedestrian['position'])
            desired_speed = pedestrian['desiredSpeed']
            self.scenario.pedestrians.append(Pedestrian(position, desired_speed))

        for obstacle in data['obstacles']:
            self.scenario.grid[obstacle[0], obstacle[1]] = Scenario.NAME2ID['OBSTACLE']

        # self.scenario.recompute_target_distances()
        self.scenario.to_image(self.canvas, self.canvas_image)
        self.scenario.update_cost()

    def restart_scenario(self):
        print('Restart not implemented yet')

    def step_scenario(self):
        if self.scenario:
            self.scenario.update_step()
            self.scenario.to_image(self.canvas, self.canvas_image)
        else:
            print('No scenario loaded. Load a scenario before stepping.')

    def exit_gui(self):
        sys.exit()

    def load_scenario(self):
        scenario_file = filedialog.askopenfilename(
            title="Select Scenario File", filetypes=(("JSON files", "*.json"), ("all files", "*.*"))
        )
        if scenario_file:
            self.create_scenario(scenario_file)

    def start_gui(self):
        self.win = tkinter.Tk()
        self.win.geometry('500x500')
        self.win.title('Cellular Automata GUI')


        menu = Menu(self.win)
        self.win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=lambda: self.load_scenario())
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)

        self.canvas = Canvas(self.win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])
        self.canvas_image = self.canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        self.canvas.pack()
        self.create_scenario('scenario.json')

        btn = Button(self.win, text='Step simulation', command=lambda: self.step_scenario())
        btn.place(x=20, y=10)
        btn = Button(self.win, text='Restart simulation', command=self.restart_scenario)
        btn.place(x=200, y=10)
        btn = Button(self.win, text='Create simulation', command=lambda: self.load_scenario())
        btn.place(x=380, y=10)

        self.win.mainloop()

