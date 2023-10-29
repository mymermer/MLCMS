import sys
import tkinter
from tkinter import Button, Canvas, Menu, filedialog, Radiobutton
from scenario_elements import Scenario, Pedestrian
import json
import time
import numpy as np


class MainGUI():

    def __init__(self):
        self.win = None
        self.canvas = None
        self.canvas_image = None
        self.scenario = None
        self.directory_of_scenario = 'Exercise01/dummy_test_for_dijkstra.json'
        self.algorithm_choice = None

    def load_scenario(self, file_path: str = ""):
        if file_path == "":
            file_path = filedialog.askopenfilename(initialdir='./Exercise01', filetypes=[("JSON Files", "*.json")])

        self.directory_of_scenario = file_path

        with open(file_path, 'r') as file:
            data = json.load(file)

        self.scenario = Scenario(data['width'], data['height'])

        for target in data['targets']:
            self.scenario.grid[target[0], target[1]] = Scenario.NAME2ID['TARGET']

        self.scenario.pedestrians = []

        for pedestrian in data['pedestrians']:
            position = tuple(pedestrian['position'])
            desired_speed = pedestrian['desiredSpeed']
            self.scenario.pedestrians.append(Pedestrian(position, desired_speed))

        for obstacle in data['obstacles']:
            self.scenario.grid[obstacle[0], obstacle[1]] = Scenario.NAME2ID['OBSTACLE']

        self.scenario.to_image(self.canvas, self.canvas_image)

        self.scenario.update_cost()

    def restart_scenario(self):
        self.load_scenario(self.directory_of_scenario)

    def step_scenario(self):
        if self.scenario:
            self.scenario.update_step(self.algorithm_choice.get())
            self.scenario.to_image(self.canvas, self.canvas_image)
        else:
            print('No scenario loaded. Load a scenario before stepping.')

    def run_simulation(self):
        while self.scenario.pedestrians:
            self.step_scenario()
            self.win.update()  # Update the GUI to reflect the changes
            time.sleep(0.05)  # Adjust the sleep time as needed

    def exit_gui(self):
        sys.exit()

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
        self.canvas_image = self.canvas.create_image(5, 80, image=None, anchor=tkinter.NW)
        self.canvas.pack()
        self.load_scenario(self.directory_of_scenario)

        self.algorithm_choice = tkinter.StringVar()

        rb1 = Radiobutton(self.win, text='Euclidean distance', variable=self.algorithm_choice,
                          value="Euclidean Distance")
        rb1.place(x=20, y=10)
        rb2 = Radiobutton(self.win, text='Dijkstra Algorithm', variable=self.algorithm_choice,
                          value="Dijkstra Algorithm")
        rb2.place(x=200, y=10)

        self.algorithm_choice.set("Dijkstra Algorithm")

        btn = Button(self.win, text='Step simulation',
                     command=lambda: (self.step_scenario(), rb1.config(state="disabled"), rb2.config(state="disabled")))
        btn.place(x=20, y=40)
        btn = Button(self.win, text='Restart simulation',
                     command=lambda: (self.restart_scenario(), rb1.config(state="normal"), rb2.config(state="normal")))
        btn.place(x=200, y=40)
        btn = Button(self.win, text='Load simulation',
                     command=lambda: (self.load_scenario(), rb1.config(state="normal"), rb2.config(state="normal")))
        btn.place(x=380, y=40)
        btn = Button(self.win, text='Run simulation',
                     command=lambda: (self.run_simulation(), rb1.config(state="normal"), rb2.config(state="normal")))
        btn.place(x=560, y=40)


        self.win.mainloop()
