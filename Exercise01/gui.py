import sys
import tkinter
from tkinter import Button, Canvas, Menu, filedialog, Radiobutton, messagebox
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
        self.directory_of_scenario = 'Exercise01/dummy_test_for_dijkstra.json' # 'dummy_test_for_dijkstra.json'
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

    def save_scenario(self):
        f = tkinter.filedialog.asksaveasfilename(confirmoverwrite='TRUE', defaultextension=".json")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        data = {
            "width": self.scenario.width,
            "height": self.scenario.height,
            "pedestrians": [],
            "targets": [],
            "obstacles": []
        }
        for pedestrian in self.scenario.pedestrians:
            data["pedestrians"].append({
                'position': pedestrian.position,
                'desiredSpeed': pedestrian.desired_speed
            })
        self.create_targets_obstacles_lists(data)
        with open(f, 'w') as json_file:
            str(json.dump(data, json_file, indent=4))

    def create_targets_obstacles_lists(self, data):
        for x in range(self.scenario.width):
            for y in range(self.scenario.height):
                if self.scenario.grid[x, y] == 1:
                    data["targets"].append(
                        [x, y]
                    )
                elif self.scenario.grid[x, y] == 2:
                    data["obstacles"].append(
                        [x, y]
                    )

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
        self.win.geometry('600x600')
        self.win.title('Cellular Automata GUI')

        menu = Menu(self.win)
        self.win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=lambda: self.load_scenario())
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)

        self.canvas = Canvas(self.win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])
        self.canvas_image = self.canvas.create_image(5, 120, image=None, anchor=tkinter.NW)
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
        btn = Button(self.win, text='Run simulation',
                     command=lambda: (self.run_simulation(), rb1.config(state="normal"), rb2.config(state="normal")))
        btn.place(x=380, y=40)
        btn = Button(self.win, text='Create simulation', command=lambda: self.popupwin())
        btn.place(x=20, y=70)
        btn = Button(self.win, text='Load simulation',
                     command=lambda: (self.load_scenario(), rb1.config(state="normal"), rb2.config(state="normal")))
        btn.place(x=200, y=70)
        btn = Button(self.win, text='Save simulation',
                     command=lambda: self.save_scenario())
        btn.place(x=380, y=70)

        self.win.mainloop()

    def popupwin(self):
        """
        Creates and shows a simple user interface with multiple labels, entries and buttons.
        The user can set a new scenario size, as well as add pedestrians, targets and obstacles.
        """

        # Create a Toplevel window
        top = tkinter.Toplevel(self.win)
        top.geometry("500x250")

        # labels
        # lbl = tkinter.Label(top, text="Set dimensions")
        # lbl.place(x=40, y=30)
        lbl = tkinter.Label(top, text="Add pedestrian")
        lbl.place(x=40, y=80)
        lbl = tkinter.Label(top, text="Add target")
        lbl.place(x=40, y=110)
        lbl = tkinter.Label(top, text="Add obstacle")
        lbl.place(x=40, y=140)

        lbl = tkinter.Label(top, text="x coordinates")
        lbl.place(x=170, y=60)
        lbl = tkinter.Label(top, text="y coordinates")
        lbl.place(x=250, y=60)

        # entry fields
        # ew = tkinter.Entry(top, width=10)
        # ew.place(x=180, y=30)
        # eh = tkinter.Entry(top, width=10)
        # eh.place(x=250, y=30)

        epx = tkinter.Entry(top, width=10)
        epx.place(x=180, y=80)
        epy = tkinter.Entry(top, width=10)
        epy.place(x=250, y=80)

        etx = tkinter.Entry(top, width=10)
        etx.place(x=180, y=110)
        ety = tkinter.Entry(top, width=10)
        ety.place(x=250, y=110)

        eox = tkinter.Entry(top, width=10)
        eox.place(x=180, y=140)
        eoy = tkinter.Entry(top, width=10)
        eoy.place(x=250, y=140)

        # buttons
        button = Button(top, text="Add more",
                        command=lambda: self.add_more(epx, epy, etx, ety, eox, eoy)) # ew, eh, epx, epy, etx, ety, eox, eoy
        button.pack(pady=5, side=tkinter.TOP)
        button.place(x=350, y=110)

        button = Button(top, text="Finished", command=lambda: self.close_win(top))
        button.pack(pady=5, side=tkinter.TOP)
        button.place(x=350, y=140)

    def add_more(self, epx, epy, etx, ety, eox, eoy): # self, ew, eh, epx, epy, etx, ety, eox, eoy
        """
        Function for setting a new scenario size, as well as adding pedestrians, targets and obstacles.
        Error handling for numbers too small or too big for the scenario size.
        """
        # if (ew.get() != '') and (eh.get() != ''):
        #     self.scenario = Scenario(int(ew.get()), int(eh.get()))

        if (epx.get() != '') and (epy.get() != ''):
            try:
                if int(epx.get()) < 0 or int(epx.get()) > self.scenario.width or int(epy.get()) < 0 or int(
                        epy.get()) > self.scenario.height:
                    raise ValueError  # this will send it to the print message and back to the input option
                self.scenario.pedestrians.append(Pedestrian((int(epx.get()), int(epy.get())), 2.1))
            except ValueError:
                messagebox.showerror('Invalid integer',
                                     'The number must be in the range of 0 - {} x 0 - {}'.format(self.scenario.width,
                                                                                                 self.scenario.height))

        if etx.get() != '' and ety.get() != '':
            try:
                if int(etx.get()) < 0 or int(etx.get()) > self.scenario.width or int(ety.get()) < 0 or int(
                        ety.get()) > self.scenario.height:
                    raise ValueError  # this will send it to the print message and back to the input option
                self.scenario.grid[int(etx.get()), int(ety.get())] = Scenario.NAME2ID['TARGET']
            except ValueError:
                messagebox.showerror('Invalid integer',
                                     'The number must be in the range of 0 - {} x 0 - {}'.format(self.scenario.width,
                                                                                                 self.scenario.height))

        if eox.get() != '' and eoy.get() != '':
            try:
                if int(eox.get()) < 0 or int(eox.get()) > self.scenario.width or int(eoy.get()) < 0 or int(
                        eoy.get()) > self.scenario.height:
                    raise ValueError  # this will send it to the print message and back to the input option
                self.scenario.grid[int(eox.get()), int(eoy.get())] = Scenario.NAME2ID['OBSTACLE']
            except ValueError:
                messagebox.showerror('Invalid integer',
                                     'The number must be in the range of 0 - {} x 0 - {}'.format(self.scenario.width,
                                                                                                 self.scenario.height))

        #self.scenario.recompute_target_distances()
        self.scenario.to_image(self.canvas, self.canvas_image)
        self.clear_entries(epx, epy, etx, ety, eox, eoy) # ew, eh, epx, epy, etx, ety, eox, eoy

    def clear_entries(self, epx, epy, etx, ety, eox, eoy): # self, ew, eh, epx, epy, etx, ety, eox, eoy
        """
        Function clears the entries in the "Create Scenario" - User Interface
        """
        # ew.delete(0, tkinter.END)
        # eh.delete(0, tkinter.END)
        epx.delete(0, tkinter.END)
        epy.delete(0, tkinter.END)
        etx.delete(0, tkinter.END)
        ety.delete(0, tkinter.END)
        eox.delete(0, tkinter.END)
        eoy.delete(0, tkinter.END)

    def close_win(self, top):
        top.destroy()
