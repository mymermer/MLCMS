import sys
import tkinter as tk
from tkinter import Button, Canvas, Menu, messagebox
from scenario_elements import Scenario, Pedestrian
import threading
import json


class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def create_scenario(self, ):
        print('create not implemented yet')

    def restart_scenario(self, ):
        print('restart not implemented yet')

    def step_scenario(self, scenario, canvas, canvas_image):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        scenario.update_step()
        scenario.to_image(canvas, canvas_image)

    def exit_gui(self, ):
        """
        Close the GUI.
        """
        sys.exit()

    def start_gui(self, ):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.
        Only one button works at the moment: "step simulation".
        Also creates a rudimentary, fixed Scenario instance with three Pedestrian instances and multiple targets.
        """
        win = tk.Tk()
        win.geometry('500x500')  # setting the size of the window
        win.title('Cellular Automata GUI')

        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=self.create_scenario)
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)

        canvas = Canvas(win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])  # creating the canvas
        canvas_image = canvas.create_image(5, 50, image=None, anchor=tk.NW)
        canvas.pack()


        # sc = Scenario(5, 5)

        # sc.grid[3, 2] = Scenario.NAME2ID['TARGET']
 
        # sc.pedestrians = [
        #     Pedestrian((1, 2), 2.3),

        # ]





        sc = Scenario(100, 100)

        sc.grid[23, 25] = Scenario.NAME2ID['TARGET']
        sc.grid[23, 45] = Scenario.NAME2ID['TARGET']
        sc.grid[43, 55] = Scenario.NAME2ID['TARGET']


        #dummy obstacle numbers

        

        sc.grid [23,24] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [23,26] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [24,24] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [24,25] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [24,26] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [22,24] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [22,25] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [22,26] = Scenario.NAME2ID["OBSTACLE"]



        sc.grid [11,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [12,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [13,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [14,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [15,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [16,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [17,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [18,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [19,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [20,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [21,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [22,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [23,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [24,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [25,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [26,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [27,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [28,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [29,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [30,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [31,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [32,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [33,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [35,1] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [34,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [35,2] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [35,3] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [35,4] = Scenario.NAME2ID["OBSTACLE"]
        sc.grid [35,0] = Scenario.NAME2ID["OBSTACLE"]


        sc.pedestrians = [
            Pedestrian((31, 2), 2.3),
            Pedestrian((1, 10), 2.1),
            Pedestrian((80, 70), 2.1),
        ]

        # can be used to show pedestrians and targets
        sc.to_image(canvas, canvas_image)

        # can be used to show the target grid instead
        # sc.target_grid_to_image(canvas, canvas_image)
        sc.update_cost()#new function



        btn = Button(win, text='Step simulation', command=lambda: self.step_scenario(sc, canvas, canvas_image))
        btn.place(x=20, y=10)
        # btn = Button(win, text='Start simulation', command=lambda: self.simulate_loop(sc, canvas, canvas_image))
        # btn.place(x=130, y=10)
        btn = Button(win, text='Restart simulation', command=self.restart_scenario)
        btn.place(x=200, y=10)
        btn = Button(win, text='Create simulation', command=lambda: popupwin(win, sc, canvas, canvas_image))
        btn.place(x=380, y=10)
        ##
        btn = Button(win, text='Load scenario', command=lambda: self.load_scenario(sc, canvas, canvas_image))
        btn.place(x=200, y=40)
        btn = Button(win, text='Save scenario', command=lambda: self.save_scenario(sc, canvas, canvas_image))
        btn.place(x=380, y=40)

        # mouse.on_click(lambda: print(mouse.get_position()[0] - win.winfo_rootx(), mouse.get_position()[1] - win.winfo_rooty()))

        win.mainloop()

    ################################################################

    def load_scenario(self, sc, canvas, canvas_image):
        with open('scenario.json', 'r') as f:
            data = json.load(f)

            clear_grid(sc)
            sc.pedestrians.clear()

            for pedestrian in data["pedestrians"]:
                sc.pedestrians.append(
                    Pedestrian(pedestrian["position"], pedestrian["speed"]))

            for target in data["targets"]:
                sc.grid[target["x"], target["y"]] = Scenario.NAME2ID['TARGET']

            for obstacle in data["obstacles"]:
                sc.grid[obstacle["x"], obstacle["y"]] = Scenario.NAME2ID['OBSTACLE']

            sc.recompute_target_distances()
            sc.to_image(canvas, canvas_image)

    def save_scenario(self, sc, canvas, canvas_image):
        data = {
            "pedestrians": [],
            "targets": [],
            "obstacles": []
        }
        for pedestrian in sc.pedestrians:
            data["pedestrians"].append({
                'position': pedestrian.position,
                'speed': pedestrian.desired_speed
            })
        create_targets_obstacles_lists(sc, data)
        with open('person.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)


def create_targets_obstacles_lists(sc, data):
    for x in range(sc.width):
        for y in range(sc.height):
            if sc.grid[x, y] == 1:
                data["targets"].append({
                    'x': x,
                    'y': y
                })
            elif sc.grid[x, y] == 2:
                data["obstacles"].append({
                    'x': x,
                    'y': y
                })


def clear_grid(sc):
    for x in range(sc.width):
        for y in range(sc.height):
            sc.grid[x, y] = Scenario.NAME2ID['EMPTY']


# Define a function to open the Popup Dialogue
def popupwin(win, sc, canvas, canvas_image):
    """
    Creates and shows a simple user interface with multiple labels, entries and buttons.
    The user can set a new scenario size, as well as add pedestrians, targets and obstacles.
    """

    # Create a Toplevel window
    top = tk.Toplevel(win)
    top.geometry("750x250")

    # labels
    lbl = tk.Label(top, text="Set dimensions")
    lbl.place(x=40, y=30)
    lbl = tk.Label(top, text="Add pedestrian")
    lbl.place(x=40, y=80)
    lbl = tk.Label(top, text="Add target")
    lbl.place(x=40, y=110)
    lbl = tk.Label(top, text="Add obstacle")
    lbl.place(x=40, y=140)

    lbl = tk.Label(top, text="x coordinates")
    lbl.place(x=170, y=60)
    lbl = tk.Label(top, text="y coordinates")
    lbl.place(x=250, y=60)

    # entry fields
    ew = tk.Entry(top, width=10)
    ew.place(x=180, y=30)
    eh = tk.Entry(top, width=10)
    eh.place(x=250, y=30)

    epx = tk.Entry(top, width=10)
    epx.place(x=180, y=80)
    epy = tk.Entry(top, width=10)
    epy.place(x=250, y=80)

    etx = tk.Entry(top, width=10)
    etx.place(x=180, y=110)
    ety = tk.Entry(top, width=10)
    ety.place(x=250, y=110)

    eox = tk.Entry(top, width=10)
    eox.place(x=180, y=140)
    eoy = tk.Entry(top, width=10)
    eoy.place(x=250, y=140)

    # buttons
    button = Button(top, text="Add more",
                    command=lambda: add_more(sc, canvas, canvas_image, ew, eh, epx, epy, etx, ety, eox, eoy))
    button.pack(pady=5, side=tk.TOP)
    button.place(x=320, y=110)

    button = Button(top, text="Finished", command=lambda: close_win(top))
    button.pack(pady=5, side=tk.TOP)
    button.place(x=320, y=140)


def add_more(sc, canvas, canvas_image, ew, eh, epx, epy, etx, ety, eox, eoy):
    """
    Function for setting a new scenario size, as well as adding pedestrians, targets and obstacles.
    Error handling for numbers too small or too big for the scenario size.
    """
    if (ew.get() != '') and (eh.get() != ''):
        sc.__init__(int(ew.get()), int(eh.get()))

    if (epx.get() != '') and (epy.get() != ''):
        try:
            if int(epx.get()) < 0 or int(epx.get()) > sc.width or int(epy.get()) < 0 or int(epy.get()) > sc.height:
                raise ValueError  # this will send it to the print message and back to the input option
            sc.pedestrians.append(Pedestrian((int(epx.get()), int(epy.get())), 2.1))
        except ValueError:
            messagebox.showerror('Invalid integer',
                                 'The number must be in the range of 0 - {} x 0 - {}'.format(sc.width, sc.height))

    if etx.get() != '' and ety.get() != '':
        try:
            if int(etx.get()) < 0 or int(etx.get()) > sc.width or int(ety.get()) < 0 or int(ety.get()) > sc.height:
                raise ValueError  # this will send it to the print message and back to the input option
            sc.grid[int(etx.get()), int(ety.get())] = Scenario.NAME2ID['TARGET']
        except ValueError:
            messagebox.showerror('Invalid integer',
                                 'The number must be in the range of 0 - {} x 0 - {}'.format(sc.width, sc.height))

    if eox.get() != '' and eoy.get() != '':
        try:
            if int(eox.get()) < 0 or int(eox.get()) > sc.width or int(eoy.get()) < 0 or int(eoy.get()) > sc.height:
                raise ValueError  # this will send it to the print message and back to the input option
            sc.grid[int(eox.get()), int(eoy.get())] = Scenario.NAME2ID['OBSTACLE']
        except ValueError:
            messagebox.showerror('Invalid integer',
                                 'The number must be in the range of 0 - {} x 0 - {}'.format(sc.width, sc.height))

    sc.recompute_target_distances()
    sc.to_image(canvas, canvas_image)
    clear_entries(ew, eh, epx, epy, etx, ety, eox, eoy)


def clear_entries(ew, eh, epx, epy, etx, ety, eox, eoy):
    """
    Function clears the entries in the "Create Scenario" - User Interface
    """
    ew.delete(0, tk.END)
    eh.delete(0, tk.END)
    epx.delete(0, tk.END)
    epy.delete(0, tk.END)
    etx.delete(0, tk.END)
    ety.delete(0, tk.END)
    eox.delete(0, tk.END)
    eoy.delete(0, tk.END)


def close_win(top):
    top.destroy()
