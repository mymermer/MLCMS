import tkinter as tk
import random

class GridApp:
    def __init__(self, master):
        self.master = master
        master.title("Exercise 1")

        self.label = tk.Label(master, text="Enter grid size:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(master)
        self.entry.pack(pady=10)

        self.button = tk.Button(master, text="Next", command=self.next_page)
        self.button.pack(pady=10)

        self.canvas = tk.Canvas(master)
        self.canvas.pack(pady=20, padx=20, expand=True)

        self.start_set = False
        self.end_set = False
        self.random_set = False
        self.start_coords = None
        self.end_coords = None

    def next_page(self):
        size = int(self.entry.get())
        self.cell_size = 30

        self.label.config(text="Manually pick starting and ending point or choose them randomly")
        self.entry.pack_forget()
        self.button.pack_forget()

        for i in range(size):
            for j in range(size):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", tags=f"cell_{i}_{j}")

        self.setpoint_button = tk.Button(self.master, text="Set Point", command=self.activate_set_point)
        self.setpoint_button.pack(pady=10)

        self.rand_button = tk.Button(self.master, text="Random", command=self.set_random_points)
        self.rand_button.pack(pady=10)

        self.move_button = tk.Button(self.master, text="Move Towards End", command=self.move_towards_end, state=tk.DISABLED)
        self.move_button.pack(pady=10)

    def activate_set_point(self):
        if not self.random_set:
            self.canvas.bind("<Button-1>", self.set_point)

    def set_point(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        self.paint_cell(row, col, "green" if not self.start_set else "red")

        if not self.start_set:
            self.start_coords = (row, col)
            self.start_set = True
        elif not self.end_set:
            self.end_coords = (row, col)
            self.end_set = True
            self.move_button.config(state=tk.NORMAL)
            self.label.config(text="Let's go")

    def set_random_points(self):
        if self.start_set or self.end_set:
            return

        size = int(self.entry.get())

        start_row, start_col = random.randint(0, size-1), random.randint(0, size-1)
        end_row, end_col = random.randint(0, size-1), random.randint(0, size-1)

        while (start_row, start_col) == (end_row, end_col):
            end_row, end_col = random.randint(0, size-1), random.randint(0, size-1)

        self.paint_cell(start_row, start_col, "green")
        self.paint_cell(end_row, end_col, "red")

        self.start_coords = (start_row, start_col)
        self.end_coords = (end_row, end_col)
        self.start_set, self.end_set = True, True
        self.random_set = True

        self.move_button.config(state=tk.NORMAL)
        self.label.config(text="Let's go")

    def paint_cell(self, row, col, color):
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags=f"cell_{row}_{col}")

    def move_towards_end(self):
        size = int(self.entry.get())
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]

        closest_cell = None
        closest_distance = float('inf')

        for direction in directions:
            new_row, new_col = self.start_coords[0] + direction[0], self.start_coords[1] + direction[1]

            if 0 <= new_row < size and 0 <= new_col < size:
                distance = ((self.end_coords[0] - new_row)**2 + (self.end_coords[1] - new_col)**2)**0.5

                if distance < closest_distance:
                    closest_distance = distance
                    closest_cell = (new_row, new_col)

        if closest_cell:
            self.paint_cell(*self.start_coords, "white")
            self.paint_cell(*closest_cell, "green")
            self.start_coords = closest_cell

        if self.start_coords == self.end_coords:
            self.show_complete_window()

    def show_complete_window(self):
        self.complete_window = tk.Toplevel(self.master)
        self.complete_window.title("Mission Complete")
        complete_label = tk.Label(self.complete_window, text="Mission Complete!")
        complete_label.pack(pady=20)
        reset_button = tk.Button(self.complete_window, text="Reset", command=self.reset)
        reset_button.pack(pady=10)
        quit_button = tk.Button(self.complete_window, text="Quit", command=self.quit_program)
        quit_button.pack(pady=10)

    def reset(self):
        self.canvas.delete("all")
        self.start_set = False
        self.end_set = False
        self.random_set = False
        self.start_coords = None
        self.end_coords = None
        self.label.config(text="Enter grid size:")
        self.entry.pack(pady=10)
        self.button.pack(pady=10)
        self.setpoint_button.pack_forget()
        self.rand_button.pack_forget()
        self.move_button.pack_forget()
        self.complete_window.destroy()

    def quit_program(self):
        self.complete_window.destroy()
        self.master.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = GridApp(root)
    root.mainloop()
