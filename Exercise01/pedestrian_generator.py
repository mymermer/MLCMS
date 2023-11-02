import random
import json

# Define the corridor bounds
x_min = 0
x_max = 274
y_min = 16
y_max = 21

# Define the number of pedestrians
num_pedestrians = 1370

# Create a list to store pedestrian data
pedestrians = []

# Generate random coordinates and speeds for pedestrians
for i in range(num_pedestrians):
    x = int(random.uniform(x_min, x_max))
    y = int(random.uniform(y_min, y_max))
    speed = round(random.uniform(1.2, 1.4),2)
    pedestrian = {
        "position": [x,y],
        "desiredSpeed": speed
    }
    pedestrians.append(pedestrian)

# Save the data to a JSON file
with open("./Exercise01/scenarios/pedestrians_dont_use.json", "w") as json_file:
    json.dump(pedestrians, json_file)

print("Pedestrian data has been saved to pedestrians.json.")
