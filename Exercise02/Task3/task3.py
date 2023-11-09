import json
import os
from datetime import datetime

# adding pedestrians in
def add_pedestrian(scenario_path, x, y):
    # open scenario as JSON
    with open(scenario_path, 'r') as json_file:
        scenario = json.load(json_file)

    # making template to add
    
    pedestrian = {
        'position': {
            'x': x,
            'y': y
        },
        'type': 'PEDESTRIAN',  
        'targetIds': []
    }

    # Ensure the dynamicElements list exists in the scenario JSON
    if "scenario" in scenario and \
       "topography" in scenario["scenario"] and \
       "dynamicElements" in scenario["scenario"]["topography"]:
        for target in scenario["scenario"]["topography"]["targets"]:
            pedestrian["targetIds"].append(target["id"])
        scenario["scenario"]["topography"]["dynamicElements"].append(pedestrian)
    else:
        # Handle the case where dynamicElements does not exist
        print('Error: dynamicElements list not found in the scenario.')

    return scenario

# make a new JSON file in the same path as the former scenario
def update_new_scenario(scenario_path, new_scenario):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    base_name, ext = os.path.splitext(scenario_path)
    new_file_path = f"{base_name}_updated{timestamp}{ext}"
    with open(new_file_path, 'w') as json_file:
        json.dump(new_scenario, json_file, indent=4)

if __name__ == '__main__':
    scenario_path = 'C:/Users/AhnNayeon/gitrepo/MLCMS_Exercises/Exercise02/task3/scenarios/RimeaTest6.scenario'
    new_scenario = add_pedestrian(scenario_path, 8.5, 2)
    update_new_scenario(scenario_path, new_scenario)
