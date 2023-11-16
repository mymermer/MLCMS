import json
import os
from datetime import datetime

import os

def compare_files_across_folders(folders):
    # Store a list of files from each folder
    folder_files = {}
    for folder in folders:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        folder_files[folder] = sorted(files)

    # Check if all folders have the same list of files
    if len(set(tuple(files) for files in folder_files.values())) > 1:
        return "The list of files differs across folders."

    # Compare the files
    for file_name in folder_files[folders[0]]:
        contents = []
        for folder in folders:
            with open(os.path.join(folder, file_name), 'rb') as f:
                contents.append(f.read())
        
        if any(content != contents[0] for content in contents[1:]):
            return f"The file '{file_name}' differs across folders."

    return "All files in all folders are identical."


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
    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        base_name, ext = os.path.splitext(scenario_path)
        new_file_path = f"{base_name}_updated{timestamp}{ext}"
        with open(new_file_path, 'w') as json_file:
            json.dump(new_scenario, json_file, indent=4)
        print(f"Scenario successfully updated at {new_file_path}")
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")


if __name__ == '__main__':
    #comparing files with gui, console
    folders = ["C:/mlcms/MLCMS_Exercises/Exercise02/Task3/output/RimeaTest6_2023-11-08_16-40-39.498 guiver", "C:/mlcms/MLCMS_Exercises/Exercise02/Task3/output/RimeaTest6_2023-11-08_16-43-34.793 consolever"]
    result = compare_files_across_folders(folders)
    print(result)
    
    scenario_path = 'C:/mlcms/MLCMS_Exercises/Exercise02/Task3/scenarios/rimea6_mod.scenario'
    new_scenario = add_pedestrian(scenario_path, 12, 3)
    update_new_scenario(scenario_path, new_scenario)


###to checkout the new scenario
#run the new scenario in console first and open in gui 
# type below it in console.
# java -jar vadere-console.jar scenario-run  --output-dir youroutputfolder --scenario-file yourscenario file folder 
# java -jar vadere-gui.jar and then scenario tab, open

