import os
import graph_generator
import graph_merger

def access_latest_output_folder():
    # Step 1: Define the parent folder and the base name of the folders
    parent_folder = 'OutputFile'
    base_name = 'Attempt'
    
    # Step 2: List all directories inside the parent folder that match the base name
    existing_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f)) and f.startswith(base_name)]
    
    if not existing_folders:
        print("No folders found.")
        return

    # Step 3: Sort folders based on their numeric suffix and get the most recent folder
    folder_numbers = [int(f[len(base_name):]) for f in existing_folders if f[len(base_name):].isdigit()]
    most_recent_folder_number = max(folder_numbers)
    most_recent_folder_name = f'{base_name}{most_recent_folder_number}'
    most_recent_folder_path = os.path.join(parent_folder, most_recent_folder_name)

    print(f"Accessing most recent folder: {most_recent_folder_name}")
    return most_recent_folder_path

def access_latest_output_files():
    graph_paths = [
        "control_speed.html",
        "equipment_activation.html",
        "monitoring_accuracy.html",
        "procedural_errors.html",
        "task_time.html"
    ]
    list_of_files = []
    # Step 5: Access each file inside the most recent folder
    for graph_file in graph_paths:
        file_path = os.path.join(access_latest_output_folder(), graph_file)
        if os.path.exists(file_path):
            print(f"Found: {file_path}")
            list_of_files.append(file_path)
            # You can open or process the file as needed, e.g.:
            # with open(file_path, 'r') as f:
            #     content = f.read()
        else:
            print(f"File not found: {file_path}")
    return list_of_files

if __name__ == "__main__":
    graph_generator.generate_and_save_visualizations()
    template_path = "report_template.html"
    output_path = "report.html"
    graph_merger.merge_graphs(access_latest_output_files(), template_path, os.path.join(access_latest_output_folder(), output_path))