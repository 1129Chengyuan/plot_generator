import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from datetime import datetime
import os

# Load Control Input Speed data
def load_control_input_speed(file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv('Source/control_input_speed.csv')
        df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names
        df['Command Issued (Timestamp)'] = pd.to_datetime(df['Command Issued (Timestamp)'])
        df['Control Action Executed (Timestamp)'] = pd.to_datetime(df['Control Action Executed (Timestamp)'])
    else:
        with open('Source/control_input_speed.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['events'])
        df['command_issued'] = pd.to_datetime(df['command_issued'])
        df['control_executed'] = pd.to_datetime(df['control_executed'])
    return df

# Load Task Completion Time data
def load_task_completion_time(file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv('Source/task_completion_time.csv')
        df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
    else:
        with open('Source/task_completion_time.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['tasks'])
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
    return df

# Load Correct Activation of Equipment data
def load_equipment_activation(file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv('Source/equipment_activation.csv')
        df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names
        df['Time of Activation'] = pd.to_datetime(df['Time of Activation'])
    else:
        with open('Source/equipment_activation.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['equipment_activation'])
        df['activation_time'] = pd.to_datetime(df['activation_time'])
    return df

# Load Number of Procedural Errors data
def load_procedural_errors(file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv('Source/procedural_errors.csv')
        df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names
    else:
        with open('Source/procedural_errors.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['errors'])
    return df

# Load Accuracy in Monitoring System Parameters data
def load_monitoring_accuracy(file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv('Source/monitoring_accuracy.csv')
    else:
        with open('Source/monitoring_accuracy.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['monitoring_accuracy'])
    return df

# Create visualizations
def create_visualizations(control_speed_df, task_time_df, equipment_df, errors_df, accuracy_df):
    # 1. Histogram: Control Input Speed
    fig_control_speed = px.histogram(control_speed_df, x='control_input_speed',
                                 title='Control Input Speed Distribution')

    # 2. Bar Chart: Task Completion Time
    fig_task_time = px.bar(task_time_df, x='task', y='task_completion_time',
                       labels={'task_completion_time': 'Task Completion Time (Seconds)'},
                       title='Task Completion Time')
    
    # 3. Pie Chart: Correct Activation of Equipment
    equipment_df['activation_status'] = equipment_df['activation_status'].map({0: 'Incorrect', 1: 'Correct'})
    fig_equipment = px.pie(equipment_df, names='activation_status', title='Equipment Activation Status')

    # 4. Bar Chart: Number of Procedural Errors
    fig_errors = px.bar(errors_df, x='task', y='errors', title='Number of Procedural Errors per Task')

    # 5. Heatmap: Accuracy in Monitoring System Parameters
    fig_accuracy = px.imshow(accuracy_df.pivot(index='parameter', columns='expected_value', values='accuracy'),
                             title='Accuracy in Monitoring System Parameters',
                             labels=dict(color="Accuracy (%)"))

    return [fig_control_speed, fig_task_time, fig_equipment, fig_errors, fig_accuracy]

# Function to save plots as HTML
def save_plots_as_html(figures, filenames):
    parent_folder = 'OutputFile'
    base_name = 'Attempt'
    existing_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f)) and f.startswith(base_name)]
    folder_numbers = [int(f[len(base_name):]) for f in existing_folders if f[len(base_name):].isdigit()]
    next_folder_number = max(folder_numbers, default=0) + 1
    new_folder_name = f'{base_name}{next_folder_number}'
    new_folder_path = os.path.join(parent_folder, new_folder_name)
    os.makedirs(new_folder_path)
    print(f"New folder created: {new_folder_name} in {parent_folder}")
    for fig, filename in zip(figures, filenames):
        file_path = os.path.join(new_folder_path, f"{filename}.html")
        fig.write_html(file_path)
        print(f"Successfully saved {filename}.html in {new_folder_name}")

def generate_and_save_visualizations(file_type='json'):
    # Load data
    control_speed_df = load_control_input_speed(file_type)
    task_time_df = load_task_completion_time(file_type)
    equipment_df = load_equipment_activation(file_type)
    errors_df = load_procedural_errors(file_type)
    accuracy_df = load_monitoring_accuracy(file_type)
    
    # Create visualizations
    figures = create_visualizations(control_speed_df, task_time_df, equipment_df, errors_df, accuracy_df)
    
    # Save plots as HTML
    filenames = ['control_speed', 'task_time', 'equipment_activation', 'procedural_errors', 'monitoring_accuracy']
    save_plots_as_html(figures, filenames)
    
    print("Visualizations have been generated and saved as HTML files.")

# Example usage
# if __name__ == "__main__":
#     generate_and_save_visualizations('json')