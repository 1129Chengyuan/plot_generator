import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load Control Input Speed data
def load_control_input_speed(file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv('Source/control_input_speed.csv')
        df.columns = df.columns.str.strip()
        df['Command Issued (Timestamp)'] = pd.to_datetime(df['Command Issued (Timestamp)'])
        df['Control Action Executed (Timestamp)'] = pd.to_datetime(df['Control Action Executed (Timestamp)'])
    else:
        with open('Source/control_input_speed.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['events'])
        df['command_issued'] = pd.to_datetime(df['command_issued'], format='%Y-%m-%dT%H:%M:%SZ')
        df['control_executed'] = pd.to_datetime(df['control_executed'], format='%Y-%m-%dT%H:%M:%SZ')
    return df

# Load Task Completion Time data
def load_task_completion_time(file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv('Source/task_completion_time.csv')
        df.columns = df.columns.str.strip()
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
        df.columns = df.columns.str.strip()
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
        df.columns = df.columns.str.strip()
    else:
        with open('Source/procedural_errors.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['errors'])
    return df

# Load Accuracy in Monitoring System Parameters data
def load_monitoring_accuracy(file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv('Source/monitoring_accuracy.csv')
        df.columns = df.columns.str.strip()
    else:
        with open('Source/monitoring_accuracy.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['monitoring_accuracy'])
    return df

# Create visualizations
def create_visualizations(control_speed_df, task_time_df, equipment_df, errors_df, accuracy_df):
    # 1. Box Plot and Animation: Task Completion Time
    fig_task_time = go.Figure()
    for task in task_time_df['task'].unique():
        task_data = task_time_df[task_time_df['task'] == task]
        fig_task_time.add_trace(go.Box(y=task_data['task_completion_time'], name=task))
    fig_task_time.update_layout(title='Task Completion Time Distribution', yaxis_title='Time (Seconds)')
    
    # Animation frames
    frames = [go.Frame(data=[go.Box(y=task_time_df[task_time_df['task'] == task]['task_completion_time'], name=task)]) 
              for task in task_time_df['task'].unique()]
    fig_task_time.frames = frames
    fig_task_time.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                                  buttons=[dict(label='Play',
                                                                method='animate',
                                                                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])])

    # 2. Pie Chart: Number of Procedural Errors
    # Calculate total errors and correct procedures
    total_errors = errors_df['errors'].sum()
    total_tasks = len(errors_df)

    # Calculate the number of correct procedures (tasks with zero errors)
    correct_procedures = (errors_df['errors'] == 0).sum()

    # Prepare the pie chart
    fig_errors = px.pie(
        names=['Correct Procedures', 'Errors'], 
        values=[correct_procedures, total_errors], 
        title='Proportion of Correct Procedures vs Errors',
        color_discrete_sequence=['#F44336', '#4CAF50']  # Green for correct, Red for errors
    )

    # Add annotation to show exact numbers
    fig_errors.add_annotation(
        text=f"Total Tasks: {total_tasks}<br>Correct: {correct_procedures}<br>Errors: {total_errors}",
        xref="paper", yref="paper",
        x=0.95, y=0.95, showarrow=False
    )
    # 3. Heat Map: Correct Activation of Equipment
    equipment_pivot = equipment_df.pivot_table(values='activation_status', 
                                               index='equipment', 
                                               columns='activation_time', 
                                               aggfunc='first')
    fig_equipment = px.imshow(equipment_pivot, 
                              title='Equipment Activation Status Over Time',
                              labels=dict(x='Time', y='Equipment', color='Activation Status'))

    # 4. Box Plot and Animation: Command Execution Latency
    control_speed_df['latency'] = (control_speed_df['control_executed'] - control_speed_df['command_issued']).dt.total_seconds()
    fig_latency = go.Figure()
    for event in control_speed_df['event'].unique():
        event_data = control_speed_df[control_speed_df['event'] == event]
        fig_latency.add_trace(go.Box(y=event_data['latency'], name=event))
    fig_latency.update_layout(title='Command Execution Latency', yaxis_title='Latency (Seconds)')
    
    # Animation frames for latency
    latency_frames = [go.Frame(data=[go.Box(y=control_speed_df[control_speed_df['event'] == event]['latency'], name=event)]) 
                      for event in control_speed_df['event'].unique()]
    fig_latency.frames = latency_frames
    fig_latency.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                                buttons=[dict(label='Play',
                                                              method='animate',
                                                              args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])])

    # 5. Control Input Speed with KNN Classification
    if len(control_speed_df) >= 3 and control_speed_df['control_input_speed'].nunique() >= 3:
        scaler = StandardScaler()
        X = scaler.fit_transform(control_speed_df[['control_input_speed']])
        knn = KNeighborsClassifier(n_neighbors=min(3, len(control_speed_df)))
        knn.fit(X, np.zeros(len(X)))  # Fit the model (labels don't matter for unsupervised)
        control_speed_df['speed_category'] = knn.predict(X)
        control_speed_df['speed_category'] = control_speed_df['speed_category'].map({0: 'Slow', 1: 'Medium', 2: 'Fast'})
    else:
        # Simple classification if we don't have enough data or variation for KNN
        speed_median = control_speed_df['control_input_speed'].median()
        if control_speed_df['control_input_speed'].nunique() == 1:
            control_speed_df['speed_category'] = 'Medium'
        else:
            control_speed_df['speed_category'] = np.where(control_speed_df['control_input_speed'] < speed_median, 'Slow',
                                                          np.where(control_speed_df['control_input_speed'] > speed_median, 'Fast', 'Medium'))
    
    # Create scatter plot with Plotly
    fig_control_speed = px.scatter(control_speed_df, x='command_issued', y='control_input_speed', 
                                   color='speed_category', title='Control Input Speed Classification')
    

    return [fig_task_time, fig_errors, fig_equipment, fig_latency, fig_control_speed]

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
    filenames = ['task_time', 'procedural_errors', 'equipment_activation', 'command_execution_latency', 'control_input_speed']
    save_plots_as_html(figures, filenames)
    
    print("Visualizations have been generated and saved as HTML files.")

# Example usage
if __name__ == "__main__":
    generate_and_save_visualizations('json')