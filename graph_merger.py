from jinja2 import Environment, FileSystemLoader
import os
from typing import List

def merge_graphs(graph_paths: List[str], template_path: str, output_path: str):
    # Read the contents of each graph HTML file with UTF-8 encoding
    graphs = []
    for graph_path in graph_paths:
        with open(graph_path, 'r', encoding='utf-8') as file:
            graphs.append(file.read())

    # Set up Jinja2 environment and load the template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_path)

    # Render the template with the graphs
    output = template.render(graphs=graphs)

    # Save the rendered HTML to a file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(output)

    print("Report has been generated and saved as report.html.")

# # List of HTML graph files
# graph_paths = [
#     "control_speed.html",
#     "equipment_activation.html",
#     "monitoring_accuracy.html",
#     "procedural_errors.html",
#     "task_time.html"
# ]
# template_path = "report_template.html"
# output_path = "report.html"

# # Call the function to merge graphs into the final report
# merge_graphs(graph_paths, template_path, output_path)
