import argparse
import json
import os

parser = argparse.ArgumentParser(description="Create a JSON file with experiment name and initialized keys.")
parser.add_argument("--experiment", type=str, help="Name of the experiment for the JSON file", required=True)
parser.add_argument("--data_num", type=int, help="Number of data points for the JSON file keys", required=True)
args = parser.parse_args()

experiment_name = args.experiment
data_num = args.data_num

data = {i: 0 for i in range(data_num+1)}
os.makedirs("running_file", exist_ok=True)  # Save index file in running_file dir
filename = f"running_file/index_{experiment_name}.json"
if not os.path.exists(filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Created Index File: {filename}")
else:
    print(f"Index File Already Exists: {filename}; Continue Training ...")