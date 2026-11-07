import json

with open("experiments/results/evaluation_raw.json") as f:
    data = json.load(f)

for inst in data["instances"]:
    if inst["n_customers"] == 10:
        print(f"Name: {inst['name']}, R: {inst['n_repairmen']}, B_sol: {inst['B_sol']}, Ref: {inst['reference']}")
        print(f"Routes: {inst['routes']}")
        print("-" * 50)
    elif inst["name"] in ["20.7.txt_2", "20.15.txt_2", "20.24.txt_3"]:
        print(f"Name: {inst['name']}, R: {inst['n_repairmen']}, B_sol: {inst['B_sol']}, Ref: {inst['reference']}")
        print(f"Routes: {inst['routes']}")
        print("-" * 50)
