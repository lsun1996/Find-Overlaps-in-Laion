import json

file_to_check = "/teamspace/studios/this_studio/data/final/sun397-test/duplicate_categories/only_correct.json"
file_to_check_1 = "/teamspace/studios/this_studio/data/final/sun397-test/duplicate_categories/only_correct-1.json"

with open(file_to_check, "r") as f:
    data = json.load(f)
    print("1.", len(data))

with open(file_to_check_1, "r") as f:
    data = json.load(f)
    print("2.", len(data))