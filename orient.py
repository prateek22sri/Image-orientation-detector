# /usr/bin/env python
import sys
import csv
import itertools
import pprint
import random
from Model import NNet
from Model import Nearest, AdaBoost

train_file = "train_file.txt"
test_file = "test_file.txt"
method = ""
parameter = None
model_file = "best.model"
metadata_file = "metadata.csv"
try:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    method = sys.argv[3]
except IndexError:
    print("Usage: python orient.py train_file test_file algorithm [model-parameter]")
    sys.exit(1)
if method in ["adaboost", "nnet"]:
    try:
        parameter = int(sys.argv[4])
    except IndexError:
        print("Usage: python orient.py train_file test_file algorithm model-parameter")
        print("model-parameter (stump_count for adaboost and hidden_count for nnet) is required")
        sys.exit(2)
if method in ["best"]:
    try:
	method="nearest"
	parameter=1
        model_file = sys.argv[4]
    except IndexError:
        print("We are running the best model. This will take around half hour to run")
header = ["id", "orientation"]
color_indices = [color + item for item in
                 [str(i[0]) + str(i[1]) for i in itertools.product([1, 2, 3, 4, 5, 6, 7, 8], repeat=2)] for color in
                 ['r', 'g', 'b']]
indices = header + color_indices
select_indices = [color + item for item in
                  [str(i[0]) + str(i[1]) for i in itertools.product([1, 2, 3, 4, 5, 6, 7, 8], repeat=2) if
                   i[0] == 1 or i[1] == 1 or i[0] == 8 or i[1] == 8] for color in ['r', 'g']]
# for color in ['r', 'g', 'b']:
#     for corner in 1, 8:
#         for other in [1, 2, 3, 4, 5, 6, 7, 8]:
#             select_indices.append(color + str(corner) + str(other))


train_rows = []
train_rows_net = []
test_rows_net = []
test_rows = []
csv.register_dialect(
    'space_dialect',
    delimiter=' ',
)
with open(train_file, "r") as train_file_handler:
    reader = csv.reader(train_file_handler, dialect="space_dialect")
    for row in reader:
        current_row = []
        for column in row:
            try:
                current_row.append(int(column))
            except ValueError:
                current_row.append(column)
        current_dict = dict(zip(indices, current_row))
        train_rows.append(current_dict)
        train_rows_net.append(current_row)

with open(test_file, "r") as test_file_handler:
    reader = csv.reader(test_file_handler, dialect="space_dialect")
    for row in reader:
        current_row = []
        for column in row:
            try:
                current_row.append(int(column))
            except ValueError:
                current_row.append(column)
        current_dict = dict(zip(indices, current_row))
        test_rows.append(current_dict)
        test_rows_net.append(current_row)
print("Data set ready")
model = None
if method == "nearest":
    model = Nearest(color_indices, int(parameter))
    for train_item in train_rows:
        model.train(train_item)
elif method == "nnet":
    model = NNet(parameter, len(train_rows_net[0]) - 2)
    model.train(train_rows_net)
elif method == "adaboost":
    model = AdaBoost(color_indices, int(parameter))
    model.load("test.model")
    model.train(train_rows)
    # model.save("adaboost_model.model")

successes = 0
totals = 0
print("Training complete")
confusion_matrix = {"0": {"0": 0, "90": 0, "180": 0, "270": 0}, "90": {"0": 0, "90": 0, "180": 0, "270": 0},
                    "180": {"0": 0, "90": 0, "180": 0, "270": 0}, "270": {"0": 0, "90": 0, "180": 0, "270": 0}}

row_counter = 0
totals = len(test_rows)
correct_ids = []
incorrect_ids = []
for test_index, test_item in enumerate(test_rows):
    id = None
    orientation = None
    if method == "nnet":
        id, orientation = model.test(test_rows_net[test_index])
        if str(orientation) == str(test_rows_net[test_index][1]):
            successes += 1
            correct_ids.append((id, orientation))
        else:
            incorrect_ids.append((id, orientation))
        confusion_matrix[str(test_rows_net[test_index][1])][str(orientation)] += 1
    else:
        id, orientation = model.test(test_item)
        if str(orientation) == str(test_item["orientation"]):
            if len(correct_ids) < 5:
                correct_ids.append(id)
            successes += 1
        else:
            if len(incorrect_ids) < 5:
                incorrect_ids.append((id, orientation))
        confusion_matrix[str(test_item["orientation"])][str(orientation)] += 1
# pprint.pprint(correct_ids)
# pprint.pprint(incorrect_ids)
print("Confusion Matrix")
print("\t0\t90\t180\t270\t")
# for key in confusion_matrix.iterkeys():
print("0\t" + str(confusion_matrix["0"]["0"]) + "\t" + str(confusion_matrix["0"]["90"]) + "\t" + str(
    confusion_matrix["0"]["180"]) + "\t" + str(confusion_matrix["0"]["270"]))
print("90\t" + str(confusion_matrix["90"]["0"]) + "\t" + str(confusion_matrix["90"]["90"]) + "\t" + str(
    confusion_matrix["90"]["180"]) + "\t" + str(confusion_matrix["90"]["270"]))
print("180\t" + str(confusion_matrix["180"]["0"]) + "\t" + str(confusion_matrix["180"]["90"]) + "\t" + str(
    confusion_matrix["180"]["180"]) + "\t" + str(confusion_matrix["180"]["270"]))
print("270\t" + str(confusion_matrix["270"]["0"]) + "\t" + str(confusion_matrix["270"]["90"]) + "\t" + str(
    confusion_matrix["270"]["180"]) + "\t" + str(confusion_matrix["270"]["270"]))
# print(successes)
# print(totals)
print(1.0 * successes / totals)
