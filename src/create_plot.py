import os
from collections import defaultdict

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas
import numpy

plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})

legend = []
plots = []


def plot_csv_data(file_input, data_name):
    csv_input = pandas.read_csv(f"../data/{file_input}")
    data = defaultdict(list)
    order = defaultdict(int)
    legend.append(data_name)
    for model_name, success_rate in zip(csv_input["model_name"], csv_input["success_rate"]):
        # model_name = data['model_name']
        # success_rate = data['success_rate']
        split_model_name = model_name.split("_")
        corrective_index = split_model_name.index("corr") - 1
        evaluative_index = split_model_name.index("eval") - 1
        feedback_ratio = (
            f"Corrective: {split_model_name[corrective_index]}%, Evaluative: {split_model_name[evaluative_index]}%"
        )
        order[feedback_ratio] = int(split_model_name[corrective_index])
        data[feedback_ratio].append(success_rate)

    # order dictionary
    data = {key: data[key] for key, value in sorted(order.items(), key=lambda item: item[1])}
    order = {key: value for key, value in sorted(order.items(), key=lambda item: item[1])}

    data = pandas.DataFrame.from_dict(data, orient="index")
    std = data.std(axis=1)
    mean = data.mean(axis=1)

    # Make smooth lines for the plot
    ratio_array = numpy.array(list(order.values()))
    smooth_mean = make_interp_spline(ratio_array, mean.values)
    smooth_std = make_interp_spline(ratio_array, std.values)

    X_ = numpy.linspace(ratio_array.min(), ratio_array.max(), 500)
    Y_ = smooth_mean(X_)
    error = smooth_std(X_)
    plot = plt.plot(X_, Y_)
    error_band = plt.fill_between(X_, Y_ - error, Y_ + error, alpha=0.2)
    plots.append((plot[0], error_band))


# plot_title = 'Original CEILing'
# graph_titles = ['Original CEILing']
# csv_files = ['original_ceiling.csv']

plot_title = "CEILing with Penalized Actions"
graph_titles = ["Original CEILing", "CEILing with Penalized Actions"]
csv_files = ["original_ceiling.csv", "ceiling_penalized_actions.csv"]

for csv_file, graph_title in zip(csv_files, graph_titles):
    plot_csv_data(csv_file, graph_title)

plt.title(plot_title, fontsize=18)
plt.ylim(0, 1)
plt.ylabel("Success Rate", fontsize=12)
plt.xlabel("Corrective Feedback Ratio [\%]", fontsize=12)
plt.legend(plots, legend)

image_name = plot_title.replace(" ", "_").lower()
image_path = "../data/output/"
if not os.path.isdir(image_path):
    os.makedirs(image_path)
plt.savefig(os.path.join(image_path, f"{image_name}.pdf"))
plt.show()
