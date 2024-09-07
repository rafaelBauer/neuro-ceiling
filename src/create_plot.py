from collections import defaultdict

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas
import numpy

# ceiling_no_noise_penalized
graph_title = 'CEILing with Penalized Actions'
csv_file = 'ceiling_penalized_actions.csv'

csv_data = pandas.read_csv(f'../data/{csv_file}')

data = defaultdict(list)
order = defaultdict(int)
for model_name, success_rate in zip(csv_data["model_name"], csv_data["success_rate"]):
    # model_name = data['model_name']
    # success_rate = data['success_rate']
    split_model_name = model_name.split("_")
    corrective_index = split_model_name.index("corr") - 1
    evaluative_index = split_model_name.index("eval") - 1
    feedback_ratio = f'Corrective: {split_model_name[corrective_index]}%, Evaluative: {split_model_name[evaluative_index]}%'
    order[feedback_ratio] = int(split_model_name[corrective_index])
    data[feedback_ratio].append(success_rate)

# order dictionary
data = {key: data[key] for key, value in sorted(order.items(), key=lambda item: item[1])}

data = pandas.DataFrame.from_dict(data, orient='index')
std = data.std(axis=1)
mean = data.mean(axis=1)

# Make smooth lines for the plot
ratio_array = numpy.array(list(sorted(order.values())))
smooth_mean = make_interp_spline(ratio_array, mean.values)
smooth_std = make_interp_spline(ratio_array, std.values)

X_ = numpy.linspace(ratio_array.min(), ratio_array.max(), 500)
Y_ = smooth_mean(X_)
error = smooth_std(X_)

plt.plot(X_, Y_)
plt.fill_between(X_, Y_ - error, Y_ + error, alpha=0.2)
plt.title(graph_title, fontsize=18)
plt.ylabel('Success Rate', fontsize=14)
plt.xlabel('Corrective Feedback Ratio', fontsize=14)
plt.ylim(0, 1)

plt.show()
