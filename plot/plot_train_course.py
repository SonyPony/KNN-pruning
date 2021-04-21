import numpy as np
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from json import loads
from operator import itemgetter
from matplotlib.lines import Line2D


with open("../../result/pruning_training_course_table_elated_water_37.json", "r") as f:
    data = loads(f.read())

suffix_to_remove = "train"
suffix_to_keep = list({"val", "train"} - {suffix_to_remove})[0]
data = data.get("data")

# filter data with specific suffix
data = list(filter(lambda x: not (suffix_to_remove in x[1]), data))

# format data
steps = set()
best_acc = 0.
best_p_m = None
formatted_data = dict()


for data_point in data:
    steps.add(data_point[0])

for data_point in data:
    key = data_point[1].replace(" - {}".format(suffix_to_keep), "").replace("p% = ", "")

    if formatted_data.get(key, None) is None:
        formatted_data[key] = set()

    if len(formatted_data.get(key)) < len(steps):
        if best_acc < data_point[2]:
            best_acc = data_point[2]
            best_p_m = float(key)
        formatted_data[key].add((data_point[0], data_point[2]))

steps = sorted(list(steps))

# sort by iteration step
for key, samples in formatted_data.items():
    formatted_data[key] = sorted(list(samples), key=lambda x: x[0])
    formatted_data[key] = list(map(itemgetter(1), formatted_data[key]))

# plot
cmap = cm.get_cmap("plasma_r")
fig, (main_plt, colorbar) = plt.subplots(1, 2, figsize=(8,4), gridspec_kw={'width_ratios': [20, 1]})
pruning_level_list = list(map(float, formatted_data.keys()))
min_pruning_level = min(pruning_level_list)
max_pruning_level = max(pruning_level_list)

main_plt.grid(linestyle="--")
main_plt.set_xlim((min(steps), max(steps)))
main_plt.set_xlabel("Iteration")
main_plt.set_ylabel("Validation Top-1 Accuracy")

# hightlight some values
base_model_line_color = (0, 0, 0)
top_acc_line_color = (0.21960784, 0.78039216, 0.63137255)

custom_lines = [Line2D([0], [0], color=top_acc_line_color, lw=2, linestyle="-."),
                Line2D([0], [0], color=base_model_line_color, lw=2, linestyle="-.")]

main_plt.axhline(y=max(formatted_data.get("1.00")), xmin=0, xmax=max(steps), linewidth=1, linestyle="-.", color=base_model_line_color)
main_plt.axhline(y=best_acc, xmin=0, xmax=max(steps), linewidth=1, linestyle="-.", color=top_acc_line_color)

# main plot
for pruning_level, accs in formatted_data.items():
    pruning_level = float(pruning_level)

    cmap_level = (pruning_level) / max_pruning_level
    main_plt.plot(steps, accs, color=cmap(cmap_level))

main_plt.legend(custom_lines, ["$P_m$ = {:.1f}%".format(best_p_m * 100), "$P_m$ = 100%"])

# colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=max_pruning_level)
fig.subplots_adjust(right=0.5)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=colorbar, orientation='vertical', label='$P_m$[%], min $P_m$ = {:.1f}%'.format(min_pruning_level * 100), ticks=[i / 5 for i in range(6)])
cbar.ax.set_yticklabels(["{}".format(int((i / 5) * 100)) for i in range(6)])
fig.tight_layout()
plt.show()