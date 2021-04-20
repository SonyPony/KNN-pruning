import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.array([
    [1, 0.0947, 0.465, 0.974, 0.989],
    [0.85, 0.0947, 0.765, 0.975, 0.989],
    [0.72, 0.109, 0.934, 0.976, 0.989],
    [0.61, 0.196, 0.96, 0.976, 0.989],
    [0.52, 0.265, 0.979, 0.976, 0.989],
    [0.44, 0.425, 0.982, 0.976, 0.989],
    [0.38, 0.692, 0.983, 0.977, 0.989],
    [0.32, 0.627, 0.971, 0.976, 0.989],
    [0.27, 0.661, 0.968, 0.976, 0.989],
    [0.23, 0.613, 0.936, 0.976, 0.989],
    [0.20, 0.6, 0.968, 0.976, 0.989],
    [0.17, 0.787, 0.974, 0.975, 0.989],
    [0.12, 0.856, 0.985, 0.975, 0.989],
    [0.1, 0.81, 0.987, 0.975, 0.989],
    [0.09, 0.871, 0.984, 0.974, 0.989],
    [0.06, 0.914, 0.987, 0.973, 0.989],
    [0.03, 0.847, 0.984, 0.971, 0.989],
    [0.024, 0.761, 0.979, 0.968, 0.989],
])

df = pd.DataFrame(data, columns=("p%", "init_val_acc1", "init_val_acc5", "val_acc1", "val_acc5"))
indices = np.array(df.index)

plt.figure(figsize=(12,8))
plt.plot(df["p%"] * 100, df["init_val_acc1"])
plt.plot(df["p%"] * 100, df["init_val_acc5"])
plt.gca().invert_xaxis()
plt.grid(linestyle="--")
plt.xlabel("Percentage of Weights Remaining")
plt.ylabel("Accuracy after 0 Iterations")
plt.xscale("log")
plt.xticks(df["p%"] * 100, df["p%"] * 100)

plt.legend(("Top-1", "Top-5"))

plt.show()
plt.figure(figsize=(8,5))
plt.plot(df["p%"] * 100, df["val_acc1"])
plt.plot(df["p%"] * 100, df["val_acc5"])
plt.gca().invert_xaxis()
plt.grid(linestyle="--")
plt.xlabel("Percentage of Weights Remaining")
plt.ylabel("Accuracy after 10 Epochs")
plt.legend(("Top-1", "Top-5"))

plt.show()