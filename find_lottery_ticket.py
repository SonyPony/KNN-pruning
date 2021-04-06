import os
import  multiprocessing
from shrinkbench.experiment import LotteryTicketExperiment
from shrinkbench.plot import plot_df, df_from_results

multiprocessing.freeze_support()

# setup environment
os.environ["DATAPATH"] = "../data"
os.environ["WEIGHTSPATH"] = "../pretrained"

# settings
STRATEGY = "GlobalMagWeight"
COMPRESSION = 4

if __name__ == "__main__":
    exp = LotteryTicketExperiment(
        dataset="MNIST",
        model="MnistNet",
        strategy=STRATEGY,
        compression=COMPRESSION,
        train_kwargs={"epochs": 10},
        run_on_device=False,
        pretrained=False
    )

    exp.run()

# plot results
df = df_from_results('../results')
plot_df(df, 'compression', 'pre_acc5', markers='strategy', line='--', colors='strategy', suffix=' - pre')
plot_df(df, 'compression', 'post_acc5', markers='strategy', fig=False, colors='strategy')