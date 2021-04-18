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
P_PERC = 1.#0.02
COMPRESSION = 1 / P_PERC

if __name__ == "__main__":
    exp = LotteryTicketExperiment(
        dataset="TinyImageNet",
        model="TinyImageNetVGG16",
        strategy=STRATEGY,
        compression=COMPRESSION,
        train_kwargs={"epochs": 27, "lr": 0.01},
        dl_kwargs={"split_ratio": 0.8, "batch_size": 64, "num_workers": 4},
        run_on_device=True,
        pretrained=False,
        logging=False,
        pruning_rate=0.15
    )

    exp.run()

# plot results
#df = df_from_results('../results')
#plot_df(df, 'compression', 'pre_acc5', markers='strategy', line='--', colors='strategy', suffix=' - pre')
#plot_df(df, 'compression', 'post_acc5', markers='strategy', fig=False, colors='strategy')