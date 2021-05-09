import os
import argparse
import multiprocessing
from shrinkbench.experiment import LotteryTicketExperiment
from shrinkbench.plot import plot_df, df_from_results

multiprocessing.freeze_support()

parser = argparse.ArgumentParser(description='Lottery ticket on MNIST Net')
parser.add_argument('-d', '--data', type=str, help='path to training data folder', required=True)
parser.add_argument('-m', '--model', help='path to stored weights', required=True)
parser.add_argument('--epochs', help='number of epochs', default=5)
parser.add_argument('--p_rate', help='prune percentage at each prune iteration', default=0.15)
parser.add_argument('--p_perc', help='desired prune percentage', default=0.7)
parser.add_argument('--strategy', help='pruning strategy [GlobalMagWeight|LayerMagWeight]', default='GlobalMagWeight') 

args = parser.parse_args()

# setup environment
os.environ["DATAPATH"] = args.data
os.environ["WEIGHTSPATH"] = args.model

# settings
COMPRESSION = 1 / args.p_perc

if __name__ == "__main__":
    exp = LotteryTicketExperiment(
        dataset="MNIST",
        model="MnistNet",
        strategy=args.strategy,
        compression=COMPRESSION,
        train_kwargs={"epochs": args.epochs,},
        dl_kwargs={"split_ratio": 0.8, "num_workers": 4},
        run_on_device=True,
        pretrained=False,
        logging=False,
        pruning_rate=args.p_rate
    )

    exp.run()

# plot results
#df = df_from_results('../results')
#plot_df(df, 'compression', 'pre_acc5', markers='strategy', line='--', colors='strategy', suffix=' - pre')
#plot_df(df, 'compression', 'post_acc5', markers='strategy', fig=False, colors='strategy')