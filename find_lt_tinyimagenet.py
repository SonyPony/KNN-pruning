import os
import argparse
import multiprocessing
from shrinkbench.experiment import LotteryTicketExperiment
from shrinkbench.plot import plot_df, df_from_results

multiprocessing.freeze_support()

parser = argparse.ArgumentParser(description='Lottery ticket on MNIST Net')
parser.add_argument('-d', '--data', type=str, help='path to training data folder', required=True)
parser.add_argument('-m', '--model', type=str, help='path to stored weights', required=True)
parser.add_argument('--epochs', type=int, help='number of epochs', default=5)
parser.add_argument('--p_rate', type=float, help='prune percentage at each prune iteration', default=0.15)
parser.add_argument('--p_perc', type=float, help='desired prune percentage', default=1.)
parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
parser.add_argument('--strategy', type=str, help='pruning strategy [GlobalMagWeight|LayerMagWeight]', default='GlobalMagWeight') 

args = parser.parse_args()

# setup environment
os.environ["DATAPATH"] = args.data
os.environ["WEIGHTSPATH"] = args.model

# settings
COMPRESSION = 1 / args.p_perc

if __name__ == "__main__":
    exp = LotteryTicketExperiment(
        dataset="TinyImageNet",
        model="TinyImageNetVGG16",
        strategy=STRATEGY,
        compression=COMPRESSION,
        train_kwargs={"epochs": 27, "lr": args.lr},
        dl_kwargs={"split_ratio": 0.8, "batch_size": 64, "num_workers": 4},
        run_on_device=True,
        pretrained=False,
        logging=False,
        pruning_rate=args.p_rate,
        lr_factor=0.2
    )

    exp.run()

# plot results
#df = df_from_results('../results')
#plot_df(df, 'compression', 'pre_acc5', markers='strategy', line='--', colors='strategy', suffix=' - pre')
#plot_df(df, 'compression', 'post_acc5', markers='strategy', fig=False, colors='strategy')