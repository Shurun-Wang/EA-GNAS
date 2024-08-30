import argparse


def config():
    parser = argparse.ArgumentParser(description='[IOAGraph]')
    parser.add_argument("--dataset", type=str, default="SCH", help="The dataset.")
    parser.add_argument("--site", type=str, default="all", help="The site.")
    # settings for the intelligent optimizer_hubs algorithm
    parser.add_argument('--ioa', type=str, default="ga", help='the selected optimizer algorithm')
    parser.add_argument('--num_individuals', type=int, default=10, help='the population size')
    parser.add_argument('--num_iterations', type=int, default=100, help='the total iteration')
    # settings for the dataset
    parser.add_argument('--myCost', type=float, default=0.2, help='threshold for tree search')
    # settings for the gnn model
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='number of the GNN layers')
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--cuda", type=str, default="cpu", help="cuda:1")
    args = parser.parse_args()
    return args