"""
This tools runs various Hawkes Process Simulations
"""

# python imports
import numpy as np
import matplotlib.pyplot as plt
import pprint

# project imports
import _init_paths
from base.sim_hp_arg_parser import parse_args

# project imports Hawkes Process Algorithms
# import ibhp # indian buffet hawkes process
import hp20 # proposed algorithm
import hp18 # shelton hawkes process
import hp   # ordinary hawkes process

def main():
    # parse arguments
    pp = pprint.PrettyPrinter(indent=4)

    # --------------
    # Load settings
    # --------------
    pp.print("Running with arguments")
    args = parse_args()
    cfg = load_mnist_cfg()
    if args is None: args = parse_args()
    set_cfg_from_args(cfg,args)
    if args.cfg_file: cfg = read_cfg_file(args.cfg_file) # overwrite with cfg file.
    device = cfg.device
    cfg.use_cuda = args.no_cuda and torch.cuda.is_available()
    save_cfg(cfg)


    # select hawkes process inference algorithm
    hp_sim = None
    if args.hp_type == 'ibhp':
        hp_sim = ibhp
    elif args.hp_type == 'hp20':
        hp_sim = hp20
    elif args.hp_type == 'hp18':
        hp_sim = hp18
    elif args.hp_type == 'hp':
        hp_sim = hp
    else:
        raise ValueError(f"Uknown hp_type [{args.hp_type}]")

    # load data
    data = load_data(args)

    # inference
    if args.sim_type == 'prior':
        samples = hp_sim.prior()
    elif args.sim_type == 'posterior':
        samples = hp_sim.posterior(data)
    else:
        raise ValueError(f"Uknown sim_type [{args.sim_type}]")

if __name__ == "__main__":
    main()
        


