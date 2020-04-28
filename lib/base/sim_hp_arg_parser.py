import sys
import argparse

def parse_args():
    """
    Parse input arguments
    """
    desc = "Simulator for Hawkes Process algorithms."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--hp_type', dest='hp_type',type=str,
                        help='Hawkes Process Simulation Type', default=None)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--export_cfg', dest='export_cfg',action='store_true',
                        help="export the config to file.")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
