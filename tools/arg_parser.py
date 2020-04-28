
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test an Object Detection network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_dets_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--rotate', dest='rotate',
                        help='how much should we rotate each image?',
                        default=0, type=int)
    parser.add_argument('--av_save', dest='av_save',
                        help="tells us to save the activity vectors",
                        action='store_true')
    # params for model to which active learning is applied
    parser.add_argument('--al_def', dest='al_def',
                        help='model prototxt to which active learning is applied',
                        default=None, type=str)
    parser.add_argument('--al_net', dest='al_net',
                        help='model weights to which active learning is applied',
                        default=None, type=str)
    parser.add_argument('--warp_affine_pretrain', dest='warp_affine_pretrain',
                        help='did we train the warp affine with a pretrained model?',
                        default=None, type=str)
    parser.add_argument('--name_override', dest='name_override',
                        help='overwrite the current model name with this string instead.',
                        default=None, type=str)
    parser.add_argument('--new_cache', dest='new_cache',
                        help="tells us to re-write the old cache",
                        action='store_true')
    parser.add_argument('--create_angle_dataset', dest='create_angle_dataset',
                        help="should we create an angle dataset numpy file?",
                        action='store_true')
    parser.add_argument('--siamese', dest='siamese',
                        help="is the testing model type a siamese model?",
                        action='store_true')
    parser.add_argument('--arch', dest='arch',type=str,
                        help="specify the model architecture")
    parser.add_argument('--optim', dest='optim',type=str,
                        help="specify the model optim alg")
    parser.add_argument('--export_cfg', dest='export_cfg',action='store_true',
                        help="export the config to file.")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
