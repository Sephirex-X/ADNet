import os

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='ADNet ICCV 2023')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--work_dirs', type=str, default='work_dirs',
        help='work dirs')
    parser.add_argument(
        '--load_from', default=None,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--finetune_from', default=None,
        help='whether to finetune from the checkpoint, will not remember epochs')
    parser.add_argument(
        '--view', action='store_true',
        help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int,
                        default=3407, help='random seed')
    args = parser.parse_args()

    return args

def main(args):
    from adnet.engine.runner import Runner
    from adnet.utils.config import Config
    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed
    cfg.validate = args.validate
    cfg.work_dirs = os.path.join(args.work_dirs, str(
        args.config).split('/')[1], cfg.dataset.train.type)
    runner = Runner(cfg)

    if args.validate:
        runner.validate()
    else:
        runner.train()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)
    main(args)
