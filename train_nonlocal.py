import argparse,logging,os
import mxnet as mx
import symbol.symbol_nonlocal as nonlocal_resnet
from cfgs.config import cfg, read_cfg
import pprint


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.9):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def main():
    read_cfg(args.cfg)
    if args.gpus:
        cfg.gpus = args.gpus
    if args.model_path:
        cfg.model_path = args.model_path
    pprint.pprint(cfg)
    
    # get symbol
    symbol = nonlocal_resnet.get_symbol(cfg)
    
    kv = mx.kvstore.create(cfg.kv_store)
    devs = mx.cpu() if cfg.gpus is None else [mx.gpu(int(i)) for i in cfg.gpus.split(',')]
    epoch_size = max(int(cfg.dataset.num_examples / cfg.batch_size / kv.num_workers), 1)
    begin_epoch = cfg.model_load_epoch if cfg.model_load_epoch else 0
    if not os.path.exists(cfg.model_path):
        os.mkdir(cfg.model_path)
    model_prefix = cfg.model_path + "nonlocal_resnet-{}-{}-{}-{}".format(cfg.dataset.data_type, cfg.network.depth, kv.rank, 2)
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    arg_params = None
    aux_params = None
    
    if cfg.retrain:
        print("loading pretrained parameters...")
        _, arg_params, aux_params = mx.model.load_checkpoint('model/resnet-tiny-imagenet-50-0', 100)
    if cfg.memonger:
        import memonger
        symbol = memonger.search_plan(symbol, data=(cfg.batch_size, 3, 32, 32) if cfg.dataset.data_type=="cifar10"
                                                    else (cfg.batch_size, 3, 224, 224))
    ## data rec path
    if cfg.dataset.data_type == "cifar10":
        train_rec = os.path.join(cfg.dataset.data_dir, "cifar10_train.rec")
        val_rec = os.path.join(cfg.dataset.data_dir, "cifar10_val.rec")
    elif cfg.dataset.data_type == "cifar100":
        train_rec = os.path.join(cfg.dataset.data_dir, "cifar100_train.rec")
        val_rec = os.path.join(cfg.dataset.data_dir, "cifar100_test.rec")
    elif cfg.dataset.data_type == "tiny-imagenet":
        train_rec = os.path.join(cfg.dataset.data_dir, "tiny-imagenet-10_train.rec")
        val_rec = os.path.join(cfg.dataset.data_dir, "tiny-imagenet-10_val.rec")
    else:
        val_rec = os.path.join(cfg.dataset.data_dir, "val_256_q95.rec")
        if cfg.dataset.aug_level == 1:
            train_rec = os.path.join(cfg.dataset.data_dir, "train_256_q95.rec")
        else:
            train_rec = os.path.join(cfg.dataset.data_dir, "train_480_q95.rec")

    
    train = mx.io.ImageRecordIter(
        path_imgrec         = train_rec,
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 32, 32) if cfg.dataset.data_type in ["cifar10", "cifar100"] else (3, 224, 224),
        batch_size          = cfg.batch_size,
        pad                 = 4 if cfg.dataset.data_type in ["cifar10", "cifar100"] else 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale    = 1.0 if cfg.dataset.data_type in ["cifar10", "cifar100"]
                              else 1.0 if cfg.dataset.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0 if cfg.dataset.data_type in ["cifar10", "cifar100"]
                              else 0 if cfg.dataset.aug_level == 1 else 0.25,
        random_h            = 0 if cfg.dataset.data_type in ["cifar10", "cifar100"]
                              else 0 if cfg.dataset.aug_level == 1 else 36,  # 0.4*90
        random_s            = 0 if cfg.dataset.data_type in ["cifar10", "cifar100"]
                              else 0 if cfg.dataset.aug_level == 1 else 50,  # 0.4*127
        random_l            = 0 if cfg.dataset.data_type in ["cifar10", "cifar100"]
                              else 0 if cfg.dataset.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle    = 0 if cfg.dataset.aug_level <= 2 else 10,
        max_shear_ratio     = 0 if cfg.dataset.aug_level <= 2 else 0.1,
        rand_mirror         = True,
        shuffle             = True,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    
    val = mx.io.ImageRecordIter(
        path_imgrec         = val_rec,
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = cfg.batch_size,
        data_shape          = (3, 32, 32) if cfg.dataset.data_type in ["cifar10", "cifar100"] else (3, 224, 224),
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    
    model = mx.model.FeedForward(
        ctx                 = devs,
        symbol              = symbol,
        arg_params          = arg_params,
        aux_params          = aux_params,
        num_epoch           = 200 if cfg.dataset.data_type in ["cifar10", "cifar100"] else 120,
        begin_epoch         = begin_epoch,
        learning_rate       = cfg.train.lr,
        momentum            = cfg.train.mom,
        wd                  = cfg.train.wd,
        # optimizer           = 'nag',
        optimizer          = 'sgd',
        initializer         = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        lr_scheduler        = multi_factor_scheduler(begin_epoch, epoch_size, step=cfg.train.lr_steps, factor=0.1),
        )
    
    print("Score on validation dataset after loading", model.score(val))
    
    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = ['acc', 'ce'] if cfg.data_type in ["cifar10", "cifar100"] else
                             ['acc', mx.metric.create('top_k_accuracy', top_k = 2)],
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(cfg.batch_size, cfg.frequent),
        epoch_end_callback = checkpoint)
    
    logging.info("top-1 and top-5 acc is {}".format(model.score(X = val,
                  eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k = 5)])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model_path', help='the loc to save model checkpoints', default='', type=str)
    args = parser.parse_args()
    logging.info(args)
    main()
