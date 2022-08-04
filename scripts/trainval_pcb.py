import os.path as osp
import socket
from datetime import datetime
import huepy as hue

import torch
import torchreid
from torchreid.utils import (set_random_seed, load_pretrained_weights,
                             check_isfile, resume_from_checkpoint)

import sys
sys.path.append('./')
from configs import (parse_from_cli, videodata_kwargs, optimizer_kwargs,
                     lr_scheduler_kwargs, engine_run_kwargs)

from lib.data import VideoDataManager
from lib.engine import VideoSoftmaxEngine
from lib.utils.serialization import (mkdir_if_missing, save_cfg_to_file)


def main(cfg):
    # Prepration
    test_only = cfg.test.evaluate
    if not test_only:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        cfg.data.save_dir = osp.join(
            cfg.data.save_dir, current_time + '_' + socket.gethostname())
        mkdir_if_missing(cfg.data.save_dir)
    set_random_seed(cfg.train.seed)
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
    print(hue.info(hue.bold(hue.lightgreen(
        'Working directory: {}'.format(cfg.data.save_dir)))))

    # Data
    if 'posetrackreidvo' in cfg.data.sources:
        print(hue.info('Registering new datasets...'))
        from torchreid.data.datasets import register_video_dataset
        from lib.data.datasets.video import PoseTrackReIDVO
        register_video_dataset('posetrackreidvo', PoseTrackReIDVO)
    if cfg.data.type == 'image':
        raise NotImplementedError  # patch it up later
    elif cfg.data.type == 'video':
        data_kwargs = videodata_kwargs(cfg)
        DataManagerCLS = VideoDataManager
        Engine = VideoSoftmaxEngine

    datamanager = DataManagerCLS(**data_kwargs)

    # Model
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
    if cfg.use_gpu:
        model = model.cuda()

    if not test_only:
        # Optimizer
        optim_kwargs = optimizer_kwargs(cfg)
        optimizer = torchreid.optim.build_optimizer(
            model,
            **optim_kwargs
        )
        # lr_scheduler
        schdl_kwargs = lr_scheduler_kwargs(cfg)
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            **schdl_kwargs
        )
    else:
        optimizer, scheduler = None, None

    # Resume
    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )
    if not test_only:
        save_cfg_to_file(cfg, osp.join(cfg.data.save_dir, 'cfg.yaml'))

    # Runner
    engine = Engine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=cfg.loss.softmax.label_smooth,
        pooling_method=cfg.video.pooling_method,
    )
    run_kwargs = engine_run_kwargs(cfg)
    engine.run(**run_kwargs)


if __name__ == '__main__':
    main(parse_from_cli())
