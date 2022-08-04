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
from configs import (parse_from_cli, posedata_kwargs, optimizer_kwargs,
                     lr_scheduler_kwargs, engine_run_kwargs)

from lib.models import build_model
from lib.data import PoseDataManager
from lib.engine import PoseWVideoSoftmaxEngine
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
    assert cfg.data.type == 'pose'
    data_kwargs = posedata_kwargs(cfg)
    DataManagerCLS = PoseDataManager
    Engine = PoseWVideoSoftmaxEngine

    datamanager = DataManagerCLS(**data_kwargs)

    # Model
    model = build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=False,  # cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
        num_keypoints=12 if cfg.data.headless else 17,
        input_dim=6 if cfg.data.include_dynamics else 2,
        xv_only=cfg.model.xv_only,
        xg_only=cfg.model.xg_only,
        pcb_pretrained_path=cfg.model.pcb_pretrained_path,
        pool_modes=cfg.model.gcn_pool_modes
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
