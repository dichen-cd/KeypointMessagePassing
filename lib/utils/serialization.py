import os
import json
import errno
import os.path as osp
import numpy as np
from contextlib import redirect_stdout


def save_cfg_to_file(cfg, file_name):
    with open(file_name, 'w') as f:
        with redirect_stdout(f):
            print(cfg.dump())


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    _obj = obj.copy()
    for k, v in _obj.items():
        if isinstance(v, np.ndarray):
            _obj.pop(k)
    with open(fpath, 'w') as f:
        json.dump(_obj, f, indent=4, separators=(',', ': '))
