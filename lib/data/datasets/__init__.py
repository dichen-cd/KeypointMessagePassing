from __future__ import print_function, absolute_import

from .pose import MarsPose
from .pose import PoseTrackReID
from .pose import DukeMTMCVidReIDPose


__pose_datasets = {
    'marspose': MarsPose,
    'posetrackreid': PoseTrackReID,
    'dukemtmcvidreidpose': DukeMTMCVidReIDPose
}


def init_pose_dataset(name, **kwargs):
    """Initializes a video dataset with keypoints."""
    avai_datasets = list(__pose_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __pose_datasets[name](**kwargs)


def register_pose_dataset(name, dataset):
    """Registers a new video dataset with keypoints.
    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.
    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_pose_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.PoseDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.PoseDataManager(
            root='reid-data',
            sources=['new_dataset', 'ilidsvid']
        )
    """
    global __pose_datasets
    curr_datasets = list(__pose_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __pose_datasets[name] = dataset
