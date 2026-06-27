from functools import partial

from vlmeval.dataset import *


# Keep the active release image-replay surface from importing video dataset
# variants whose classes are absent from this cleaned fork. Image benchmarks
# still import this module through vlmeval.dataset.__init__, so missing video
# symbols must not make every image dataset fail to build.
mmbench_video_dataset = {
    'MMBench_Video_8frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=False),
    'MMBench_Video_8frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=True),
    'MMBench_Video_16frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=16, pack=False),
    'MMBench_Video_64frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=False),
    'MMBench_Video_64frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=True),
    'MMBench_Video_1fps_nopack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=False),
    'MMBench_Video_1fps_pack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=True),
}

mvbench_dataset = {
    'MVBench_8frame': partial(MVBench, dataset='MVBench', nframe=8),
    'MVBench_64frame': partial(MVBench, dataset='MVBench', nframe=64),
    'MVBench_MP4_8frame': partial(MVBench_MP4, dataset='MVBench_MP4', nframe=8),
    'MVBench_MP4_1fps': partial(MVBench_MP4, dataset='MVBench_MP4', fps=1.0),
}

videomme_dataset = {
    'Video-MME_8frame': partial(VideoMME, dataset='Video-MME', nframe=8),
    'Video-MME_64frame': partial(VideoMME, dataset='Video-MME', nframe=64),
    'Video-MME_8frame_subs': partial(VideoMME, dataset='Video-MME', nframe=8, use_subtitle=True),
    'Video-MME_1fps': partial(VideoMME, dataset='Video-MME', fps=1.0),
    'Video-MME_0.5fps': partial(VideoMME, dataset='Video-MME', fps=0.5),
    'Video-MME_0.5fps_subs': partial(VideoMME, dataset='Video-MME', fps=0.5, use_subtitle=True),
}

supported_video_datasets = {}
for grp in [mmbench_video_dataset, mvbench_dataset, videomme_dataset]:
    supported_video_datasets.update(grp)
