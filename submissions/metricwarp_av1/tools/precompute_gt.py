#!/usr/bin/env python
"""Run metric nets on gt.raw, cache pose outputs and segnet argmax maps."""
import time
import numpy as np
from pathlib import Path
from metric_lib import load_net, raw_pairs, net_outputs, setup_threads

HERE = Path(__file__).resolve().parent

def main():
    setup_threads()
    net = load_net()
    poses, segs = [], []
    t0 = time.time()
    n = 0
    for chunk, batch in raw_pairs(HERE / 'gt.raw', batch_size=8):
        pose, seg = net_outputs(net, batch)
        poses.append(pose)
        segs.append(seg)
        n += len(chunk)
        if n % 80 == 0:
            el = time.time() - t0
            print(f"{n} pairs, {el:.1f}s ({el/n:.2f}s/pair)", flush=True)
    poses = np.concatenate(poses)
    segs = np.concatenate(segs)
    np.save(HERE / 'gt_pose.npy', poses)
    np.save(HERE / 'gt_seg.npy', segs)
    print(f"done: pose {poses.shape}, seg {segs.shape}, {time.time()-t0:.1f}s total")

if __name__ == '__main__':
    main()
