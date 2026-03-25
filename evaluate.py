#!/usr/bin/env python
import os, torch, math, argparse
from pathlib import Path
from tqdm import tqdm
from frame_utils import DaliVideoDataset, AVVideoDataset, TensorVideoDataset, camera_size, seq_len
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

def main():
  parser = argparse.ArgumentParser(description="Evaluate a comma2k19 compression submission.")
  parser.add_argument("--batch-size", type=int, default=16, help="dataloader batch size")
  parser.add_argument("--num-threads", type=int, default=2, help="DALI worker threads")
  parser.add_argument("--prefetch-queue-depth", type=int, default=4, help="DALI prefetch depth")
  parser.add_argument("--submission-dir", type=Path, default=Path('./submissions/baseline/'), help="compressed videos path")
  parser.add_argument("--uncompressed-dir", type=Path, default=Path('./videos/'), help="original uncompressed videos path")
  parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
  parser.add_argument("--device", type=str, default=None, help="device: 'cpu', 'cuda', or 'mps' (default: auto-detect)")
  parser.add_argument("--report", type=Path, default=Path("./report.txt"), help="output report file path")
  parser.add_argument("--video-names-file", type=Path, default=Path("./public_test_video_names.txt"), help="text file with test video names (one per line)")
  args = parser.parse_args()

  if args.device is not None:
    device = torch.device(args.device)
  elif torch.cuda.is_available():
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")

  # distributed (cuda only)
  if device.type == "cuda":
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    if device.index is None:
      device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    DefaultDatasetClass = DaliVideoDataset
  else:
    local_rank, rank, world_size, is_distributed = 0, 0, 1, False
    DefaultDatasetClass = AVVideoDataset

  if rank == 0:
    printed_args = ["=== Evaluation config ==="]
    printed_args.extend([f"  {k}: {vars(args)[k]}" for k in sorted(vars(args))])
    print("\n".join(printed_args))

  if is_distributed and not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl", device_id=local_rank)

  distortion_net = DistortionNet().eval().to(device=device)
  distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

  with open(args.video_names_file, "r") as file:
    test_video_names = [line.strip() for line in file.readlines()]

  ds_gt = DefaultDatasetClass(test_video_names, data_dir=args.uncompressed_dir, batch_size=args.batch_size, device=device, num_threads=args.num_threads, seed=args.seed, prefetch_queue_depth=args.prefetch_queue_depth)
  ds_gt.prepare_data()
  dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)

  if rank == 0:
    compressed_size = (args.submission_dir / 'archive.zip').stat().st_size
    uncompressed_size = sum(file.stat().st_size for file in args.uncompressed_dir.rglob('*') if file.is_file())
    rate = compressed_size / uncompressed_size

  ds_comp = TensorVideoDataset(test_video_names, data_dir=args.submission_dir / 'inflated', batch_size=args.batch_size, device=device, num_threads=args.num_threads, seed=args.seed, prefetch_queue_depth=args.prefetch_queue_depth)
  ds_comp.prepare_data()
  dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

  dl = zip(dl_gt, dl_comp)
  posenet_dists, segnet_dists, batch_sizes = torch.zeros([], device=device), torch.zeros([], device=device), torch.zeros([], device=device)
  with torch.inference_mode():
    for (_,_,batch_gt), (_,_,batch_comp) in tqdm(dl):
      batch_gt = batch_gt.to(device)
      batch_comp = batch_comp.to(device)
      assert list(batch_comp.shape)[1:] == [seq_len, camera_size[1], camera_size[0], 3], f"unexpected batch shape: {batch_comp.shape}"
      assert batch_gt.shape == batch_comp.shape, f"ground truth and compressed batch shape mismatch: {batch_gt.shape} vs {batch_comp.shape}"
      posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
      assert posenet_dist.shape == (batch_gt.shape[0],) and segnet_dist.shape == (batch_gt.shape[0],), f"unexpected distortion shapes: {posenet_dist.shape}, {segnet_dist.shape}"
      posenet_dists += posenet_dist.sum()
      segnet_dists += segnet_dist.sum()
      batch_sizes += batch_gt.shape[0]
    if is_distributed and torch.distributed.is_initialized():
      torch.distributed.all_reduce(posenet_dists, op=torch.distributed.ReduceOp.SUM)
      torch.distributed.all_reduce(segnet_dists, op=torch.distributed.ReduceOp.SUM)
      torch.distributed.all_reduce(batch_sizes, op=torch.distributed.ReduceOp.SUM)

    if rank == 0:
      posenet_dist = (posenet_dists / batch_sizes).item()
      segnet_dist = (segnet_dists / batch_sizes).item()
      score = 100 * segnet_dist +  math.sqrt(posenet_dist * 10)  + 25 * rate
      printed_results = [
        f"=== Evaluation results over {batch_sizes:.0f} samples ===",
        f"  Average PoseNet Distortion: {posenet_dist:.8f}",
        f"  Average SegNet Distortion: {segnet_dist:.8f}",
        f"  Submission file size: {compressed_size:,} bytes",
        f"  Original uncompressed size: {uncompressed_size:,} bytes",
        f"  Compression Rate: {rate:.8f}",
        f"  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = {score:.2f}"
      ]
      print("\n".join(printed_results))
      with open(args.report, "w") as f:
        f.write("\n".join(printed_args + printed_results) + "\n")

  # Cleanup
  if is_distributed and torch.distributed.is_initialized():
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
  main()
