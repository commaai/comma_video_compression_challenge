import os
import subprocess
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
HERE = os.path.dirname(__file__)

in_video = os.path.join(ROOT, "videos", "0.mkv")
archive_dir = os.path.join(HERE, "archive")
out_video = os.path.join(archive_dir, "0.mkv")
archive_zip = os.path.join(HERE, "archive.zip")

os.system(f"rm -rf {archive_dir}")
os.makedirs(archive_dir, exist_ok=True)

cmd = [
    "ffmpeg", "-nostdin", "-y",
    "-hide_banner", "-loglevel", "warning",
    "-r", "20",
    "-fflags", "+genpts",
    "-i", in_video,
    "-vf", "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos",
    "-c:v", "libx265",
    "-preset", "ultrafast",
    "-crf", "29",
    "-g", "60",
    "-bf", "0",
    "-x265-params", "keyint=60:min-keyint=60:scenecut=0:frame-threads=4:log-level=warning",
    "-r", "20",
    out_video,
]

subprocess.run(cmd, check=True)

with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(out_video, "0.mkv")

print("saved", archive_zip)