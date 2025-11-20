#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIG ======
SRC_ROOT="/home/haiduong/Documents/Project/BookLibraryWebsiteCanvas/data/samples"
DEST_ROOT="/home/haiduong/Documents/Project/BookLibraryWebsiteCanvas/data/extracted_frames"  # change if needed
QUALITY=2   # ffmpeg quality for JPEG (2 = very high, 1 = highest but huge)
# ====================

mkdir -p "$DEST_ROOT"

for sample_dir in "$SRC_ROOT"/*; do
    [ -d "$sample_dir" ] || continue

    sample_name="$(basename "$sample_dir")"
    video_path="$sample_dir/drone_video.mp4"

    if [ ! -f "$video_path" ]; then
        echo "⚠️  No drone_video.mp4 in $sample_dir, skipping."
        continue
    fi

    out_dir="$DEST_ROOT/$sample_name"
    mkdir -p "$out_dir"

    echo "🎥 Extracting ALL frames from: $video_path"
    echo "   → $out_dir/${sample_name}_frame_XXXXXX.jpg"

    ffmpeg -hide_banner -loglevel error \
        -i "$video_path" \
        -qscale:v "$QUALITY" \
        "$out_dir/${sample_name}_frame_%06d.jpg"
done

echo "✅ DONE. Extracted raw frames at original FPS into:"
echo "   $DEST_ROOT/<SampleName>/"
