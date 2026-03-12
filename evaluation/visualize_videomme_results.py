#!/usr/bin/env python3
"""Generate visualization GIFs/frames from Video-MME evaluation metrics JSONs."""

import os
import json
import argparse
import shutil
from visualization import VideoQAGIFGenerator


def build_question_text(entry):
    """Build a human-readable 'question' line from Video-MME metadata."""
    qid = entry.get("question_id", "")
    cat = entry.get("category", "")
    sub_cat = entry.get("sub_category", "")
    task_cat = entry.get("task_category", "")
    parts = [p for p in [qid, cat, sub_cat, task_cat] if p]
    if not parts:
        return qid or "Video-MME question"
    return " / ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Visualize Video-MME results as GIFs or frames")
    parser.add_argument(
        "--metrics_json",
        type=str,
        required=True,
        help="Path to Video-MME metrics JSON (e.g., evaluation/metrics_*.json)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/scratch/bai.xiang/eval_benchmarks/Video-MME/data",
        help="Directory containing Video-MME video files (video_id.mp4 under here)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_videomme_vis",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="frames",
        choices=["frames", "gif"],
        help="Output format: 'frames' saves key PNGs (+video), 'gif' saves animated GIF",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Max number of samples to visualize",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated indices of samples to visualize (overrides max_samples)",
    )
    parser.add_argument("--font_size", type=int, default=14)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--target_height", type=int, default=360)
    parser.add_argument("--gif_fps", type=int, default=12)
    args = parser.parse_args()

    with open(args.metrics_json, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        indices = list(range(min(args.max_samples, len(data))))

    generator = VideoQAGIFGenerator(font_size=args.font_size)

    for idx in indices:
        if idx >= len(data):
            print(f"Index {idx} out of range (dataset has {len(data)} samples), skipping.")
            continue

        entry = data[idx]
        video_id = entry.get("video_id") or entry.get("videoID")
        if not video_id:
            print(f"[{idx}] No video_id field found, skipping.")
            continue

        # Reasoning + answer text for the panel
        reasoning = entry.get("reasoning_process", "")
        if not reasoning:
            print(f"[{idx}] No reasoning_process, skipping.")
            continue

        pred = entry.get("pred_answer", entry.get("response", ""))
        gt = entry.get("answer", "")
        answer_text = f"Pred: {pred} | GT: {gt}"

        question = build_question_text(entry)

        # Resolve video path similarly to dataloader.videomme.videomme_doc_to_visual
        base = os.path.join(args.video_dir, f"{video_id}.mp4")
        candidates = [
            base,
            base.replace(".mp4", ".MP4"),
            base.replace(".mp4", ".mkv"),
        ]
        video_path = None
        for cand in candidates:
            if os.path.exists(cand):
                video_path = cand
                break

        if not video_path:
            print(f"[{idx}] Video not found for video_id={video_id} under {args.video_dir}, skipping.")
            continue

        safe_vid = str(video_id).replace("/", "_")

        print(f"\n[{idx}] Generating {args.format} for video_id={video_id}")
        print(f"  Question: {question}")
        print(f"  Answer:   {answer_text}")

        try:
            if args.format == "frames":
                frame_dir = os.path.join(args.output_dir, f"{idx:04d}_{safe_vid}")
                generator.create_keyframes(
                    video_path=video_path,
                    question=question,
                    reasoning=reasoning,
                    answer=answer_text,
                    output_dir=frame_dir,
                    target_size=(args.target_width, args.target_height),
                    prefix="frame",
                )
                # Save raw entry + copy video
                with open(os.path.join(frame_dir, "entry.json"), "w") as jf:
                    json.dump(entry, jf, indent=2, ensure_ascii=False)
                dst_video = os.path.join(frame_dir, os.path.basename(video_path))
                if not os.path.exists(dst_video):
                    shutil.copy2(video_path, dst_video)
            else:
                out_path = os.path.join(args.output_dir, f"{idx:04d}_{safe_vid}.gif")
                generator.create_demo_gif(
                    video_path=video_path,
                    question=question,
                    reasoning=reasoning,
                    answer=answer_text,
                    output_path=out_path,
                    target_size=(args.target_width, args.target_height),
                    gif_fps=args.gif_fps,
                    frames_per_word=3,
                )
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone! Visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

