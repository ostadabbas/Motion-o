#!/usr/bin/env python3
"""Generate visualization GIFs from V-STaR evaluation result JSONs."""
import os
import re
import json
import argparse
from visualization import VideoQAGIFGenerator


def extract_think_answer(raw_output):
    thinks = re.findall(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    think = "\n".join(t.strip() for t in thinks) if thinks else ""
    m = re.search(r"<answer>(.*?)</answer>", raw_output, re.DOTALL)
    answer = m.group(1).strip() if m else ""
    return think, answer


def main():
    parser = argparse.ArgumentParser(description="Visualize V-STaR results as GIFs")
    parser.add_argument("--result_json", type=str, required=True, help="Path to V-STaR result JSON")
    parser.add_argument("--video_dir", type=str, default="/scratch/bai.xiang/eval_benchmarks/V-STaR/videos",
                        help="Directory containing V-STaR video files")
    parser.add_argument("--output_dir", type=str, default="./output_gif", help="Output directory for GIFs")
    parser.add_argument("--task", type=str, default="vqa",
                        choices=["vqa", "temporal", "spatial", "temporal2", "spatial2"],
                        help="Which model output to visualize")
    parser.add_argument("--max_samples", type=int, default=5, help="Max number of GIFs to generate")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices of samples to visualize (overrides max_samples)")
    parser.add_argument("--format", type=str, default="frames", choices=["frames", "gif"],
                        help="Output format: 'frames' saves key PNGs, 'gif' saves animated GIF")
    parser.add_argument("--font_size", type=int, default=14)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--target_height", type=int, default=360)
    parser.add_argument("--gif_fps", type=int, default=12)
    args = parser.parse_args()

    with open(args.result_json, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    field_map = {
        "vqa": "answer_vqa_raw_output",
        "temporal": "answer_temporal_pre",
        "spatial": "answer_spatial_pre",
        "temporal2": "answer_temporal_pre_2",
        "spatial2": "answer_spatial_pre_2",
    }
    output_field = field_map[args.task]

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
        vid = entry["vid"]
        raw_output = entry.get(output_field, "")
        if not raw_output:
            print(f"[{idx}] No output for task '{args.task}', skipping.")
            continue

        think, answer = extract_think_answer(raw_output)
        if not think and not answer:
            print(f"[{idx}] Could not parse think/answer, skipping.")
            continue

        question_key = {
            "vqa": "question",
            "temporal": "temporal_question",
            "spatial": "spatial_question",
            "temporal2": "temporal_question",
            "spatial2": "spatial_question_2",
        }[args.task]
        question = entry.get(question_key, entry.get("question", ""))

        video_path = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            parts = vid.split("&")
            for part in parts:
                alt = os.path.join(args.video_dir, f"{part}.mp4")
                if os.path.exists(alt):
                    video_path = alt
                    break

        if not os.path.exists(video_path):
            print(f"[{idx}] Video not found: {video_path}, skipping.")
            continue

        safe_vid = vid.replace("/", "_").replace("&", "_")

        print(f"\n[{idx}] Generating {args.format} for vid={vid}, task={args.task}")
        print(f"  Question: {question[:80]}...")
        print(f"  Answer: {answer[:80]}...")

        try:
            if args.format == "frames":
                frame_dir = os.path.join(args.output_dir, f"{args.task}_{idx}_{safe_vid}")
                generator.create_keyframes(
                    video_path=video_path,
                    question=question,
                    reasoning=think,
                    answer=answer,
                    output_dir=frame_dir,
                    target_size=(args.target_width, args.target_height),
                    prefix="frame",
                )
                with open(os.path.join(frame_dir, "entry.json"), "w") as jf:
                    json.dump(entry, jf, indent=2, ensure_ascii=False)
            else:
                out_path = os.path.join(args.output_dir, f"{args.task}_{idx}_{safe_vid}.gif")
                generator.create_demo_gif(
                    video_path=video_path,
                    question=question,
                    reasoning=think,
                    answer=answer,
                    output_path=out_path,
                    target_size=(args.target_width, args.target_height),
                    gif_fps=args.gif_fps,
                    frames_per_word=3,
                )
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone! GIFs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
