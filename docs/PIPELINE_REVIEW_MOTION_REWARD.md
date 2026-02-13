# MotionR1 GRPO Pipeline Review — Motion Trajectory Reward

**Scope:** Full path trace (RL sample → prompt/completion → motion_trajectory_reward → direction/speed → reward), edge cases, reward aggregation, and integration. No code changes; analysis only.

---

## 1. Full path trace (one RL sample)

**1.1 Data → trainer**

- **Data:** `STGR-RL-subset.json` → `Dataset.from_json` → `make_conversation_image_and_video` adds `prompt`; other columns (e.g. `key_items`, `key_frames`, `answer`, `task`, `image_size`, `source`, `video_path`) are passed through.
- **Batch:** `compute_loss` receives `inputs` (list of samples; typically batch size 1 with `per_device_train_batch_size=1`). Only `inputs[0]` is used for video path and vision processing.
- **Vision:** `process_vision_info` loads video; on success, `inputs[0]['image_size'] = (video_inputs[0].size(3), video_inputs[0].size(2))` (W, H). On exception, this assignment is skipped (see **Potential issue 1**).
- **Reward kwargs:** `reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}` then for each key, `reward_kwargs[key].extend([example[key]] * num_generations)` for each example. So for 1 sample and 2 generations, `key_items`, `key_frames`, `image_size`, etc. are each a list of 2 identical copies. Index `idx` in the reward loop corresponds to completion index and thus to the right sample.

**1.2 Completions**

- Completions are decoded and wrapped as `[[{"role": "assistant", "content": "<text>"}], ...]`. So `completion[0]["content"]` in `motion_trajectory_reward` is the correct string.

**1.3 motion_trajectory_reward**

- **Task filter:** Non–temporal-spatial / non–General video QA tasks get 0.0 and skip; only `"temporal-spatial free-form QA"`, `"General video QA Free-form"`, `"General video QA MCQ"` are scored.
- **Parse think:** `<think>...</think>` is extracted; temporal-spatial claims are parsed with `<obj>...</obj><box>[...]</box>at<t>...</t>s`.
- **GT:** `gt_items = kwargs["key_items"][idx]`, `gt_frames = kwargs["key_frames"][idx]`, `image_size = kwargs.get("image_size", [(640, 480)])[idx]`.
- **GT times:** Built from `key_frames` (frame `idx` → `time`) and from `key_items.keys()` for frames not in `key_frames` (time estimated as `int(frame_idx)/30.0`).
- **Matching:** Each claim’s `pred_time` and `pred_bbox` are matched to the closest GT frame (by time, within 2 s). Pred bbox is converted to pixels if normalized (`all(0 <= c <= 1)`); GT bbox is converted with `convert_coord_format(gt_bbox, image_size)`.
- **Trajectory:** Need ≥2 matched predictions spanning ≥2 distinct GT frames. Otherwise 0.0 or 0.1 (same-frame fallback).
- **Direction:** Last `<motion>...</motion>` is parsed for direction word; GT direction from `compute_direction_from_bboxes(bbox1, bbox2)` (first vs last matched GT bbox). Binary score: 1.0 if pred_direction == gt_direction else 0.0.
- **Speed:** Pred and GT displacement magnitude over time; speed ratio = min/max of pred_speed and gt_speed; `speed_score` in [0, 1].
- **Combined:** `motion_reward = 0.5 * direction_score + 0.5 * speed_score`, clamped to [0, 1].

**1.4 Reward aggregation (grpo_trainer)**

- `rewards_per_func[:, i]` = list of per-completion rewards from each reward function; `rewards = rewards_per_func.sum(dim=1)` (sum over reward functions). So motion_trajectory is one term in the total reward. Logging: mean per reward function over the batch (`rewards/rewards/{reward_func_name}`). Advantage: `(rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)`. Loss uses per-token log probs, completion mask, advantages, and KL. This is consistent with a standard GRPO/Open-o3-style sum-of-rewards setup.

**Verdict:** End-to-end path is consistent: one RL sample → one set of kwargs per generation → correct indexing by `idx` → direction/speed → single scalar in [0,1] → summed with other rewards and used for advantage.

---

## 2. Looks correct

- **Reward signature and call:** Trainer calls `reward_func(prompts=prompts, completions=completions, **reward_kwargs)`. `motion_trajectory_reward(completions, **kwargs)` uses `completions` and kwargs only; no mismatch.
- **Indexing:** For each completion index `idx`, `key_items[idx]`, `key_frames[idx]`, `image_size[idx]` are built by repeating each sample’s value `num_generations` times; so each completion is paired with the correct sample’s GT.
- **key_items key type:** `gt_frame_times` uses `str(frame['idx'])`; `available_gt_frame_indices = set(gt_items.keys())`. JSON-loaded dicts have string keys; `closest_frame_idx in gt_items` and lookups stay consistent when keys are strings.
- **Coordinate system:** Pred and GT bboxes are converted to pixels with the same `image_size` (W, H) in `convert_coord_format`; GT is assumed normalized in key_items (same as `thk_spatial_reward`).
- **Missing `<motion>`:** If no `<motion>` tag, `pred_direction_text` is None → `direction_score = 0.0`; speed_score still computed; no crash.
- **Single-frame matches:** When all matches land on the same GT frame, reward is 0.1 (small reward for trying); otherwise 0.0 or full trajectory reward.
- **Empty key_items:** `gt_items is None` → 0.0; `gt_items == {}` → `available_gt_frame_indices` empty, `matched_predictions` stays empty → 0.0.
- **Exception handling:** In the broad `except` in `motion_trajectory_reward`, 0.0 is appended and `idx` is still incremented (after the except block), so the returned list length matches `len(completions)`.
- **train_grpo.py / sbatch:** `motion_trajectory` is in `reward_funcs` and in the registry; script passes `motion_trajectory` in `--reward_funcs`. Integration is correct.

---

## 3. Potential issues

**3.1 image_size empty or invalid (crash or wrong scale)**

- **Where:** `training/motion_reward.py` line 227:  
  `image_size = kwargs.get("image_size", [(640, 480)])[idx] if idx < len(kwargs.get("image_size", [])) else (640, 480)`
- **Problem:** If the dataset has `"image_size": []` (or any list shorter than `idx`), then `kwargs.get("image_size", [(640, 480)])` returns that list, and `...[idx]` can raise **IndexError**. You noted “image_size sometimes empty in samples”; that would trigger this.
- **Also:** If `process_vision_info` fails (`grpo_trainer.py` ~413–414), `inputs[0]['image_size']` is never set. If the sample also has no (or empty) `image_size`, the key may be missing and the default `(640, 480)` is used — no crash but possibly wrong scale for bbox conversion.
- **Suggestion:** Defend against missing/empty: e.g. ensure `image_size` is a tuple `(w, h)` before indexing; if not, use a safe default and optionally log.

**3.2 key_frames entries without `idx` (KeyError)**

- **Where:** `training/motion_reward.py` line 241: `gt_frame_times[str(frame['idx'])] = frame['time']`
- **Problem:** Doc says key_frames are `{idx, time}`; trainer also uses `key_frame["path"]`. If any entry has `time` (and `path`) but no `idx`, this line raises **KeyError**.
- **Suggestion:** Use `frame.get('idx', None)` and skip or derive an index from position/path if needed; or validate data so every key_frame has `idx`.

**3.3 GT bbox “any object” at frame**

- **Where:** `training/motion_reward.py` lines 275–279:  
  `gt_obj_bboxes = list(gt_items[closest_frame_idx].values())` then `gt_bbox = gt_obj_bboxes[0][0]`.
- **Logic:** The code takes the first object’s first bbox at that frame. If the model is reasoning about a specific object but GT has multiple objects per frame, direction/speed are computed from that first object only, which may not match the predicted object.
- **Impact:** Possible under-counting or mismatch when multiple objects exist; not a bug per se but a design limitation to be aware of.

**3.4 Frame time for key_items-only frames**

- **Where:** `training/motion_reward.py` lines 244–247: for frames in `key_items` but not in `key_frames`, time is set to `int(frame_idx) / 30.0`.
- **Problem:** Assumes frame index equals frame number at 30 fps. If key_items indices are not 0-based frame numbers or video fps differs, time alignment is wrong and matching (and thus direction/speed) can be off.
- **Suggestion:** Prefer mapping frame_idx to time from metadata or key_frames when available; use 30 fps only as a fallback and document it.

**3.5 Normalized pred bbox check**

- **Where:** `training/motion_reward.py` line 261: `if all(0 <= c <= 1 for c in pred_bbox):` then convert to pixels.
- **Problem:** If the model outputs pixel coords (e.g. 0–640), the check fails and pred_bbox is left in pixels while GT is converted with `image_size`. If `image_size` is wrong or inconsistent, pred and GT are still in the same “units” (pixels) for that path, but if one path is normalized and the other pixels, that would be a bug. Currently both end up in pixels for the reward math; the only risk is when `image_size` is wrong (see 3.1).

---

## 4. Reward aggregation and training step

- **Sum of rewards:** `rewards = rewards_per_func.sum(dim=1)` — Open-o3-style; motion_trajectory is one component.
- **Logging:** Per-reward means over the batch are logged; motion_trajectory will appear as `rewards/motion_trajectory_reward` (or the function’s `__name__`).
- **Advantage / loss:** Group mean and std over `num_generations`, then advantage; loss uses advantages and KL. No special handling for motion_trajectory; integration is correct.

---

## 5. Summary

| Item | Status |
|------|--------|
| Full path (sample → reward) | Correct |
| Reward kwargs and indexing | Correct |
| Direction/speed logic | Correct |
| key_items/key_frames types and lookups | Correct for string keys |
| Empty key_items / missing &lt;motion&gt; / single-frame | Handled |
| Reward sum and logging | Consistent with Open-o3 |
| **image_size empty or short list** | **Risk of IndexError** |
| **key_frames missing `idx`** | **Risk of KeyError** |
| GT “first object” vs predicted object | Design limitation |
| Time estimate for key_items-only frames | Assumption (30 fps) |

**Recommendation:** Before changing code, confirm in data how often `image_size` is empty and whether any `key_frames` entry lacks `idx`. If both are possible, fixing 3.1 and 3.2 will make the pipeline robust. If you see crashes in logs (IndexError/KeyError), address those first; otherwise you can keep the current run and add the fixes in a follow-up commit.
