# Test Outputs

This directory contains results from VLM spatial grounding diagnostic tests.

## Directory Structure

```
outputs/
├── DIAGNOSTIC_SUMMARY.md           # Overall findings and recommendations
├── final_tracking/                 # Single-frame detection results
│   ├── frame_XXXX_tracked.jpg     # Annotated frames
│   └── tracking_data.json         # Structured tracking data
├── motion_chain_inference/         # Full reasoning chain outputs
│   ├── motion_chain_response.txt  # Generated evidence chain
│   └── motion_chain_result.json   # Structured result
└── *.log                          # Execution logs
```

## Key Files

1. **DIAGNOSTIC_SUMMARY.md** - READ THIS FIRST
   - Comprehensive analysis of all test results
   - Identifies gaps for GRPO training
   - Provides recommended training strategy

2. **final_tracking/tracking_data.json**
   - Baseline single-frame detection performance
   - Frame-by-frame bbox predictions
   - Motion trajectory data

3. **motion_chain_inference/motion_chain_response.txt**
   - Full generated reasoning chain (the target format)
   - Shows current capability vs. desired output
   - Demonstrates hallucination problem (all bboxes identical)

## Quick Summary

**What works**: Format generation (temporal intervals, bbox syntax, structure)  
**What doesn't**: Spatial grounding (bboxes are hallucinated/copied, not detected)  
**Solution**: GRPO training with geometric rewards

See DIAGNOSTIC_SUMMARY.md for full details.
