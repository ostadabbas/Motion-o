#!/usr/bin/env python3
"""
Generate MCQ dataset using Google Gemini API (text-only, cost-optimized).
Estimated cost: FREE for first 1,500 requests/day, then ~$0.00035/request = ~$1.87 for 5,348 examples
Much cheaper than Claude!
"""

"""
conda activate dora_cuda

python scripts/generate_dora_vqa.py \
    --dataset-path outputs/dataset_text_only \
    --output-path outputs/mcq_dataset \
    --api-key YOUR_GEMINI_API_KEY
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from datasets import load_from_disk, Dataset
import google.generativeai as genai
import time
import random
import re
from collections import Counter


def classify_doravqa(question: str, answer: str, transcript: str) -> dict:
    """Classify using only reliable text-based signals."""
    
    q_lower = question.lower().strip()
    t_lower = transcript.lower()
    
    # ═══════════════════════════════════════════════════════
    # 1. REASONING CATEGORY (Direct from question pattern)
    # ═══════════════════════════════════════════════════════
    
    if 'how many' in q_lower or 'count' in q_lower:
        reasoning = 'counting'
    elif any(phrase in q_lower for phrase in ['where do we go', 'where should we go']):
        reasoning = 'navigation'  # "River → Forest → Valley"
    elif 'do you see' in q_lower or q_lower == 'where?':
        reasoning = 'spatial_location'  # "Do you see X? Where?"
    elif any(phrase in q_lower for phrase in ['which', 'should we choose']):
        reasoning = 'object_selection'  # "Which boat?"
    elif any(phrase in q_lower for phrase in ['how are we', 'how can we', 'how do we']):
        reasoning = 'problem_solving'  # "How are we going to..."
    elif any(phrase in q_lower for phrase in ['who do we', 'who helps', 'what do we use when']):
        reasoning = 'recall_knowledge'  # "Who do we ask for help?"
    elif 'can you say' in q_lower or '¿' in question:
        reasoning = 'language'  # "Can you say azul?"
    else:
        reasoning = 'other'
    
    # ═══════════════════════════════════════════════════════
    # 2. COMPOSITIONAL COMPLEXITY (Count constraints)
    # ═══════════════════════════════════════════════════════
    
    # For selection questions, count explicit requirements in transcript
    num_constraints = 0
    if reasoning in ['object_selection', 'navigation']:
        # Look for constraint patterns in transcript
        constraint_indicators = [
            ('without', t_lower.count('without')),
            ('must', t_lower.count('must')),
            ('need', t_lower.count('need')),
            ('should', t_lower.count('should')),
        ]
        num_constraints = sum(count for _, count in constraint_indicators if count > 0)
    
    # Simplify to bins
    if num_constraints == 0:
        complexity = 'simple'
    elif num_constraints == 1:
        complexity = 'constrained'
    else:
        complexity = 'multi-constrained'
    
    # ═══════════════════════════════════════════════════════
    # 3. TEMPORAL SPAN (Sequence indicators in question)
    # ═══════════════════════════════════════════════════════
    
    sequence_words = ['first', 'then', 'after', 'before', 'across', 'through']
    has_sequence = any(word in q_lower for word in sequence_words)
    
    if has_sequence:
        temporal_span = 'sequential'  # Multi-step reasoning
    else:
        temporal_span = 'immediate'  # Single-step
    
    # ═══════════════════════════════════════════════════════
    # 4. MODALITY (What's needed to answer)
    # ═══════════════════════════════════════════════════════
    
    # Strong visual indicators
    visual_indicators = [
        'do you see' in q_lower,
        'where?' == q_lower,
        'which' in q_lower,
        reasoning == 'counting',
    ]
    requires_visual = any(visual_indicators)
    
    # Strong text indicators
    text_indicators = [
        reasoning in ['recall_knowledge', 'language'],
        num_constraints > 0,  # Constraints mentioned in dialogue
        has_sequence,  # Navigation sequences spoken
        'can you say' in q_lower,
    ]
    requires_transcript = any(text_indicators)
    
    # Classify modality
    if requires_visual and requires_transcript:
        modality = 'multimodal'
    elif requires_visual:
        modality = 'visual-only'
    elif requires_transcript:
        modality = 'text-only'
    else:
        modality = 'multimodal'  # Default safe choice
    
    return {
        'reasoning_category': reasoning,
        'compositional_complexity': complexity,
        'temporal_span': temporal_span,
        'requires_visual': requires_visual,
        'requires_transcript': requires_transcript,
        'modality': modality,
        'question_word_count': len(question.split()),
        'answer_word_count': len(answer.split()),
    }


class RateLimitError(Exception):
    """Custom exception for rate limit errors with retry delay."""
    def __init__(self, message, retry_delay):
        super().__init__(message)
        self.retry_delay = retry_delay

def call_gemini_for_mcq(
    question: str,
    gold_answer: str,
    context: str,
    model: genai.GenerativeModel,
    num_distractors: int = 3
) -> Dict:
    """
    Generate MCQ distractors using Gemini API (text-only).
    
    Cost: FREE for first 1,500/day, then ~$0.00035 per call
    (Gemini 1.5 Flash: $0.075 per 1M input tokens, $0.30 per 1M output tokens)
    """
    
    # Truncate context to save tokens (keep relevant parts)
    if len(context) > 2000:
        context = context[-2000:]  # Keep last 2000 chars (most recent context)
    
    prompt = f"""Create a multiple-choice question with {num_distractors} wrong answer options.

            VIDEO CONTEXT:
            {context[:1000]}

            QUESTION: {question}
            ORIGINAL CORRECT ANSWER: {gold_answer}

            TASK: 
            1. First, simplify the original correct answer to be concise and MCQ-appropriate (similar length to distractors, clear and direct)
            2. Then generate {num_distractors} wrong but plausible answers that match the simplified answer's style and length

            REQUIREMENTS:
            - Simplified answer: Short, clear, direct (1-2 sentences max, ideally 1 phrase)
            - Distractors: Each must be incorrect, believable, simple words for ages 3-7, unique, and similar length to the simplified answer

            OUTPUT FORMAT - Copy this structure exactly and fill in all fields:
            {{
                "simplified_answer": "concise version of the correct answer",
                "distractor_1": "your first wrong answer",
                "distractor_2": "your second wrong answer",
                "distractor_3": "your third wrong answer"
            }}

            IMPORTANT: 
            - Simplify the answer to be concise (like the distractors will be)
            - You must provide simplified_answer, distractor_1, distractor_2, AND distractor_3. All fields are required."""

    try:
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,  # Lower temperature for more consistent output
                max_output_tokens=2048,  # Increased significantly to ensure all 3 distractors fit
                top_p=0.95,
            )
        )
        
        # Extract text
        response_text = response.text.strip()
        
        # Check if response was truncated (common issue with Gemini)
        if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
            finish_reason = response.candidates[0].finish_reason
            if finish_reason == 'MAX_TOKENS':
                print(f"\n  ⚠️  Response truncated (MAX_TOKENS), may be incomplete")
        
        # Clean up response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Sometimes Gemini adds extra text before/after JSON, extract JSON only
        # Look for JSON pattern (improved regex to handle incomplete JSON)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        # Fix common JSON issues
        # 1. Remove trailing commas before closing brace/bracket
        response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
        
        # 2. Check if JSON looks complete (has closing brace)
        if not response_text.strip().endswith('}'):
            # First, try to extract complete distractor entries (handles truncated responses better)
            distractor_pattern = r'"distractor_\d+":\s*"[^"]*"'
            distractor_matches = list(re.finditer(distractor_pattern, response_text))
            
            if distractor_matches and len(distractor_matches) >= 2:
                # Reconstruct from complete distractor entries only
                complete_parts = []
                for match in distractor_matches:
                    complete_parts.append(match.group(0))
                response_text = '{' + ', '.join(complete_parts) + '}'
            else:
                # Fallback: try to find all complete key-value pairs
                matches = list(re.finditer(r'"([^"]+)":\s*"([^"]*)"', response_text))
                
                if matches and len(matches) >= 2:  # Need at least 2 complete pairs
                    # Reconstruct JSON from complete pairs only
                    pairs = []
                    for match in matches:
                        key = match.group(1)
                        value = match.group(2)
                        pairs.append(f'"{key}": "{value}"')
                    
                    response_text = '{' + ', '.join(pairs) + '}'
                else:
                    # Last resort: try to close at last complete entry
                    last_complete = response_text.rfind('",')
                    if last_complete > 0:
                        temp = response_text[:last_complete+2]
                        temp = re.sub(r',(\s*$)', r'\1', temp)  # Remove trailing comma
                        response_text = temp + '}'
                    else:
                        raise ValueError(f"Incomplete JSON response (missing closing brace): {response_text[:200]}")
        
        # 3. Remove any trailing commas again (in case we added them back)
        response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
        
        # Parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            # If parsing still fails, try one more fix: remove incomplete last entry
            # This handles cases where the last distractor is cut off mid-string
            try:
                # Find the last complete distractor entry
                distractor_pattern = r'"distractor_\d+":\s*"[^"]*"'
                matches = list(re.finditer(distractor_pattern, response_text))
                if matches and len(matches) >= 2:  # Need at least 2 distractors
                    # Reconstruct from complete entries only
                    complete_parts = []
                    for match in matches:
                        complete_parts.append(match.group(0))
                    response_text = '{' + ', '.join(complete_parts) + '}'
                    result = json.loads(response_text)
                else:
                    raise  # Re-raise original error
            except:
                # If all fixes fail, show the error
                print(f"\n  ⚠️  JSON parse error: {e}")
                print(f"  Response text (first 500 chars): {response_text[:500]}")
                raise
        
        # Validate we got at least some distractors
        found_distractors = []
        for i in range(1, num_distractors + 1):
            key = f"distractor_{i}"
            if key in result:
                found_distractors.append(i)
        
        # For 4-choice MCQ, we need at least 3 distractors (1 correct + 3 distractors = 4 total)
        min_required = 3 if num_distractors >= 3 else 2
        if len(found_distractors) < min_required:
            raise ValueError(f"Not enough distractors found. Need at least {min_required} for {min_required+1}-choice MCQ, got {len(found_distractors)}: {found_distractors}")
        
        # If we got fewer than requested, that's okay - we'll use what we have
        if len(found_distractors) < num_distractors:
            print(f"\n  ⚠️  Only found {len(found_distractors)}/{num_distractors} distractors, but proceeding with available ones")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"\nJSON parse error: {e}")
        if 'response_text' in locals():
            print(f"Response was: {response_text[:300]}")
        raise
    except Exception as e:
        error_str = str(e)
        # Check for rate limit errors (429)
        if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
            # Try to extract retry delay from error message
            retry_delay = 60  # Default to 60 seconds if we can't parse it
            if "retry_delay" in error_str or "retry in" in error_str.lower():
                delay_match = re.search(r'retry in ([\d.]+)', error_str, re.IGNORECASE)
                if delay_match:
                    retry_delay = float(delay_match.group(1)) + 5  # Add 5 second buffer
            raise RateLimitError(f"Rate limit exceeded. Retry after {retry_delay:.1f} seconds", retry_delay)
        
        print(f"\nAPI error: {e}")
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            print(f"Prompt feedback: {response.prompt_feedback}")
        raise


def generate_dataset_stats(mcq_data: List[Dict]) -> Dict:
    """Generate statistics for dataset figure."""
    total = len(mcq_data)
    if total == 0:
        return {}
    
    # Count reasoning categories
    reasoning_counts = Counter(item.get('reasoning_category', 'other') for item in mcq_data)
    reasoning_pct = {k: (v / total * 100) for k, v in reasoning_counts.items()}
    
    # Count modality
    modality_counts = Counter(item.get('modality', 'multimodal') for item in mcq_data)
    modality_pct = {k: (v / total * 100) for k, v in modality_counts.items()}
    
    # Count compositional complexity
    complexity_counts = Counter(item.get('compositional_complexity', 'simple') for item in mcq_data)
    complexity_pct = {k: (v / total * 100) for k, v in complexity_counts.items()}
    
    # Count temporal span
    temporal_counts = Counter(item.get('temporal_span', 'immediate') for item in mcq_data)
    temporal_pct = {k: (v / total * 100) for k, v in temporal_counts.items()}
    
    return {
        'total': total,
        'reasoning_category': {
            'counts': dict(reasoning_counts),
            'percentages': reasoning_pct
        },
        'modality': {
            'counts': dict(modality_counts),
            'percentages': modality_pct
        },
        'compositional_complexity': {
            'counts': dict(complexity_counts),
            'percentages': complexity_pct
        },
        'temporal_span': {
            'counts': dict(temporal_counts),
            'percentages': temporal_pct
        }
    }


def create_mcq_dataset_gemini(
    dataset_path: str,
    output_path: str,
    api_key: str,
    num_distractors: int = 3,
    batch_size: int = 50,  # Gemini has higher rate limits
    save_every: int = 100,
    max_examples: Optional[int] = None,
    resume_from: Optional[int] = None,
    model_name: str = "gemini-2.5-flash",
    skip_llm: bool = True  # Using Gemini 2.5 Flash (latest stable)
):
    """
    Create MCQ dataset using Google Gemini API (text-only, optimized for cost).
    
    Args:
        dataset_path: Path to your GRPO dataset
        output_path: Where to save MCQ dataset
        api_key: Google AI Studio API key
        num_distractors: Number of wrong answers (default 3)
        batch_size: Pause after N requests (rate limiting)
        save_every: Save checkpoint every N examples
        max_examples: Limit processing (for testing)
        resume_from: Resume from checkpoint index
        model_name: "gemini-1.5-flash" (faster, cheaper) or "gemini-1.5-pro" (better quality)
    
    Estimated cost:
    - gemini-1.5-flash: FREE for 1,500/day, then ~$0.00035/call = ~$1.87 total
    - gemini-1.5-pro: ~$0.0035/call = ~$18.72 total
    """
    if not skip_llm:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # List available models first
        print("\nListing available Gemini models...")
        try:
            available_models = genai.list_models()
            print("Available models:")
            for m in available_models:
                if 'generateContent' in m.supported_generation_methods:
                    print(f"  - {m.name} (display_name: {m.display_name})")
        except Exception as e:
            print(f"  Could not list models: {e}")
        
        # Initialize model
        print(f"\nUsing model: {model_name}")
        model = genai.GenerativeModel(model_name)
        print("API configured successfully")
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    print(f"Processing {len(dataset)} examples")
    
    # Setup output paths
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Load existing progress if resuming
    mcq_data = []
    start_idx = 0
    
    if resume_from is not None:
        checkpoint_file = checkpoint_path / f"checkpoint_{resume_from}.json"
        if checkpoint_file.exists():
            print(f"Resuming from checkpoint {resume_from}...")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                mcq_data = json.load(f)
            start_idx = resume_from
            print(f"Loaded {len(mcq_data)} existing examples")
    
    # Track errors and stats
    errors = []
    api_calls = 0
    start_time = time.time()
    
    # Process examples
    print(f"\nStarting MCQ generation...")
    print(f"{'='*60}")
    
    for i in tqdm(range(start_idx, len(dataset)), desc="Generating MCQ", initial=start_idx, total=len(dataset)):
        item = dataset[i]
        
        try:
            # Extract data
            context = item.get('transcript', '')
            question = item.get('question', '')
            gold_answer = item.get('answer', '')
            
            # Normalize gold answer: if it's all uppercase, convert to sentence case (first letter only)
            gold_answer_stripped = gold_answer.strip()
            if gold_answer_stripped.isupper() and len(gold_answer_stripped) > 1:
                # Convert all uppercase to sentence case (only first letter capitalized)
                gold_answer = gold_answer_stripped.capitalize()
            else:
                gold_answer = gold_answer_stripped
            
            # Validate inputs
            if not question or not gold_answer:
                raise ValueError("Missing question or answer")
            
            # Skip if gold answer is too short (might be problematic)
            if len(gold_answer.strip()) < 2:
                print(f"\n  ⚠️  Skipping example {i}: answer too short ('{gold_answer}')")
                continue
            
            if not skip_llm:
            # Generate distractors with retry logic
                max_retries = 5  # Increased retries for rate limits
                result = None
                
                for retry in range(max_retries):
                    try:
                        result = call_gemini_for_mcq(
                            question=question,
                            gold_answer=gold_answer,
                            context=context,
                            model=model,
                            num_distractors=num_distractors
                        )
                        # Extract simplified answer
                        simplified_answer = result.get("simplified_answer", gold_answer).strip()
                        if not simplified_answer:
                            simplified_answer = gold_answer
                        
                        # Validate we got enough distractors
                        found_count = sum(1 for k in result.keys() if k.startswith('distractor_'))
                        if found_count >= 3:
                            break  # Success - got enough distractors
                        else:
                            print(f"\n  ⚠️  Retry {retry+1}/{max_retries}: Only got {found_count} distractors, retrying...")
                            if retry < max_retries - 1:
                                time.sleep(2)  # Brief pause before retry
                                continue
                            else:
                                raise ValueError(f"Failed to get enough distractors after {max_retries} retries. Got {found_count}, need 3.")
                    except RateLimitError as e:
                        if retry < max_retries - 1:
                            wait_time = e.retry_delay
                            print(f"\n  ⏳ Rate limit hit. Waiting {wait_time:.1f} seconds before retry {retry+1}/{max_retries}...")
                            time.sleep(wait_time)
                        else:
                            raise  # Give up after retries
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"\n  Retry {retry+1}/{max_retries} for example {i}")
                            time.sleep(2)  # Wait before retry
                        else:
                            raise  # Give up after retries
                
                if result is None:
                    raise ValueError("Failed to generate distractors after retries")
                
                api_calls += 1
            
                # Extract distractors
                distractors = [
                    result[f'distractor_{j+1}'].strip()
                    for j in range(num_distractors)
                    if f'distractor_{j+1}' in result
                ]
                
                # Extract simplified answer from result
                simplified_answer = result.get("simplified_answer", gold_answer).strip()
                if not simplified_answer:
                    simplified_answer = gold_answer
                
                # Validate distractors are different from simplified answer and from each other
                unique_distractors = []
                for d in distractors:
                    # Check not equal to simplified answer (case-insensitive)
                    if d.lower().strip() == simplified_answer.lower().strip():
                        continue
                    # Check not duplicate of existing distractor
                    if d.lower().strip() in [ud.lower().strip() for ud in unique_distractors]:
                        continue
                    unique_distractors.append(d)
                
                distractors = unique_distractors
            
                # If we lost too many distractors, log warning
                if len(distractors) < num_distractors:
                    print(f"\n  ⚠️  Example {i}: Only {len(distractors)}/{num_distractors} unique distractors")
                
                # Need at least 3 distractors for a 4-choice MCQ (1 correct + 3 distractors = 4 total)
                if len(distractors) < 3:
                    raise ValueError(f"Not enough unique distractors. Need at least 3 for 4-choice MCQ, got {len(distractors)}")
                
                # Create MCQ item using simplified answer
                choices = [simplified_answer] + distractors
                random.shuffle(choices)
                correct_index = choices.index(simplified_answer)
                
                # Classify the MCQ for statistics
                classification = classify_doravqa(
                    question=question,
                    answer=simplified_answer,
                    transcript=item.get('transcript', '')
                )
            
            mcq_item = {
                'transcript': item.get('transcript', ''),
                'question': question,
                'answer': simplified_answer,  # Use simplified answer for MCQ
                'original_answer': gold_answer,  # Keep original for reference
                'choices': choices,
                'correct_index': correct_index,
                'correct_choice': chr(65 + correct_index),  # A, B, C, D
                'num_choices': len(choices),
                'generation_method': f'gemini-{model_name}',
                'season': item.get('season'),
                'episode': item.get('episode', ''),  # Preserve episode from source dataset
                'entry_id': item.get('entry_id', ''),  # Preserve entry_id from source dataset
                # Classification fields for statistics
                **classification
            }
            
            mcq_data.append(mcq_item)
            
            # Save checkpoint periodically (also helps with memory management)
            if (i + 1) % save_every == 0:
                checkpoint_file = checkpoint_path / f"checkpoint_{i+1}.json"
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(mcq_data, f, indent=2, ensure_ascii=False)
                
                # Also save intermediate dataset to disk to free up memory gradually
                # This allows resuming even if the script crashes
                try:
                    temp_dataset = Dataset.from_list(mcq_data)
                    temp_dataset.save_to_disk(str(output_path / "mcq_dataset_temp"))
                except Exception as e:
                    print(f"  ⚠️  Could not save temp dataset: {e}")
                
                elapsed = time.time() - start_time
                rate = api_calls / elapsed if elapsed > 0 else 0
                
                # Estimate cost (very rough)
                if "flash" in model_name.lower():
                    # Flash: $0.075 per 1M input, $0.30 per 1M output
                    # Estimate: 600 input tokens, 50 output tokens per call
                    estimated_cost = api_calls * ((600 * 0.075 + 50 * 0.30) / 1_000_000)
                else:
                    # Pro: $1.25 per 1M input, $5.00 per 1M output
                    estimated_cost = api_calls * ((600 * 1.25 + 50 * 5.00) / 1_000_000)
                
                print(f"\n✓ Checkpoint {i+1}/{len(dataset)}")
                print(f"  Processed: {len(mcq_data)} MCQs")
                print(f"  Rate: {rate:.1f} examples/sec")
                print(f"  Est. cost: ${estimated_cost:.2f}")
                print(f"  Errors: {len(errors)}")
            
            # Rate limiting
            # Free tier: 5 requests/min per model = 1 request every 12 seconds
            # Be more conservative: wait 15 seconds to avoid hitting limits
            # Note: This is slow but necessary for free tier. Consider upgrading to paid plan for better limits.
            time.sleep(15)  # Wait 15 seconds between requests to stay well under 5/min limit
                
        except Exception as e:
            error_info = {
                'index': i,
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'error': str(e),
                'error_type': type(e).__name__
            }
            errors.append(error_info)
            print(f"\n⚠️  Error on example {i}: {e}")
            
            # Continue with next example
            continue
    
    # Save final dataset
    print(f"\n{'='*60}")
    print("Saving final MCQ dataset...")
    print(f"Total items in memory: {len(mcq_data)}")
    
    # Save as HuggingFace Dataset (more memory efficient than JSON)
    print("  Creating HuggingFace dataset...")
    mcq_dataset = Dataset.from_list(mcq_data)
    print(f"  Saving to disk...")
    mcq_dataset.save_to_disk(str(output_path / "mcq_dataset"))
    print(f"  ✓ Saved HuggingFace dataset: {len(mcq_dataset)} items")
    
    # Save as JSON backup (only if reasonable size to avoid memory issues)
    # For large datasets, JSON can be very memory-intensive
    if len(mcq_data) < 10000:  # Only save JSON if < 10k items
        print("  Saving JSON backup...")
        with open(output_path / "mcq_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(mcq_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved JSON backup")
    else:
        print(f"  ⚠️  Skipping JSON backup (too large: {len(mcq_data)} items).")
        print(f"     Use HuggingFace dataset (mcq_dataset/) instead, or load from checkpoints.")
    
    # Save errors log
    if errors:
        with open(output_path / "errors.json", 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
    
    # Generate and save statistics for figure
    if len(mcq_data) > 0:
        print(f"\n{'='*60}")
        print("Generating dataset statistics...")
        stats = generate_dataset_stats(mcq_data)
        
        # Save statistics
        stats_file = output_path / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved statistics to: {stats_file}")
        
        # Print summary
        print("\nDataset Statistics Summary:")
        print(f"  Total MCQs: {stats['total']}")
        print(f"  Reasoning Categories: {len(stats['reasoning_category']['counts'])} types")
        print(f"  Modality Distribution:")
        for mod, pct in sorted(stats['modality']['percentages'].items(), key=lambda x: -x[1]):
            print(f"    {mod}: {pct:.1f}%")
        print(f"  Compositional Complexity:")
        for comp, pct in sorted(stats['compositional_complexity']['percentages'].items(), key=lambda x: -x[1]):
            print(f"    {comp}: {pct:.1f}%")
        print(f"  Temporal Span:")
        for temp, pct in sorted(stats['temporal_span']['percentages'].items(), key=lambda x: -x[1]):
            print(f"    {temp}: {pct:.1f}%")
    
    # Calculate stats
    total_time = time.time() - start_time
    
    # Cost estimate
    if "flash" in model_name.lower():
        total_cost = api_calls * ((600 * 0.075 + 50 * 0.30) / 1_000_000)
    else:
        total_cost = api_calls * ((600 * 1.25 + 50 * 5.00) / 1_000_000)
    
    success_rate = (len(mcq_data) / len(dataset)) * 100 if len(dataset) > 0 else 0
    
    # Print summary
    print("="*60)
    print("MCQ Dataset Generation Complete!")
    print("="*60)
    print(f"Model used: {model_name}")
    print(f"Total examples processed: {len(dataset)}")
    print(f"Successful MCQs created: {len(mcq_data)}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Errors: {len(errors)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average rate: {len(mcq_data)/total_time:.2f} examples/sec")
    print(f"Total API calls: {api_calls}")
    print(f"Estimated cost: ${total_cost:.2f}")
    print(f"\nOutput saved to: {output_path / 'mcq_dataset'}")
    
    # Show sample MCQ
    if mcq_data:
        print("\n" + "="*60)
        print("Sample MCQs:")
        print("="*60)
        
        # Show 3 random samples
        samples = random.sample(mcq_data, min(3, len(mcq_data)))
        
        for idx, sample in enumerate(samples, 1):
            print(f"\n[Sample {idx}]")
            print(f"Question: {sample['question']}")
            print(f"Context: {sample['transcript'][:100]}...")
            print("\nChoices:")
            for j, choice in enumerate(sample['choices']):
                marker = "✓ CORRECT" if j == sample['correct_index'] else ""
                print(f"  {chr(65+j)}. {choice} {marker}")
    
    return mcq_dataset


def validate_mcq_sample(mcq_dataset, n: int = 10):
    """Quick quality check on random samples."""
    
    if isinstance(mcq_dataset, Dataset):
        samples = random.sample(range(len(mcq_dataset)), min(n, len(mcq_dataset)))
        samples = [mcq_dataset[i] for i in samples]
    else:
        samples = random.sample(mcq_dataset, min(n, len(mcq_dataset)))
    
    print("\n" + "="*60)
    print(f"Validating {len(samples)} random samples")
    print("="*60)
    
    issues_found = 0
    
    for i, item in enumerate(samples, 1):
        print(f"\n[Sample {i}]")
        print(f"Q: {item['question']}")
        print("Choices:")
        for j, choice in enumerate(item['choices']):
            marker = "✓" if j == item['correct_index'] else " "
            print(f"  {chr(65+j)}. {choice} {marker}")
        
        # Auto-checks
        gold = item['choices'][item['correct_index']]
        distractors = [c for j, c in enumerate(item['choices']) if j != item['correct_index']]
        
        issues = []
        
        # Check 1: Gold not duplicated
        if any(gold.lower().strip() == d.lower().strip() for d in distractors):
            issues.append("⚠️  Gold answer duplicated in distractors")
        
        # Check 2: No duplicate choices
        if len(set(c.lower().strip() for c in item['choices'])) < len(item['choices']):
            issues.append("⚠️  Duplicate choices detected")
        
        # Check 3: All choices non-empty
        if any(not c.strip() for c in item['choices']):
            issues.append("⚠️  Empty choice detected")
        
        if issues:
            issues_found += len(issues)
            for issue in issues:
                print(f"  {issue}")
    
    print(f"\n{'='*60}")
    print(f"Validation complete: {issues_found} issues found in {len(samples)} samples")
    print(f"{'='*60}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MCQ dataset using Google Gemini API")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to GRPO dataset")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Output path for MCQ dataset")
    parser.add_argument("--api-key", type=str, default="XXXXXXXXXXXXXXXXXXXXXXX",
                       help="Google AI Studio API key")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                       choices=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-flash-latest", "gemini-pro-latest"],
                       help="Gemini model to use (flash=faster/cheaper, pro=better quality)")
    parser.add_argument("--num-distractors", type=int, default=3,
                       help="Number of wrong answers (default: 3)")
    parser.add_argument("--test-run", action="store_true",
                       help="Test on first 100 examples only")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum number of examples to process (overrides --test-run)")
    parser.add_argument("--resume-from", type=int, default=None,
                       help="Resume from checkpoint index")
    parser.add_argument("--validate", action="store_true",
                       help="Validate after generation")
    
    args = parser.parse_args()
    
    # Get API key from environment variable if not provided
    api_key = args.api_key
    if api_key == "XXXXXXXXXXXXXXXXXXXXXXX" or not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("❌ Error: GEMINI_API_KEY not found in environment and --api-key not provided")
            print("   Set it with: export GEMINI_API_KEY=your_key_here")
            sys.exit(1)
    
    print("="*60)
    print("Gemini MCQ Generator")
    print("="*60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Model: {args.model}")
    print(f"Test run: {args.test_run}")
    print("="*60)
    
    # Run generation
    max_examples = args.max_examples if args.max_examples is not None else (100 if args.test_run else None)
    
    mcq_dataset = create_mcq_dataset_gemini(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        api_key=api_key,
        model_name=args.model,
        num_distractors=args.num_distractors,
        batch_size=50,
        save_every=100,
        max_examples=max_examples,
        resume_from=args.resume_from
    )
    
    # Validate if requested
    if args.validate:
        validate_mcq_sample(mcq_dataset, n=20)
    
    print("\n✅ Done!")