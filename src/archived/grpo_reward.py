"""
Reward function for GRPO training on Dora Q&A dataset.

Computes rewards based on how well the model's answer matches the ground truth.
Uses F1 score (token-level) for answer comparison.
"""

from typing import List, Dict, Any, Optional
import re


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from model completion.
    
    Looks for patterns like:
    - "Final answer: <answer>"
    - "Answer: <answer>"
    - Or just returns the text if no pattern found
    
    Args:
        text: Model completion text
        
    Returns:
        Extracted answer string
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Pattern 1: "Final answer: <answer>"
    pattern1 = r"(?:final\s+answer|answer)\s*:?\s*(.+?)(?:\n|$)"
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Remove any trailing punctuation that might be part of the pattern
        answer = re.sub(r'[.!?]+$', '', answer).strip()
        if answer:
            return answer
    
    # Pattern 2: Look for text after common prefixes (but capture full answer)
    prefixes = [
        r"the answer is\s*:?\s*(.+?)(?:\n|$)",
        # r"it is\s*:?\s*(.+?)(?:\n|$)",
    ]
    for prefix_pattern in prefixes:
        match = re.search(prefix_pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'[.!?]+$', '', answer).strip()
            if answer:
                return answer
    
    # Pattern 3: "we need to" - capture everything after it (full answer)
    if "we need to" in text.lower():
        match = re.search(r"we need to\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Don't remove trailing punctuation - keep full answer
            return answer
    
    # Fallback: return the first sentence or the whole text if short
    sentences = re.split(r'[.!?]\s+', text)
    if sentences:
        last_sentence = sentences[0].strip()
        if len(last_sentence) > 0:
            return last_sentence
    
    # Final fallback: return cleaned text
    return text.strip()


def tokenize_answer(text: str) -> List[str]:
    """
    Tokenize answer into words for F1 computation.
    
    Args:
        text: Answer text
        
    Returns:
        List of lowercase tokens
    """
    if not text:
        return []
    
    # Normalize: lowercase, remove punctuation, split on whitespace
    text = text.lower()
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    
    return tokens


def compute_f1_score(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    """
    Compute F1 score between predicted and gold tokens.
    
    Args:
        pred_tokens: Predicted tokens
        gold_tokens: Gold tokens
        
    Returns:
        F1 score between 0.0 and 1.0
    """
    if not gold_tokens:
        # If no gold answer, return 0.0
        return 0.0
    
    if not pred_tokens:
        # If no prediction, return 0.0
        return 0.0
    
    # Count overlaps
    gold_set = set(gold_tokens)
    pred_set = set(pred_tokens)
    
    # True positives: tokens in both
    tp = len(gold_set & pred_set)
    
    if tp == 0:
        return 0.0
    
    # Precision: tp / (tp + fp) = tp / len(pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    
    # Recall: tp / (tp + fn) = tp / len(gold_set)
    recall = tp / len(gold_set) if gold_set else 0.0
    
    # F1: 2 * precision * recall / (precision + recall)
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_dora_reward(
    completions: List[str] = None,
    answers: Optional[List[str]] = None,
    questions: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """
    Compute rewards for GRPO training.
    
    This function is called by TRL's GRPOTrainer with various arguments.
    We extract completions and ground truth answers, then compute F1 scores.
    
    Args:
        completions: List of model completion strings (or message dicts)
        answers: List of ground truth answers (optional, may be in kwargs)
        questions: List of questions (optional, for logging)
        **kwargs: Additional arguments that may contain completions, answers, etc.
    
    Returns:
        List of reward scores (one per completion), each between 0.0 and 1.0
    """
    # DEBUG: Log function call signature
    import inspect
    frame = inspect.currentframe()
    args_info = inspect.getargvalues(frame)
    print(f"\n[DEBUG REWARD] Function called with:")
    print(f"  - Positional args: completions={completions is not None}, answers={answers is not None}, questions={questions is not None}")
    print(f"  - kwargs count: {len(kwargs)}")
    
    # Handle different calling conventions from TRL
    # TRL may pass completions as positional args or in kwargs
    
    # Extract completions
    if not completions:
        completions = kwargs.get('completions', [])
    
    if not completions:
        # Try to get from args if passed as positional
        if len(kwargs) == 0:
            return [0.0]
        # Last resort: check if completions are in kwargs with different key
        for key in ['completion', 'outputs', 'responses']:
            if key in kwargs:
                completions = kwargs[key]
                if not isinstance(completions, list):
                    completions = [completions]
                break
    
    if not completions:
        print("[REWARD] WARNING: No completions provided!")
        return [0.0]
    
    # Extract ground truth answers
    if answers is None:
        answers = kwargs.get('answer', kwargs.get('answers', []))
    
    if not answers:
        print("[REWARD] WARNING: No ground truth answers provided!")
        return [0.0] * len(completions)
    
    # Extract questions for logging
    if questions is None:
        questions = kwargs.get('question', kwargs.get('questions', []))
    
    # DEBUG: Log what we received
    print(f"\n[DEBUG REWARD] compute_dora_reward called")
    print(f"  - completions type: {type(completions)}")
    print(f"  - completions length: {len(completions) if completions else 0}")
    if completions:
        print(f"  - first completion type: {type(completions[0])}")
        print(f"  - first completion repr (first 200 chars): {repr(completions[0])[:200]}")
    print(f"  - kwargs keys: {list(kwargs.keys())}")
    if 'completion_ids' in kwargs:
        print(f"  - completion_ids type: {type(kwargs['completion_ids'])}")
        print(f"  - completion_ids length: {len(kwargs['completion_ids']) if kwargs['completion_ids'] else 0}")
    
    # Process each completion
    rewards = []
    num_items = len(completions)
    
    for i in range(num_items):
        completion = completions[i]
        gold_answer = answers[i] if i < len(answers) else ""
        question = questions[i] if questions and i < len(questions) else ""
        
        # DEBUG: Log raw completion
        print(f"\n[DEBUG REWARD] Processing completion {i}:")
        print(f"  - Raw completion type: {type(completion)}")
        print(f"  - Raw completion repr (first 300 chars): {repr(completion)[:300]}")
        
        # Handle different completion formats
        completion_text = ""
        
        if isinstance(completion, str):
            completion_text = completion
            print(f"  - Detected as string, length: {len(completion_text)}")
        elif isinstance(completion, list):
            # Message format: [{"role": "assistant", "content": "..."}]
            if completion and isinstance(completion[0], dict):
                for msg in completion:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            completion_text = content
                            break
                        elif isinstance(content, list):
                            # Multimodal content - extract text parts
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            completion_text = " ".join(text_parts)
                            break
            else:
                completion_text = str(completion)
        elif isinstance(completion, dict):
            # Single message dict
            if completion.get("role") == "assistant":
                content = completion.get("content", "")
                if isinstance(content, str):
                    completion_text = content
                else:
                    completion_text = str(content)
            else:
                completion_text = str(completion)
        else:
            completion_text = str(completion) if completion else ""
        
        # DEBUG: Log extracted text
        print(f"  - Extracted completion_text length: {len(completion_text)}")
        print(f"  - Extracted completion_text (first 200 chars): {repr(completion_text[:200])}")
        
        # Extract final answer from completion
        pred_answer = extract_final_answer(completion_text)
        
        # DEBUG: Log answer extraction
        print(f"  - Extracted pred_answer: {repr(pred_answer)}")
        
        # Compute F1 score
        pred_tokens = tokenize_answer(pred_answer)
        gold_tokens = tokenize_answer(gold_answer)
        
        # DEBUG: Log tokenization
        print(f"  - Pred tokens: {pred_tokens}")
        print(f"  - Gold tokens: {gold_tokens}")
        
        f1_score = compute_f1_score(pred_tokens, gold_tokens)
        rewards.append(float(f1_score))
        
        # Log first few examples for debugging
        if i < 2:
            print(f"\n[REWARD] Example {i}:")
            print(f"  Question: {question[:100] if question else 'N/A'}")
            print(f"  Gold: {gold_answer[:100]}")
            print(f"  Pred: {pred_answer[:100]}")
            print(f"  F1: {f1_score:.3f}")
    
    return rewards

