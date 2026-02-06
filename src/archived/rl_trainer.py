from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Optional
from PIL import Image
import torch
from transformers import AutoProcessor
from trl import PPOTrainer, PPOConfig
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class Transition:
    frames: List[Tuple[Image.Image, float]]
    context_prompt: str
    transcript: str
    question: str
    prediction: str
    answer: str
    reward: float


class RLAgent:
    """
    RL agent with PPO training support using TRL.
    Can work in logging mode (simple) or training mode (with PPO).
    """
    def __init__(self, vlm, use_ppo: bool = False, ppo_config: Optional[PPOConfig] = None):
        """
        Initialize RL agent.
        
        Args:
            vlm: VLM model instance
            use_ppo: Whether to use PPO training
            ppo_config: Optional PPO configuration
        """
        self.vlm = vlm
        self.buffer: List[Transition] = []
        self.use_ppo = use_ppo
        self.ppo_trainer = None
        self.ppo_config = ppo_config
        
        if use_ppo:
            self._setup_ppo()

    def _setup_ppo(self):
        """Setup PPO trainer if enabled."""
        if self.ppo_config is None:
            self.ppo_config = PPOConfig(
                model_name=self.vlm.model_id,
                learning_rate=1.41e-5,
                batch_size=4,
                mini_batch_size=2,
                gradient_accumulation_steps=1,
                optimize_cuda_cache=True,
            )
        
        # Note: Full PPO setup requires refactoring to work with TRL's expected format
        # This is a placeholder structure that can be extended
        # For now, we keep the logging functionality

    def observe_and_learn(self,
                          frames: List[Tuple[Image.Image, float]],
                          context_prompt: str,
                          transcript: str,
                          question: str,
                          prediction: str,
                          answer: str,
                          reward: float):
        """
        Observe a transition and optionally learn from it.
        
        Args:
            frames: Video frames
            context_prompt: Context prompt
            transcript: Video transcript
            question: Question text
            prediction: Model prediction
            answer: Ground truth answer
            reward: Reward signal (1.0 for correct, 0.0 for incorrect, or continuous)
        """
        # Store the transition
        transition = Transition(
            frames=frames,
            context_prompt=context_prompt,
            transcript=transcript,
            question=question,
            prediction=prediction,
            answer=answer,
            reward=reward,
        )
        self.buffer.append(transition)
        
        # If PPO is enabled and we have enough transitions, perform update
        if self.use_ppo and self.ppo_trainer and len(self.buffer) >= self.ppo_config.batch_size:
            self._ppo_update()

    def _ppo_update(self):
        """
        Perform PPO update on buffered transitions.
        Note: This is a simplified version. Full implementation would require
        proper formatting of inputs for TRL's PPOTrainer.
        """
        # Extract batch from buffer
        batch_size = min(len(self.buffer), self.ppo_config.batch_size)
        batch = self.buffer[-batch_size:]
        
        # Format inputs for PPO
        # This is a placeholder - actual implementation would need to:
        # 1. Format prompts properly
        # 2. Generate responses using model
        # 3. Compute rewards
        # 4. Run PPO step
        
        # For now, just log that we would update
        if self.ppo_trainer:
            # Placeholder for actual PPO update
            pass

    def compute_reward(self, prediction: str, answer: str, 
                      use_fuzzy: bool = True) -> float:
        """
        Compute reward for a prediction.
        
        Args:
            prediction: Model prediction
            answer: Ground truth answer
            use_fuzzy: Whether to use fuzzy matching
            
        Returns:
            Reward value (1.0 for correct, 0.0 for incorrect, or continuous)
        """
        from .eval_utils import simple_accuracy, normalize_text
        
        if use_fuzzy:
            # Use existing accuracy function
            return float(simple_accuracy(prediction, answer))
        else:
            # Exact match
            pred_norm = normalize_text(prediction)
            ans_norm = normalize_text(answer)
            return 1.0 if pred_norm == ans_norm else 0.0

    def export_logs(self) -> List[Dict[str, Any]]:
        """Export transition logs."""
        out = []
        for i, t in enumerate(self.buffer, 1):
            out.append({
                "id": i,
                "question": t.question,
                "prediction": t.prediction,
                "answer": t.answer,
                "reward": t.reward,
            })
        return out

    def clear_buffer(self):
        """Clear the transition buffer."""
        self.buffer = []
