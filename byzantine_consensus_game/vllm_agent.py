"""
vLLM-based Agent Base Class
Provides a reusable foundation for LLM agents running on cluster with vLLM.
"""

import os
import sys
import warnings
import io
import logging
import json
from typing import Optional, Dict, Any, List, Tuple
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


# Suppress warnings and logging
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

# Suppress all vLLM logging
logging.getLogger('vllm').setLevel(logging.ERROR)
logging.getLogger('vllm.engine').setLevel(logging.ERROR)
logging.getLogger('vllm.worker').setLevel(logging.ERROR)
logging.getLogger('vllm.executor').setLevel(logging.ERROR)

VERBOSE = os.environ.get('VERBOSE', '0') == '1'


def _auto_configure_attention_backend():
    """Auto-detect GPU compute capability and set appropriate attention backend."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get compute capability of first GPU
            capability = torch.cuda.get_device_capability(0)
            compute_cap = capability[0] + capability[1] / 10.0
            
            # Flash Attention 2 requires compute capability >= 8.0
            if compute_cap < 8.0:
                os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
                if VERBOSE:
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"GPU: {gpu_name} (compute {compute_cap:.1f}) - using XFORMERS backend", flush=True)
            else:
                if VERBOSE:
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"GPU: {gpu_name} (compute {compute_cap:.1f}) - using Flash Attention 2", flush=True)
    except Exception as e:
        if VERBOSE:
            print(f"Could not auto-detect GPU, using vLLM defaults: {e}", flush=True)


class VLLMAgent:
    """
    Base agent class using vLLM for efficient LLM inference.
    All agents share a single model instance for efficiency.
    """
    
    # Shared model instance across all agents
    _shared_llm: Optional[LLM] = None
    _shared_model_name: Optional[str] = None
    _shared_model_config: Optional[Dict[str, Any]] = None
    
    def __init__(
        self,
        agent_id: str,
        model_name: str = "Qwen/Qwen3-8B",
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize vLLM agent.
        
        Args:
            agent_id: Unique identifier for this agent
            model_name: HuggingFace model path
            model_config: Model configuration (max_model_len, gpu_memory_utilization, etc.)
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.model_config = model_config or {
            'max_model_len': 4096,
            'gpu_memory_utilization': 0.85,
            'tensor_parallel_size': 1,
            'max_num_seqs': 64,
        }
        
        # Load model if not already loaded or if different model requested
        if (VLLMAgent._shared_llm is None or 
            VLLMAgent._shared_model_name != model_name or
            VLLMAgent._shared_model_config != self.model_config):
            self._load_model()
        
        self.llm = VLLMAgent._shared_llm
    
    def _load_model(self):
        """Load vLLM model (shared across all agents)."""
        if VERBOSE:
            print(f"Loading model {self.model_name}... (this may take 1-2 minutes)", flush=True)
        
        # Auto-configure attention backend based on GPU compute capability
        _auto_configure_attention_backend()
        
        try:
            # Set environment variables if specified in config (overrides auto-detection)
            env_vars = self.model_config.get('env_vars', {})
            for key, value in env_vars.items():
                os.environ[key] = value
            
            # Show progress only in verbose mode
            if VERBOSE:
                print("Initializing vLLM engine...", flush=True)
            
            # Suppress ALL output during model loading
            _stderr = sys.stderr
            _stdout = sys.stdout
            if not VERBOSE:
                sys.stderr = io.StringIO()
                sys.stdout = io.StringIO()
            
            # Build LLM arguments
            llm_args = {
                'model': self.model_name,
                'trust_remote_code': True,
                'max_model_len': self.model_config.get('max_model_len', 8192),
                'gpu_memory_utilization': self.model_config.get('gpu_memory_utilization', 0.85),
                'tensor_parallel_size': self.model_config.get('tensor_parallel_size', 1),
                'disable_log_stats': True,
            }
            
            # Add max_num_seqs if specified in config
            if 'max_num_seqs' in self.model_config:
                llm_args['max_num_seqs'] = self.model_config['max_num_seqs']
            
            # Only use distributed_executor_backend if tensor_parallel_size > 1
            # Single GPU doesn't need multiprocessing executor
            if self.model_config.get('tensor_parallel_size', 1) > 1:
                llm_args['distributed_executor_backend'] = 'mp'
            
            VLLMAgent._shared_llm = LLM(**llm_args)
            VLLMAgent._shared_model_name = self.model_name
            VLLMAgent._shared_model_config = self.model_config.copy()
            
            # Restore stdout/stderr
            if not VERBOSE:
                sys.stderr = _stderr
                sys.stdout = _stdout
            
            if VERBOSE:
                print("✓ Model loaded successfully", flush=True)
        except Exception as e:
            print(f"\n✗ Error loading model: {e}", flush=True)
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt using vLLM.
        
        Args:
            prompt: Input prompt (user message)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            system_prompt: Optional system prompt with static game rules/role info
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated text (stripped)
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Format as chat messages if system_prompt is provided
        if system_prompt:
            full_prompt = self._format_chat_prompt(system_prompt, prompt)
        else:
            full_prompt = prompt
        
        outputs = self.llm.generate([full_prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response
    
    def _format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format system and user prompts into a chat-style prompt.
        Auto-detects model family and uses appropriate template.
        
        Args:
            system_prompt: Static system instructions (game rules, role, etc.)
            user_prompt: Dynamic per-round information
            
        Returns:
            Formatted prompt string
        """
        model_lower = self.model_name.lower()
        
        # Check if Qwen3 thinking mode should be disabled (from config)
        disable_thinking = self.model_config.get('disable_qwen3_thinking', True)
        
        # Qwen3 models - disable thinking mode for structured JSON output
        if 'qwen3' in model_lower or 'qwen-3' in model_lower:
            # Qwen3 Instruct 2507 models don't have thinking mode - no /no_think needed
            is_instruct_2507 = 'instruct-2507' in model_lower or 'instruct_2507' in model_lower
            
            if is_instruct_2507:
                # Qwen3 Instruct 2507 - clean non-thinking model
                formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
                if VERBOSE:
                    print(f"Using Qwen3-Instruct-2507 template (no thinking mode)")
            elif disable_thinking:
                # Base Qwen3 with thinking disabled via /no_think soft switch
                formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt} /no_think<|im_end|>
<|im_start|>assistant
"""
                if VERBOSE:
                    print(f"Using Qwen3 template with /no_think (thinking disabled)")
            else:
                # Base Qwen3 with thinking enabled (will output <think>...</think>)
                formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
                if VERBOSE:
                    print(f"Using Qwen3 template with thinking enabled")
        # Qwen2.5 and other Qwen models (ChatML format)
        elif 'qwen' in model_lower:
            formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
            if VERBOSE:
                print(f"Using Qwen2.5 ChatML template")
        # Llama 3 models
        elif 'llama-3' in model_lower or 'llama3' in model_lower:
            formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            if VERBOSE:
                print(f"Using Llama3 template")
        # Other chat models (Mistral, etc.)
        elif 'llama' in model_lower or 'mistral' in model_lower:
            formatted = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""
            if VERBOSE:
                print(f"Using standard chat template")
        # ChatML format (default fallback - works with many models)
        else:
            formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
            if VERBOSE:
                print(f"Using default ChatML template")
        
        return formatted
    
    def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using vLLM's guided decoding.
        Guarantees output matches the provided JSON schema.
        
        Args:
            prompt: Input prompt (should instruct model to output JSON)
            schema: JSON schema dict - enforced by guided decoding
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt with static game rules/role info
            
        Returns:
            Parsed JSON dict, or dict with "error" key if generation fails
        """
        try:
            # Use guided decoding to enforce JSON schema
            guided_params = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                guided_decoding=guided_params,
            )
            
            # Format as chat messages if system_prompt is provided
            if system_prompt:
                full_prompt = self._format_chat_prompt(system_prompt, prompt)
            else:
                full_prompt = prompt
            
            outputs = self.llm.generate([full_prompt], sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            # Strip leading/trailing newlines that might appear before JSON
            response_text = response_text.strip()
            while response_text.startswith('\n'):
                response_text = response_text[1:].strip()
            
            # With guided decoding, the output should be valid JSON
            # But we still parse carefully in case of edge cases
            json_obj = None
            
            # Strategy 1: Direct parse (should work with guided decoding)
            try:
                json_obj = json.loads(response_text)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Find JSON by brace matching (fallback)
            if json_obj is None:
                brace_start = response_text.find("{")
                if brace_start >= 0:
                    brace_count = 0
                    for i in range(brace_start, len(response_text)):
                        if response_text[i] == "{":
                            brace_count += 1
                        elif response_text[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                try:
                                    json_obj = json.loads(response_text[brace_start:i+1])
                                    break
                                except (json.JSONDecodeError, ValueError):
                                    pass
            
            if json_obj is not None:
                return json_obj
            else:
                return {"error": "json_parse_failed", "message": "Could not extract valid JSON from model output", "raw": response_text[:200]}
        
        except json.JSONDecodeError as e:
            if VERBOSE:
                print(f"⚠️ JSON parsing failed: {e}")
                print(f"Raw output: {response_text[:200] if 'response_text' in locals() else 'N/A'}")
            return {"error": "json_parse_failed", "message": str(e)}
        except Exception as e:
            if VERBOSE:
                print(f"⚠️ JSON generation failed: {e}")
            return {"error": str(e), "message": "JSON generation failed"}
    
    def batch_generate_json(
        self,
        prompts: List[Tuple[str, str, Dict[str, Any]]],
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> List[Dict[str, Any]]:
        """
        Generate structured JSON output for multiple prompts in batch.
        Uses vLLM's guided decoding for each prompt with its specific schema.
        
        This is the key performance optimization - all agents' prompts are 
        processed in a single batched call to the LLM.
        
        Args:
            prompts: List of tuples (system_prompt, user_prompt, schema)
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
            
        Returns:
            List of parsed JSON dicts (or dicts with "error" key if parsing fails)
        """
        if not prompts:
            return []
        
        # For vLLM batched generation with guided decoding, we need to:
        # 1. All prompts must use the same schema (vLLM limitation for batched guided decoding)
        # 2. Format each prompt with chat template
        
        formatted_prompts = []
        schemas = []
        
        for system_prompt, user_prompt, schema in prompts:
            formatted = self._format_chat_prompt(system_prompt, user_prompt)
            formatted_prompts.append(formatted)
            schemas.append(schema)
        
        # Check if all schemas are identical (required for batched guided decoding)
        all_same_schema = all(s == schemas[0] for s in schemas)
        
        if all_same_schema:
            # Efficient: single batched call with guided decoding
            try:
                guided_params = GuidedDecodingParams(json=schemas[0])
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    guided_decoding=guided_params,
                )
                
                outputs = self.llm.generate(formatted_prompts, sampling_params)
                
                results = []
                for output in outputs:
                    response_text = output.outputs[0].text.strip()
                    try:
                        json_obj = json.loads(response_text)
                        results.append(json_obj)
                    except json.JSONDecodeError:
                        # Fallback: try brace matching
                        json_obj = self._extract_json(response_text)
                        results.append(json_obj if json_obj else {"error": "json_parse_failed", "raw": response_text[:100]})
                
                return results
                
            except Exception as e:
                if VERBOSE:
                    print(f"⚠️ Batched JSON generation failed: {e}")
                return [{"error": str(e)} for _ in prompts]
        else:
            # Fallback: individual calls (slower but handles different schemas)
            results = []
            for system_prompt, user_prompt, schema in prompts:
                result = self.generate_json(user_prompt, schema, temperature, max_tokens, system_prompt)
                results.append(result)
            return results
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text using brace matching."""
        brace_start = text.find("{")
        if brace_start >= 0:
            brace_count = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[brace_start:i+1])
                        except (json.JSONDecodeError, ValueError):
                            pass
        return None
    
    def batch_generate(
        self,
        prompts: list[str],
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 1.0,
        **kwargs
    ) -> list[str]:
        """
        Generate text for multiple prompts in batch (efficient).
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
            top_p: Nucleus sampling parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated texts
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
        return responses
    
    @classmethod
    def shutdown(cls):
        """Shutdown shared LLM instance and clean up GPU resources completely."""
        if cls._shared_llm is not None:
            try:
                # First, delete the LLM instance
                del cls._shared_llm
                cls._shared_llm = None
                cls._shared_model_name = None
                cls._shared_model_config = None
            except Exception:
                pass
        
        # Force garbage collection to release references
        import gc
        gc.collect()
        
        # Clean up PyTorch CUDA memory
        try:
            import torch
            if torch.cuda.is_available():
                # Synchronize all CUDA streams before cleanup
                torch.cuda.synchronize()
                # Empty the CUDA cache to free GPU memory
                torch.cuda.empty_cache()
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                if VERBOSE:
                    print("✓ CUDA memory cache cleared", flush=True)
        except Exception as e:
            if VERBOSE:
                print(f"Warning: CUDA cleanup failed: {e}", flush=True)
        
        # Clean up PyTorch distributed process groups
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        
        # Run garbage collection again after CUDA cleanup
        gc.collect()
        
        if VERBOSE:
            print("✓ vLLM engine shutdown complete", flush=True)
