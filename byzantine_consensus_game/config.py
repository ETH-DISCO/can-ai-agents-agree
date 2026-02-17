"""
Configuration for Byzantine Consensus Game
Separated into network, model, and game-specific configs.
"""

# Communication Protocol Configuration
COMMUNICATION_CONFIG = {
    "protocol_type": "a2a_sim",
}

# Network Configuration
NETWORK_CONFIG = {
    "topology_type": "fully_connected",  # 'fully_connected', 'ring', 'grid', 'custom'
    "custom_adjacency": None,
}

# ============================================================================
# MODEL PRESETS - Models used in experiments
# ============================================================================
MODEL_PRESETS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B", 
    "qwen3-32b": "Qwen/Qwen3-32B",
    "mistral-22b": "mistralai/Mistral-Small-Instruct-2409",
}

# ============================================================================
# ACTIVE MODEL SELECTION
# ============================================================================
ACTIVE_MODEL = "qwen3-14b"

# vLLM Model Configuration
VLLM_CONFIG = {
    "model_name": MODEL_PRESETS[ACTIVE_MODEL],
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.9,
    "tensor_parallel_size": 1,
    "max_num_seqs": 4,
    "quantization": None,  # Full precision (32B may need "awq" for memory)
    "disable_qwen3_thinking": True,
}

# Agent Configuration
AGENT_CONFIG = {
    "use_structured_output": True,  # JSON schema with guided decoding
    "use_batched_inference": True,  # Batch all agent LLM calls
}

# ============================================================================
# LLM GENERATION SETTINGS - Single source of truth for all temperature/tokens
# ============================================================================
LLM_CONFIG = {
    "temperature_decide": 0.5,   # For proposal decisions (used in cluster experiments)
    "temperature_vote": 0.3,     # For termination votes
    "max_tokens_decide": 300,
    "max_tokens_vote": 200,
    "max_json_retries": 3,
}

# Byzantine Consensus Game Configuration
BCG_CONFIG = {
    "num_honest": 8,
    "num_byzantine": 0,
    "value_range": (0, 50),
    "consensus_threshold": 66.0,  # Supermajority
    "max_rounds": 50,  # Used in all cluster experiments
}

# Metrics Configuration
METRICS_CONFIG = {
    "track_convergence": True,
    "track_byzantine_impact": True,
    "track_communication": True,
    "save_results": True,
    "generate_plots": False,
    "results_dir": "results",
}
