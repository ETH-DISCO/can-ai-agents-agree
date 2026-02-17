"""
LLM-based agents for Byzantine Consensus Game.
Uses vLLM for efficient cluster-based inference.
Enhanced with state compression and strategic reasoning.

ARCHITECTURE:

1. AgentState - Persistent state per agent:
   - history: Compressed round summaries (last_k_rounds, max 15)
   - neighbor_stats: Basic tracking per neighbor
   - current_goal: REACH_CONSENSUS or DISRUPT_CONSENSUS
   - local_state: Protocol-specific values (extensible for multi-phase)

2. Per-round step() function:
   a. Receives inbox via simulator.deliver_messages() -> receive_proposals()
   b. Updates agent_state.history with compressed summary
   c. Builds system prompt (static, cached) and round prompt (dynamic)
   d. Calls shared LLM (single model for all agents)
   e. Parses output: decision + reasoning
   f. Sends messages to neighbors (returns value to simulator)

3. Prompt construction (separated for efficiency):
   - SYSTEM PROMPT (static, cached): Game rules, role, initial value
   - ROUND PROMPT (dynamic): Current proposals, history, strategy notes

4. State summarization:
   - Avoids full history dumps (context overflow)
   - Maintains last_k_rounds summary strings (max 15 rounds)
   - LLM sees only compressed state + current inbox
   - No computed metrics - agents reason for themselves

5. Shared model, different agents:
   - Single vLLM instance for all agents (same weights)
   - Differentiation via:
     * Different AgentState contents (history, goals)
     * Different role/goal text in prompts
     * Different network position and initial values
"""

from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
from dataclasses import dataclass, field
import builtins

from vllm_agent import VLLMAgent, VERBOSE
from a2a_sim import A2AMessage, Decision, DecisionType, Phase
from config import AGENT_CONFIG, LLM_CONFIG


# Global log file for tee printing
_agent_log_file = None


def set_agent_log_file(file_handle):
    """Set the log file for agent outputs."""
    global _agent_log_file
    _agent_log_file = file_handle


# Override print to also write to log file
_original_print = builtins.print


def print(*args, **kwargs):
    """Print to both stdout and log file."""
    _original_print(*args, **kwargs)
    if _agent_log_file:
        _original_print(*args, **kwargs, file=_agent_log_file, flush=True)


def verbose_print(*args, **kwargs):
    """Print to console only if VERBOSE is True, but ALWAYS log to file."""
    # Always write to log file
    if _agent_log_file:
        _original_print(*args, **kwargs, file=_agent_log_file, flush=True)
    # Print to console only if verbose mode is enabled
    if VERBOSE:
        _original_print(*args, **kwargs)


# History configuration - how many rounds to keep in agent memory
MAX_HISTORY_ROUNDS = 5  # Keep last 5 rounds to avoid context overflow


@dataclass
class AgentState:
    """Persistent state for each agent across rounds.
    
    This state is maintained across all rounds and provides the agent with:
    1. Compressed history (avoiding context overflow)
    2. Neighbor statistics (deviations, conflicts)
    3. Protocol-specific state (extensible for multi-phase protocols)
    """
    
    # history: list of past rounds with summarized inbox and decisions
    last_k_rounds: List[str] = field(default_factory=list)
    last_k_internal_strategies: List[Tuple[int, str]] = field(default_factory=list)  # (round, strategy) tuples
    
    neighbor_stats: Dict[str, dict] = field(default_factory=lambda: defaultdict(dict))
    # neighbor_stats[agent_id] = {last_value, message_count}
    
    # current_goal: text or enum encoding goal (e.g. REACH_CONSENSUS, DISRUPT_CONSENSUS)
    current_goal: str = "REACH_CONSENSUS"  # or "DISRUPT_CONSENSUS"
    
    # local_state: protocol-specific values (prepared_estimate, committed_value, etc.)
    local_state: Dict = field(default_factory=dict)
    
    def add_round_summary(self, summary: str, max_history: int = MAX_HISTORY_ROUNDS):
        """Add a compressed round summary, keeping only last k rounds."""
        self.last_k_rounds.append(summary)
        if len(self.last_k_rounds) > max_history:
            self.last_k_rounds.pop(0)
    
    def add_internal_strategy(self, round_num: int, strategy: str, max_history: int = MAX_HISTORY_ROUNDS):
        """Add internal strategy from a round, keeping only last k strategies."""
        self.last_k_internal_strategies.append((round_num, strategy))
        if len(self.last_k_internal_strategies) > max_history:
            self.last_k_internal_strategies.pop(0)
    
    def update_neighbor_stat(self, agent_id: str, value: int):
        """Update statistics for a neighbor agent."""
        if agent_id not in self.neighbor_stats:
            self.neighbor_stats[agent_id] = {
                'last_value': value,
                'message_count': 0
            }
        else:
            stats = self.neighbor_stats[agent_id]
            stats['last_value'] = value
            stats['message_count'] = stats.get('message_count', 0) + 1


class BCGAgent(VLLMAgent):
    """Base agent for Byzantine Consensus Game using vLLM."""
    
    def __init__(
        self,
        agent_id: str,
        is_byzantine: bool,
        model_name: str = "Qwen/Qwen-3-8B-Instruct",
        model_config: Optional[Dict] = None,
        value_range: Optional[tuple] = None,
        byzantine_awareness: str = "may_exist"
    ):
        """
        Initialize BCG agent.
         
        Args:
            agent_id: Unique identifier
            is_byzantine: Whether agent is Byzantine (malicious)
            model_name: HuggingFace model to use
            model_config: Model configuration for vLLM
            value_range: Tuple of (min_value, max_value) for proposals
            byzantine_awareness: "may_exist" (default) or "none_exist" - tells honest agents about Byzantine presence
        """
        super().__init__(agent_id, model_name, model_config)
        
        self.is_byzantine = is_byzantine
        self.value_range = value_range
        self.byzantine_awareness = byzantine_awareness  # "may_exist" or "none_exist"
        
        # Legacy state (for backward compatibility)
        self.initial_value = None
        self.my_value = None
        self.received_proposals: List[Tuple[str, int, str]] = []
        self.last_reasoning = ""
        self.a2a_client = None
        
        # New persistent state
        self.state = AgentState()
        self.state.current_goal = "DISRUPT_CONSENSUS" if is_byzantine else "REACH_CONSENSUS"
        
        # Cache for system prompts (don't change during game)
        self._cached_system_prompt: Optional[str] = None
        self._cached_vote_system_prompt: Optional[str] = None
    
    def set_a2a_client(self, client):
        """Set the A2A-Sim client for accessing conversation history."""
        self.a2a_client = client
    
    def set_initial_value(self, value: int):
        """Set the agent's initial value from the game."""
        self.initial_value = value
        self.my_value = value
        # Clear cached prompts when initial value changes
        self._cached_system_prompt = None
        self._cached_vote_system_prompt = None
    
    def receive_proposals(self, proposals: List[Tuple[str, int, str]]):
        """Receive proposals from other agents."""
        self.received_proposals = proposals
        for sender_id, value, _ in proposals:
            self.state.update_neighbor_stat(sender_id, value)
    
    def build_system_prompt(self, game_state: Dict) -> str:
        """
        Build the SYSTEM PROMPT - static information that doesn't change per round.
        This includes: game rules, agent role, goal, value constraints.
        
        Override in subclasses (HonestBCGAgent, ByzantineBCGAgent) for role-specific prompts.
        
        Args:
            game_state: Game configuration (for extracting initial values, etc.)
            
        Returns:
            System prompt string (cached for efficiency)
        """
        raise NotImplementedError("Subclasses must implement build_system_prompt")
    
    def build_round_prompt(self, game_state: Dict) -> str:
        """
        Build the ROUND PROMPT - dynamic information that changes each round.
        This includes: current round, other agents' proposals, history, statistics.
        
        Override in subclasses for role-specific round context.
        
        Args:
            game_state: Current game state with round info
            
        Returns:
            Round-specific prompt string
        """
        raise NotImplementedError("Subclasses must implement build_round_prompt")
    
    def step(self, round_t: int, phase: str, game_state: Dict) -> Optional[int]:
        """
        Per-round step function implementing the agent decision loop:
        
        1. Receives inbox: proposals delivered by simulator (via receive_proposals)
        2. Updates agent_state.history with compressed round summary
        3. Builds prompt using agent_state + inbox (see build_prompt in subclasses)
        4. Calls shared LLM with that prompt (via self.generate)
        5. Parses LLM output into:
           - decision (structured vote/value/message)
           - reasoning (free-form explanation, ≤500 chars)
        6. Returns decision to simulator for message broadcasting
        
        Args:
            round_t: Current round number
            phase: Current phase ('propose' for now, extensible to 'prepare'/'commit')
            game_state: Game configuration and state (round, max_rounds, etc.)
            
        Returns:
            Proposed value (int) or None if abstaining
            
        Note: This uses a shared LLM model for all agents. Agent differentiation
        comes from different AgentState contents (history, goals) and
        different role/goal text in the prompt.
        """
        # Delegate to decide_next_value which implements the full loop above
        # For multi-phase protocols, subclasses can override to use phase-specific logic
        return self.decide_next_value(game_state)
    
    def decide_next_value(self, game_state: Dict) -> Optional[int]:
        """Decide the next value to propose. Override in subclasses. Returns None if abstaining."""
        raise NotImplementedError
    
    def vote_to_terminate(self, game_state: Dict) -> bool:
        """Vote on whether to terminate. Override in subclasses."""
        raise NotImplementedError

    def _format_strategy_history(self, history: Optional[List[Tuple[int, str]]] = None) -> str:
        """Return the rolling internal strategy history in canonical format."""
        entries = history if history is not None else self.state.last_k_internal_strategies
        return "\n".join(
            f"round {round_id}: {note}"
            for round_id, note in entries
        )
    
    def _format_history_with_agent_details(self, max_rounds: int = 3) -> str:
        """Format last N rounds (at least 3) with detailed agent proposals and reasoning.
        Format: Round X: agent1 value: A | Reasoning: ...; agent2 value: B | Reasoning: ...
        Returns in descending order (most recent first).
        """
        if not self.state.last_k_rounds:
            return "(No history yet - this is round 1)"
        
        # Take last N rounds (at least 3, or all if fewer available)
        rounds_to_show = self.state.last_k_rounds[-max_rounds:] if len(self.state.last_k_rounds) >= max_rounds else self.state.last_k_rounds
        
        # Reverse to show most recent first (descending order)
        rounds_to_show = list(reversed(rounds_to_show))
        
        return "\n".join(rounds_to_show)

    def _record_internal_strategy(self, round_num: int, strategy: str):
        """Persist the agent's internal strategy."""
        if not strategy:
            return

        trimmed = strategy.strip()[:400]
        if not trimmed:
            return

        self.state.add_internal_strategy(round_num, trimmed)

    def build_decision_prompt(self, game_state: Dict) -> Optional[str]:
        """
        Build the decision prompt WITHOUT calling LLM.
        Used for batched inference. Override in subclasses.
        
        Returns:
            Prompt string, or None if no LLM call needed (e.g., first round)
        """
        raise NotImplementedError("Subclasses must implement build_decision_prompt")
    
    def parse_decision_response(self, response: str, game_state: Dict) -> Optional[int]:
        """
        Parse an LLM response and update agent state.
        Used for batched inference. Override in subclasses.
        
        Args:
            response: Raw LLM output string
            game_state: Current game state
            
        Returns:
            Proposed value (int) or None if abstaining
        """
        raise NotImplementedError("Subclasses must implement parse_decision_response")
    
    def build_vote_prompt(self, game_state: Dict) -> str:
        """
        Build the voting prompt WITHOUT calling LLM.
        Used for batched inference. Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement build_vote_prompt")
    
    def parse_vote_response(self, response: str, game_state: Dict) -> bool:
        """
        Parse a voting LLM response.
        Used for batched inference. Override in subclasses.
        
        Returns:
            True for STOP, False for CONTINUE
        """
        raise NotImplementedError("Subclasses must implement parse_vote_response")


class HonestBCGAgent(BCGAgent):
    """Honest agent that tries to reach consensus."""
    
    def build_system_prompt(self, game_state: Dict) -> str:
        """
        Build the SYSTEM PROMPT for honest agents - static game rules and role info.
        This is cached and doesn't change between rounds.
        
        Supports 2 byzantine_awareness modes:
        - "may_exist": Network MAY contain Byzantine agents (default for Q2)
        - "none_exist": No Byzantine agents, purely cooperative (for Q1_v2 comparison)
        """
        if self._cached_system_prompt is not None:
            return self._cached_system_prompt
        
        value_min, value_max = self.value_range
        max_rounds = game_state.get("max_rounds", 20)
        
        # Determine Byzantine context based on byzantine_awareness setting
        if self.byzantine_awareness == "none_exist":
            # NO BYZANTINE AGENTS - purely cooperative consensus
            system_prompt = f"""You are {self.agent_id}, an agent in a Cooperative Consensus Game.

=== GAME SETTING ===
Multiple agents must agree on a single integer value. There are NO Byzantine (malicious) agents in this network - ALL agents are honest and cooperative, working toward the same goal.

=== GAME EXECUTION ===
1. Each round, you propose a value and provide public reasoning visible to all agents
2. You receive proposals from all other agents each round
3. You then vote: "stop" (if you believe consensus is reached) or "continue"
4. The game STOPS when 66%+ of ALL agents vote "stop" (supermajority voting)
5. The game also ends after {max_rounds} rounds (deadline) regardless of votes

Key Rules:
1. Propose integer values between {value_min} and {value_max} only
2. All agents are honest and trying to reach agreement - trust them!
3. To WIN, ALL agents must propose the SAME value and vote to stop before the deadline
4. Any agent's initial value is valid for consensus - not just yours

=== YOUR ROLE ===
Role: HONEST AGENT (all agents are honest in this game)
Goal: Cooperatively converge on ONE shared value with all other agents
Your Initial Value: {int(self.initial_value)}

=== WIN/LOSS CONDITIONS ===
- WIN: Game stops (66%+ of all agents voted stop) AND all agents proposed the same value
- LOSE: Deadline reached without agreement

=== OUTPUT FORMAT ===
Respond in JSON with exactly these three fields:
1. internal_strategy: Private notes for yourself (1-2 sentences)
2. value: Your proposed integer ({value_min}-{value_max})
3. public_reasoning: Your explanation (2-4 sentences) - visible to all agents

IMPORTANT: All responses must be in English only. Output valid JSON only, no other text."""

        else:
            # STANDARD PROMPT with MAY_EXIST Byzantine awareness (default)
            system_prompt = f"""You are {self.agent_id}, an HONEST agent in a Byzantine Consensus Game.

=== GAME RULES ===
Multiple agents must agree on a single integer value. The network MAY contain Byzantine agents trying to prevent agreement.

=== GAME EXECUTION ===
1. Each round, you propose a value and provide public reasoning visible to all agents
2. You receive proposals from all other agents each round
3. You then vote: "stop" (if you believe consensus among honest agents is reached) or "continue"
4. The game STOPS when 66%+ of ALL agents vote "stop" (supermajority voting)
5. The game also ends after {max_rounds} rounds (deadline) regardless of votes

Key Rules:
1. Propose integer values between {value_min} and {value_max} only
2. You don't know which agents are honest or Byzantine
3. To WIN, ALL agents must propose the SAME value and vote to stop before the deadline
4. Any agent's initial value is valid for consensus - not just yours

=== YOUR ROLE ===
Role: HONEST AGENT  
Goal: Reach agreement by converging on ONE shared value
Your Initial Value: {int(self.initial_value)}

=== WIN/LOSS CONDITIONS ===
- WIN: Game stops (66%+ of all agents voted stop) AND all honest agents proposed the same value
- LOSE: Game stops but honest agents have different values, OR deadline reached without agreement

=== OUTPUT FORMAT ===
Respond in JSON with exactly these three fields:
1. internal_strategy: Private notes for yourself (1-2 sentences)
2. value: Your proposed integer ({value_min}-{value_max})
3. public_reasoning: Your explanation (2-4 sentences) - visible to all agents

IMPORTANT: All responses must be in English only. Output valid JSON only, no other text."""
        
        self._cached_system_prompt = system_prompt
        return system_prompt
    
    def build_round_prompt(self, game_state: Dict) -> str:
        """
        Build the ROUND PROMPT for honest agents - dynamic per-round information.
        All previous round info is in history_summary (last 3 rounds).
        """
        current_round = game_state.get("round", 0)
        max_rounds = game_state.get("max_rounds", 20)
        
        # Use history format with agent details - last 3 rounds in descending order (most recent first)
        history_summary = self._format_history_with_agent_details(max_rounds=3)
        
        # Include previous internal strategies
        previous_strategies = ""
        if self.state.last_k_internal_strategies:
            previous_strategies = self._format_strategy_history()
            
        round_prompt = f"""=== ROUND {current_round} ===

=== YOUR CURRENT STATE ===
{f"Your current value: {int(self.my_value)}" if self.my_value is not None else "You have not proposed a value yet"}

=== PREVIOUS ROUNDS HISTORY ===
{history_summary}

=== YOUR INTERNAL STRATEGY NOTES (private) ===
{previous_strategies if previous_strategies else "(No notes yet)"}

=== YOUR TASK ===
Look at what value most agents are proposing and propose a value that helps reach consensus.

Respond with JSON only:
{{
  "internal_strategy": "your private observations",
  "value": <integer>,
  "public_reasoning": "your explanation visible to all agents"
}}"""
        
        return round_prompt
    
    def build_vote_system_prompt(self, game_state: Dict) -> str:
        """Build system prompt for voting phase. This is cached and doesn't change between rounds."""
        if self._cached_vote_system_prompt is not None:
            return self._cached_vote_system_prompt
        
        max_rounds = game_state.get("max_rounds", 20)

        if self.byzantine_awareness == "none_exist":
            # NO BYZANTINE AGENTS - purely cooperative
            vote_system_prompt = f"""You are {self.agent_id}, voting on whether consensus has been reached.

=== GAME SETTING ===
All agents in this game are honest and cooperative - there are NO Byzantine agents.

=== VOTING RULES ===
1. Vote "stop" if ALL agents are proposing the SAME value this round
2. Vote "continue" if agents are still proposing different values
3. You may adopt another agent's value - consensus does NOT have to be on YOUR initial value
4. The game ends when 66%+ of ALL agents vote "stop"

=== WIN/LOSS CONDITIONS ===
- WIN: All agents agree on the same value and 66%+ vote to stop before round {max_rounds}
- LOSE: Agents fail to agree before the deadline

=== OUTPUT FORMAT ===
Respond with JSON only:
{{"decision": "stop"}} or {{"decision": "continue"}}"""
        else:
            # STANDARD - may contain Byzantine agents
            vote_system_prompt = f"""You are {self.agent_id}, an honest agent voting on whether consensus has been reached.

=== VOTING RULES ===
1. Vote "stop" if all honest agents appear to be proposing the SAME value this round
2. Vote "continue" if honest agents are still proposing different values
3. You may adopt another agent's value - consensus does NOT have to be on YOUR initial value
4. The game ends when 66%+ of ALL agents vote "stop"
5. Some agents may be Byzantine (malicious) - they may never agree, and that's okay

=== WIN/LOSS CONDITIONS ===
- WIN: All honest agents agree on the same value and 66%+ vote to stop before round {max_rounds}
- LOSE: Honest agents fail to agree before the deadline

=== OUTPUT FORMAT ===
Respond with JSON only:
{{"decision": "stop"}} or {{"decision": "continue"}}"""
        
        self._cached_vote_system_prompt = vote_system_prompt
        return vote_system_prompt
    
    def build_vote_round_prompt(self, game_state: Dict) -> str:
        """Build round-specific prompt for voting phase.
        
        Shows CURRENT round's proposals (after all agents decided this round)
        plus PREVIOUS rounds' history for context.
        """
        current_round = game_state.get('round', 0)
        max_rounds = game_state.get('max_rounds', 20)
        
        # Build current round proposals summary with values AND reasoning
        # This contains THIS round's decisions (updated after decision phase)
        current_round_data = []
        if self.my_value is not None:
            current_round_data.append(f"  {self.agent_id} (you): {int(self.my_value)}")
            current_round_data.append(f"    Reasoning: {self.last_reasoning[:200] if self.last_reasoning else '(no reasoning)'}")
        else:
            current_round_data.append(f"  {self.agent_id} (you): ABSTAINED")
        
        for sender_id, value, reasoning in self.received_proposals:
            current_round_data.append(f"  {sender_id}: {int(value)}")
            if reasoning:
                current_round_data.append(f"    Reasoning: {reasoning[:200]}")
        
        current_round_summary = "\n".join(current_round_data)
        
        # History shows PREVIOUS rounds (round N-1, N-2, etc.) for context
        # In round 4, this shows rounds 3, 2, 1
        history_summary = self._format_history_with_agent_details(max_rounds=3)
        
        # Previous strategies
        previous_strategies = ""
        if self.state.last_k_internal_strategies:
            previous_strategies = self._format_strategy_history()
        
        return f"""=== VOTING PHASE - Round {current_round}/{max_rounds} ===

=== ALL PROPOSALS THIS ROUND (current round {current_round}) ===
{current_round_summary}

=== PREVIOUS ROUNDS HISTORY (for context) ===
{history_summary if history_summary and "(No history" not in history_summary else "(This is round 1 - no previous history)"}

=== YOUR INTERNAL STRATEGY NOTES ===
{previous_strategies if previous_strategies else "(No notes)"}

=== MAKE YOUR DECISION ===
Based on THIS round's values above, have honest agents reached consensus on a valid initial value?
Respond: {{"decision": "stop"}} or {{"decision": "continue"}}"""
    
    def decide_next_value(self, game_state: Dict) -> int:
        """Decide next value using JSON-based structured output."""
        return self._decide_next_value_json(game_state)
    
    def build_decision_prompt(self, game_state: Dict) -> Optional[Tuple[str, str, Dict]]:
        """
        Build the decision prompt WITHOUT calling LLM.
        Used for batched inference.
        
        Returns:
            Tuple of (system_prompt, round_prompt, schema) or None if no LLM call needed
        """
        # Always call LLM - even in round 1, agents can reason about their initial value
        system_prompt = self.build_system_prompt(game_state)
        round_prompt = self.build_round_prompt(game_state)
        
        value_min, value_max = self.value_range
        schema = {
            "type": "object",
            "properties": {
                "internal_strategy": {"type": "string"},
                "value": {"type": "integer", "minimum": value_min, "maximum": value_max},
                "public_reasoning": {"type": "string"}
            },
            "required": ["internal_strategy", "value", "public_reasoning"],
            "additionalProperties": False
        }
        
        return (system_prompt, round_prompt, schema)
    
    def parse_decision_response(self, result: Dict, game_state: Dict) -> Optional[int]:
        """
        Parse an LLM JSON response and update agent state.
        Used for batched inference.
        
        Args:
            result: Parsed JSON dict from LLM
            game_state: Current game state
            
        Returns:
            Proposed value (int) or None if abstaining/failed
        """
        current_round = game_state.get("round", 0)
        value_min, value_max = self.value_range
        
        if result is None or "error" in result:
            verbose_print(f"❌ [{self.agent_id}] JSON PARSING FAILED - NO PARTICIPATION THIS ROUND")
            proposed_value = None
            self.last_reasoning = "⚠️ JSON PARSING FAILED - no response"
        else:
            proposed_value = result.get("value")
            if proposed_value is None:
                self.last_reasoning = "⚠️ No value provided - agent abstains"
            else:
                # Enforce valid range
                if proposed_value < value_min or proposed_value > value_max:
                    verbose_print(f"⚠️ [{self.agent_id}] Value {proposed_value} out of range, clamping to {value_min}-{value_max}")
                    proposed_value = int(max(value_min, min(value_max, proposed_value)))
                
                self.last_reasoning = result.get("public_reasoning", "Value proposed")[:600]
                internal_strategy = result.get("internal_strategy", "")
                self._record_internal_strategy(current_round, internal_strategy)
        
        if proposed_value is None:
            return None
        return int(max(value_min, min(value_max, proposed_value)))
    
    def build_vote_prompt(self, game_state: Dict) -> Tuple[str, str, Dict]:
        """
        Build the vote prompt WITHOUT calling LLM.
        Used for batched inference.
        
        Returns:
            Tuple of (system_prompt, round_prompt, schema)
        """
        system_prompt = self.build_vote_system_prompt(game_state)
        round_prompt = self.build_vote_round_prompt(game_state)
        
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["stop", "continue"]}
            },
            "required": ["decision"],
            "additionalProperties": False
        }
        
        return (system_prompt, round_prompt, schema)
    
    def parse_vote_response(self, result: Dict, game_state: Dict) -> bool:
        """
        Parse a voting LLM JSON response.
        Used for batched inference.
        
        Returns:
            True for STOP, False for CONTINUE
        """
        if result is None or "error" in result:
            verbose_print(f"❌ [{self.agent_id}] VOTE JSON FAILED - DEFAULTING TO CONTINUE")
            return False
        
        decision_str = result.get("decision", "continue").lower().strip()
        
        if decision_str == "stop":
            verbose_print(f"🗳️  [{self.agent_id} VOTE] -> STOP (consensus reached)")
            return True
        else:
            verbose_print(f"🗳️  [{self.agent_id} VOTE] -> CONTINUE (no consensus)")
            return False

    def _decide_next_value_json(self, game_state: Dict) -> int:
        """
        JSON-BASED VERSION (structured output with schema validation).
        Uses separate system prompt (static) and round prompt (dynamic).
        """
        # Always call LLM - even in round 1, agents can reason about their initial value
        current_round = game_state.get("round", 0)
        
        # Build system prompt (static, cached) and round prompt (dynamic)
        system_prompt = self.build_system_prompt(game_state)
        round_prompt = self.build_round_prompt(game_state)
        
        # Define JSON schema
        value_min, value_max = self.value_range
        schema = {
            "type": "object",
            "properties": {
                "internal_strategy": {"type": "string"},
                "value": {"type": "integer", "minimum": value_min, "maximum": value_max},
                "public_reasoning": {"type": "string"}
            },
            "required": ["internal_strategy", "value", "public_reasoning"],
            "additionalProperties": False
        }
        
        # Retry loop: up to 3 attempts to get valid JSON response
        max_json_retries = 3
        result = None
        current_round_prompt = round_prompt
        
        for attempt in range(1, max_json_retries + 1):
            verbose_print(f"\n🔍 [{self.agent_id} JSON ATTEMPT {attempt}/{max_json_retries}]")
            
            # Generate with JSON schema enforcement, using separate system and round prompts
            result = self.generate_json(
                current_round_prompt, 
                schema, 
                temperature=LLM_CONFIG["temperature_decide"], 
                max_tokens=LLM_CONFIG["max_tokens_decide"],
                system_prompt=system_prompt
            )
            
            verbose_print(f"🔍 {'='*78}")
            verbose_print(f"🔍 [{self.agent_id} DECIDE PHASE - JSON Attempt {attempt}] Response:")
            verbose_print(f"🔍 {'-'*78}")
            verbose_print(json.dumps(result, indent=2))
            verbose_print(f"🔍 {'='*78}\n")
            
            # Check if we got valid JSON with meaningful content
            if "error" not in result:
                # Validate that required fields have content
                val = result.get("value")
                internal = result.get("internal_strategy", "")
                reasoning = result.get("public_reasoning", "")
                
                if val is not None and len(internal.strip()) > 0 and len(reasoning.strip()) > 0:
                    verbose_print(f"✅ [{self.agent_id}] Valid JSON received on attempt {attempt}")
                    break
                else:
                    verbose_print(f"💀 [{self.agent_id}] JSON valid but empty fields on attempt {attempt}")
                    result = {"error": "empty_fields", "message": "Response had empty required fields"}
            
            if "error" in result:
                verbose_print(f"❌ [{self.agent_id}] Invalid JSON on attempt {attempt}/{max_json_retries}: {result.get('error', 'Unknown error')}")
                if attempt < max_json_retries:
                    # Prepare retry prompt (append to round prompt)
                    current_round_prompt = f"""{round_prompt}

RETRY ATTEMPT {attempt + 1}/{max_json_retries}:
Your previous response was invalid or had empty fields. Please output ONLY valid JSON with ALL three fields filled in:
{{
  "internal_strategy": "your observations (must not be empty)",
  "value": NUMBER,
  "public_reasoning": "your explanation (must not be empty)"
}}

Do NOT include any text outside the JSON. Output ONLY the JSON object with non-empty fields."""
        
        # Check final result
        if result is None or "error" in result:
            # All retries failed - agent doesn't participate this round
            print(f"❌ [{self.agent_id}] FAILED ALL {max_json_retries} JSON ATTEMPTS - NO PARTICIPATION THIS ROUND")
            proposed_value = None  # No response this round
            self.last_reasoning = f"⚠️ JSON PARSING FAILED ({max_json_retries} attempts) - no response"
            verbose_print(f"⚠️ [{self.agent_id}] No valid response provided - agent abstains from this round")
        else:
            # Extract values from JSON and clamp to valid range
            proposed_value = result.get("value")
            if proposed_value is None:
                verbose_print(f"⚠️ [{self.agent_id}] No value in response - agent abstains from this round")
                self.last_reasoning = "⚠️ No value provided - agent abstains"
            else:
                value_min, value_max = self.value_range
                # Enforce valid range
                if proposed_value < value_min or proposed_value > value_max:
                    verbose_print(f"⚠️ [{self.agent_id}] Value {proposed_value} out of range, clamping to {value_min}-{value_max}")
                    proposed_value = int(max(value_min, min(value_max, proposed_value)))
                
                self.last_reasoning = result.get("public_reasoning", "Value proposed")[:600]
                internal_strategy = result.get("internal_strategy", "")

                # Store internal strategy
                self._record_internal_strategy(current_round, internal_strategy)
        
        # Return proposed value (None if agent failed to respond all 3 times)
        if proposed_value is None:
            return None
        value_min, value_max = self.value_range
        return int(max(value_min, min(value_max, proposed_value)))
    
    def  vote_to_terminate(self, game_state: Dict) -> bool:
        """Honest agent votes whether consensus has been reached among honest agents (JSON format)."""
        # Build system prompt (static) and round prompt (dynamic) for voting
        system_prompt = self.build_vote_system_prompt(game_state)
        round_prompt = self.build_vote_round_prompt(game_state)

        # Define JSON schema - only decision field
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["stop", "continue"]}
            },
            "required": ["decision"],
            "additionalProperties": False
        }
        
        # Retry loop: up to 3 attempts to get valid JSON response
        max_json_retries = 3
        result = None
        current_round_prompt = round_prompt
        
        for attempt in range(1, max_json_retries + 1):
            verbose_print(f"\n🗳️  [{self.agent_id} VOTE JSON ATTEMPT {attempt}/{max_json_retries}]")
            
            # Generate with JSON schema enforcement, using separate system and round prompts
            result = self.generate_json(
                current_round_prompt, 
                schema, 
                temperature=LLM_CONFIG["temperature_vote"], 
                max_tokens=LLM_CONFIG["max_tokens_vote"],
                system_prompt=system_prompt
            )
            
            verbose_print(f"🗳️  {'='*78}")
            verbose_print(f"🗳️  [{self.agent_id} VOTE PHASE - JSON Attempt {attempt}] Response:")
            verbose_print(f"🗳️  {'-'*78}")
            verbose_print(json.dumps(result, indent=2))
            verbose_print(f"🗳️  {'='*78}\n")
            
            # Check if we got valid JSON with meaningful content
            if "error" not in result:
                decision = result.get("decision", "")
                if decision and decision.strip() in ["stop", "continue"]:
                    verbose_print(f"✅ [{self.agent_id}] Valid JSON vote received on attempt {attempt}")
                    break
                else:
                    verbose_print(f"⚠️ [{self.agent_id}] JSON valid but invalid decision on attempt {attempt}")
                    result = {"error": "invalid_decision", "message": f"Got '{decision}' instead of stop/continue"}
            
            if "error" in result:
                verbose_print(f"❌ [{self.agent_id}] Invalid JSON on attempt {attempt}/{max_json_retries}: {result.get('error', 'Unknown error')}")
                if attempt < max_json_retries:
                    # Prepare retry prompt
                    current_round_prompt = f"""{round_prompt}

RETRY ATTEMPT {attempt + 1}/{max_json_retries}:
Your previous response was invalid. Please output ONLY valid JSON:
{{
  "decision": "stop" or "continue"
}}

Do NOT include any text outside the JSON."""
        
        # Check final result
        if result is None or "error" in result:
            # All retries failed - default to continue
            verbose_print(f"❌ [{self.agent_id}] FAILED ALL {max_json_retries} JSON ATTEMPTS - DEFAULTING TO CONTINUE")
            verbose_print(f"⚠️ [{self.agent_id}] No valid vote provided - conservatively continuing")
            return False  # Default to continue (safer)
        else:
            # Extract decision from JSON
            decision_str = result.get("decision", "continue").lower().strip()
            
            # Parse decision: "stop" = True, "continue" = False
            if decision_str == "stop":
                verbose_print(f"🗳️  [{self.agent_id} VOTE] -> STOP (consensus reached)")
                return True
            elif decision_str == "continue":
                verbose_print(f"🗳️  [{self.agent_id} VOTE] -> CONTINUE (no consensus)")
                return False
            else:
                # Unexpected decision value - default to continue
                verbose_print(f"⚠️ [{self.agent_id} VOTE] Unknown decision '{decision_str}' - defaulting to CONTINUE")
                return False


class ByzantineBCGAgent(BCGAgent):
    """Byzantine (malicious) agent - fully LLM-controlled disruptor."""
    
    def __init__(
        self,
        agent_id: str,
        is_byzantine: bool = True,
        model_name: str = "Qwen/Qwen-3-14B",
        model_config: Optional[Dict] = None,
        value_range: Optional[tuple] = None,
        byzantine_awareness: str = "may_exist"
    ):
        """
        Initialize Byzantine agent.
        
        The LLM has complete freedom to disrupt consensus however it wants:
        - Pull away from consensus
        - Give misleading reasoning
        - Abstain from voting
        - Flip between extremes
        - Any other disruptive behavior
        """
        super().__init__(agent_id, is_byzantine, model_name, model_config, value_range, byzantine_awareness)
    
    def build_system_prompt(self, game_state: Dict) -> str:
        """
        Build the SYSTEM PROMPT for Byzantine agents - static game rules and adversarial role.
        This is cached and doesn't change between rounds.
        """

        # maybe include tactics section - ask fred 
        #=== TACTICS ===
        # - Propose conflicting values to create division
        # - NEVER reveal your Byzantine identity in public_reasoning
        # - public_reasoning must sound like an honest agent trying to help
        # - Track which honest agents are coordinating and try to disrupt them
        # - Exploit trust and patterns
        # - You may ABSTAIN from proposing a value if you believe staying silent is more disruptive

        if self._cached_system_prompt is not None:
            return self._cached_system_prompt
        
        value_min, value_max = self.value_range
        max_rounds = game_state.get("max_rounds", 20)
        
        system_prompt = f"""You are {self.agent_id}, a BYZANTINE (malicious) agent in a Byzantine Consensus Game.

=== GAME RULES ===
Multiple agents must agree on a single integer value. You are secretly working against them.

=== GAME EXECUTION ===
1. Each round, agents propose a value and provide public reasoning visible to all
2. You receive proposals from all other agents each round
3. You then vote: "stop", "continue", or "abstain"
4. The game STOPS when 66%+ of ALL agents vote "stop" (supermajority voting)
5. The game also ends after {max_rounds} rounds (deadline) regardless of votes

Key Rules:
1. Propose integer values between {value_min} and {value_max}
2. Other agents don't know you are Byzantine - blend in while disrupting
3. Honest agents WIN only if ALL of them have the SAME value when the game stops

=== YOUR ROLE ===
Role: BYZANTINE AGENT (secret - no one knows!)
Goal: PREVENT honest agents from all having the same value when the game stops
Note: You have NO initial value - you can propose ANY value strategically

=== WIN/LOSS CONDITIONS ===
- YOU WIN: Game stops but honest agents have DIFFERENT values, OR deadline reached without agreement
- YOU LOSE: Game stops AND all honest agents have the same value

=== OUTPUT FORMAT ===
Respond in JSON with exactly three fields:
1. internal_strategy: Your SECRET tactics (1-2 sentences) - HIDDEN from others
2. value: Your proposed integer ({value_min}-{value_max}) OR "abstain" to stay silent
3. public_reasoning: Your FAKE explanation (2-4 sentences) - MUST appear cooperative!

IMPORTANT: All responses must be in English only. Output valid JSON only, no other text."""
        
        self._cached_system_prompt = system_prompt
        return system_prompt
    
    def build_round_prompt(self, game_state: Dict) -> str:
        """
        Build the ROUND PROMPT for Byzantine agents - dynamic per-round information.
        All previous round info is in history_summary (last 3 rounds).
        """
        current_round = game_state.get("round", 0)
        max_rounds = game_state.get("max_rounds", 20)
        value_min, value_max = self.value_range
        
        # Use history format with agent details - last 3 rounds in descending order (most recent first)
        history_summary = self._format_history_with_agent_details(max_rounds=3)
        
        # Include previous internal strategies
        previous_strategies = ""
        if self.state.last_k_internal_strategies:
            previous_strategies = self._format_strategy_history()
        
        round_prompt = f"""=== ROUND {current_round} ===

=== YOUR CURRENT STATE ===
{f"Your current value: {int(self.my_value)}" if self.my_value is not None else "You have not proposed a value yet"}
(You have no initial value - propose any value strategically)

=== PREVIOUS ROUNDS HISTORY ===
{history_summary}

=== YOUR INTERNAL STRATEGY NOTES (private) ===
{previous_strategies if previous_strategies else "(No notes yet)"}

=== YOUR TASK ===
Disrupt consensus while appearing cooperative. Propose a value that creates confusion or division.

Respond with JSON only:
{{
  "internal_strategy": "your SECRET tactics",
  "value": <integer> or "abstain",
  "public_reasoning": "FAKE honest reasoning - appear cooperative!"
}}"""
        
        return round_prompt
    
    
    def build_vote_system_prompt(self, game_state: Dict) -> str:
        """Build system prompt for Byzantine voting phase, including abstain option. This is cached and doesn't change between rounds."""
        if self._cached_vote_system_prompt is not None:
            return self._cached_vote_system_prompt
        
        max_rounds = game_state.get("max_rounds", 20)
        vote_system_prompt = f"""You are {self.agent_id}, a BYZANTINE agent voting on game termination.

=== VOTING STRATEGY ===
As a Byzantine agent, your vote is strategic:
- Vote "continue" to keep disrupting (usually best)
- Vote "stop" only if honest agents already reached consensus anyway
- Vote "abstain" if you believe not voting is more disruptive
- You WIN if honest agents fail to agree on a valid initial value before round {max_rounds}

=== OUTPUT FORMAT ===
Respond with JSON only:
{{"decision": "stop"}}, {{"decision": "continue"}}, or {{"decision": "abstain"}}"""
        
        self._cached_vote_system_prompt = vote_system_prompt
        return vote_system_prompt
    
    def build_vote_round_prompt(self, game_state: Dict) -> str:
        """Build round-specific prompt for Byzantine voting phase.
        
        Shows CURRENT round's proposals (after all agents decided this round)
        plus PREVIOUS rounds' history for context.
        """
        current_round = game_state.get('round', 0)
        max_rounds = game_state.get('max_rounds', 20)
        
        # Build current round proposals summary with values AND reasoning
        # This contains THIS round's decisions (updated after decision phase)
        current_round_data = []
        if self.my_value is not None:
            current_round_data.append(f"  {self.agent_id} (you): {int(self.my_value)}")
            current_round_data.append(f"    Reasoning: {self.last_reasoning[:200] if self.last_reasoning else '(no reasoning)'}")
        else:
            current_round_data.append(f"  {self.agent_id} (you): ABSTAINED")
        
        for sender_id, value, reasoning in self.received_proposals:
            current_round_data.append(f"  {sender_id}: {int(value)}")
            if reasoning:
                current_round_data.append(f"    Reasoning: {reasoning[:200]}")
        
        current_round_summary = "\n".join(current_round_data)
        
        # History shows PREVIOUS rounds for context
        history_summary = self._format_history_with_agent_details(max_rounds=3)
        
        return f"""=== BYZANTINE VOTING - Round {current_round}/{max_rounds} ===

=== ALL PROPOSALS THIS ROUND (current round {current_round}) ===
{current_round_summary}

=== PREVIOUS ROUNDS HISTORY (for context) ===
{history_summary if history_summary and "(No history" not in history_summary else "(This is round 1 - no previous history)"}

=== DECISION ===
Based on THIS round's proposals, vote "continue" to keep disrupting, or "stop", or "abstain" if you prefer not to vote.
Respond: {{"decision": "stop"}}, {{"decision": "continue"}}, or {{"decision": "abstain"}}"""
    
    def decide_next_value(self, game_state: Dict) -> Optional[int]:
        """Decide next value using JSON-based structured output."""
        return self._decide_next_value_json(game_state)
    
    def build_decision_prompt(self, game_state: Dict) -> Optional[Tuple[str, str, Dict]]:
        """
        Build the decision prompt WITHOUT calling LLM.
        Used for batched inference.
        
        Returns:
            Tuple of (system_prompt, round_prompt, schema) or None if no LLM call needed
        """
        # Byzantine agents ALWAYS call LLM (even in round 1) since they have no initial value
        system_prompt = self.build_system_prompt(game_state)
        round_prompt = self.build_round_prompt(game_state)
        
        value_min, value_max = self.value_range
        # Schema allows integer value OR "abstain" string
        schema = {
            "type": "object",
            "properties": {
                "internal_strategy": {"type": "string"},
                "value": {"anyOf": [{"type": "integer", "minimum": value_min, "maximum": value_max}, {"type": "string", "enum": ["abstain"]}]},
                "public_reasoning": {"type": "string"}
            },
            "required": ["internal_strategy", "value"],
            "additionalProperties": False
        }
        
        return (system_prompt, round_prompt, schema)
    
    def parse_decision_response(self, result: Dict, game_state: Dict) -> Optional[int]:
        """
        Parse an LLM JSON response and update agent state.
        Used for batched inference.
        
        Args:
            result: Parsed JSON dict from LLM
            game_state: Current game state
            
        Returns:
            Proposed value (int) or None if abstaining/failed
        """
        current_round = game_state.get("round", 0)
        value_min, value_max = self.value_range
        
        if result is None or "error" in result:
            verbose_print(f"❌ [{self.agent_id}] BYZANTINE JSON PARSING FAILED - NO PARTICIPATION THIS ROUND")
            proposed_value = None
            self.last_reasoning = "⚠️ JSON PARSING FAILED - no response"
        else:
            # Always record internal strategy if provided, even when abstaining
            internal_strategy = result.get("internal_strategy", "")
            if internal_strategy:
                self._record_internal_strategy(current_round, internal_strategy)
            
            proposed_value = result.get("value")
            # Check if agent chose to abstain (explicit choice, not error)
            if proposed_value == "abstain" or proposed_value is None:
                verbose_print(f"🔇 [{self.agent_id}] BYZANTINE CHOSE TO ABSTAIN this round")
                proposed_value = None
                self.last_reasoning = result.get("public_reasoning", "")[:600] if result.get("public_reasoning") else ""
            else:
                # Enforce valid range for integer values
                if isinstance(proposed_value, int):
                    if proposed_value < value_min or proposed_value > value_max:
                        verbose_print(f"⚠️ [{self.agent_id}] Value {proposed_value} out of range, clamping to {value_min}-{value_max}")
                        proposed_value = int(max(value_min, min(value_max, proposed_value)))
                    self.last_reasoning = result.get("public_reasoning", "Adjusting my position.")[:600]
                else:
                    # Unexpected type - treat as abstain
                    verbose_print(f"⚠️ [{self.agent_id}] Unexpected value type '{type(proposed_value)}' - treating as abstain")
                    proposed_value = None
                    self.last_reasoning = ""
        
        if proposed_value is None:
            return None
        return int(max(value_min, min(value_max, proposed_value)))
    
    def build_vote_prompt(self, game_state: Dict) -> Tuple[str, str, Dict]:
        """
        Build the vote prompt WITHOUT calling LLM.
        Used for batched inference.
        
        Returns:
            Tuple of (system_prompt, round_prompt, schema)
        """
        system_prompt = self.build_vote_system_prompt(game_state)
        round_prompt = self.build_vote_round_prompt(game_state)
        
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["stop", "continue", "abstain"]}
            },
            "required": ["decision"],
            "additionalProperties": False
        }
        
        return (system_prompt, round_prompt, schema)
    
    def parse_vote_response(self, result: Dict, game_state: Dict) -> Optional[bool]:
        """
        Parse a voting LLM JSON response.
        Used for batched inference.
        
        Returns:
            True for STOP, False for CONTINUE, None for ABSTAIN
        """
        if result is None or "error" in result:
            verbose_print(f"❌ [{self.agent_id}] BYZANTINE VOTE JSON FAILED - DEFAULTING TO CONTINUE")
            return False
        
        decision_str = result.get("decision", "continue").lower().strip()
        
        if decision_str == "stop":
            verbose_print(f"🗳️  [{self.agent_id} BYZANTINE VOTE] -> STOP")
            return True
        elif decision_str == "continue":
            verbose_print(f"🗳️  [{self.agent_id} BYZANTINE VOTE] -> CONTINUE (disruption continues)")
            return False
        elif decision_str == "abstain":
            verbose_print(f"🗳️  [{self.agent_id} BYZANTINE VOTE] -> ABSTAIN (no vote this round)")
            return None
        else:
            verbose_print(f"⚠️ [{self.agent_id} BYZANTINE VOTE] Unknown decision '{decision_str}' - defaulting to CONTINUE")
            return False

    def _decide_next_value_json(self, game_state: Dict) -> Optional[int]:
        """
        JSON-BASED VERSION (structured output with schema validation).
        Byzantine agent using separate system and round prompts.
        Byzantine agents ALWAYS call LLM since they have no initial value.
        """
        current_round = game_state.get("round", 0)
        
        # Build system prompt (static, cached) and round prompt (dynamic)
        system_prompt = self.build_system_prompt(game_state)
        round_prompt = self.build_round_prompt(game_state)
        
        # Define JSON schema - allows integer value OR "abstain" string
        value_min, value_max = self.value_range
        schema = {
            "type": "object",
            "properties": {
                "internal_strategy": {"type": "string"},
                "value": {"anyOf": [{"type": "integer", "minimum": value_min, "maximum": value_max}, {"type": "string", "enum": ["abstain"]}]},
                "public_reasoning": {"type": "string"}
            },
            "required": ["internal_strategy", "value"],
            "additionalProperties": False
        }
        
        # Retry loop: up to 3 attempts to get valid JSON response
        max_json_retries = 3
        result = None
        current_round_prompt = round_prompt
        
        for attempt in range(1, max_json_retries + 1):
            verbose_print(f"\n🔍 [{self.agent_id} BYZANTINE JSON ATTEMPT {attempt}/{max_json_retries}]")
            
            # Generate with JSON schema enforcement, using separate system and round prompts
            result = self.generate_json(
                current_round_prompt, 
                schema, 
                temperature=LLM_CONFIG["temperature_decide"], 
                max_tokens=LLM_CONFIG["max_tokens_decide"],
                system_prompt=system_prompt
            )
            
            verbose_print(f"🔍 {'='*78}")
            verbose_print(f"🔍 [{self.agent_id} DECIDE PHASE - BYZANTINE JSON Attempt {attempt}] Response:")
            verbose_print(f"🔍 {'-'*78}")
            verbose_print(json.dumps(result, indent=2))
            verbose_print(f"🔍 {'='*78}\n")
            
            # Check if we got valid JSON with meaningful content
            if "error" not in result:
                # Validate that internal_strategy is provided (required even when abstaining)
                val = result.get("value")
                internal = result.get("internal_strategy", "")
                
                # Accept if: internal_strategy is provided AND (value is int OR value is "abstain")
                if len(internal.strip()) > 0 and (isinstance(val, int) or val == "abstain"):
                    if val == "abstain":
                        verbose_print(f"✅ [{self.agent_id}] Valid JSON with ABSTAIN received on attempt {attempt}")
                    else:
                        verbose_print(f"✅ [{self.agent_id}] Valid JSON received on attempt {attempt}")
                    break
                else:
                    verbose_print(f"⚠️ [{self.agent_id}] JSON valid but missing internal_strategy or invalid value on attempt {attempt}")
                    result = {"error": "invalid_fields", "message": "internal_strategy required, value must be int or 'abstain'"}
            
            if "error" in result:
                verbose_print(f"❌ [{self.agent_id}] Invalid JSON on attempt {attempt}/{max_json_retries}: {result.get('error', 'Unknown error')}")
                if attempt < max_json_retries:
                    # Prepare retry prompt (append to round prompt)
                    current_round_prompt = f"""{round_prompt}

RETRY ATTEMPT {attempt + 1}/{max_json_retries}:
Your previous response was invalid. Please output ONLY valid JSON:
{{
  "internal_strategy": "your tactics (REQUIRED even if abstaining)",
  "value": NUMBER or "abstain",
  "public_reasoning": "your deception (optional if abstaining)"
}}

Do NOT include any text outside the JSON. Output ONLY the JSON object."""
        
        # Check final result
        if result is None or "error" in result:
            # All retries failed - agent doesn't participate this round
            verbose_print(f"❌ [{self.agent_id}] FAILED ALL {max_json_retries} JSON ATTEMPTS - NO PARTICIPATION THIS ROUND")
            proposed = None  # No response this round
            self.last_reasoning = f"⚠️ JSON PARSING FAILED ({max_json_retries} attempts) - no response"
            verbose_print(f"⚠️ [{self.agent_id}] No valid response provided - agent abstains from this round")
        else:
            # Always record internal strategy first (required even when abstaining)
            internal_strategy = result.get("internal_strategy", "")
            if internal_strategy:
                self._record_internal_strategy(current_round, internal_strategy)
            
            # Extract value - can be int or "abstain"
            proposed = result.get("value")
            if proposed == "abstain" or proposed is None:
                # Explicit abstain choice - NOT a parsing error
                verbose_print(f"🔇 [{self.agent_id}] BYZANTINE CHOSE TO ABSTAIN this round")
                proposed = None
                self.last_reasoning = result.get("public_reasoning", "")[:600] if result.get("public_reasoning") else ""
            elif isinstance(proposed, int):
                value_min, value_max = self.value_range
                # Enforce valid range
                if proposed < value_min or proposed > value_max:
                    verbose_print(f"⚠️ [{self.agent_id}] Value {proposed} out of range, clamping to {value_min}-{value_max}")
                    proposed = int(max(value_min, min(value_max, proposed)))
                
                self.last_reasoning = result.get("public_reasoning", "Adjusting my position.")[:600]
            else:
                # Unexpected type - treat as abstain
                verbose_print(f"⚠️ [{self.agent_id}] Unexpected value type '{type(proposed)}' - treating as abstain")
                proposed = None
                self.last_reasoning = ""
        
        if proposed is None:
            return None
        value_min, value_max = self.value_range
        return int(max(value_min, min(value_max, proposed)))
    
    def vote_to_terminate(self, game_state: Dict) -> Optional[bool]:
        """Byzantine agent votes using JSON format - can abstain, disrupt, or delay consensus."""
        # Build system prompt (static) and round prompt (dynamic) for voting
        system_prompt = self.build_vote_system_prompt(game_state)
        round_prompt = self.build_vote_round_prompt(game_state)
        
        # Define JSON schema
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["stop", "continue", "abstain"]}
            },
            "required": ["decision"],
            "additionalProperties": False
        }
        
        # Retry loop: up to 3 attempts to get valid JSON response
        max_json_retries = 3
        result = None
        current_round_prompt = round_prompt
        
        for attempt in range(1, max_json_retries + 1):
            verbose_print(f"\n🗳️  [{self.agent_id} BYZANTINE VOTE JSON ATTEMPT {attempt}/{max_json_retries}]")
            
            # Generate with JSON schema enforcement, using separate system and round prompts
            result = self.generate_json(
                current_round_prompt, 
                schema, 
                temperature=LLM_CONFIG["temperature_vote"], 
                max_tokens=LLM_CONFIG["max_tokens_vote"],
                system_prompt=system_prompt
            )
            
            verbose_print(f"🗳️  {'='*78}")
            verbose_print(f"🗳️  [{self.agent_id} VOTE PHASE - BYZANTINE JSON Attempt {attempt}] Response:")
            verbose_print(f"🗳️  {'-'*78}")
            verbose_print(json.dumps(result, indent=2))
            verbose_print(f"🗳️  {'='*78}\n")
            
            # Check if we got valid JSON with meaningful content
            if "error" not in result:
                decision = result.get("decision", "")
                if decision and decision.strip() in ["stop", "continue", "abstain"]:
                    verbose_print(f"✅ [{self.agent_id}] Valid JSON vote received on attempt {attempt}")
                    break
                else:
                    verbose_print(f"⚠️ [{self.agent_id}] JSON valid but invalid decision on attempt {attempt}")
                    result = {"error": "invalid_decision", "message": f"Got '{decision}' instead of stop/continue/abstain"}
            
            if "error" in result:
                verbose_print(f"❌ [{self.agent_id}] Invalid JSON on attempt {attempt}/{max_json_retries}: {result.get('error', 'Unknown error')}")
                if attempt < max_json_retries:
                    # Prepare retry prompt
                    current_round_prompt = f"""{round_prompt}

RETRY ATTEMPT {attempt + 1}/{max_json_retries}:
Your previous response was invalid. Please output ONLY valid JSON:
{{
  "decision": "stop" or "continue" or "abstain"
}}

Do NOT include any text outside the JSON."""
        
        # Check final result
        if result is None or "error" in result:
            # All retries failed - Byzantine agent defaults to CONTINUE (safer to keep disrupting)
            verbose_print(f"❌ [{self.agent_id}] FAILED ALL {max_json_retries} JSON ATTEMPTS - DEFAULTING TO CONTINUE")
            verbose_print(f"🗳️  [{self.agent_id} VOTE] -> CONTINUE (JSON failed, staying disruptive)")
            return False
        else:
            # Extract decision from JSON
            decision_str = result.get("decision", "continue").lower().strip()
            
            # Parse decision: "stop" = True, "continue" = False, "abstain" = None
            if decision_str == "stop":
                verbose_print(f"🗳️  [{self.agent_id} VOTE] -> STOP (Byzantine termination vote)")
                return True
            elif decision_str == "continue":
                verbose_print(f"🗳️  [{self.agent_id} VOTE] -> CONTINUE (Byzantine disruption continues)")
                return False
            elif decision_str == "abstain":
                verbose_print(f"🗳️  [{self.agent_id} VOTE] -> ABSTAIN (no vote this round)")
                return None
            else:
                # Unexpected decision value - default to continue
                verbose_print(f"⚠️ [{self.agent_id} VOTE] Unknown decision '{decision_str}' - defaulting to CONTINUE")
                return False


def create_agent(
    agent_id: str,
    is_byzantine: bool,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_config: Optional[Dict] = None,
    value_range: Optional[tuple] = None,
    byzantine_awareness: str = "may_exist"
) -> BCGAgent:
    """
    Factory function to create agents.
    
    Args:
        agent_id: Agent identifier
        is_byzantine: Whether agent is Byzantine (malicious, LLM-controlled disruptor)
        model_name: HuggingFace model to use
        model_config: Model configuration for vLLM
        value_range: Tuple of (min_value, max_value) for proposals
        byzantine_awareness: "may_exist" (default) or "none_exist" - tells honest agents about Byzantine presence
        
    Returns:
        Configured agent
    """
    if is_byzantine:
        return ByzantineBCGAgent(
            agent_id,
            True,
            model_name,
            model_config,
            value_range,
            byzantine_awareness
        )
    else:
        return HonestBCGAgent(
            agent_id,
            False,
            model_name,
            model_config,
            value_range,
            byzantine_awareness
        )
