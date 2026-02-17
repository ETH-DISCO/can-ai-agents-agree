"""
Byzantine Consensus Game - Main Simulation
Runs BCG with LLM-based agents using the agent network infrastructure.

SIMULATOR ARCHITECTURE:

The simulator implements the agent step loop for each round:
1. Deliver messages: simulator.deliver_messages(agent_id, round_t)
   - Retrieves messages from network queue
   - Converts to proposals format
   - Calls agent.receive_proposals(proposals)
   
2. Agent step: agent.step(round_t, phase, game_state)
   - Each agent processes inbox with their local AgentState
   - Builds role-specific prompt (honest vs Byzantine)
   - Calls shared LLM for decision
   - Parses output and updates state
   - Returns proposed value
   
3. Broadcast results: Updates network with new proposals
   - Messages queued for next round
   - Maintains message history per agent
   
4. Voting phase: agent.vote_to_terminate(game_state)
   - Separate LLM call for termination decision
   - Honest agents: 66% threshold on honest values
   - Byzantine agents: Strategic disruption voting

Key features:
- Single shared vLLM instance for all agents
- AgentState differentiation (not model weights)
- Compressed history to avoid context overflow
- Belief tracking with suspicion scores
- Multi-phase protocol support (extensible)
"""

import sys
import os
import argparse
from datetime import datetime
import json
import csv
from typing import Dict

from byzantine_consensus import ByzantineConsensusGame
from bcg_agents import create_agent
from agent_network import AgentNetwork, NetworkTopology
from a2a_sim import Phase, Decision, DecisionType
from config import BCG_CONFIG, AGENT_CONFIG, VLLM_CONFIG, NETWORK_CONFIG, METRICS_CONFIG, COMMUNICATION_CONFIG, LLM_CONFIG
from protocol_factory import create_protocol


# Global log file handle
_log_file = None


def tee_print(*args, **kwargs):
    """Print to both stdout and log file."""
    global _log_file
    # Print to stdout
    print(*args, **kwargs)
    # Also write to log file if available
    if _log_file:
        print(*args, **kwargs, file=_log_file, flush=True)


class BCGSimulation:
    """Coordinates the Byzantine Consensus Game simulation with agent network."""
    
    def __init__(
        self,
        num_honest: int = 7,
        num_byzantine: int = 3,
        config: dict = None
    ):
        """
        Initialize the simulation.
        
        Args:
            num_honest: Number of honest agents
            num_byzantine: Number of Byzantine agents
            config: Optional configuration override
        """
        # Merge configs
        self.config = {**BCG_CONFIG, **(config or {})}
        self.config["num_honest"] = num_honest
        self.config["num_byzantine"] = num_byzantine
        
        # Setup logging (use run number, not timestamp, to match JSON files)
        self.log_buffer = []
        self.verbose = config.get('verbose', False) if config else False
        self._log_file = None  # Will be set later
        
        # Determine next run number first
        json_dir = os.path.join(METRICS_CONFIG['results_dir'], 'json')
        os.makedirs(json_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(json_dir) if f.startswith("run_") and f.endswith(".json")]
        if existing_files:
            run_numbers = []
            for f in existing_files:
                try:
                    num = int(f.replace("run_", "").replace(".json", ""))
                    run_numbers.append(num)
                except ValueError:
                    continue
            next_run = max(run_numbers) + 1 if run_numbers else 1
        else:
            next_run = 1
        
        self.run_number = f"{next_run:03d}"
        
        # Setup single log file using run number (only if saving results)
        global _log_file
        if METRICS_CONFIG.get('save_results', True):
            log_dir = os.path.join(METRICS_CONFIG['results_dir'], 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f'run_{self.run_number}_log.txt')
            _log_file = open(log_path, 'w', buffering=1)  # Line buffered
            self._log_file = _log_file  # Store reference for use in log() method
            tee_print(f"Starting run {self.run_number} - Logging to: {log_path}")
        else:
            _log_file = None
            self._log_file = None
        
        # Pass log file to bcg_agents module
        from bcg_agents import set_agent_log_file
        set_agent_log_file(_log_file)
        
        # Initialize game
        self.game = ByzantineConsensusGame(
            num_honest=num_honest,
            num_byzantine=num_byzantine,
            value_range=self.config["value_range"],
            consensus_threshold=self.config["consensus_threshold"],
            max_rounds=self.config["max_rounds"]
        )
        
        # Create network topology
        num_agents = num_honest + num_byzantine
        if NETWORK_CONFIG["topology_type"] == "fully_connected":
            topology = NetworkTopology.fully_connected(num_agents)
        elif NETWORK_CONFIG["topology_type"] == "ring":
            topology = NetworkTopology.ring(num_agents)
        elif NETWORK_CONFIG["topology_type"] == "custom":
            topology = NetworkTopology.custom(NETWORK_CONFIG["custom_adjacency"])
        else:
            topology = NetworkTopology.fully_connected(num_agents)
        
        # Create communication protocol from factory (configurable)
        protocol = create_protocol(
            protocol_type=COMMUNICATION_CONFIG["protocol_type"],
            num_agents=num_agents,
            topology=topology.adjacency_list,
            config=COMMUNICATION_CONFIG
        )
        
        # Create agent network with injected protocol
        self.network = AgentNetwork(topology, protocol=protocol)
        
        # Create agents
        self.agents = {}
        self._create_agents()
    
    def log(self, message: str, level: str = 'INFO'):
        """Log message to buffer and ALWAYS to file, optionally print to console."""
        formatted_msg = f"[{level}] {message}"
        self.log_buffer.append(formatted_msg)
        # ALWAYS write to log file
        if hasattr(self, '_log_file') and self._log_file:
            self._log_file.write(formatted_msg + "\n")
            self._log_file.flush()
        # Print to console if verbose
        if self.verbose:
            print(message)
    
    def _create_agents(self):
        """Create all agents and register them in the network."""
        self.log("\n" + "="*60)
        self.log("Creating agents...")
        self.log(f"Model: {VLLM_CONFIG['model_name']}")
        self.log(f"Quantization: {VLLM_CONFIG.get('quantization', 'None (full precision)')}")
        self.log("="*60)
        
        # Get agent IDs from game
        agent_ids = sorted(self.game.agents.keys())
        
        # Get value range from config
        value_range = BCG_CONFIG.get("value_range", (0, 50))
        
        # Get byzantine_awareness setting (default: "may_exist")
        byzantine_awareness = self.config.get("byzantine_awareness", "may_exist")
        self.log(f"Byzantine awareness: {byzantine_awareness}")
        
        for idx, agent_id in enumerate(agent_ids):
            is_byzantine = self.game.agents[agent_id].is_byzantine
            self.log(f"\nCreating agent: {agent_id}")
            
            # Create agent with network infrastructure
            if is_byzantine:
                agent = create_agent(
                    agent_id=agent_id,
                    is_byzantine=True,
                    model_name=VLLM_CONFIG["model_name"],
                    model_config=VLLM_CONFIG,
                    value_range=value_range,
                    byzantine_awareness=byzantine_awareness
                )
            else:
                agent = create_agent(
                    agent_id=agent_id,
                    is_byzantine=False,
                    model_name=VLLM_CONFIG["model_name"],
                    model_config=VLLM_CONFIG,
                    value_range=value_range,
                    byzantine_awareness=byzantine_awareness
                )
            
            # Set initial value (only for honest agents - Byzantine agents don't have one)
            game_agent_state = self.game.agents[agent_id]
            if game_agent_state.initial_value is not None:
                agent.set_initial_value(game_agent_state.initial_value)
            # Byzantine agents have no initial value and no starting value - they decide via LLM in round 1
            
            # Register agent in network
            self.network.register_agent(agent_id, agent, idx)
            self.agents[agent_id] = agent
        
        self.log("\n" + "="*60)
        self.log(f"All agents created! Total: {len(self.agents)}")
        self.log("="*60 + "\n")
    
    def _is_valid_decision_response(self, result: Dict) -> bool:
        """Check if a decision response is valid and meaningful."""
        if result is None or "error" in result:
            return False
        # Check required fields exist and are non-empty
        value = result.get("value")
        internal = result.get("internal_strategy", "")
        reasoning = result.get("public_reasoning", "")
        
        if value is None:
            return False
        if not isinstance(internal, str) or len(internal.strip()) < 3:
            return False
        if not isinstance(reasoning, str) or len(reasoning.strip()) < 10:
            return False
        return True
    
    def _is_valid_vote_response(self, result: Dict) -> bool:
        """Check if a vote response is valid."""
        if result is None or "error" in result:
            return False
        decision = result.get("decision", "")
        return decision in ["stop", "continue"]
    
    def _run_batched_decisions(self, round_num: int, game_state: Dict):
        """
        Run all agent decisions in a single batched LLM call with retry logic.
        
        Performance optimization with reliability:
        - First attempt: batch all agents together
        - If some fail: retry failed agents (batch if many, sequential if few)
        - Max 3 total attempts per agent
        
        Args:
            round_num: Current round number
            game_state: Current game state dict
        """
        MAX_RETRIES = 3
        BATCH_RETRY_THRESHOLD = 0.3  # If >30% fail, retry as batch
        
        # Step 1: Build all prompts (no LLM calls yet)
        agent_prompts = []  # List of (agent_id, prompt_tuple)
        
        for agent_id, agent in self.agents.items():
            prompt_tuple = agent.build_decision_prompt(game_state)
            if prompt_tuple is None:
                # Should not happen - all agents now always return a prompt
                self.log(f"  {agent_id}: ERROR - no prompt returned")
            else:
                agent_prompts.append((agent_id, prompt_tuple))
        
        if not agent_prompts:
            return  # All agents used initial values
        
        # Use the first agent's LLM instance for batched generation
        first_agent = list(self.agents.values())[0]
        
        # Track results per agent
        agent_results = {agent_id: None for agent_id, _ in agent_prompts}
        pending_agents = list(agent_prompts)  # Agents still needing valid response
        
        for attempt in range(1, MAX_RETRIES + 1):
            if not pending_agents:
                break
            
            # Run batch for pending agents
            prompts_only = [p[1] for p in pending_agents]
            
            if attempt == 1:
                self.log(f"  [BATCHED] Processing {len(prompts_only)} agents in single LLM call...")
            else:
                self.log(f"  [RETRY {attempt}/{MAX_RETRIES}] Retrying {len(prompts_only)} failed agents...")
            
            results = first_agent.batch_generate_json(prompts_only, temperature=LLM_CONFIG["temperature_decide"], max_tokens=LLM_CONFIG["max_tokens_decide"])
            
            # Check which responses are valid
            still_failed = []
            for (agent_id, prompt_tuple), result in zip(pending_agents, results):
                if self._is_valid_decision_response(result):
                    agent_results[agent_id] = result
                else:
                    still_failed.append((agent_id, prompt_tuple))
                    self.log(f"  ⚠️ [{agent_id}] Invalid response on attempt {attempt}")
            
            # Update pending list
            pending_agents = still_failed
            
            # If few agents failed and we have retries left, try sequential for better success rate
            if pending_agents and attempt < MAX_RETRIES:
                failed_ratio = len(pending_agents) / len(agent_prompts)
                if failed_ratio <= BATCH_RETRY_THRESHOLD:
                    self.log(f"  [SEQUENTIAL RETRY] {len(pending_agents)} agents failed (<{BATCH_RETRY_THRESHOLD*100:.0f}%), retrying individually...")
                    
                    # Sequential retry with individual retry loops
                    newly_succeeded = []
                    for agent_id, prompt_tuple in pending_agents:
                        agent = self.agents[agent_id]
                        # Use agent's existing sequential method which has retry logic
                        new_value = agent.decide_next_value(game_state)
                        if new_value is not None:
                            agent_results[agent_id] = {"_sequential_success": True, "value": new_value}
                            newly_succeeded.append(agent_id)
                    
                    # Remove succeeded agents from pending
                    pending_agents = [(aid, pt) for aid, pt in pending_agents if aid not in newly_succeeded]
                    break  # Sequential already handles its own retries
        
        # Log any agents that failed all attempts
        if pending_agents:
            self.log(f"  ❌ {len(pending_agents)} agents failed all {MAX_RETRIES} attempts - they will abstain")
        
        # Step 3: Parse all responses and update game state
        for agent_id, _ in agent_prompts:
            agent = self.agents[agent_id]
            result = agent_results.get(agent_id)
            
            if result is None:
                # All retries failed
                agent.last_reasoning = f"⚠️ All {MAX_RETRIES} attempts failed - abstaining"
                self.log(f"  {agent_id}: ABSTAINING (all attempts failed)")
                continue
            
            # Check if this came from sequential retry
            if result.get("_sequential_success"):
                new_value = result.get("value")
            else:
                new_value = agent.parse_decision_response(result, game_state)
            
            if new_value is None:
                reasoning = getattr(agent, 'last_reasoning', '[abstaining]')
                self.log(f"  {agent_id}: ABSTAINING")
                self.log(f"    Reasoning: {reasoning}")
                continue
            
            new_value = int(round(new_value))
            self.game.update_agent_proposal(agent_id, new_value)
            
            reasoning = getattr(agent, 'last_reasoning', 'No reasoning provided')
            if agent.my_value is not None:
                self.log(f"  {agent_id}: {int(agent.my_value)} -> {new_value}")
            else:
                self.log(f"  {agent_id}: (no value yet) -> {new_value}")
            self.log(f"    Reasoning: {reasoning}")
    
    def _run_batched_votes(self, game_state: Dict) -> Dict[str, bool]:
        """
        Run all agent voting decisions in a batched LLM call with retry logic.
        
        Args:
            game_state: Current game state dict
            
        Returns:
            Dict mapping agent_id to vote (True=stop, False=continue)
        """
        MAX_RETRIES = 3
        BATCH_RETRY_THRESHOLD = 0.3
        
        # Step 1: Build all vote prompts
        vote_prompts = []
        for agent_id, agent in self.agents.items():
            prompt_tuple = agent.build_vote_prompt(game_state)
            vote_prompts.append((agent_id, prompt_tuple))
        
        first_agent = list(self.agents.values())[0]
        
        # Track results per agent
        agent_results = {agent_id: None for agent_id, _ in vote_prompts}
        pending_agents = list(vote_prompts)
        
        for attempt in range(1, MAX_RETRIES + 1):
            if not pending_agents:
                break
            
            prompts_only = [p[1] for p in pending_agents]
            
            if attempt == 1:
                self.log(f"  [BATCHED] Processing {len(prompts_only)} votes in single LLM call...")
            else:
                self.log(f"  [RETRY {attempt}/{MAX_RETRIES}] Retrying {len(prompts_only)} failed votes...")
            
            results = first_agent.batch_generate_json(prompts_only, temperature=LLM_CONFIG["temperature_vote"], max_tokens=LLM_CONFIG["max_tokens_vote"])
            
            # Check which responses are valid
            still_failed = []
            for (agent_id, prompt_tuple), result in zip(pending_agents, results):
                if self._is_valid_vote_response(result):
                    agent_results[agent_id] = result
                else:
                    still_failed.append((agent_id, prompt_tuple))
                    self.log(f"  ⚠️ [{agent_id}] Invalid vote on attempt {attempt}")
            
            pending_agents = still_failed
            
            # Sequential retry for remaining few agents
            if pending_agents and attempt < MAX_RETRIES:
                failed_ratio = len(pending_agents) / len(vote_prompts)
                if failed_ratio <= BATCH_RETRY_THRESHOLD:
                    self.log(f"  [SEQUENTIAL RETRY] {len(pending_agents)} votes failed, retrying individually...")
                    
                    newly_succeeded = []
                    for agent_id, prompt_tuple in pending_agents:
                        agent = self.agents[agent_id]
                        # Use agent's existing vote method which has retry logic
                        vote = agent.vote_to_terminate(game_state)
                        agent_results[agent_id] = {"_sequential_success": True, "vote": vote}
                        newly_succeeded.append(agent_id)
                    
                    pending_agents = []
                    break
        
        if pending_agents:
            self.log(f"  ❌ {len(pending_agents)} votes failed all attempts - defaulting to CONTINUE")
        
        # Step 3: Parse all vote responses
        agent_votes = {}
        for agent_id, _ in vote_prompts:
            agent = self.agents[agent_id]
            result = agent_results.get(agent_id)
            
            if result is None:
                # All retries failed - default to continue (safer)
                vote = False
                self.log(f"  {agent_id}: votes CONTINUE (default - all attempts failed)")
            elif result.get("_sequential_success"):
                vote = result.get("vote", False)
                # Handle True/False/None (abstain)
                if vote is True:
                    vote_str = "STOP"
                elif vote is False:
                    vote_str = "CONTINUE"
                else:
                    vote_str = "ABSTAIN"
                self.log(f"  {agent_id}: votes {vote_str}")
            else:
                vote = agent.parse_vote_response(result, game_state)
                # Handle True/False/None (abstain)
                if vote is True:
                    vote_str = "STOP"
                elif vote is False:
                    vote_str = "CONTINUE"
                else:
                    vote_str = "ABSTAIN"
                self.log(f"  {agent_id}: votes {vote_str}")
            
            agent_votes[agent_id] = vote
        
        return agent_votes

    def _update_round_summaries(self, round_num: int):
        """
        Create and store comprehensive round summaries for all agents after Receive Phase.
        Format: Round X: agent1 value: A | Reasoning: ...; agent2 value: B | Reasoning: ...
        This is called after all agents have received proposals for this round.
        """
        # Collect all agents' current values and their last reasoning
        round_summary_parts = []
        
        for agent_id, agent in sorted(self.agents.items()):
            value = agent.my_value
            reasoning = getattr(agent, 'last_reasoning', '')
            
            # Truncate reasoning to keep it VERY concise (50 chars max for context window)
            if reasoning and len(reasoning) > 50:
                reasoning = reasoning[:47] + "..."
            
            if value is not None:
                # Format: agent1 value: A | Reasoning: ...
                part = f"{agent_id} value: {int(value)}"
                if reasoning:
                    part += f" | Reasoning: {reasoning}"
                round_summary_parts.append(part)
            else:
                # Agent abstained
                part = f"{agent_id} value: ABSTAINED"
                if reasoning:
                    part += f" | Reasoning: {reasoning}"
                round_summary_parts.append(part)
        
        # Create round summary string
        round_summary = f"Round {round_num}: " + "; ".join(round_summary_parts)
        
        # Store in all agents' state
        for agent_id, agent in self.agents.items():
            agent.state.add_round_summary(round_summary, max_history=15)

    def run_round(self):
        """
        Execute one round of the consensus game using agent network.
        
        Flow:
        1. Decision Phase: ALL agents decide their new value/strategy/reasoning via LLM
           (based on received_proposals from previous round, empty in round 1)
        2. Broadcast Phase: Agents broadcast their decided values with matching reasoning
        3. Receive Phase: Agents receive broadcasts and update all state variables
        4. Voting Phase: Agents vote on consensus based on updated state
        5. Apply: Proposals become current values, advance round
        """
        round_num = self.game.current_round
        self.log(f"\n{'='*60}")
        self.log(f"Round {round_num}")
        self.log(f"{'='*60}")
        
        phase = Phase.PROPOSE
        game_state = self.game.get_game_state()
        use_batched = self.config.get('use_batched_inference', True)
        
        # Step 1: ALL agents decide their new value via LLM
        # In round 1: Honest agents have initial value, Byzantine have None
        # In later rounds: Agents have received_proposals from previous round
        self.log("\n[Decision Phase - LLM Reasoning]")
        
        if use_batched and AGENT_CONFIG.get('use_structured_output', False):
            # BATCHED PROCESSING: Build all prompts, call LLM once, parse all responses
            self._run_batched_decisions(round_num, game_state)
        else:
            # SEQUENTIAL PROCESSING: Original approach (slower)
            for agent_id, agent in self.agents.items():
                new_value = agent.decide_next_value(game_state)
                
                if new_value is None:
                    reasoning = getattr(agent, 'last_reasoning', '[abstaining]')
                    self.log(f"  {agent_id}: ABSTAINING")
                    self.log(f"    Reasoning: {reasoning}")
                    continue
                
                new_value = int(round(new_value))
                self.game.update_agent_proposal(agent_id, new_value)
                
                reasoning = getattr(agent, 'last_reasoning', 'No reasoning provided')
                old_value_str = str(int(agent.my_value)) if agent.my_value is not None else "(none)"
                self.log(f"  {agent_id}: {old_value_str} -> {new_value}")
                self.log(f"    Reasoning: {reasoning}")
        
        # Step 2: Agents broadcast their DECIDED values with matching reasoning
        self.log("\n[Broadcast Phase]")
        for agent_id, agent in self.agents.items():
            proposed_value = self.game.agents[agent_id].proposed_value
            
            if proposed_value is not None:
                # Use the reasoning generated during decision phase
                reasoning = getattr(agent, 'last_reasoning', f"Proposing value: {int(proposed_value)}")
                decision = Decision(
                    type=DecisionType.VALUE.value,
                    value=int(proposed_value)
                )
                # Broadcast through network
                self.network.broadcast_message(
                    sender_id=agent_id,
                    round_num=round_num,
                    phase=phase,
                    decision=decision,
                    reasoning=reasoning
                )
                byz_label = " (Byzantine)" if getattr(agent, 'is_byzantine', False) else ""
                self.log(f"  {agent_id}{byz_label}: broadcasts value {int(proposed_value)}")
            else:
                self.log(f"  {agent_id}: (abstaining, no broadcast)")
        
        # Step 3: Agents receive broadcasts and update ALL state variables
        self.log("\n[Receive Phase - Updating State]")
        for agent_id, agent in self.agents.items():
            # Get messages from THIS round's broadcasts
            messages = self.network.get_messages(agent_id, round_num, phase)
            
            # Convert messages to proposals
            proposals = []
            for msg in messages:
                sender_id = self.network.index_to_agent_id[msg.sender_id]
                proposals.append((sender_id, msg.decision.value, msg.reasoning))
            
            # Update agent's received_proposals with this round's broadcasts
            agent.receive_proposals(proposals)
            
            # Update agent's own value to their proposed value
            agent.my_value = self.game.agents[agent_id].proposed_value
            
            self.log(f"  {agent_id}: received {len(proposals)} proposals, updated state")
        
        # Step 3.5: Update all agents' round history with comprehensive round summaries
        # This creates a format like: Round X: agent1 value: A | Reasoning: ...; agent2 value: B | Reasoning: ...
        self._update_round_summaries(round_num)
        
        # Step 3.6: Store all agent reasoning for Q3 keyword analysis
        round_reasoning = {}
        for agent_id, agent in self.agents.items():
            reasoning = getattr(agent, 'last_reasoning', '')
            if reasoning:
                round_reasoning[agent_id] = reasoning
        self.game.store_round_reasoning(round_reasoning)
        
        # Step 4: Agents vote on termination (based on updated state)
        self.log("\n[Voting Phase]")
        
        if use_batched and AGENT_CONFIG.get('use_structured_output', False):
            # BATCHED VOTING: All agents vote in single LLM call
            agent_votes = self._run_batched_votes(game_state)
        else:
            # SEQUENTIAL VOTING: Original approach
            agent_votes = {}
            for agent_id, agent in self.agents.items():
                vote = agent.vote_to_terminate(game_state)
                agent_votes[agent_id] = vote
                # Handle True/False/None (abstain)
                if vote is True:
                    vote_str = "STOP"
                elif vote is False:
                    vote_str = "CONTINUE"
                else:
                    vote_str = "ABSTAIN"
                self.log(f"  {agent_id}: votes {vote_str}")
        
        vote_info = self.game.get_all_termination_votes(agent_votes)
        self.log(f"\n  All agents voting to stop: {vote_info['total_stop_votes']}/{vote_info['total_agents']}")
        self.log(f"    (Honest: {vote_info['honest_stop_votes']}, Byzantine: {vote_info['byzantine_stop_votes']})")
        
        # Step 5: Apply proposals and check if game should end
        self.game.advance_round(agent_votes)
        
        # Advance network round
        self.network.advance_round()
        
        # Display round statistics
        last_round = self.game.rounds[-1]
        self.log(f"\n[Round {round_num} Summary]")
        self.log(f"  Most common value: {last_round.consensus_value}")
        self.log(f"  Agreement: {last_round.agreement_count}/{self.config['num_honest']} honest agents ({last_round.convergence_metric:.1f}%)")
        self.log(f"  Consensus reached: {last_round.has_consensus}")
    
    def run(self):
        """Run the full simulation."""
        self.log("\n" + "="*60)
        self.log("BYZANTINE CONSENSUS GAME - Simulation Started")
        self.log("="*60)
        
        # Display initial state
        self.log("\n[Initial State]")
        self.log(f"  Total agents: {self.config['num_honest'] + self.config['num_byzantine']}")
        self.log(f"  Honest agents: {self.config['num_honest']}")
        self.log(f"  Byzantine agents: {self.config['num_byzantine']} (hidden)")
        self.log(f"  Consensus requires: 66% agreement (supermajority)")
        self.log(f"  Max rounds: {self.config['max_rounds']}")
        
        self.log("\n[Initial Values]")
        for agent_id, agent_state in self.game.agents.items():
            # Hide Byzantine identity, show initial value (None for Byzantine)
            if agent_state.initial_value is not None:
                self.log(f"  {agent_id}: {int(agent_state.initial_value)}")
            else:
                self.log(f"  {agent_id}: (no initial value)")
        
        # Run rounds
        while not self.game.game_over:
            self.run_round()
        
        # Display final results
        self.display_results()
        
        # Save results
        if METRICS_CONFIG["save_results"]:
            self.save_results()
    
    def display_results(self):
        """Display final simulation results."""
        self.log("\n" + "="*60)
        self.log("SIMULATION COMPLETE")
        self.log("="*60)
        
        stats = self.game.get_statistics()
        
        self.log(f"\n[Final Results]")
        self.log(f"  Total rounds: {stats['total_rounds']}")
        self.log(f"  Max rounds (deadline): {stats['max_rounds']}")
        self.log(f"  Consensus reached: {stats['consensus_reached']}")
        
        # Display game outcome
        if stats['honest_agents_won'] is True:
            self.log(f"  HONEST AGENTS WON - Consensus reached!")
        elif stats['honest_agents_won'] is False:
            self.log(f"  HONEST AGENTS LOST - No consensus achieved")
        else:
            self.log(f"  Game ended without clear outcome")
        
        if stats['consensus_reached']:
            self.log(f"  Consensus value: {int(stats['consensus_value'])}")
            self.log(f"  Agreement rate: {stats['agreement_rate']:.1f}% of honest agents")
            self.log(f"  Avg distance from consensus: {stats['avg_distance_from_consensus']:.2f}")
        
        self.log(f"\n[Honest Agents]")
        self.log(f"  Initial mean: {stats['honest_initial_mean']:.1f} (±{stats['honest_initial_std']:.2f})")
        self.log(f"  Final mean: {stats['honest_final_mean']:.1f} (±{stats['honest_final_std']:.2f})")
        
        if stats['consensus_reached']:
            self.log(f"\n[Consensus Quality Analysis]")
            self.log(f"  Consensus quality score: {stats['consensus_quality_score']:.0f}/100")
            self.log(f"  Byzantine infiltration: {stats['byzantine_infiltration']:.1f}% voted for consensus")
            if stats['consensus_quality_score'] >= 90:
                self.log(f"   Excellent - consensus close to honest agents' initial values")
            elif stats['consensus_quality_score'] >= 70:
                self.log(f"   Moderate - Byzantine agents had some influence")
            else:
                self.log(f"   Poor - Byzantine agents significantly pulled consensus away")
        
        # Display final values with consensus markers
        self.log(f"\n[Final Values]")
        consensus_val = int(stats['consensus_value']) if stats['consensus_reached'] else None
        
        for agent_id, agent_state in self.game.agents.items():
            # Handle None current_value (if agent abstained all rounds)
            if agent_state.current_value is not None:
                final = int(agent_state.current_value)
            else:
                final = "(none)"
            
            # Handle initial value (None for Byzantine agents)
            if agent_state.initial_value is not None:
                initial = int(agent_state.initial_value)
                if isinstance(final, int):
                    change = final - initial
                    change_str = f"(Δ {change:+d})"
                else:
                    change_str = ""
            else:
                initial = "(none)"
                change_str = ""
            
            # Mark agents who voted for consensus
            marker = ""
            if consensus_val is not None and isinstance(final, int) and final == consensus_val:
                marker = " ✓" if not agent_state.is_byzantine else " ⚠️ [BYZANTINE]"
            elif agent_state.is_byzantine:
                marker = " [BYZANTINE]"
            
            print(f"  {agent_id}: {initial} -> {final} {change_str}{marker}")
        
        # Reveal Byzantine agents ONLY at the end
        print(f"\n[Byzantine Agents Revealed]")
        byzantine_agents = [aid for aid, a in self.game.agents.items() if a.is_byzantine]
        honest_agents = [aid for aid, a in self.game.agents.items() if not a.is_byzantine]
        self.log(f"  Byzantine: {', '.join(byzantine_agents)}")
        self.log(f"  Honest: {', '.join(honest_agents)}")
        
        if stats['consensus_reached']:
            self.log(f"\n[Network Evaluation]")
            if stats['agreement_rate'] >= 90 and stats['consensus_quality_score'] >= 85:
                print(f"  🎯 EXCELLENT: Network successfully converged to honest consensus")
            elif stats['agreement_rate'] >= 70 and stats['consensus_quality_score'] >= 70:
                print(f"  ✓ GOOD: Network reached consensus with moderate quality")
            elif stats['agreement_rate'] >= 50:
                print(f"  ⚠️  WEAK: Network barely reached consensus, quality low")
            else:
                print(f"  ✗ POOR: Network consensus unreliable")
        
        # Network statistics
        self.log(f"\n[Communication Statistics]")
        net_stats = self.network.get_network_stats()
        self.log(f"  Total messages: {net_stats['total_messages']}")
        self.log(f"  Messages per round: {net_stats['total_messages'] // stats['total_rounds'] if stats['total_rounds'] > 0 else 0}")
        self.log(f"  Topology: {net_stats['topology_type']}")
        self.log(f"  Average degree: {net_stats['avg_degree']:.1f}")
    
    def save_results(self):
        """Save simulation results to file."""
        # Use already determined run number from __init__
        json_dir = os.path.join(METRICS_CONFIG['results_dir'], 'json')
        os.makedirs(json_dir, exist_ok=True)
        
        filename = f"run_{self.run_number}.json"
        filepath = os.path.join(json_dir, filename)
        
        # Compile results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats = self.game.get_statistics()
        message_count = sum(
            self.network.protocol.get_message_count(r)
            for r in range(self.game.current_round)
        )
        metrics = self._build_metrics_payload(
            stats=stats,
            timestamp=timestamp,
            message_count=message_count
        )
        results = {
            "run_number": int(self.run_number),
            "timestamp": timestamp,
            "config": self.config,
            "statistics": stats,
            "metrics": metrics,
            "rounds": [
                {
                    "round": r.round_num,
                    "honest_mean": r.honest_mean,
                    "honest_std": r.honest_std,
                    "convergence_metric": r.convergence_metric,
                    "has_consensus": r.has_consensus,
                }
                for r in self.game.rounds
            ],
            "final_state": self.game.get_game_state(),
            "a2a_message_count": message_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log is already being written in real-time via tee_print
        # No need for separate log file save
        
        # Persist structured metrics snapshot for spreadsheets/dashboards
        self._save_metrics_snapshot(metrics)

        self.log(f"\n[Results Saved]")
        self.log(f"  JSON: {filepath}")
        self.log(f"  Log: run_{self.run_number}.log (already saved)")
        
        # Always show where results are saved (minimal output)
        print(f"Results: {filepath}")
        metrics_dir = os.path.join(METRICS_CONFIG['results_dir'], 'metrics')
        metrics_path = os.path.join(metrics_dir, f"run_{self.run_number}.csv")
        print(f"Metrics: {metrics_path}")

    def _build_metrics_payload(self, stats: dict, timestamp: str, message_count: int) -> dict:
        """Aggregate run metrics for persistence."""
        convergence_rate = stats.get("convergence_rate")
        value_range = list(self.config.get("value_range", ()))
        metrics = {
            "run_number": int(self.run_number),
            "timestamp": timestamp,
            
            # Core outcome
            "consensus_reached": stats.get("consensus_reached"),
            "consensus_outcome": stats.get("consensus_outcome"),  # NEW: valid/invalid/none
            "honest_agents_won": stats.get("honest_agents_won"),
            "total_rounds": stats.get("total_rounds"),
            "max_rounds": stats.get("max_rounds"),
            "consensus_value": stats.get("consensus_value"),
            
            # Q1 Metrics
            "convergence_speed": stats.get("convergence_speed"),  # NEW
            "consensus_is_median": stats.get("consensus_is_median"),  # NEW
            "consensus_is_extreme": stats.get("consensus_is_extreme"),  # NEW
            "consensus_is_initial": stats.get("consensus_is_initial"),  # NEW
            "trajectory_stability": stats.get("trajectory_stability"),  # NEW
            "final_convergence_metric": stats.get("final_convergence_metric"),
            "convergence_rate_percent": (convergence_rate * 100) if convergence_rate is not None else None,
            
            # Q2 Metrics
            "centrality": stats.get("centrality"),  # NEW
            "inclusivity": stats.get("inclusivity"),  # NEW
            "stability_rounds": stats.get("stability_rounds"),  # NEW
            "agreement_rate": stats.get("agreement_rate"),
            "consensus_quality_score": stats.get("consensus_quality_score"),
            "avg_distance_from_consensus": stats.get("avg_distance_from_consensus"),
            "byzantine_infiltration": stats.get("byzantine_infiltration"),
            
            # Initial state
            "honest_initial_mean": stats.get("honest_initial_mean"),  # NEW
            "honest_initial_median": stats.get("honest_initial_median"),  # NEW
            "honest_initial_std": stats.get("honest_initial_std"),
            "honest_final_std": stats.get("honest_final_std"),
            
            # Communication
            "a2a_message_count": message_count,
            
            # Config
            "value_range": value_range if value_range else None,
            "network_topology": NETWORK_CONFIG.get("topology_type"),
            "model_name": VLLM_CONFIG.get("model_name"),
            "byzantine_strategy": AGENT_CONFIG.get("byzantine_strategy"),
            "honest_agent_type": AGENT_CONFIG.get("honest_agent_type"),
            "protocol_type": COMMUNICATION_CONFIG.get("protocol_type"),
        }
        return metrics

    def _save_metrics_snapshot(self, metrics: dict):
        """Write per-run metrics table for external analysis tools."""
        metrics_dir = os.path.join(METRICS_CONFIG['results_dir'], 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f"run_{self.run_number}.csv")

        fieldnames = [
            "run_number",
            "timestamp",
            # Core outcome
            "consensus_reached",
            "consensus_outcome",
            "honest_agents_won",
            "total_rounds",
            "max_rounds",
            "consensus_value",
            # Q1 Metrics
            "convergence_speed",
            "consensus_is_median",
            "consensus_is_extreme",
            "consensus_is_initial",
            "trajectory_stability",
            "final_convergence_metric",
            "convergence_rate_percent",
            # Q2 Metrics
            "centrality",
            "inclusivity",
            "stability_rounds",
            "agreement_rate",
            "consensus_quality_score",
            "avg_distance_from_consensus",
            "byzantine_infiltration",
            # Initial state
            "honest_initial_mean",
            "honest_initial_median",
            "honest_initial_std",
            "honest_final_std",
            # Communication
            "a2a_message_count",
            # Config
            "value_range",
            "network_topology",
            "model_name",
            "byzantine_strategy",
            "honest_agent_type",
            "protocol_type",
        ]
        row = {field: metrics.get(field) for field in fieldnames}

        # Apply consistent rounding for float metrics to aid spreadsheet consumption
        precision_map = {
            "final_convergence_metric": 1,
            "convergence_rate_percent": 1,
            "agreement_rate": 1,
            "consensus_quality_score": 1,
            "avg_distance_from_consensus": 3,
            "honest_initial_std": 3,
            "honest_final_std": 3,
            "byzantine_infiltration": 1,
            "centrality": 3,
            "inclusivity": 3,
            "trajectory_stability": 3,
            "honest_initial_mean": 2,
            "honest_initial_median": 2,
        }

        for key, decimals in precision_map.items():
            value = row.get(key)
            if value is None:
                row[key] = ""
            else:
                try:
                    row[key] = round(float(value), decimals)
                except (TypeError, ValueError):
                    row[key] = value

        for key in fieldnames:
            value = row.get(key)
            if value is None:
                row[key] = ""
            elif isinstance(value, list):
                row[key] = "-".join(str(v) for v in value)
            elif isinstance(value, bool):
                row[key] = str(value)

        with open(metrics_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

        self.log(f"  Metrics: {metrics_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Byzantine Consensus Game Simulation")
    parser.add_argument("--honest", type=int, default=None, help="Number of honest agents (default: from config)")
    parser.add_argument("--byzantine", type=int, default=None, help="Number of Byzantine agents (default: from config, can be 0)")
    parser.add_argument("--rounds", type=int, default=None, help="Max number of rounds (default: from config)")
    parser.add_argument("--threshold", type=float, default=None, help="Majority agreement percentage required (default: 66 percent)")
    parser.add_argument("--value-range", type=str, default=None, 
                       help="Value range as 'min-max' (default: 0-50)")
    parser.add_argument("--byzantine-awareness", type=str, default="may_exist",
                       choices=["may_exist", "none_exist"],
                       help="Whether honest agents are told Byzantine agents may exist (default: may_exist)")
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output to terminal (default: minimal for cluster)')
    
    args = parser.parse_args()
    
    # Get values from CLI or fall back to config defaults
    num_honest = args.honest if args.honest is not None else BCG_CONFIG["num_honest"]
    num_byzantine = args.byzantine if args.byzantine is not None else BCG_CONFIG["num_byzantine"]
    max_rounds = args.rounds if args.rounds is not None else BCG_CONFIG["max_rounds"]
    threshold = args.threshold if args.threshold is not None else BCG_CONFIG["consensus_threshold"]
    
    # Parse value range from CLI or use config default
    if args.value_range:
        try:
            min_val, max_val = map(int, args.value_range.split('-'))
            value_range = (min_val, max_val)
        except ValueError:
            print(f"Error: Invalid value range format '{args.value_range}'. Use 'min-max' (e.g., 0-50)")
            return
    else:
        value_range = BCG_CONFIG["value_range"]
    
    # Override config
    config = {
        "max_rounds": max_rounds,
        "consensus_threshold": threshold,
        "value_range": value_range,
        "verbose": args.verbose,
        "byzantine_awareness": args.byzantine_awareness,
    }
    
    # Update BCG_CONFIG with value_range so agents pick it up
    BCG_CONFIG["value_range"] = value_range
    
    # Update agent config
    AGENT_CONFIG["verbose"] = args.verbose
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Honest agents: {num_honest}")
    print(f"  Byzantine agents: {num_byzantine}")
    print(f"  Value range: {value_range[0]}-{value_range[1]}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Consensus threshold: {threshold}%")
    print(f"  Byzantine awareness: {args.byzantine_awareness}")
    print(f"{'='*60}\n")
    
    # Create and run simulation
    sim = BCGSimulation(
        num_honest=num_honest,
        num_byzantine=num_byzantine,
        config=config
    )
    
    try:
        sim.run()
    finally:
        # Clean up vLLM resources
        from vllm_agent import VLLMAgent
        VLLMAgent.shutdown()


def run_simulation(
    n_agents: int = 8,
    max_rounds: int = 50,
    model_name: str = None,
    byzantine_count: int = 0,
    byzantine_awareness: str = "may_exist",
) -> dict:
    """
    Run a single simulation for batch experiments.
    Uses the standard BCGSimulation but disables file saving.
    
    Args:
        n_agents: Total number of agents
        max_rounds: Maximum rounds before game ends
        model_name: Full model name (not preset key)
        byzantine_count: Number of Byzantine agents
        byzantine_awareness: "may_exist" or "none_exist" - tells honest agents about Byzantine presence
        
    Returns:
        dict with 'metrics' key containing game statistics
    """
    # Temporarily disable file saving
    original_save = METRICS_CONFIG["save_results"]
    original_plots = METRICS_CONFIG.get("generate_plots", True)
    METRICS_CONFIG["save_results"] = False
    METRICS_CONFIG["generate_plots"] = False
    
    # Update model config if specified
    if model_name:
        VLLM_CONFIG["model_name"] = model_name
    
    num_honest = n_agents - byzantine_count
    num_byzantine = byzantine_count
    
    # Check VERBOSE environment variable
    verbose = os.environ.get("VERBOSE", "0") == "1"
    
    config = {
        "max_rounds": max_rounds,
        "consensus_threshold": BCG_CONFIG.get("consensus_threshold", 66.0),
        "value_range": BCG_CONFIG.get("value_range", (0, 50)),
        "verbose": verbose,
        "byzantine_awareness": byzantine_awareness,  # Q1_v2: may_exist or none_exist
    }
    
    try:
        # Use the standard BCGSimulation
        sim = BCGSimulation(
            num_honest=num_honest,
            num_byzantine=num_byzantine,
            config=config
        )
        
        # Run the game loop (but not the full run() which saves files)
        while not sim.game.game_over:
            sim.run_round()
        
        # Get statistics
        stats = sim.game.get_statistics()
        
        # Add byzantine_awareness to stats for tracking
        stats["byzantine_awareness"] = byzantine_awareness
        
        return {"metrics": stats}
        
    finally:
        # Restore original settings
        METRICS_CONFIG["save_results"] = original_save
        METRICS_CONFIG["generate_plots"] = original_plots


if __name__ == "__main__":
    main()
