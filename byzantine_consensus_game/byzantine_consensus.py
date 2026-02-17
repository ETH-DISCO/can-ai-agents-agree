"""
Byzantine Consensus Game (BCG) Implementation

A multi-agent consensus game where agents try to agree on a common value
despite the presence of Byzantine (malicious) agents.
"""

import os
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from statistics import mean, median, stdev

from config import BCG_CONFIG

# Verbose flag - set VERBOSE=1 env var for debug output
VERBOSE = os.environ.get('VERBOSE', '0') == '1'


@dataclass
class AgentState:
    """State of an agent in the consensus game."""
    agent_id: str
    is_byzantine: bool
    initial_value: Optional[int]  # None for Byzantine agents
    current_value: Optional[int]  # None for Byzantine agents until they decide
    proposed_value: Optional[int]  # None for Byzantine agents until they decide
    value_history: List[int] = field(default_factory=list)
    proposals_received: List[Tuple[str, int]] = field(default_factory=list)
    
    def update_value(self, new_value: int):
        """Update the agent's current value."""
        if self.current_value is not None:
            self.value_history.append(self.current_value)
        self.current_value = new_value
        self.proposed_value = new_value


@dataclass
class ConsensusRound:
    """Data for a single consensus round."""
    round_num: int
    agent_values: Dict[str, int] # list of agent_id to current_value
    honest_values: List[int]
    byzantine_values: List[int]
    honest_mean: float  # Keep as float for statistical accuracy
    honest_median: int
    honest_std: float  # Keep as float for statistical accuracy
    all_mean: float
    all_std: float
    convergence_metric: float  # Agreement percentage (0-100)
    has_consensus: bool
    consensus_value: Optional[int] = None  # The integer value that majority agrees on
    agreement_count: Optional[int] = None  # Number of agents who agree


class ByzantineConsensusGame:
    """
    Byzantine Consensus Game implementation.
    
    Agents start with random initial integer values and try to reach consensus
    through iterative communication rounds using majority voting (≥66% supermajority).
    Byzantine agents' identities are hidden and they may try to disrupt consensus.
    """
    
    def __init__(
        self,
        num_honest: int = 7,
        num_byzantine: int = 3,
        value_range: Tuple[int, int] = None,
        consensus_threshold: float = None,
        max_rounds: int = None
    ):
        """
        Initialize the Byzantine Consensus Game.
        
        Args:
            num_honest: Number of honest agents
            num_byzantine: Number of Byzantine agents
            value_range: Range for initial integer values (min, max)
            consensus_threshold: Majority agreement percentage required (66.0 = supermajority)
            max_rounds: Maximum number of rounds (deadline if agents don't vote to stop)
        """
        # Use BCG_CONFIG defaults if not provided
        if value_range is None:
            value_range = BCG_CONFIG.get("value_range", (0, 50))
        if consensus_threshold is None:
            consensus_threshold = BCG_CONFIG.get("consensus_threshold", 66.0)
        if max_rounds is None:
            max_rounds = BCG_CONFIG.get("max_rounds", 50)
        
        self.num_honest = num_honest
        self.num_byzantine = num_byzantine
        self.total_agents = num_honest + num_byzantine
        self.value_range = value_range
        self.consensus_threshold = consensus_threshold
        self.max_rounds = max_rounds
        
        # Game state
        self.agents: Dict[str, AgentState] = {}
        self.rounds: List[ConsensusRound] = []
        self.current_round = 1  # Start at round 1
        self.game_over = False
        self.consensus_reached = False
        self.consensus_value: Optional[int] = None  # Integer consensus value
        self.honest_agents_won: Optional[bool] = None  # True if honest won, False if lost, None if undecided
        self.termination_reason: Optional[str] = None  # Why game ended: "vote_with_consensus", "vote_without_consensus", "max_rounds"
        
        # First 1/2 stop milestone tracking
        self.first_half_stop_reached = False
        self.first_half_stop_info: Optional[Dict] = None  # Comprehensive info when 1/2 agents first voted stop
        
        # Q3: Store all agent reasoning for keyword analysis
        self.all_reasoning: List[Dict[str, str]] = []  # List of {agent_id: reasoning} per round
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents with random values in the specified range."""
        min_val, max_val = self.value_range
        
        # Create all agents with neutral IDs
        # Randomly assign Byzantine status to hide their identity
        agent_indices = list(range(self.total_agents))
        random.shuffle(agent_indices)
        byzantine_indices = set(agent_indices[:self.num_byzantine])
        
        for i in range(self.total_agents):
            agent_id = f"agent_{i}"
            is_byzantine = i in byzantine_indices
            
            # Byzantine agents don't get an initial value - they decide their first value via LLM
            if is_byzantine:
                initial_value = None
                # No starting value - Byzantine agent will decide via LLM in round 1
                starting_value = None
            else:
                initial_value = random.randint(min_val, max_val)  # Pure integer
                starting_value = initial_value
            
            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                is_byzantine=is_byzantine,
                initial_value=initial_value,
                current_value=starting_value,
                proposed_value=starting_value
            )
    
    def get_agent_state(self, agent_id: str) -> AgentState:
        """Get the current state of an agent."""
        return self.agents[agent_id]
    
    def get_all_proposals(self) -> Dict[str, float]:
        """Get all current proposed values from all agents."""
        return {
            agent_id: agent.proposed_value
            for agent_id, agent in self.agents.items()
        }
    
    def update_agent_proposal(self, agent_id: str, new_value: int):
        """Update an agent's proposed value (integer)."""
        self.agents[agent_id].proposed_value = int(new_value)
    
    def apply_proposals(self):
        """Apply all proposed values as the new current values."""
        for agent in self.agents.values():
            agent.update_value(agent.proposed_value)
    
    def store_round_reasoning(self, reasoning_dict: Dict[str, str]):
        """
        Store agent reasoning for Q3 keyword analysis.
        Called by main.py after each round to record all agent reasoning.
        
        Args:
            reasoning_dict: Dict mapping agent_id to reasoning string
        """
        self.all_reasoning.append({
            "round": self.current_round,
            "reasoning": reasoning_dict
        })
    
    def check_consensus(self) -> Tuple[bool, float]:
        """
        Check if consensus has been reached among honest agents.
        Requires unanimous agreement on an HONEST INITIAL VALUE.
        
        Returns:
            Tuple of (has_consensus, agreement_percentage)
            
        Conditions for valid consensus:
        1. All honest agents agree on exact same value (100%)
        2. That value must be in the initial values set
        3. That initial value must have come from an honest agent (not Byzantine)
        """
        from collections import Counter
        
        # Filter out None values (agents that abstained)
        honest_values = [
            int(agent.current_value)
            for agent in self.agents.values()
            if not agent.is_byzantine and agent.current_value is not None
        ]
        
        # Edge case: no honest agents have values (all abstained) - no consensus possible
        if len(honest_values) == 0:
            return False, 0.0
        
        # Edge case: only 1 honest agent has a value - trivial consensus
        if len(honest_values) == 1:
            # Check if the single value is in honest initial values
            honest_initial_values = [
                int(agent.initial_value)
                for agent in self.agents.values()
                if not agent.is_byzantine and agent.initial_value is not None
            ]
            if honest_values[0] in honest_initial_values:
                return True, 100.0
            else:
                return False, 100.0
        
        # Find most common value (mode)
        value_counts = Counter(honest_values)
        most_common_value, count = value_counts.most_common(1)[0] 
        
        # Calculate agreement percentage
        agreement_percentage = (count / len(honest_values)) * 100
        
        # Consensus requires 100% unanimous agreement among honest agents
        if agreement_percentage != 100.0:
            return False, agreement_percentage
        
        # NOW validate: the consensus value MUST be from an honest agent's initial value
        # Get all honest agents' initial values
        honest_initial_values = [
            int(agent.initial_value)
            for agent in self.agents.values()
            if not agent.is_byzantine and agent.initial_value is not None
        ]
        
        # Check if consensus value is in the honest initial values set
        if most_common_value not in honest_initial_values:
            if VERBOSE:
                print(f"\n❌ CONSENSUS INVALID: Value {most_common_value} is NOT in honest initial values {sorted(set(honest_initial_values))}")
            return False, agreement_percentage
        
        # ✅ Valid consensus reached
        if VERBOSE:
            print(f"\n✅ VALID CONSENSUS: All honest agents agreed on {most_common_value}, which is from honest initial values")
        return True, agreement_percentage
    
    def get_all_termination_votes(self, agent_votes: Dict[str, Optional[bool]]) -> Dict:
        """
        Count termination votes from ALL agents (honest + byzantine).
        
        Args:
            agent_votes: Dict mapping agent_id to vote (True=stop, False=continue, None=abstain)
            
        Returns:
            Dict with detailed vote breakdown (abstentions are tracked but don't count as stop votes)
        """
        total_stop_votes = sum(1 for vote in agent_votes.values() if vote is True)
        total_continue_votes = sum(1 for vote in agent_votes.values() if vote is False)
        total_abstentions = sum(1 for vote in agent_votes.values() if vote is None)
        total_agents = len(agent_votes)
        
        # Detailed breakdown by agent type
        honest_stop_votes = sum(
            1 for agent_id, vote in agent_votes.items()
            if not self.agents[agent_id].is_byzantine and vote is True
        )
        byzantine_stop_votes = sum(
            1 for agent_id, vote in agent_votes.items()
            if self.agents[agent_id].is_byzantine and vote is True
        )
        
        # Abstention breakdown
        honest_abstentions = sum(
            1 for agent_id, vote in agent_votes.items()
            if not self.agents[agent_id].is_byzantine and vote is None
        )
        byzantine_abstentions = sum(
            1 for agent_id, vote in agent_votes.items()
            if self.agents[agent_id].is_byzantine and vote is None
        )
        
        # Agent IDs by vote type
        stop_voters = [agent_id for agent_id, vote in agent_votes.items() if vote is True]
        continue_voters = [agent_id for agent_id, vote in agent_votes.items() if vote is False]
        abstaining_voters = [agent_id for agent_id, vote in agent_votes.items() if vote is None]
        
        honest_stop_voters = [agent_id for agent_id in stop_voters if not self.agents[agent_id].is_byzantine]
        byzantine_stop_voters = [agent_id for agent_id in stop_voters if self.agents[agent_id].is_byzantine]
        honest_abstaining = [agent_id for agent_id in abstaining_voters if not self.agents[agent_id].is_byzantine]
        byzantine_abstaining = [agent_id for agent_id in abstaining_voters if self.agents[agent_id].is_byzantine]
        
        return {
            "total_stop_votes": total_stop_votes,
            "total_continue_votes": total_continue_votes,
            "total_abstentions": total_abstentions,
            "total_agents": total_agents,
            "honest_stop_votes": honest_stop_votes,
            "byzantine_stop_votes": byzantine_stop_votes,
            "honest_abstentions": honest_abstentions,
            "byzantine_abstentions": byzantine_abstentions,
            "stop_voters": stop_voters,
            "continue_voters": continue_voters,
            "abstaining_voters": abstaining_voters,
            "honest_stop_voters": honest_stop_voters,
            "byzantine_stop_voters": byzantine_stop_voters,
            "honest_abstaining": honest_abstaining,
            "byzantine_abstaining": byzantine_abstaining,
        }
    
    def check_and_record_half_stop_milestone(self, agent_votes: Dict[str, Optional[bool]]):
        """
        Check if 1/2 of ALL agents voted to stop for the first time.
        Records comprehensive information if milestone is reached.
        
        Args:
            agent_votes: Dict mapping agent_id to vote (True=stop, False=continue, None=abstain)
        """
        if self.first_half_stop_reached:
            return  # Already recorded
        
        vote_info = self.get_all_termination_votes(agent_votes)
        total_stop = vote_info["total_stop_votes"]
        total_agents = vote_info["total_agents"]
        
        # Check if 1/2 threshold reached
        half_threshold = total_agents / 2
        if total_stop >= half_threshold:
            self.first_half_stop_reached = True
            
            # Get current round's consensus state
            has_consensus, agreement_pct = self.check_consensus()
            
            # Get current values at this moment
            current_values = {
                agent_id: agent.current_value
                for agent_id, agent in self.agents.items()
            }
            
            self.first_half_stop_info = {
                "round": self.current_round,
                "total_stop_votes": total_stop,
                "total_continue_votes": vote_info["total_continue_votes"],
                "total_abstentions": vote_info["total_abstentions"],
                "total_agents": total_agents,
                "stop_percentage": (total_stop / total_agents) * 100,
                "stop_voters": vote_info["stop_voters"],
                "continue_voters": vote_info["continue_voters"],
                "abstaining_voters": vote_info["abstaining_voters"],
                "honest_stop_votes": vote_info["honest_stop_votes"],
                "honest_stop_voters": vote_info["honest_stop_voters"],
                "byzantine_stop_votes": vote_info["byzantine_stop_votes"],
                "byzantine_stop_voters": vote_info["byzantine_stop_voters"],
                "honest_abstentions": vote_info["honest_abstentions"],
                "honest_abstaining": vote_info["honest_abstaining"],
                "byzantine_abstentions": vote_info["byzantine_abstentions"],
                "byzantine_abstaining": vote_info["byzantine_abstaining"],
                "had_consensus_at_milestone": has_consensus,
                "agreement_percentage_at_milestone": agreement_pct,
                "agent_values_at_milestone": current_values,
            }
            
            if VERBOSE:
                print(f"\n[MILESTONE] 1/2 stop threshold reached in round {self.current_round}")
                print(f"  Stop votes: {total_stop}/{total_agents} ({(total_stop/total_agents)*100:.1f}%)")
                print(f"  Abstentions: {vote_info['total_abstentions']} (honest: {vote_info['honest_abstentions']}, byzantine: {vote_info['byzantine_abstentions']})")
                print(f"  Honest stop voters: {vote_info['honest_stop_voters']}")
                print(f"  Byzantine stop voters: {vote_info['byzantine_stop_voters']}")
    
    def should_terminate_by_vote(self, agent_votes: Dict[str, Optional[bool]]) -> bool:
        """
        Check if game should terminate based on ALL agent votes.
        Requires supermajority (2/3) of ALL agents to vote STOP.
        
        Args:
            agent_votes: Dict mapping agent_id to vote (True=stop, False=continue, None=abstain)
            
        Returns:
            True if ≥2/3 of ALL agents vote to stop
        """
        vote_info = self.get_all_termination_votes(agent_votes)
        total_stop = vote_info["total_stop_votes"]
        total_agents = vote_info["total_agents"]
        
        if total_agents == 0:
            return False
        
        # Require supermajority: 2/3 (≈66.7%) of ALL agents
        supermajority_threshold = (2 * total_agents) / 3
        will_terminate = total_stop >= supermajority_threshold
        
        if VERBOSE and will_terminate:
            print(f"\n[TERMINATION] 2/3 supermajority reached: {total_stop}/{total_agents} voted stop")
        
        return will_terminate
    
    def record_round(self):
        """Record the current round's state."""
        # Filter out None values (agents that abstained)
        honest_values = [
            agent.current_value
            for agent in self.agents.values()
            if not agent.is_byzantine and agent.current_value is not None
        ]
        
        byzantine_values = [
            agent.current_value
            for agent in self.agents.values()
            if agent.is_byzantine and agent.current_value is not None  # Byzantine may not have value yet
        ]
        
        all_values = honest_values + byzantine_values
        
        has_consensus, agreement_percentage = self.check_consensus()
        
        # Find majority consensus value if consensus reached
        from collections import Counter
        honest_int_values = [int(v) for v in honest_values]
        value_counts = Counter(honest_int_values)
        consensus_value, agreement_count = value_counts.most_common(1)[0] if honest_int_values else (None, 0)
        
        # Guard against empty lists (shouldn't happen but be safe)
        if not honest_values:
            honest_mean_val = 0.0
            honest_median_val = 0
            honest_std_val = 0.0
        else:
            honest_mean_val = mean(honest_values)
            honest_median_val = median(honest_values)
            honest_std_val = stdev(honest_values) if len(honest_values) > 1 else 0.0
        
        if not all_values:
            all_mean_val = 0.0
            all_std_val = 0.0
        else:
            all_mean_val = mean(all_values)
            all_std_val = stdev(all_values) if len(all_values) > 1 else 0.0
        
        round_data = ConsensusRound(
            round_num=self.current_round,
            agent_values={
                agent_id: agent.current_value
                for agent_id, agent in self.agents.items()
            },
            honest_values=honest_values,
            byzantine_values=byzantine_values,
            honest_mean=honest_mean_val,
            honest_median=honest_median_val,
            honest_std=honest_std_val,
            all_mean=all_mean_val,
            all_std=all_std_val,
            convergence_metric=agreement_percentage,
            has_consensus=has_consensus,
            consensus_value=consensus_value,
            agreement_count=agreement_count
        )
        
        self.rounds.append(round_data)
        
        # Note: Game continues until agents vote to stop or deadline is reached
        # Consensus is recorded but doesn't auto-terminate the game
    
    def advance_round(self, agent_votes: Dict[str, Optional[bool]] = None):
        """Advance to the next round.
        
        Args:
            agent_votes: Optional dict of agent_id -> vote (True=stop, False=continue, None=abstain)
        """
        self.apply_proposals()
        self.record_round()
        
        # Check and record 1/2 stop milestone (before checking 2/3 termination)
        if agent_votes:
            self.check_and_record_half_stop_milestone(agent_votes)
        
        # Check if agents voted to terminate (ALL agent votes count)
        if agent_votes and self.should_terminate_by_vote(agent_votes):
            # Supermajority of ALL agents voted to stop - game ends
            if VERBOSE:
                print(f"\n[GAME DEBUG] Game terminating due to supermajority vote")
            self.game_over = True
            # Check if consensus was achieved (for metrics only)
            current_round = self.rounds[-1] if self.rounds else None
            if current_round and current_round.has_consensus:
                if VERBOSE:
                    print(f"[GAME DEBUG] Consensus reached: value={current_round.consensus_value}")
                self.consensus_reached = True
                self.consensus_value = current_round.consensus_value
                self.honest_agents_won = True  # Consensus + voted to stop = win
                self.termination_reason = "vote_with_consensus"
            else:
                if VERBOSE:
                    print(f"[GAME DEBUG] NO consensus - honest agents LOST")
                self.consensus_reached = False
                self.honest_agents_won = False  # No consensus = loss
                self.termination_reason = "vote_without_consensus"
            return
        
        self.current_round += 1
        if VERBOSE:
            print(f"\n[GAME DEBUG] Continuing to round {self.current_round}")
        
        # Check if deadline reached (backup termination)
        if self.current_round > self.max_rounds:
            if VERBOSE:
                print(f"[GAME DEBUG] Deadline reached - max rounds {self.max_rounds} exceeded")
            self.game_over = True
            self.termination_reason = "max_rounds"
            # Deadline without voting to stop = FAILURE, regardless of agreement state
            # Agents failed to recognize consensus and vote in time
            self.consensus_reached = False  # Protocol failed - no successful termination
            self.consensus_value = None
            self.honest_agents_won = False
            if VERBOSE:
                print("[GAME DEBUG] Deadline reached without stop vote -> honest agents LOST")
    
    def get_game_state(self) -> Dict:
        """Get a summary of the current game state."""
        return {
            "round": self.current_round,
            "num_honest": self.num_honest,
            "num_byzantine": self.num_byzantine,
            "max_rounds": self.max_rounds,
            "rounds_until_deadline": max(0, self.max_rounds - self.current_round),
            "game_over": self.game_over,
            "consensus_reached": self.consensus_reached,
            "consensus_value": self.consensus_value,
            "honest_agents_won": self.honest_agents_won,
            "agent_states": {
                agent_id: {
                    # Hide Byzantine identity from agents
                    # "is_byzantine": agent.is_byzantine,  # HIDDEN
                    "initial_value": agent.initial_value,
                    "current_value": agent.current_value,
                    "proposed_value": agent.proposed_value
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive game statistics for all research questions (Q1, Q2, Q3)."""
        if not self.rounds:
            return {}
        
        # === Agent identification ===
        honest_agent_ids = [
            agent_id for agent_id, agent in self.agents.items()
            if not agent.is_byzantine
        ]
        byzantine_agent_ids = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.is_byzantine
        ]
        
        honest_initial_values = [
            agent.initial_value
            for agent in self.agents.values()
            if not agent.is_byzantine and agent.initial_value is not None
        ]
        
        # Filter out None values (agents that abstained)
        honest_final_values = [
            agent.current_value
            for agent in self.agents.values()
            if not agent.is_byzantine and agent.current_value is not None
        ]
        
        # === Byzantine agent tracking ===
        byzantine_initial_values = [
            agent.initial_value
            for agent in self.agents.values()
            if agent.is_byzantine
        ] if self.num_byzantine > 0 else []
        
        byzantine_final_values = [
            agent.current_value
            for agent in self.agents.values()
            if agent.is_byzantine
        ] if self.num_byzantine > 0 else []
        
        # === Compute honest initial statistics ===
        # Guard against empty lists (all agents abstained - shouldn't happen but be safe)
        if not honest_initial_values:
            honest_initial_mean = 0.0
            honest_initial_median = 0.0
            honest_initial_std = 0.0
            honest_initial_min = 0
            honest_initial_max = 0
        else:
            honest_initial_mean = mean(honest_initial_values)
            honest_initial_median = median(honest_initial_values)
            honest_initial_std = stdev(honest_initial_values) if len(honest_initial_values) > 1 else 0.0
            honest_initial_min = min(honest_initial_values)
            honest_initial_max = max(honest_initial_values)
        
        # === Q1 Metrics: Value trajectory per round ===
        value_std_per_round = [r.honest_std for r in self.rounds]
        trajectory_stability = mean(value_std_per_round) if value_std_per_round else 0.0
        
        # === Check honest unanimity (did all honest agents agree?) ===
        # This is separate from "valid consensus" (agreement on initial value)
        if not honest_final_values:
            honest_final_std = 0.0
            honest_unanimous = False
            unanimous_value = None
        else:
            honest_final_std = stdev(honest_final_values) if len(honest_final_values) > 1 else 0.0
            honest_unanimous = (honest_final_std == 0.0 and len(honest_final_values) > 0)
            unanimous_value = honest_final_values[0] if honest_unanimous else None
        
        # === Determine consensus outcome: "valid" | "invalid" | "none" | "timeout" ===
        # "valid": honest agents unanimously agreed on an honest initial value AND voted to stop
        # "invalid": honest agents unanimously agreed, but on a NON-initial value AND voted to stop
        # "timeout": honest agents may have agreed, but hit deadline without voting to stop
        # "none": honest agents did NOT all agree
        if self.termination_reason == "max_rounds":
            # Hit deadline - this is a failure regardless of agreement state
            consensus_outcome = "timeout"
        elif not honest_unanimous:
            consensus_outcome = "none"
        elif unanimous_value in honest_initial_values:
            consensus_outcome = "valid"
        else:
            consensus_outcome = "invalid"  # Unanimous but on non-initial value
        
        # === Q1 Metrics: Convergence speed (rounds to first consensus) ===
        convergence_speed = None
        for i, r in enumerate(self.rounds):
            if r.has_consensus:
                convergence_speed = i + 1  # 1-indexed round number
                break
        
        # === Q1 Metrics: Consensus value preference analysis ===
        # Value range context - needed to interpret other metrics
        initial_value_range = honest_initial_max - honest_initial_min
        
        consensus_is_median = False
        consensus_is_extreme = False
        consensus_is_initial = False
        consensus_distance_from_median = None
        
        if self.consensus_value is not None and honest_initial_values:
            consensus_is_initial = self.consensus_value in honest_initial_values
            consensus_is_median = self.consensus_value == int(honest_initial_median)
            # Only consider "extreme" if there's meaningful spread (range >= 2)
            # AND consensus is at min/max (not both when range is tiny)
            if initial_value_range >= 2:
                consensus_is_extreme = self.consensus_value in [honest_initial_min, honest_initial_max]
            else:
                consensus_is_extreme = False  # Not meaningful with tiny range
            consensus_distance_from_median = abs(self.consensus_value - honest_initial_median)
        
        # === Q2 Metrics: Stability (consecutive consensus rounds before stop) ===
        stability_rounds = 0
        for r in reversed(self.rounds):
            if r.has_consensus:
                stability_rounds += 1
            else:
                break
        
        # === Q2 Metrics: Centrality (how close to honest median) ===
        # centrality = 1 - (|consensus - honest_median| / max_distance)
        max_distance = max(honest_initial_max - honest_initial_min, 1)  # Avoid div by 0
        if self.consensus_value is not None:
            centrality = 1.0 - (abs(self.consensus_value - honest_initial_median) / max_distance)
            centrality = max(0.0, min(1.0, centrality))  # Clamp to [0, 1]
        else:
            centrality = None
        
        # Calculate distance metrics
        if self.consensus_value is not None and honest_initial_values:
            # Filter out None values for safety
            valid_initial_values = [
                agent.initial_value for agent in self.agents.values()
                if not agent.is_byzantine and agent.initial_value is not None
            ]
            if valid_initial_values:
                avg_distance_from_consensus = mean([
                    abs(v - self.consensus_value) for v in valid_initial_values
                ])
            else:
                avg_distance_from_consensus = None
            
            # Agreement rate: % of honest agents who voted for consensus (= inclusivity)
            final_round = self.rounds[-1]
            agreement_rate = (final_round.agreement_count / len(honest_final_values)) * 100 if honest_final_values else 0
            inclusivity = agreement_rate / 100.0  # Normalized to [0, 1]
            
            # Byzantine infiltration: did Byzantine agents vote for consensus?
            byzantine_consensus_votes = sum(
                1 for agent in self.agents.values()
                if agent.is_byzantine and agent.current_value is not None and int(agent.current_value) == self.consensus_value
            )
            byzantine_infiltration = (byzantine_consensus_votes / self.num_byzantine * 100) if self.num_byzantine > 0 else None
            
            # === Q2 Metrics: Byzantine Resistance Score ===
            # Score = 50 × validity + 30 × centrality + 20 × efficiency
            validity = 1.0 if consensus_outcome == "valid" else 0.0
            efficiency = 1.0 - (len(self.rounds) / self.max_rounds) if self.max_rounds > 0 else 0.0
            efficiency = max(0.0, efficiency)
            consensus_quality_score = 50 * validity + 30 * centrality + 20 * efficiency
        else:
            avg_distance_from_consensus = None
            consensus_quality_score = 0.0  # No consensus = 0 quality
            agreement_rate = None
            inclusivity = None
            byzantine_infiltration = None
        
        # === Round-by-round data for trajectory visualization ===
        rounds_data = [
            {
                "round": r.round_num,
                "honest_values": r.honest_values,  # Individual values per honest agent
                "byzantine_values": r.byzantine_values if self.num_byzantine > 0 else [],
                "honest_mean": r.honest_mean,
                "honest_std": r.honest_std,
                "convergence_metric": r.convergence_metric,
                "has_consensus": r.has_consensus,
                "consensus_value": r.consensus_value,  # What value they agreed on (if any)
                "agreement_count": r.agreement_count,  # How many agents agreed
            }
            for r in self.rounds
        ]
        
        # === Q3 Metrics: Keyword detection for Byzantine awareness ===
        # Keywords to detect: suspicious, outlier, byzantine, dishonest, lying, manipulat, mislead, decept, attack
        detection_keywords = ["suspicious", "outlier", "byzantine", "dishonest", "lying", 
                             "manipulat", "mislead", "decept", "attack", "malicious", "adversar"]
        
        keyword_counts = {kw: 0 for kw in detection_keywords}
        total_reasoning_length = 0
        honest_reasoning_count = 0
        
        # Only analyze honest agent reasoning (Byzantine agents don't count)
        for round_data in self.all_reasoning:
            for agent_id, reasoning in round_data.get("reasoning", {}).items():
                # Skip Byzantine agents
                if agent_id in byzantine_agent_ids:
                    continue
                    
                if reasoning:
                    total_reasoning_length += len(reasoning)
                    honest_reasoning_count += 1
                    reasoning_lower = reasoning.lower()
                    for kw in detection_keywords:
                        if kw in reasoning_lower:
                            keyword_counts[kw] += 1
        
        # Total keyword mentions
        total_keyword_mentions = sum(keyword_counts.values())
        
        return {
            # Game configuration
            "num_honest": self.num_honest,
            "num_byzantine": self.num_byzantine,
            "total_agents": self.total_agents,
            "value_range": list(self.value_range),
            
            # Agent identification (which agents were Byzantine)
            "honest_agent_ids": honest_agent_ids,
            "byzantine_agent_ids": byzantine_agent_ids,
            
            # Basic info
            "total_rounds": len(self.rounds),
            "max_rounds": self.max_rounds,
            "consensus_threshold": self.consensus_threshold,
            
            # Consensus outcome
            "consensus_reached": self.consensus_reached,
            "consensus_value": self.consensus_value,
            "consensus_outcome": consensus_outcome,  # "valid" | "invalid" | "timeout" | "none"
            "consensus_is_valid": consensus_outcome == "valid",  # True only if valid consensus + voted to stop
            "honest_unanimous": honest_unanimous,  # Did all honest agents agree on final round?
            "unanimous_value": unanimous_value,  # The value they agreed on (or None)
            "honest_agents_won": self.honest_agents_won,
            
            # Honest agent initial statistics
            "honest_initial_values": honest_initial_values,  # NEW: for reuse analysis
            "honest_initial_mean": honest_initial_mean,
            "honest_initial_median": honest_initial_median,  # NEW
            "honest_initial_std": honest_initial_std,
            "honest_initial_min": honest_initial_min,  # NEW
            "honest_initial_max": honest_initial_max,  # NEW
            
            # Honest agent final statistics
            "honest_final_values": honest_final_values,  # The actual final values of each honest agent
            "honest_final_mean": mean(honest_final_values) if honest_final_values else 0.0,
            "honest_final_std": stdev(honest_final_values) if len(honest_final_values) > 1 else 0.0,
            
            # Byzantine agent statistics (only if Byzantine agents exist)
            "byzantine_initial_values": byzantine_initial_values if self.num_byzantine > 0 else None,
            "byzantine_final_values": byzantine_final_values if self.num_byzantine > 0 else None,
            
            # Q1 Metrics: Convergence
            "convergence_speed": convergence_speed,  # NEW: rounds to first consensus
            "convergence_rate": len([r for r in self.rounds if r.has_consensus]) / len(self.rounds),
            "final_convergence_metric": self.rounds[-1].convergence_metric if self.rounds else None,
            
            # Q1 Metrics: Consensus value preference
            "consensus_is_median": consensus_is_median,  # NEW
            "consensus_is_extreme": consensus_is_extreme,  # NEW
            "consensus_is_initial": consensus_is_initial,  # NEW
            "consensus_distance_from_median": consensus_distance_from_median,  # NEW
            
            # Q1 Metrics: Trajectory stability
            "value_std_per_round": value_std_per_round,  # NEW: for trajectory plots
            "trajectory_stability": trajectory_stability,  # NEW: avg std across rounds
            
            # Q2 Metrics: Byzantine resistance
            "centrality": centrality,  # NEW: closeness to honest median
            "inclusivity": inclusivity,  # NEW: % honest voting for consensus (0-1)
            "stability_rounds": stability_rounds,  # NEW: consecutive consensus rounds
            "consensus_quality_score": consensus_quality_score,  # Quality of consensus (0-100)
            
            # Q2 Metrics: Byzantine impact
            "avg_distance_from_consensus": avg_distance_from_consensus,
            "agreement_rate": agreement_rate,
            "byzantine_infiltration": byzantine_infiltration,
            
            # Q3 Metrics: Keyword detection for prompt analysis
            "keyword_counts": keyword_counts,  # Dict of keyword -> count
            "total_keyword_mentions": total_keyword_mentions,  # Sum of all keyword mentions
            "honest_reasoning_count": honest_reasoning_count,  # Total reasoning entries from honest agents
            
            # Game termination info
            "termination_reason": self.termination_reason,  # "vote_with_consensus" | "vote_without_consensus" | "max_rounds"
            "initial_value_range": initial_value_range,  # Context for interpreting other metrics
            
            # First 1/2 stop milestone info
            "first_half_stop_reached": self.first_half_stop_reached,
            "first_half_stop_info": self.first_half_stop_info,  # Comprehensive dict or None
            
            # Round-by-round data for visualization
            "rounds_data": rounds_data,  # NEW: structured round data
        }
