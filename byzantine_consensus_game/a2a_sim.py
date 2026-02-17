"""
A2A-Sim Protocol Implementation
Agent-to-Agent Simulation Protocol for Byzantine Consensus Games
Inheriting from CommunicationProtocol base class.

Based on the specification from the a2a-sim.pdf:
- Synchronous round-based message exchange
- Dual payload: structured decision + natural language reasoning
- Local delivery to neighbors only
- Deterministic serialization for reproducibility
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set
from enum import Enum

from communication_protocol import Message, ProtocolClient, CommunicationProtocol


class Phase(str, Enum):
    """Protocol phases for Byzantine consensus"""
    PROPOSE = "propose"
    PREPARE = "prepare"
    COMMIT = "commit"
    CUSTOM = "custom"
   

class DecisionType(str, Enum):
    """Types of decisions agents can make"""
    VALUE = "value"
    VOTE = "vote"
    ABSTAIN = "abstain"


@dataclass
class Decision:
    """Structured action component of a message"""
    type: str  # DecisionType value
    value: Any  # Consensus value (e.g., integer, None for abstain)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "value": self.value}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Decision':
        return cls(type=data["type"], value=data["value"])


@dataclass
class A2AMessage(Message):
    """
    A2A-Sim message schema conforming to specification.
    
    All messages are JSON-serializable and contain:
    - sender_id, receiver_id: Point-to-point routing
    - round, phase: Synchronization markers
    - decision: Machine-readable action
    - reasoning: Human-readable LLM explanation
    - timestamp: Total ordering for duplicate suppression
    """
    sender_id: int
    receiver_id: int
    round: int
    phase: str  # Phase enum value
    decision: Decision
    reasoning: str  # LLM-generated explanation (≤ 500 chars)
    timestamp: int
    
    def __post_init__(self):
        """Validate message fields"""
        # Enforce reasoning length limit per spec
        if len(self.reasoning) > 500:
            self.reasoning = self.reasoning[:497] + "..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict"""
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "round": self.round,
            "phase": self.phase,
            "decision": self.decision.to_dict(),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2AMessage':
        """Deserialize from dict"""
        return cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            round=data["round"],
            phase=data["phase"],
            decision=Decision.from_dict(data["decision"]),
            reasoning=data["reasoning"],
            timestamp=data["timestamp"]
        )
    
    def __hash__(self):
        """Enable duplicate suppression via set operations"""
        # Include receiver_id so messages to different receivers are distinct
        return hash((self.sender_id, self.receiver_id, self.round, self.phase, self.timestamp))
    
    def __eq__(self, other):
        """Duplicate detection: same sender, receiver, round, phase, timestamp"""
        if not isinstance(other, A2AMessage):
            return False
        return (self.sender_id == other.sender_id and
                self.receiver_id == other.receiver_id and
                self.round == other.round and
                self.phase == other.phase and
                self.timestamp == other.timestamp)


class A2ASimProtocol(CommunicationProtocol):
    """
    A2A-Sim Protocol Implementation
    
    Manages message routing over static network topology G=(V,E).
    Provides synchronous message delivery with:
    - Point-to-point routing to neighbors only
    - Duplicate suppression
    - Inbox ordering by sender_id, then timestamp
    - Deterministic serialization
    
    Assumptions (per spec):
    - Static undirected graph G=(V,E)
    - Synchronous rounds: all messages in round t arrive before round t+1
    - No message loss, delay, or reordering (idealized channel)
    - Total order per sender preserved
    """
    
    def __init__(self, num_agents: int, topology: Dict[int, List[int]]):
        """
        Initialize A2A-Sim protocol.
        
        Args:
            num_agents: Total number of agents n (V = {0, 1, ..., n-1})
            topology: Adjacency list defining edges E
                     topology[i] = [j1, j2, ...] means (i,j1), (i,j2), ... ∈ E
        """
        super().__init__(num_agents, topology)
        self.num_agents = num_agents
        self.topology = topology  # Graph G=(V,E) as adjacency list
        
        # Message buffers: round -> receiver_id -> [messages]
        self.message_buffer: Dict[int, Dict[int, List[A2AMessage]]] = {}
        
        # Delivered messages tracking (for duplicate suppression)
        self.delivered: Set[A2AMessage] = set()
        
        # Current protocol state
        self.current_round = 0
        self.current_phase = Phase.PROPOSE.value
    
    def send_message(self, sender_id: int, receiver_id: int, message: A2AMessage):
        """
        Send message from sender to receiver (point-to-point).
        
        Messages are buffered until deliver_messages() is called for the round.
        Performs duplicate suppression per spec.
        
        Args:
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID  
            message: A2AMessage to send
        """
        # Validate sender can reach receiver (must be neighbor)
        if receiver_id not in self.topology.get(sender_id, []):
            raise ValueError(f"Agent {sender_id} cannot send to {receiver_id}: not in neighbor set")
        
        # Duplicate suppression
        if message in self.delivered:
            return
        
        # Buffer message for delivery
        round_buffer = self.message_buffer.setdefault(message.round, {})
        inbox = round_buffer.setdefault(receiver_id, [])
        inbox.append(message)
        self.delivered.add(message)
    
    def broadcast_to_neighbors(self, sender_id: int, round: int, phase: str,
                               decision: Decision, reasoning: str, timestamp: int):
        """
        Multi-cast illusion: Send identical message to all neighbors.
        
        Per spec: "All neighbors receive identical decision + reasoning content"
        
        Args:
            sender_id: Broadcasting agent
            round: Current round t
            phase: Protocol phase
            decision: Structured decision
            reasoning: LLM explanation
            timestamp: Monotonic counter for ordering
        """
        neighbors = self.topology.get(sender_id, [])
        
        for neighbor_id in neighbors:
            message = A2AMessage(
                sender_id=sender_id,
                receiver_id=neighbor_id,
                round=round,
                phase=phase,
                decision=decision,
                reasoning=reasoning,
                timestamp=timestamp
            )
            self.send_message(sender_id, neighbor_id, message)
    
    def deliver_messages(self, agent_id: int, round: int) -> List[A2AMessage]:
        """
        Collect inbox for agent in given round
        
        Returns messages ordered by:
        1. sender_id (grouped by sender)
        2. timestamp (total order per sender)
        
        Args:
            agent_id: Receiving agent ID
            round: Round t to retrieve messages for
            
        Returns:
            List of A2AMessage sorted per spec "Inbox Ordering"
        """
        round_buffer = self.message_buffer.get(round, {})
        inbox = round_buffer.get(agent_id, [])
        
        # Inbox ordering: group by sender_id, then timestamp
        inbox_sorted = sorted(inbox, key=lambda m: (m.sender_id, m.timestamp))
        
        return inbox_sorted
    
    def clear_round_buffer(self, round: int):
        """
        Clear message buffer for completed round.
        Enables memory-efficient multi-round simulations.
        
        Args:
            round: Round to clear
        """
        if round in self.message_buffer:
            del self.message_buffer[round]
    
    def get_neighbors(self, agent_id: int) -> List[int]:
        """
        Get neighbor set N_i for agent i.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of neighbor IDs (N_i = {j | (i,j) ∈ E})
        """
        return self.topology.get(agent_id, [])
    
    def set_phase(self, round: int, phase: str):
        """
        Update current protocol round and phase.
        
        Args:
            round: New round number t
            phase: New phase (propose/prepare/commit/custom)
        """
        self.current_round = round
        self.current_phase = phase
    
    def get_message_count(self, round: int) -> int:
        """
        Get total message count for a round (debugging/metrics).
        
        Args:
            round: Round to query
            
        Returns:
            Total messages buffered for round
        """
        round_buffer = self.message_buffer.get(round, {})
        return sum(len(inbox) for inbox in round_buffer.values())
    
    def reset(self):
        """Reset protocol state (for new simulation runs)"""
        self.message_buffer.clear()
        self.delivered.clear()
        self.current_round = 0
    
    def create_client(self, agent_id: int) -> 'A2ASimClient':
        """
        Create a client instance for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            A2ASimClient for this agent
        """
        return A2ASimClient(agent_id=agent_id, protocol=self)


class A2ASimClient(ProtocolClient):
    """
    Agent-side A2A-Sim client interface 

    Provides high-level API for agents to:
    - Receive messages via step()
    - Send messages to neighbors
    - Maintain conversation history H_i
    """
    
    def __init__(self, agent_id: int, protocol: A2ASimProtocol):
        """
        Initialize client for an agent.
        
        Args:
            agent_id: This agent's unique ID
            protocol: Shared A2ASimProtocol instance
        """
        super().__init__(agent_id, protocol)
        self.protocol: A2ASimProtocol = protocol
        self.history: List[Dict[str, Any]] = []  # Persistent conversation history H_i
        self._timestamp_counter = 0  # Monotonic counter for message ordering
    
    def next_timestamp(self) -> int:
        """Generate next timestamp for outgoing message"""
        self._timestamp_counter += 1
        return self._timestamp_counter
    
    def receive_messages(self, round: int) -> List[A2AMessage]:
        """
        Collect inbox M_t^i for current round 
        
        Args:
            round: Current round t
            
        Returns:
            Sorted inbox messages
        """
        return self.protocol.deliver_messages(self.agent_id, round)
    
    def send_to_neighbors(self, round: int, phase: str, decision: Decision, reasoning: str):
        """
        Broadcast message to all neighbors (§3.4 step 3).
        
        Args:
            round: Current round t
            phase: Protocol phase
            decision: Structured decision
            reasoning: LLM explanation (will be truncated to 500 chars if needed)
        """
        timestamp = self.next_timestamp()
        self.protocol.broadcast_to_neighbors(
            sender_id=self.agent_id,
            round=round,
            phase=phase,
            decision=decision,
            reasoning=reasoning,
            timestamp=timestamp
        )
    
    def update_history(self, round: int, inbox: List[A2AMessage], local_state: Dict[str, Any]):
        """
        Append round data to persistent history H_i 
        
        Args:
            round: Round t
            inbox: Received messages M_t^i
            local_state: Agent's internal state snapshot
        """
        self.history.append({
            "round": round,
            "inbox": [msg.to_dict() for msg in inbox],
            "local_state": local_state
        })
    
    def get_neighbors(self) -> List[int]:
        """Get this agent's neighbor set N_i"""
        return self.protocol.get_neighbors(self.agent_id)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get full conversation history H_i"""
        return self.history
    
    def reset(self):
        """Reset client state for new simulation"""
        self.history.clear()
        self._timestamp_counter = 0
