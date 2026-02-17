"""
Agent Network Infrastructure
Base layer for multi-agent communication networks with configurable topology.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from communication_protocol import CommunicationProtocol, ProtocolClient, Message
from a2a_sim import Decision, Phase


@dataclass
class NetworkTopology:
    """
    Defines the communication topology for agent network.
    """
    num_agents: int
    adjacency_list: Dict[int, List[int]]  # agent_id -> [neighbor_ids]
    topology_type: str  # 'fully_connected', 'ring', 'grid', 'custom'
    
    @classmethod
    def fully_connected(cls, num_agents: int) -> 'NetworkTopology':
        """Create a fully connected network (all-to-all)."""
        adjacency_list = {
            i: [j for j in range(num_agents) if j != i]
            for i in range(num_agents)
        }
        return cls(
            num_agents=num_agents,
            adjacency_list=adjacency_list,
            topology_type='fully_connected'
        )
    
    @classmethod
    def ring(cls, num_agents: int) -> 'NetworkTopology':
        """Create a ring network (each agent connected to 2 neighbors)."""
        adjacency_list = {
            i: [(i - 1) % num_agents, (i + 1) % num_agents]
            for i in range(num_agents)
        }
        return cls(
            num_agents=num_agents,
            adjacency_list=adjacency_list,
            topology_type='ring'
        )
    
    @classmethod
    def grid(cls, rows: int, cols: int) -> 'NetworkTopology':
        """Create a 2D grid network."""
        num_agents = rows * cols
        adjacency_list = {}
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                neighbors = []
                
                # Up
                if i > 0:
                    neighbors.append((i - 1) * cols + j)
                # Down
                if i < rows - 1:
                    neighbors.append((i + 1) * cols + j)
                # Left
                if j > 0:
                    neighbors.append(i * cols + (j - 1))
                # Right
                if j < cols - 1:
                    neighbors.append(i * cols + (j + 1))
                
                adjacency_list[idx] = neighbors
        
        return cls(
            num_agents=num_agents,
            adjacency_list=adjacency_list,
            topology_type='grid'
        )
    
    @classmethod
    def custom(cls, adjacency_list: Dict[int, List[int]]) -> 'NetworkTopology':
        """Create a custom topology from adjacency list."""
        num_agents = len(adjacency_list)
        return cls(
            num_agents=num_agents,
            adjacency_list=adjacency_list,
            topology_type='custom'
        )


class AgentNetwork:
    """
    Base agent network infrastructure.
    Manages agent communication over configurable topology using pluggable protocols.
    """
    
    def __init__(
        self,
        topology: NetworkTopology,
        protocol: CommunicationProtocol,
        agents: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent network.
        
        Args:
            topology: Network topology configuration
            protocol: Communication protocol instance (injected dependency)
            agents: Optional dict of agent_id -> agent instance
        """
        self.topology = topology
        self.num_agents = topology.num_agents
        
        # Use injected protocol (any implementation of CommunicationProtocol)
        self.protocol = protocol
        
        # Agent management
        self.agents: Dict[str, Any] = agents or {}
        self.agent_id_to_index: Dict[str, int] = {}
        self.index_to_agent_id: Dict[int, str] = {}
        self.clients: Dict[str, ProtocolClient] = {}
        
        # Network state
        self.current_round = 0
        self.message_history: List[Message] = []
    
    def register_agent(self, agent_id: str, agent: Any, agent_index: int):
        """
        Register an agent in the network.
        
        Args:
            agent_id: Unique agent identifier
            agent: Agent instance
            agent_index: Integer index for A2A protocol (0 to n-1)
        """
        self.agents[agent_id] = agent
        self.agent_id_to_index[agent_id] = agent_index
        self.index_to_agent_id[agent_index] = agent_id
        
        # Create protocol client for this agent using factory method
        client = self.protocol.create_client(agent_index)
        self.clients[agent_id] = client
        
        # Set protocol client on agent if it has the method
        if hasattr(agent, 'set_a2a_client'):
            agent.set_a2a_client(client)
    
    def broadcast_message(
        self,
        sender_id: str,
        round_num: int,
        phase: Phase,
        decision: Decision,
        reasoning: str
    ):
        """
        Broadcast a message from sender to all neighbors.
        
        Args:
            sender_id: Sender agent ID
            round_num: Current round number
            phase: Protocol phase
            decision: Structured decision
            reasoning: Natural language reasoning
        """
        client = self.clients[sender_id]
        
        # Send to all neighbors using protocol client
        # Note: For A2A-Sim, this requires phase, decision, reasoning
        # Other protocols might have different parameters
        client.send_to_neighbors(
            round=round_num,
            phase=phase.value,
            decision=decision,
            reasoning=reasoning
        )
    
    def get_messages(
        self,
        receiver_id: str,
        round_num: int,
        phase: Phase
    ) -> List[Message]:
        """
        Get messages for a specific receiver in a round/phase.
        
        Args:
            receiver_id: Receiver agent ID
            round_num: Round number
            phase: Protocol phase
            
        Returns:
            List of messages
        """
        client = self.clients[receiver_id]
        return client.receive_messages(round=round_num)
    
    def advance_round(self):
        """Advance to next round."""
        self.current_round += 1
    
    def get_conversation_history(
        self,
        agent_id: str,
        max_messages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for an agent.
        
        Args:
            agent_id: Agent ID
            max_messages: Maximum number of recent messages to return
            
        Returns:
            List of history entries (each contains round, inbox, local_state)
        """
        client = self.clients[agent_id]
        history = client.get_history()
        
        if max_messages:
            return history[-max_messages:]
        return history
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        # Calculate total messages from protocol
        total_messages = sum(
            self.protocol.get_message_count(r) 
            for r in range(self.current_round)
        )
        
        return {
            'num_agents': self.num_agents,
            'topology_type': self.topology.topology_type,
            'current_round': self.current_round,
            'total_messages': total_messages,
            'avg_degree': sum(len(neighbors) for neighbors in self.topology.adjacency_list.values()) / self.num_agents
        }
