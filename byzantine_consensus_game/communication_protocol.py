"""
Abstract Communication Protocol Interface
Defines the base classes that all communication protocols must implement.

This allows pluggable communication protocols - just implement these interfaces
and register in the protocol factory to use in any game.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class Message(ABC):
    """
    Base message class that all protocol messages must extend.
    
    Required fields:
        sender_id: ID of sending agent
        receiver_id: ID of receiving agent
        round: Round number when message was sent
    """
    sender_id: int
    receiver_id: int
    round: int
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize message to JSON-compatible dictionary.
        
        Returns:
            Dictionary representation of message
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Deserialize message from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Message instance
        """
        pass
    
    @abstractmethod
    def __hash__(self):
        """Enable message deduplication via set operations"""
        pass
    
    @abstractmethod
    def __eq__(self, other):
        """Enable message equality checking for deduplication"""
        pass


class ProtocolClient(ABC):
    """
    Base client interface that agents use to interact with communication protocol.
    
    Each agent gets one client instance to send and receive messages.
    """
    
    def __init__(self, agent_id: int, protocol: 'CommunicationProtocol'):
        """
        Initialize client for an agent.
        
        Args:
            agent_id: Unique agent identifier
            protocol: Protocol instance this client belongs to
        """
        self.agent_id = agent_id
        self.protocol = protocol
    
    @abstractmethod
    def receive_messages(self, round: int) -> List[Message]:
        """
        Receive messages for a specific round.
        
        Args:
            round: Round number to receive messages for
            
        Returns:
            List of messages received in this round
        """
        pass
    
    @abstractmethod
    def send_to_neighbors(self, round: int, **kwargs):
        """
        Send message to all neighbors.
        
        Args:
            round: Current round number
            **kwargs: Protocol-specific message content
        """
        pass
    
    @abstractmethod
    def get_neighbors(self) -> List[int]:
        """
        Get list of neighbor agent IDs.
        
        Returns:
            List of neighbor IDs this agent can communicate with
        """
        pass
    
    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history for this agent.
        
        Returns:
            List of historical communication data
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset client state for new simulation run"""
        pass


class CommunicationProtocol(ABC):
    """
    Base communication protocol that manages message routing and delivery.
    
    Protocols define how agents communicate, what information messages contain,
    and how messages are routed through the network.
    """
    
    def __init__(self, num_agents: int, topology: Dict[int, List[int]]):
        """
        Initialize protocol.
        
        Args:
            num_agents: Total number of agents in network
            topology: Network adjacency list {agent_id: [neighbor_ids]}
        """
        self.num_agents = num_agents
        self.topology = topology
    
    @abstractmethod
    def create_client(self, agent_id: int) -> ProtocolClient:
        """
        Create a client instance for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Protocol client for this agent
        """
        pass
    
    @abstractmethod
    def send_message(self, sender_id: int, receiver_id: int, message: Message):
        """
        Send a message from sender to receiver.
        
        Args:
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID
            message: Message to send
        """
        pass
    
    @abstractmethod
    def deliver_messages(self, agent_id: int, round: int) -> List[Message]:
        """
        Deliver all messages to an agent for a specific round.
        
        Args:
            agent_id: Receiving agent ID
            round: Round number
            
        Returns:
            List of messages for this agent in this round
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, agent_id: int) -> List[int]:
        """
        Get neighbor set for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of neighbor IDs
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset protocol state for new simulation run"""
        pass
    
    def get_message_count(self, round: int) -> int:
        """
        Get total message count for a round (optional, for metrics).
        
        Args:
            round: Round number
            
        Returns:
            Number of messages in round
        """
        return 0
