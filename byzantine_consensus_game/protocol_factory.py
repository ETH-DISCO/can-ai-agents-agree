"""
Protocol Factory
Creates communication protocol instances based on configuration.
"""

from typing import Dict, List, Any, Optional
from communication_protocol import CommunicationProtocol
from a2a_sim import A2ASimProtocol


def create_protocol(
    protocol_type: str,
    num_agents: int,
    topology: Dict[int, List[int]],
    config: Optional[Dict[str, Any]] = None
) -> CommunicationProtocol:
    """
    Factory function to create protocol instances based on type.
    
    Args:
        protocol_type: Type of protocol to create ('a2a_sim')
        num_agents: Number of agents in the network
        topology: Network topology as adjacency list {agent_id: [neighbor_ids]}
        config: Optional protocol-specific configuration dictionary
        
    Returns:
        CommunicationProtocol instance
        
    Raises:
        ValueError: If protocol_type is unknown
    """
    config = config or {}
    
    if protocol_type == "a2a_sim":
        # A2A-Sim: Synchronous message exchange with dual payload
        return A2ASimProtocol(
            num_agents=num_agents,
            topology=topology
        )
    else:
        raise ValueError(
            f"Unknown protocol type: '{protocol_type}'. "
            f"Supported types: 'a2a_sim'"
        )
