# agents/__init__.py
"""
Package des agents spécialisés pour l'analyse d'appels d'offres
"""

from .base_agent import BaseExtractionAgent
from .identity_agent_arch import IdentityExtractionAgent
from .volume_agent import VolumeExtractionAgent
from .financial_agent import FinancialExtractionAgent
from .legal_agent import LegalExtractionAgent
from .operational_agent import OperationalExtractionAgent
from .timeline_agent import TimelineExtractionAgent
from .orchestrator import AOExtractionOrchestrator

__all__ = [
    'BaseExtractionAgent',
    'IdentityExtractionAgent',
    'VolumeExtractionAgent',
    'FinancialExtractionAgent',
    'LegalExtractionAgent',
    'OperationalExtractionAgent',
    'TimelineExtractionAgent',
    'AOExtractionOrchestrator'
]

# Mapping des agents par catégorie pour faciliter l'utilisation
AGENT_REGISTRY = {
    'identity': IdentityExtractionAgent,
    'volume': VolumeExtractionAgent,
    'financial': FinancialExtractionAgent,
    'legal': LegalExtractionAgent,
    'operational': OperationalExtractionAgent,
    'timeline': TimelineExtractionAgent
}

def get_agent(category: str):
    """
    Factory pour obtenir un agent par catégorie
    
    Args:
        category: Nom de la catégorie d'agent
        
    Returns:
        Instance de l'agent correspondant
        
    Raises:
        ValueError: Si la catégorie n'existe pas
    """
    if category not in AGENT_REGISTRY:
        raise ValueError(f"Agent category '{category}' not found. Available: {list(AGENT_REGISTRY.keys())}")
    
    return AGENT_REGISTRY[category]()