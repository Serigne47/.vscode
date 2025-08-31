# utils/__init__.py
"""
Utilitaires pour l'extraction d'informations des appels d'offres
"""

import logging

logger = logging.getLogger(__name__)

# Imports conditionnels avec gestion d'erreurs
try:
    from .arch0.enhanced_retrieval1 import EnhancedAORetriever
    from .arch0.vectorstore1 import get_vectorstore
    logger.info("✅ Modules utils chargés avec succès")
except ImportError as e:
    logger.warning(f"⚠️ Certains modules utils non disponibles: {e}")
    EnhancedAORetriever = None
    get_vectorstore = None

__version__ = "1.0.0"