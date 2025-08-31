# agents/identity_agent.py - Version corrigée
"""
Agent d'identité optimisé avec Enhanced Retrieval intégré
"""

import logging
import json
import re
import asyncio
import sys
from pathlib import Path

# Ajuster le path pour les imports
sys.path.append(str(Path(__file__).parent.parent))  # Remonter d'un niveau pour accéder à utils

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# Imports depuis base_agent
from base_agent import YAMLBaseAgent, QuestionResult, ExtractionStatus, ExtractionSource, QuestionConfig

# Imports conditionnels pour Enhanced Retrieval
try:
    from aoagentc.utils.arch0.enhanced_retrieval1 import EnhancedAORetriever
    from aoagentc.utils.arch0.vectorstore1 import get_vectorstore
    ENHANCED_RETRIEVAL_AVAILABLE = True
except ImportError:
    ENHANCED_RETRIEVAL_AVAILABLE = False
    logging.warning("Enhanced Retrieval non disponible - fonctionnement en mode basique")

logger = logging.getLogger(__name__)

class IdentityExtractionAgent(YAMLBaseAgent):
    def __init__(
        self,
        config_path: str = "aoagentc/configs/prompts/identity/en.yaml",
        model: str = "gpt-4o-mini",
        enable_cache: bool = True,
        enable_parallel: bool = True,
        vectorstore=None,
        use_enhanced_retrieval: bool = True  # Par défaut False pour éviter les erreurs
    ):
        """
        Initialise l'agent d'extraction d'identité
        
        Args:
            config_path: Chemin vers le fichier YAML de configuration
            model: Modèle OpenAI à utiliser
            enable_cache: Activer le cache
            enable_parallel: Activer l'extraction parallèle
            vectorstore: Vectorstore pour Enhanced Retrieval (optionnel)
            use_enhanced_retrieval: Utiliser Enhanced Retrieval si disponible
        """
        # Initialiser la classe parente avec les bons paramètres
        super().__init__(
            config_path=config_path,
            model=model,
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
            max_parallel=5
        )
        
        # Configuration Enhanced Retrieval
        self.use_enhanced_retrieval = use_enhanced_retrieval and ENHANCED_RETRIEVAL_AVAILABLE
        self.enhanced_retriever = None
        
        if self.use_enhanced_retrieval:
            try:
                if vectorstore is None:
                    vectorstore = get_vectorstore(collection_name="ao_documents")
                
                self.enhanced_retriever = EnhancedAORetriever(
                    vectorstore=vectorstore,
                    use_reranking=True,
                    use_compression=False
                )
                logger.info("✅ Enhanced Retrieval intégré avec succès")
            except Exception as e:
                logger.warning(f"⚠️ Impossible d'initialiser Enhanced Retrieval: {e}")
                self.use_enhanced_retrieval = False
                self.enhanced_retriever = None
    
    # ============================================================================
    # OVERRIDE: MÉTHODES D'EXTRACTION SPÉCIALISÉES
    # ============================================================================
    
    async def extract(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Override de la méthode extract pour l'identité
        """
        # Si Enhanced Retrieval est activé et qu'on n'a pas de chunks, on peut quand même chercher
        if self.use_enhanced_retrieval and not chunks:
            logger.info("🔍 Utilisation d'Enhanced Retrieval sans chunks initiaux")
            chunks = await self._get_initial_chunks()
        
        # Appeler la méthode parente
        return await super().extract(chunks)
    
    async def _get_initial_chunks(self) -> List[Document]:
        """
        Récupère des chunks initiaux via Enhanced Retrieval
        """
        if not self.enhanced_retriever:
            return []
        
        try:
            # Recherche générale pour l'identité
            initial_chunks = self.enhanced_retriever.retrieve_with_context(
                query="appel offres tender RFP client reference deadline submission",
                category="identity",
                k=10,
                fetch_k=30
            )
            logger.info(f"📚 {len(initial_chunks)} chunks initiaux récupérés")
            return initial_chunks
        except Exception as e:
            logger.error(f"Erreur récupération chunks initiaux: {e}")
            return []
    
    def _prepare_context(self, chunks: List[Document]) -> str:
        """
        Override: Prépare le contexte avec Enhanced Retrieval si disponible
        """
        if not self.use_enhanced_retrieval or not self.enhanced_retriever:
            # Utiliser la méthode parente
            return super()._prepare_context(chunks)
        
        try:
            # Enrichir le contexte avec Enhanced Retrieval
            enriched_context = self._prepare_enhanced_context(chunks)
            if enriched_context:
                return enriched_context
        except Exception as e:
            logger.warning(f"Erreur Enhanced Context, fallback sur méthode standard: {e}")
        
        return super()._prepare_context(chunks)
    
    def _prepare_enhanced_context(self, chunks: List[Document]) -> str:
        """
        Prépare un contexte enrichi avec Enhanced Retrieval
        """
        # Construire une requête globale pour l'identité
        identity_query = self._build_identity_search_query()
        
        try:
            # Recherche optimisée
            optimized_chunks = self.enhanced_retriever.retrieve_with_context(
                query=identity_query,
                category="identity",
                k=12,
                fetch_k=30
            )
            
            if not optimized_chunks:
                return ""
            
            # Construire le contexte enrichi
            context_parts = []
            seen_content = set()  # Pour éviter les doublons
            
            for i, chunk in enumerate(optimized_chunks):
                # Éviter les doublons
                content_hash = hash(chunk.page_content[:100])
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                
                # Métadonnées
                source = chunk.metadata.get('source', f'Document {i+1}')
                page = chunk.metadata.get('page', '')
                rerank_score = chunk.metadata.get('rerank_score', 0)
                
                # En-tête
                header = f"[SOURCE: {source}"
                if page:
                    header += f" | PAGE: {page}"
                if rerank_score > 0:
                    header += f" | RELEVANCE: {rerank_score:.2f}"
                header += "]"
                
                # Ajouter le contenu
                context_parts.append(f"{header}\n{chunk.page_content}")
            
            logger.info(f"✨ Contexte enrichi avec {len(context_parts)} sections optimisées")
            return "\n\n---\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Erreur préparation contexte enrichi: {e}")
            return ""
    
    def _build_identity_search_query(self) -> str:
        """
        Construit une requête de recherche globale pour l'identité
        """
        # Mots-clés essentiels pour l'identité d'un appel d'offres
        keywords = [
            "client issuer company",
            "tender reference number RFP RFQ",
            "deadline submission date",
            "submit platform portal email",
            "documents required deliverables",
            "countries regions geographical",
            "service scope objective",
            "contract type duration term"
        ]
        
        return " OR ".join(keywords)
    
    # ============================================================================
    # EXTRACTION POUR UNE QUESTION SPÉCIFIQUE
    # ============================================================================
    
    async def _extract_single_question(
        self,
        question_config: QuestionConfig,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Override: Extraction spécialisée pour les questions d'identité
        """
        # Si Enhanced Retrieval est disponible, enrichir le contexte pour cette question
        if self.use_enhanced_retrieval and self.enhanced_retriever:
            try:
                enhanced_context = await self._get_question_specific_context(
                    question_config,
                    chunks
                )
                if enhanced_context:
                    context = enhanced_context
                    logger.debug(f"🎯 Contexte spécialisé pour {question_config.id}")
            except Exception as e:
                logger.warning(f"Erreur contexte spécialisé: {e}")
        
        # Appeler la méthode parente
        return await super()._extract_single_question(question_config, context, chunks)
    
    async def _get_question_specific_context(
        self,
        question_config: QuestionConfig,
        chunks: List[Document]
    ) -> Optional[str]:
        """
        Obtient un contexte spécifique pour une question
        """
        if not self.enhanced_retriever:
            return None
        
        # Mapping des questions vers des requêtes spécialisées
        query_mapping = {
            'identity.client_name': "client name company issuer ordering party who issued tender",
            'identity.tender_reference': "reference number tender RFP RFQ procurement ID",
            'identity.timeline_milestones': "dates deadline submission calendar milestones key dates",
            'identity.submission_channel': "submit how platform portal email upload channel method",
            'identity.expected_deliverables': "documents required deliverables must provide attachments",
            'identity.operating_countries': "countries regions geographical scope areas coverage",
            'identity.service_main_scope': "objective scope service work transport logistics main",
            'identity.contract_type': "contract type framework agreement spot call-off",
            'identity.contract_duration': "duration term period validity renewable contract length"
        }
        
        # Obtenir la requête spécifique
        base_query = query_mapping.get(question_config.id)
        if not base_query:
            return None
        
        try:
            # Recherche ciblée
            specific_chunks = self.enhanced_retriever.retrieve_with_context(
                query=base_query,
                category="identity",
                k=5,
                fetch_k=15
            )
            
            if not specific_chunks:
                return None
            
            # Construire le contexte
            context_parts = []
            for chunk in specific_chunks:
                source = chunk.metadata.get('source', 'Document')
                context_parts.append(f"[{source}]\n{chunk.page_content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Erreur recherche spécifique pour {question_config.id}: {e}")
            return None
    
    # ============================================================================
    # MÉTHODES UTILITAIRES SPÉCIFIQUES À L'IDENTITÉ
    # ============================================================================
    
    def _calculate_confidence(
        self,
        answer: Any,
        sources: List[ExtractionSource],
        chunks: List[Document]
    ) -> float:
        """
        Override: Calcul de confiance spécifique pour l'identité
        """
        base_confidence = super()._calculate_confidence(answer, sources, chunks)
        
        # Ajustements spécifiques pour l'identité
        if answer:
            # Bonus pour les réponses structurées
            if isinstance(answer, dict) and len(answer) > 2:
                base_confidence += 0.1
            
            # Bonus pour les références explicites
            if isinstance(answer, str):
                if any(keyword in answer.lower() for keyword in ['ltd', 'inc', 'sa', 'gmbh']):
                    base_confidence += 0.05  # Probablement un nom d'entreprise
                if re.match(r'.*\d{4,}.*', answer):
                    base_confidence += 0.05  # Contient un numéro de référence
        
        return min(base_confidence, 1.0)
    
    def get_identity_summary(self, results: Dict) -> Dict:
        """
        Génère un résumé structuré de l'identité extraite
        """
        summary = {
            "client": None,
            "reference": None,
            "deadline": None,
            "submission_method": None,
            "scope": None,
            "countries": None,
            "contract_info": None
        }
        
        questions = results.get("questions", {})
        
        # Mapper les résultats
        mappings = {
            "identity.client_name": "client",
            "identity.tender_reference": "reference",
            "identity.timeline_milestones": "deadline",
            "identity.submission_channel": "submission_method",
            "identity.service_main_scope": "scope",
            "identity.operating_countries": "countries",
            "identity.contract_type": "contract_info"
        }
        
        for question_id, summary_key in mappings.items():
            if question_id in questions:
                result = questions[question_id]
                if result.get("status") == "success" and result.get("answer"):
                    summary[summary_key] = result["answer"]
        
        return summary


# ============================================================================
# FONCTION PRINCIPALE POUR TEST
# ============================================================================

async def main():
    """Fonction de test avec Enhanced Retrieval"""
    import logging
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Ajouter le chemin parent
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from aoagentc.utils.arch0.vectorstore1 import get_vectorstore
    
    # Configuration paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    config_file = project_root / "configs" / "prompts" / "identity" / "en.yaml"
    
    # Récupérer le vectorstore avec les 1755 documents
    vectorstore = get_vectorstore(collection_name="ao_documents")
    
    # Créer l'agent AVEC le vectorstore
    agent = IdentityExtractionAgent(
        config_path=str(config_file),
        vectorstore=vectorstore.vectorstore,  # ← Passer l'objet vectorstore interne
        use_enhanced_retrieval=True  # ← ACTIVER Enhanced Retrieval
    )
    
    print("✅ Agent initialisé avec 1755 documents!")
    print("🔍 Extraction en cours...")
    
    # L'agent va maintenant chercher dans les documents
    results = await agent.extract([])
    
    # Afficher les résultats
    print("\n" + "="*60)
    print("RÉSULTATS D'EXTRACTION D'IDENTITÉ")
    print("="*60)
    
    for question_id, result in results["questions"].items():
        print(f"\n📋 {question_id}:")
        print(f"   Status: {result['status']}")
        print(f"   Answer: {result['answer']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        if result.get('sources'):
            print(f"   Sources: {len(result['sources'])} documents")
    
    # Résumé
    summary = agent.get_identity_summary(results)
    print("\n" + "="*60)
    print("RÉSUMÉ DE L'IDENTITÉ")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\n⏱️ Temps total: {results['stats']['total_time']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())