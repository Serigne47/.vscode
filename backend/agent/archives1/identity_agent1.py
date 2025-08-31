# agents/identity_agent.py - Version finale fonctionnelle
"""
Agent d'identité pour extraction depuis vectorstore
"""

import logging
import json
import re
import asyncio
import sys
import os
from pathlib import Path

# Configuration des paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# Imports depuis base_agent
from agent.base_agent import YAMLBaseAgent, QuestionResult, ExtractionStatus, ExtractionSource, QuestionConfig

logger = logging.getLogger(__name__)

class IdentityExtractionAgent(YAMLBaseAgent):
    def __init__(
        self,
        config_path: str = None,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,  # Temperature par défaut pour gpt-4o-mini
        enable_cache: bool = True,
        enable_parallel: bool = True,
        vectorstore=None
    ):
        """
        Initialise l'agent d'extraction d'identité
        """
        # Si aucun chemin n'est fourni, utiliser le chemin par défaut
        if config_path is None:
            config_path = str(project_root / "configs" / "prompts" / "identity" / "en.yaml")
        
        # Initialiser la classe parente
        super().__init__(
            config_path=config_path,
            model=model,
            temperature=temperature,
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
            max_parallel=5
        )
        
        self.vectorstore = vectorstore
        logger.info(f"✅ IdentityExtractionAgent initialisé avec {len(self.questions)} questions")
    
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
        
        # Mapper les résultats selon les IDs du YAML
        mappings = {
            "identity.client_name": "client",
            "identity.tender_reference": "reference",
            "identity.timeline_milestones": "deadline",
            "identity.submission_channel": "submission_method",
            "identity.service_main_scope": "scope",
            "identity.operating_countries_corridors": "countries",
            "identity.contract_type": "contract_info"
        }
        
        for question_id, summary_key in mappings.items():
            if question_id in questions:
                result = questions[question_id]
                if result.get("status") == "success" and result.get("answer"):
                    summary[summary_key] = result["answer"]
        
        return summary


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

async def main():
    """Fonction principale avec récupération manuelle des chunks"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import du vectorstore
    from aoagentc.utils.arch0.vectorstore1 import get_vectorstore
    
    print("\n" + "="*60)
    print("AGENT D'EXTRACTION D'IDENTITÉ")
    print("="*60)
    
    # 1. Charger le vectorstore
    print("\n📚 Chargement du vectorstore...")
    vs = get_vectorstore(collection_name="ao_documents")
    stats = vs.get_collection_stats()
    print(f"✅ Vectorstore chargé: {stats['document_count']} documents")
    
    if stats['document_count'] == 0:
        print("❌ ERREUR: Le vectorstore est vide!")
        print("Exécutez d'abord: python utils/vectorize_documents.py")
        return
    
    # 2. Récupérer des chunks pertinents pour l'identité
    print("\n🔍 Recherche de chunks pertinents pour l'identité...")
    
    # Requêtes ciblées pour chaque aspect de l'identité
    search_queries = [
        "client company name issuer who issued tender RFP",
        "tender reference number ID RFP RFQ procurement",
        "deadline submission date when submit response",
        "submit email platform portal how send proposal",
        "deliverables documents required attachments provide",
        "countries regions geographical scope areas coverage",
        "service transport maritime air road rail scope",
        "contract type framework agreement spot duration",
        "price cost budget payment terms conditions"
    ]
    
    all_chunks = []
    seen_content = set()  # Pour éviter les doublons
    
    for query in search_queries:
        try:
            # Recherche avec le vectorstore
            results = vs.similarity_search(query, k=5)
            
            # Ajouter sans doublons
            for doc in results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    all_chunks.append(doc)
                    seen_content.add(content_hash)
            
            print(f"  ✓ '{query[:40]}...': {len(results)} chunks")
            
        except Exception as e:
            print(f"  ✗ Erreur recherche '{query[:40]}...': {e}")
    
    print(f"\n📄 Total: {len(all_chunks)} chunks uniques récupérés")
    
    if not all_chunks:
        print("❌ Aucun chunk trouvé! Vérifiez votre vectorstore.")
        return
    
    # 3. Créer l'agent
    print("\n🤖 Initialisation de l'agent d'identité...")
    agent = IdentityExtractionAgent(
        temperature=1.0  # Temperature correcte pour gpt-4o-mini
    )
    
    # 4. Extraire les informations
    print(f"\n🔄 Extraction en cours depuis {len(all_chunks)} chunks...")
    print("⏳ Cela peut prendre quelques secondes...")
    
    try:
        results = await agent.extract(all_chunks)
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Afficher les résultats détaillés
    print("\n" + "="*60)
    print("RÉSULTATS D'EXTRACTION D'IDENTITÉ")
    print("="*60)
    
    questions = results.get("questions", {})
    
    for question_id, result in questions.items():
        status = result.get('status', 'unknown')
        answer = result.get('answer')
        confidence = result.get('confidence', 0)
        
        # Symbole selon le statut
        if status == "success":
            symbol = "✅"
        elif status == "partial":
            symbol = "⚠️"
        else:
            symbol = "❌"
        
        print(f"\n{symbol} {question_id}:")
        print(f"   Status: {status}")
        print(f"   Answer: {answer}")
        print(f"   Confidence: {confidence:.2%}")
        
        # Afficher les sources si disponibles
        sources = result.get('sources', [])
        if sources:
            print(f"   Sources: {len(sources)} document(s)")
            for i, source in enumerate(sources[:2], 1):  # Max 2 sources
                if isinstance(source, dict):
                    doc_name = source.get('document', 'Unknown')
                    snippet = source.get('context_snippet', '')[:100]
                    print(f"     {i}. {doc_name}: {snippet}...")
    
    # 6. Résumé consolidé
    summary = agent.get_identity_summary(results)
    
    print("\n" + "="*60)
    print("RÉSUMÉ DE L'IDENTITÉ DU TENDER")
    print("="*60)
    
    has_results = False
    for key, value in summary.items():
        if value:
            print(f"✅ {key.upper()}: {value}")
            has_results = True
        else:
            print(f"❌ {key.upper()}: Non trouvé")
    
    if not has_results:
        print("\n⚠️ Aucune information d'identité extraite.")
        print("Causes possibles:")
        print("- Les documents ne contiennent pas ces informations")
        print("- Les prompts dans le YAML nécessitent un ajustement")
        print("- Le format des documents est différent de ce qui est attendu")
    
    # 7. Statistiques
    stats = results.get('stats', {})
    print(f"\n📊 STATISTIQUES:")
    print(f"   Questions traitées: {stats.get('total_questions', 0)}")
    print(f"   Extractions réussies: {stats.get('successful_extractions', 0)}")
    print(f"   Extractions échouées: {stats.get('failed_extractions', 0)}")
    print(f"   Temps total: {stats.get('total_time', 0):.2f}s")
    
    # 8. Afficher un échantillon de chunks pour debug (optionnel)
    if len(all_chunks) > 0:
        print("\n" + "="*60)
        print("ÉCHANTILLON DE CHUNKS ANALYSÉS (pour debug)")
        print("="*60)
        
        for i, chunk in enumerate(all_chunks[:2], 1):
            print(f"\n📄 Chunk {i}:")
            print(f"   Source: {chunk.metadata.get('source', 'N/A')}")
            print(f"   Type: {chunk.metadata.get('type', 'N/A')}")
            print(f"   Page: {chunk.metadata.get('page', 'N/A')}")
            
            # Afficher le début du contenu
            content_preview = chunk.page_content[:300].replace('\n', ' ')
            print(f"   Contenu: {content_preview}...")
    
    print("\n" + "="*60)
    print("FIN DE L'EXTRACTION")
    print("="*60)


if __name__ == "__main__":
    # Vérifier que l'API key est configurée
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERREUR: OPENAI_API_KEY non configurée!")
        print("Configurez la variable d'environnement OPENAI_API_KEY")
        sys.exit(1)
    
    # Lancer l'extraction
    asyncio.run(main())