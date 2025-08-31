# agents/identity_agent.py
"""
Agent d'identité simplifié pour GPT-4o-mini
Version épurée sans sur-paramétrage
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Import des composants
from utils.vectorstore import IntelligentVectorStore
from utils.enhanced_retrieval import IntelligentRetriever, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class IdentityQuestion:
    """Structure d'une question d'identité"""
    id: str
    system_prompt: str
    validator: Dict[str, Any] = None
    
@dataclass
class IdentityAnswer:
    """Réponse structurée"""
    question_id: str
    answer: Any
    sources: List[Dict[str, str]]
    confidence: float
    status: str  # success, partial, failed
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SimpleIdentityAgent:
    """
    Agent d'extraction d'identité simplifié pour GPT-4o-mini
    """
    
    def __init__(
        self,
        vectorstore: IntelligentVectorStore,
        retriever: IntelligentRetriever,
        config_path: str = "configs/prompts/identity/en.yaml"
    ):
        """
        Initialise l'agent simplifié
        
        Args:
            vectorstore: IntelligentVectorStore configuré
            retriever: IntelligentRetriever configuré
            config_path: Chemin vers le YAML des questions
        """
        self.vectorstore = vectorstore
        self.retriever = retriever
        
        # Un seul LLM simple pour tout
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,  # Faible pour plus de précision
            max_tokens=2000
        )
        
        # Charger les questions
        self.questions = self._load_questions(config_path)
        
        # Stats basiques
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0
        }
        
        logger.info(f"✅ Agent initialisé avec {len(self.questions)} questions")
    
    def _load_questions(self, config_path: str) -> List[IdentityQuestion]:
        """Charge les questions depuis le YAML"""
        questions = []
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            for item in config:
                question = IdentityQuestion(
                    id=item['id'],
                    system_prompt=item['system_prompt'],
                    validator=item.get('validator', {})
                )
                questions.append(question)
            
            logger.info(f"📋 {len(questions)} questions chargées")
            
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            raise
        
        return questions
    
    def extract_all(self) -> Dict[str, Any]:
        """
        Extrait toutes les informations d'identité
        
        Returns:
            Dictionnaire avec toutes les réponses
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 EXTRACTION D'IDENTITÉ")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        results = {
            'timestamp': start_time.isoformat(),
            'questions': {},
            'summary': {}
        }
        
        # Traiter chaque question
        for question in self.questions:
            answer = self.extract_single(question)
            results['questions'][answer.question_id] = answer.to_dict()
        
        # Générer le résumé
        results['summary'] = self._generate_summary(results['questions'])
        
        # Stats finales
        elapsed = (datetime.now() - start_time).total_seconds()
        results['stats'] = {
            **self.stats,
            'time_seconds': elapsed
        }
        
        logger.info(f"\n✅ Extraction terminée en {elapsed:.1f}s")
        logger.info(f"   Succès: {self.stats['success']}/{self.stats['total']}")
        
        return results
    
    def extract_single(self, question: IdentityQuestion) -> IdentityAnswer:
        """
        Extrait l'information pour une question
        
        Args:
            question: Question à traiter
            
        Returns:
            IdentityAnswer avec la réponse
        """
        self.stats['total'] += 1
        logger.info(f"\n🔍 Traitement: {question.id}")
        
        try:
            # 1. Créer une requête de recherche simple
            search_query = self._create_search_query(question)
            
            # 2. Rechercher dans les documents
            retrieval_result = self.retriever.retrieve_and_answer(
                query=search_query,
                category="identity",
                require_source=True,
                max_chunks=3
            )
            
            # 3. Extraire la réponse structurée
            answer = self._extract_answer(question, retrieval_result)
            
            if answer.status == "success":
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
            
            logger.info(f"   ✅ Status: {answer.status} (confiance: {answer.confidence:.0%})")
            
            return answer
            
        except Exception as e:
            logger.error(f"   ❌ Erreur: {e}")
            self.stats['failed'] += 1
            
            return IdentityAnswer(
                question_id=question.id,
                answer=None,
                sources=[],
                confidence=0.0,
                status="failed"
            )
    
    def _create_search_query(self, question: IdentityQuestion) -> str:
        """
        Crée une requête de recherche simple basée sur l'ID
        """
        # Mapping simple des questions aux mots-clés
        query_map = {
            "identity.client_name": "client company name who issued tender",
            "identity.tender_reference": "tender reference number RFP code ID",
            "identity.timeline_milestones": "deadline submission date timeline",
            "identity.submission_channel": "submit submission email portal how",
            "identity.expected_deliverables": "deliverables documents required",
            "identity.operating_countries": "countries regions geographical scope",
            "identity.service_main_scope": "service scope transport logistics",
            "identity.contract_type": "contract type framework agreement",
            "identity.contract_duration": "contract duration period years"
        }
        
        return query_map.get(question.id, "tender information details")
    
    def _extract_answer(
        self,
        question: IdentityQuestion,
        retrieval_result: RetrievalResult
    ) -> IdentityAnswer:
        """
        Extrait la réponse depuis les chunks trouvés
        """
        # Préparer le contexte
        context = self._prepare_context(retrieval_result.chunks[:3])
        
        # Prompt simplifié pour extraction JSON
        prompt = f"""{question.system_prompt}

Context from documents:
{context}

IMPORTANT: Return ONLY a valid JSON object. No explanations.
The JSON must start with {{ and end with }}.
"""
        
        try:
            # Appeler le LLM
            response = self.llm.invoke(prompt)
            result_text = response.content.strip()
            
            # Extraire le JSON de la réponse
            json_obj = self._extract_json(result_text)
            
            # Créer la réponse
            return IdentityAnswer(
                question_id=question.id,
                answer=json_obj.get('answer'),
                sources=self._format_sources(retrieval_result),
                confidence=retrieval_result.confidence,
                status="success" if json_obj.get('answer') else "partial"
            )
            
        except Exception as e:
            logger.debug(f"Erreur extraction: {e}")
            
            # Fallback: utiliser la réponse directe du retrieval
            return IdentityAnswer(
                question_id=question.id,
                answer=retrieval_result.answer if retrieval_result.answer else None,
                sources=self._format_sources(retrieval_result),
                confidence=retrieval_result.confidence * 0.5,
                status="partial"
            )
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """
        Prépare le contexte depuis les chunks
        """
        context_parts = []
        
        for chunk in chunks:
            source = chunk.get('source', {})
            text = chunk.get('text', '')[:1000]  # Limiter la taille
            
            source_info = f"[Doc: {source.get('document', 'Unknown')}]"
            context_parts.append(f"{source_info}\n{text}")
        
        return "\n---\n".join(context_parts)
    
    def _extract_json(self, text: str) -> Dict:
        """
        Extrait un objet JSON depuis du texte
        """
        # Nettoyer le texte
        text = text.strip()
        
        # Chercher le JSON
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            json_str = text[start:end]
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Essayer de nettoyer et réessayer
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                return json.loads(json_str)
        
        # Si pas de JSON trouvé, retourner un dict vide
        return {}
    
    def _format_sources(self, retrieval_result: RetrievalResult) -> List[Dict]:
        """
        Formate les sources de manière simple
        """
        sources = []
        
        for source in retrieval_result.sources[:3]:  # Max 3 sources
            sources.append({
                'document': source.get('document', 'Unknown'),
                'page': source.get('page'),
                'snippet': source.get('text', '')[:100]
            })
        
        return sources
    
    def _generate_summary(self, questions: Dict[str, Dict]) -> Dict:
        """
        Génère un résumé simple de l'identité extraite
        """
        summary = {}
        
        # Mapping des questions aux clés du résumé
        mapping = {
            "identity.client_name": "client",
            "identity.tender_reference": "reference",
            "identity.timeline_milestones": "deadline",
            "identity.submission_channel": "submission",
            "identity.expected_deliverables": "deliverables",
            "identity.operating_countries": "countries",
            "identity.service_main_scope": "scope",
            "identity.contract_type": "contract_type",
            "identity.contract_duration": "duration"
        }
        
        for question_id, key in mapping.items():
            if question_id in questions:
                result = questions[question_id]
                if result['status'] in ['success', 'partial']:
                    summary[key] = result['answer']
                else:
                    summary[key] = None
        
        return summary


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Test de l'agent simplifié"""
    import os
    
    # Vérifier la clé API
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY non trouvée")
        return
    
    print("\n" + "="*60)
    print("🤖 AGENT D'IDENTITÉ SIMPLIFIÉ")
    print("="*60)
    
    # Imports
    from utils.vectorstore import IntelligentVectorStore
    from utils.enhanced_retrieval import IntelligentRetriever
    
    # 1. Charger le vectorstore
    print("\n📚 Chargement du vectorstore...")
    vectorstore = IntelligentVectorStore(
        persist_directory=Path("data/intelligent_store"),
        llm_model="gpt-4o-mini"  # ← Ajoutez cette ligne
    )
    stats = vectorstore.get_stats()
    print(f"✅ Vectorstore: {stats['total_documents']} documents")
    
    if stats['total_documents'] == 0:
        print("❌ Vectorstore vide! Lancez d'abord la vectorisation.")
        return
    
    # 2. Créer le retriever
    print("\n🔍 Initialisation du retriever...")
    retriever = IntelligentRetriever(
        vectorstore=vectorstore,
        llm_model="gpt-4o-mini"
    )
    
    # 3. Créer l'agent
    print("\n🤖 Création de l'agent...")
    agent = SimpleIdentityAgent(
        vectorstore=vectorstore,
        retriever=retriever,
        config_path="configs/prompts/identity/en.yaml"
    )
    
    # 4. Lancer l'extraction
    print("\n🚀 Lancement de l'extraction...")
    results = agent.extract_all()
    
    # 5. Afficher les résultats
    print("\n" + "="*60)
    print("📊 RÉSULTATS")
    print("="*60)
    
    # Résumé
    print("\n📋 IDENTITÉ EXTRAITE:")
    print("-" * 40)
    
    for key, value in results['summary'].items():
        if value:
            print(f"✅ {key.upper()}: {value}")
        else:
            print(f"❌ {key.upper()}: Non trouvé")
    
    # Stats
    stats = results['stats']
    print(f"\n📊 Statistiques:")
    print(f"   Total: {stats['total']}")
    print(f"   Succès: {stats['success']}")
    print(f"   Échecs: {stats['failed']}")
    print(f"   Temps: {stats['time_seconds']:.1f}s")
    
    print("\n✅ Extraction terminée!")


if __name__ == "__main__":
    main()