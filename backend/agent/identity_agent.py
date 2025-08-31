# agents/identity_agent.py
"""
Agent d'identité simplifié pour GPT-4o-mini
Version améliorée avec traçabilité complète des chunks
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
import yaml

# 250822 Nouveau : 
import pickle  # AJOUTER
import hashlib  # AJOUTER
from typing import Dict, List, Any, Optional, Tuple  # Ajouter Tuple si pas déjà présent

import sys
sys.path.append(str(Path(__file__).parent.parent))

from openai import OpenAI
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
    """
    Réponse structurée avec traçabilité complète
    Inclut maintenant les chunks complets utilisés pour l'extraction
    """
    question_id: str
    answer: Any
    sources: List[Dict[str, str]]
    confidence: float
    status: str  # success, partial, failed
    evidence_chunks: List[Dict[str, Any]] = field(default_factory=list)  # NOUVEAU: chunks complets avec métadonnées
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire avec tous les champs"""
        return asdict(self)


class SimpleIdentityAgent:
    """
    Agent d'extraction d'identité simplifié pour GPT-4o-mini
    Version améliorée avec traçabilité complète des sources
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
        
        # Client OpenAI direct pour GPT-5-mini
        self.client = OpenAI()
        self.model = "gpt-5-mini"  # ou "gpt-5" pour plus de puissance
        
        # Charger les questions
        self.questions = self._load_questions(config_path)

        # NOUVEAU : Initialiser le système de cache
        self.cache_dir = Path("data/cache/identity_agent")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
        # NOUVEAU : Queries optimisées pour chaque question
        self.optimized_queries = {
            "identity.client_name": "client company name official legal entity issuing tender tendering party",
            "identity.tender_reference": "tender reference number RFP RFQ code identifier official",
            "identity.timeline_milestones": "deadline submission date timeline milestones key dates schedule",
            "identity.submission_channel": "submit submission email portal platform how where send",
            "identity.expected_deliverables": "deliverables documents required expected submit provide",
            "identity.operating_countries": "countries regions geographical scope territories location operation",
            "identity.service_main_scope": "service scope main primary transport logistics activities",
            "identity.contract_type": "contract type framework agreement nature legal structure",
            "identity.contract_duration": "contract duration period years months term length validity"
        }
    
        # NOUVEAU : Pré-charger ou calculer les embeddings
        self.embeddings_cache = self._initialize_embeddings_cache()
        
        # Stats basiques
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_chunks_used': 0,  # NOUVEAU: tracking des chunks utilisés
            'cache_hits': 0,  # NOUVEAU
            'cache_misses': 0  # NOUVEAU
        }
        
        logger.info(f"✅ Agent initialisé avec {len(self.questions)} questions")

    def _initialize_embeddings_cache(self) -> Dict[str, Any]:
        """
        Initialise le cache d'embeddings persistant
        Charge depuis le disque ou calcule si nécessaire
        """
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        # Essayer de charger le cache existant
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    logger.info(f"✅ Cache chargé depuis {cache_file}")
                    
                    # Vérifier que toutes les questions sont dans le cache
                    missing = set(self.optimized_queries.keys()) - set(cache.keys())
                    if missing:
                        logger.info(f"⚠️ Questions manquantes dans le cache: {missing}")
                        # Compléter le cache avec les questions manquantes
                        for question_id in missing:
                            cache = self._add_to_cache(cache, question_id)
                        # Sauvegarder le cache mis à jour
                        self._save_cache(cache)
                    
                    return cache
            except Exception as e:
                logger.warning(f"⚠️ Impossible de charger le cache: {e}")
        
        # Si pas de cache, le créer
        logger.info("🔄 Création du cache d'embeddings...")
        cache = {}
        
        for question_id, query in self.optimized_queries.items():
            cache = self._add_to_cache(cache, question_id)
        
        # Sauvegarder le cache
        self._save_cache(cache)
        logger.info(f"✅ Cache créé avec {len(cache)} embeddings")
        
        return cache

    def _add_to_cache(self, cache: Dict, question_id: str) -> Dict:
        """
        Ajoute un embedding au cache
        """
        query = self.optimized_queries.get(question_id)
        if query:
            logger.info(f"  📝 Calcul embedding pour: {question_id}")
            embedding = self.vectorstore.embeddings.embed_query(query)
            cache[question_id] = {
                'query': query,
                'embedding': embedding,
                'created_at': datetime.now().isoformat()
            }
        return cache

    def _save_cache(self, cache: Dict):
        """
        Sauvegarde le cache sur disque
        """
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"💾 Cache sauvegardé: {cache_file}")
    
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
        Extrait toutes les informations d'identité avec traçabilité complète
        
        Returns:
            Dictionnaire avec toutes les réponses et leurs sources détaillées
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 EXTRACTION D'IDENTITÉ AVEC TRAÇABILITÉ")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        results = {
            'timestamp': start_time.isoformat(),
            'questions': {},
            'summary': {},
            'metadata': {  # NOUVEAU: métadonnées globales
                'total_chunks_analyzed': 0,
                'documents_referenced': set()
            }
        }
        
        # Traiter chaque question
        for question in self.questions:
            answer = self.extract_single(question)
            results['questions'][answer.question_id] = answer.to_dict()
            
            # Collecter les métadonnées globales
            for chunk in answer.evidence_chunks:
                if 'metadata' in chunk and 'source' in chunk['metadata']:
                    results['metadata']['documents_referenced'].add(
                        chunk['metadata']['source']
                    )
            results['metadata']['total_chunks_analyzed'] += len(answer.evidence_chunks)
        
        # Convertir le set en liste pour la sérialisation JSON
        results['metadata']['documents_referenced'] = list(
            results['metadata']['documents_referenced']
        )
        
        # Générer le résumé
        results['summary'] = self._generate_summary(results['questions'])
        
        # Stats finales
        elapsed = (datetime.now() - start_time).total_seconds()
        results['stats'] = {
            **self.stats,
            'time_seconds': elapsed,
            'avg_chunks_per_question': (
                self.stats['total_chunks_used'] / self.stats['total'] 
                if self.stats['total'] > 0 else 0
            )
        }
        
        logger.info(f"\n✅ Extraction terminée en {elapsed:.1f}s")
        logger.info(f"   Succès: {self.stats['success']}/{self.stats['total']}")
        logger.info(f"   Chunks utilisés: {self.stats['total_chunks_used']}")
        
        return results
    
    def extract_single(self, question: IdentityQuestion) -> IdentityAnswer:
        """
        Extrait l'information pour une question avec cache optimisé
        
        Args:
            question: Question à traiter
            
        Returns:
            IdentityAnswer avec la réponse et tous les chunks utilisés
        """
        self.stats['total'] += 1
        logger.info(f"\n🔍 Traitement: {question.id}")
        
        try:
            # NOUVEAU : Utiliser le cache si disponible
            if question.id in self.embeddings_cache:
                # Recherche optimisée avec embedding caché
                retrieval_result = self._retrieve_with_cached_embedding(
                    question_id=question.id,
                    category="identity"
                )
                self.stats['cache_hits'] += 1
                logger.debug(f"   ✅ Cache hit pour {question.id}")
            else:
                # Fallback : méthode normale (pour questions custom)
                search_query = self._create_search_query(question)
                retrieval_result = self.retriever.retrieve_and_answer(
                    query=search_query,
                    category="identity",
                    require_source=True,
                    max_chunks=3
                )
                self.stats['cache_misses'] += 1
                logger.debug(f"   ⚠️ Cache miss pour {question.id}")
            
            # 3. Extraire la réponse structurée avec les chunks complets
            answer = self._extract_answer_with_evidence(question, retrieval_result)
            
            # Mise à jour des stats
            if answer.status == "success":
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
            
            self.stats['total_chunks_used'] += len(answer.evidence_chunks)
            
            logger.info(f"   ✅ Status: {answer.status} (confiance: {answer.confidence:.0%})")
            logger.info(f"   📄 Chunks utilisés: {len(answer.evidence_chunks)}")
            
            return answer
            
        except Exception as e:
            logger.error(f"   ❌ Erreur: {e}")
            self.stats['failed'] += 1
            
            return IdentityAnswer(
                question_id=question.id,
                answer=None,
                sources=[],
                confidence=0.0,
                status="failed",
                evidence_chunks=[]
            )
    def _retrieve_with_cached_embedding(
        self,
        question_id: str,
        category: str
    ) -> RetrievalResult:
        """
        Effectue une recherche en utilisant l'embedding pré-calculé
        Évite l'appel API pour embed_query
        """
        cached_data = self.embeddings_cache[question_id]
        query = cached_data['query']
        embedding = cached_data['embedding']
        
        # Recherche directe dans le vectorstore avec l'embedding
        # Au lieu d'appeler intelligent_search qui recalculerait l'embedding
        results = self._search_vectorstore_with_embedding(
            embedding=embedding,
            k=3
        )
        
        # Construire un RetrievalResult compatible
        chunks = []
        sources = []
        
        for result in results:
            chunks.append(result)
            if 'source' in result:
                sources.append(result['source'])
        
        # Créer une réponse simple basée sur les chunks
        answer_text = f"Information found in {len(chunks)} chunks"
        
        return RetrievalResult(
            query=query,
            answer=answer_text,
            chunks=chunks,
            sources=sources[:3],
            confidence=0.8 if chunks else 0.0,
            metadata={'from_cache': True}
        )

    def _search_vectorstore_with_embedding(
        self,
        embedding: List[float],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recherche directe dans FAISS avec un embedding pré-calculé
        """
        import numpy as np
        
        # Convertir en numpy array
        query_vector = np.array([embedding], dtype=np.float32)
        
        # Normaliser pour FAISS (si nécessaire)
        if hasattr(self.vectorstore, 'index') and self.vectorstore.index:
            import faiss
            faiss.normalize_L2(query_vector)
            
            # Recherche dans FAISS
            distances, indices = self.vectorstore.index.search(query_vector, k)
            
            # Récupérer les chunks correspondants
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx in self.vectorstore.id_to_text:
                    chunk_result = {
                        'text': self.vectorstore.id_to_text[idx],
                        'metadata': self.vectorstore.id_to_metadata.get(idx, {}),
                        'score': float(dist),
                        'source': self.vectorstore.id_to_metadata.get(idx, {})
                    }
                    results.append(chunk_result)
            
            return results
        else:
            # Fallback si pas d'accès direct à FAISS
            logger.warning("Pas d'accès direct à FAISS, utilisation de la recherche normale")
            return []
    
    def _create_search_query(self, question: IdentityQuestion) -> str:
        """
        Retourne la query optimisée depuis le mapping
        """
        # Utiliser les queries optimisées définies dans __init__
        return self.optimized_queries.get(
            question.id, 
            "tender information details"  # Fallback générique
        )
    
    def _extract_answer_with_evidence(
        self,
        question: IdentityQuestion,
        retrieval_result: RetrievalResult
    ) -> IdentityAnswer:
        """
        Extrait la réponse depuis les chunks trouvés avec traçabilité complète
        
        Args:
            question: La question à traiter
            retrieval_result: Résultats du retrieval avec chunks
            
        Returns:
            IdentityAnswer enrichi avec evidence_chunks
        """
        # Préparer le contexte ET capturer les chunks info
        context, chunks_details = self._prepare_context_with_tracking(
            retrieval_result.chunks[:3]
        )
        
        # Construire evidence_chunks avec toutes les métadonnées
        evidence_chunks = self._build_evidence_chunks(
            retrieval_result.chunks[:3],
            chunks_details
        )
        
        # Prompt simplifié pour extraction JSON
        prompt = f"""{question.system_prompt}

Context from documents:
{context}

IMPORTANT: Return ONLY a valid JSON object. No explanations.
The JSON must start with {{ and end with }}.
"""
        
        try:
            # Appeler le LLM GPT-5-mini avec le contexte COMPLET
            response = self.client.responses.create(
                model=self.model,
                input=f"{question.system_prompt}\n\nContext from documents:\n{context}",
                reasoning={"effort": "minimal"},  # Pour extraction rapide
                text={"verbosity": "low"}  # Réponses concises JSON
            )
            result_text = response.output_text.strip()
            
            # Extraire le JSON de la réponse
            json_obj = self._extract_json(result_text)
            
            # Créer la réponse enrichie
            return IdentityAnswer(
                question_id=question.id,
                answer=json_obj.get('answer'),
                sources=[],
                confidence=retrieval_result.confidence,
                status="success" if json_obj.get('answer') else "partial",
                evidence_chunks=evidence_chunks  # NOUVEAU: chunks complets avec métadonnées
            )
            
        except Exception as e:
            logger.debug(f"Erreur extraction: {e}")
            
            # Fallback: utiliser la réponse directe du retrieval
            return IdentityAnswer(
                question_id=question.id,
                answer=retrieval_result.answer if retrieval_result.answer else None,
                sources=[],
                confidence=retrieval_result.confidence * 0.5,
                status="partial",
                evidence_chunks=evidence_chunks  # Inclure même en cas de fallback
            )
    
    def _prepare_context_with_tracking(
        self, 
        chunks: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Prépare le contexte depuis les chunks avec TEXTE COMPLET
        et retourne aussi les détails pour traçabilité
        
        Args:
            chunks: Liste des chunks du retrieval
            
        Returns:
            Tuple (context_string, chunks_details)
        """
        context_parts = []
        chunks_details = []
        
        for i, chunk in enumerate(chunks):
            # Extraire les métadonnées
            source = chunk.get('source', {})
            metadata = chunk.get('metadata', {})
            
            # Utiliser le TEXTE COMPLET du chunk (pas de limitation!)
            text_full = chunk.get('text', '')
            
            # Info source pour le contexte
            source_info = f"[Doc: {source.get('document', 'Unknown')} | Section: {source.get('section', 'N/A')}]"
            
            # Ajouter au contexte avec le texte COMPLET
            context_parts.append(f"{source_info}\n{text_full}")
            
            # Sauvegarder les détails pour evidence_chunks
            chunks_details.append({
                'index': i,
                'text': text_full,  # Texte COMPLET
                'source': source.get('document', 'Unknown'),
                'section': source.get('section', 'N/A'),
                'metadata': metadata  # Toutes les métadonnées disponibles
            })
        
        context_string = "\n---\n".join(context_parts)
        
        logger.debug(f"   Context préparé: {len(context_string)} caractères depuis {len(chunks)} chunks")
        
        return context_string, chunks_details
    
    def _build_evidence_chunks(
        self,
        original_chunks: List[Dict],
        chunks_details: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Construit la structure evidence_chunks avec toutes les métadonnées
        
        Args:
            original_chunks: Chunks originaux du retrieval
            chunks_details: Détails extraits par _prepare_context_with_tracking
            
        Returns:
            Liste structurée pour evidence_chunks
        """
        evidence_chunks = []
        
        for chunk, details in zip(original_chunks, chunks_details):
            evidence_chunk = {
                "chunk_index": details['index'],
                "text_full": details['text'],  # Texte COMPLET du chunk
                "metadata": {
                    "source": details['source'],
                    "section": details['section']
                },
                "relevance_score": chunk.get('score', 0.0) if 'score' in chunk else chunk.get('confidence', 0.0)
            }
            
            
            evidence_chunks.append(evidence_chunk)
        
        return evidence_chunks
    
    def _extract_json(self, text: str) -> Dict:
        """
        Extrait un objet JSON depuis du texte
        Gère les cas où le LLM ajoute du texte avant/après
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
            except json.JSONDecodeError as e:
                logger.debug(f"Première tentative JSON échouée: {e}")
                # Essayer de nettoyer et réessayer
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                # Retirer les commentaires potentiels
                json_str = ' '.join(line.split('//')[0] for line in json_str.split('\n'))
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e2:
                    logger.error(f"Impossible d'extraire JSON: {e2}")
                    logger.debug(f"Texte reçu: {text[:500]}")
        
        # Si pas de JSON trouvé, retourner un dict vide
        logger.warning("Aucun JSON valide trouvé dans la réponse")
        return {}
    
    def _format_sources(self, retrieval_result: RetrievalResult) -> List[Dict]:
        """
        Formate les sources pour compatibilité avec l'ancien format
        """
        sources = []
        seen_sources = set()  # Pour éviter les doublons
        
        for source in retrieval_result.sources[:3]:  # Max 3 sources
            source_key = f"{source.get('document')}_{source.get('page')}"
            
            if source_key not in seen_sources:
                sources.append({
                    'document': source.get('document', 'Unknown'),
                    'section': source.get('section', '')
                })
                seen_sources.add(source_key)
        
        return sources
    
    def _generate_summary(self, questions: Dict[str, Dict]) -> Dict:
        """
        Génère un résumé simple de l'identité extraite
        Inclut maintenant un indicateur de qualité basé sur les evidence_chunks
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
                    # Ajouter la valeur extraite
                    summary[key] = result['answer']
                    
                    # NOUVEAU: Ajouter un indicateur de qualité basé sur les evidence
                    if 'evidence_chunks' in result and result['evidence_chunks']:
                        summary[f"{key}_quality"] = {
                            'confidence': result['confidence'],
                            'sources_count': len(result['evidence_chunks']),
                            'status': result['status']
                        }
                else:
                    summary[key] = None
                    summary[f"{key}_quality"] = {
                        'confidence': 0.0,
                        'sources_count': 0,
                        'status': 'failed'
                    }
        
        return summary
    
    def get_extraction_report(self, results: Dict) -> str:
        """
        Génère un rapport textuel détaillé de l'extraction
        NOUVEAU: méthode helper pour affichage
        """
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append("📊 RAPPORT D'EXTRACTION D'IDENTITÉ")
        report_lines.append("="*60)
        
        # Résumé des réponses
        report_lines.append("\n📋 INFORMATIONS EXTRAITES:")
        report_lines.append("-" * 40)
        
        for key, value in results['summary'].items():
            if not key.endswith('_quality'):
                quality = results['summary'].get(f"{key}_quality", {})
                if value:
                    confidence = quality.get('confidence', 0) * 100
                    sources = quality.get('sources_count', 0)
                    report_lines.append(
                        f"✅ {key.upper()}: {value} "
                        f"(Confiance: {confidence:.0f}%, Sources: {sources})"
                    )
                else:
                    report_lines.append(f"❌ {key.upper()}: Non trouvé")
        
        # Documents référencés
        if 'metadata' in results:
            report_lines.append(f"\n📚 DOCUMENTS ANALYSÉS:")
            report_lines.append("-" * 40)
            for doc in results['metadata']['documents_referenced']:
                report_lines.append(f"   • {doc}")
            
            report_lines.append(
                f"\n📄 Total chunks analysés: {results['metadata']['total_chunks_analyzed']}"
            )
        
        # Statistiques
        if 'stats' in results:
            stats = results['stats']
            report_lines.append(f"\n📊 STATISTIQUES:")
            report_lines.append("-" * 40)
            report_lines.append(f"   Questions traitées: {stats['total']}")
            report_lines.append(f"   Succès: {stats['success']}")
            report_lines.append(f"   Échecs: {stats['failed']}")
            report_lines.append(f"   Chunks utilisés: {stats.get('total_chunks_used', 0)}")
            report_lines.append(f"   Temps: {stats['time_seconds']:.1f}s")
        
        return "\n".join(report_lines)
    
    def clear_cache(self):
        """
        Vide le cache des embeddings (utile pour forcer le recalcul)
        """
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
            logger.info("🗑️ Cache supprimé")
        
        # Réinitialiser
        self.embeddings_cache = self._initialize_embeddings_cache()


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Test de l'agent amélioré avec traçabilité complète et cache optimisé"""
    import os
    
    # Vérifier la clé API
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY non trouvée")
        print("   Définissez-la avec: export OPENAI_API_KEY='votre-clé'")
        return
    
    print("\n" + "="*60)
    print("🤖 AGENT D'IDENTITÉ AVEC TRAÇABILITÉ COMPLÈTE ET CACHE")
    print("="*60)
    
    # Imports
    from utils.vectorstore import IntelligentVectorStore
    from utils.enhanced_retrieval import IntelligentRetriever
    
    # 1. Charger le vectorstore
    print("\n📚 Chargement du vectorstore...")
    vectorstore = IntelligentVectorStore(
        persist_directory=Path("data/intelligent_store"),
        llm_model="gpt-4o-mini"
    )
    stats = vectorstore.get_stats()
    print(f"✅ Vectorstore: {stats['total_documents']} documents")
    
    if stats['total_documents'] == 0:
        print("❌ Vectorstore vide! Lancez d'abord la vectorisation.")
        print("   Commande: python utils/vectorize_documents.py --source <dossier>")
        return
    
    # 2. Créer le retriever
    print("\n🔍 Initialisation du retriever...")
    retriever = IntelligentRetriever(
        vectorstore=vectorstore,
        llm_model="gpt-4o-mini"
    )
    
    # 3. Créer l'agent
    print("\n🤖 Création de l'agent avec cache intelligent...")
    agent = SimpleIdentityAgent(
        vectorstore=vectorstore,
        retriever=retriever,
        config_path="configs/prompts/identity/en.yaml"
    )
    
    # NOUVEAU : Afficher l'état du cache au démarrage
    if hasattr(agent, 'embeddings_cache'):
        print(f"💾 Cache d'embeddings: {len(agent.embeddings_cache)} queries pré-calculées")
    
    # 4. Lancer l'extraction
    print("\n🚀 Lancement de l'extraction avec traçabilité...")
    start_time = datetime.now()
    results = agent.extract_all()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    # 5. Afficher le rapport détaillé
    report = agent.get_extraction_report(results)
    print(report)
    
    # NOUVEAU : Section dédiée aux statistiques de cache
    print("\n" + "="*60)
    print("💾 PERFORMANCE DU CACHE")
    print("="*60)
    
    if 'stats' in results:
        stats = results['stats']
        
        # Calculer les métriques du cache
        cache_hits = stats.get('cache_hits', 0)
        cache_misses = stats.get('cache_misses', 0)
        total_cache_ops = cache_hits + cache_misses
        
        if total_cache_ops > 0:
            cache_hit_ratio = (cache_hits / total_cache_ops) * 100
            
            print(f"📊 Statistiques du Cache:")
            print(f"   ├─ Cache hits: {cache_hits}")
            print(f"   ├─ Cache misses: {cache_misses}")
            print(f"   ├─ Hit ratio: {cache_hit_ratio:.1f}%")
            print(f"   └─ Total opérations: {total_cache_ops}")
            
            # Calculer les économies
            embedding_cost_per_call = 0.00002  # Coût approximatif par embedding
            saved_embedding_calls = cache_hits
            saved_cost = saved_embedding_calls * embedding_cost_per_call
            time_saved = saved_embedding_calls * 0.1  # ~100ms par appel API évité
            
            print(f"\n💰 Économies Réalisées:")
            print(f"   ├─ Appels API évités: {saved_embedding_calls}")
            print(f"   ├─ Coût économisé: ~${saved_cost:.5f}")
            print(f"   └─ Temps gagné: ~{time_saved:.1f}s")
            
            # Comparaison avec/sans cache
            print(f"\n📈 Comparaison Performance:")
            print(f"   ├─ Temps total avec cache: {elapsed_time:.1f}s")
            print(f"   ├─ Temps estimé sans cache: ~{elapsed_time + time_saved:.1f}s")
            print(f"   └─ Amélioration: {(time_saved/(elapsed_time + time_saved)*100):.0f}% plus rapide")
        else:
            print("⚠️ Aucune statistique de cache disponible")
    
    # 6. Afficher un exemple de traçabilité détaillée
    print("\n" + "="*60)
    print("🔍 EXEMPLE DE TRAÇABILITÉ DÉTAILLÉE")
    print("="*60)
    
    # Prendre la première question réussie comme exemple
    example_shown = False
    for question_id, result in results['questions'].items():
        if result['status'] == 'success' and result.get('evidence_chunks'):
            print(f"\n📌 Question: {question_id}")
            print(f"📝 Réponse: {result['answer']}")
            print(f"🎯 Confiance: {result['confidence']*100:.0f}%")
            
            # NOUVEAU : Indiquer si la réponse vient du cache
            if result.get('metadata', {}).get('from_cache'):
                print(f"💾 Source: Cache (embedding pré-calculé)")
            
            print(f"\n📄 Evidence Chunks ({len(result['evidence_chunks'])} chunks utilisés):")
            
            for i, chunk in enumerate(result['evidence_chunks'], 1):
                print(f"\n   Chunk {i}:")
                print(f"   ├─ Source: {chunk['metadata']['source']}")
                print(f"   ├─ Section: {chunk['metadata']['section']}")
                print(f"   ├─ Score: {chunk['relevance_score']:.2f}")
                print(f"   └─ Extrait: {chunk['text_full'][:150]}...")
            
            example_shown = True
            break  # Afficher juste un exemple
    
    if not example_shown:
        print("ℹ️ Aucun exemple de traçabilité disponible")
    
    # 7. Optionnel: Sauvegarder les résultats
    output_file = Path("extraction_results_with_evidence.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Résultats sauvegardés dans: {output_file}")
    
    # NOUVEAU : Résumé final avec métriques de performance
    print("\n" + "="*60)
    print("✅ EXTRACTION TERMINÉE AVEC SUCCÈS!")
    print("="*60)
    
    print(f"\n📊 Résumé Final:")
    print(f"   ├─ Questions traitées: {stats.get('total', 0)}")
    print(f"   ├─ Succès: {stats.get('success', 0)}/{stats.get('total', 0)}")
    print(f"   ├─ Temps total: {elapsed_time:.1f}s")
    
    if total_cache_ops > 0:
        print(f"   ├─ Cache utilisé: {cache_hit_ratio:.0f}% des requêtes")
        print(f"   └─ Efficacité: ${saved_cost:.5f} économisés")
    else:
        print(f"   └─ Cache: Non utilisé")
    
    # NOUVEAU : Suggestion d'optimisation
    if cache_misses > 0:
        print(f"\n💡 Conseil: {cache_misses} requêtes n'étaient pas dans le cache.")
        print(f"   Relancez l'extraction pour bénéficier du cache à 100%!")


if __name__ == "__main__":
    main()