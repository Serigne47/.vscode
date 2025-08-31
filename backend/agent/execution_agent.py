# agents/execution_agent.py
"""
Agent d'exécution pour l'analyse des aspects opérationnels des appels d'offres
Version adaptée de l'Identity Agent avec support GPT-5-mini
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
import yaml
import pickle
import hashlib

import sys
sys.path.append(str(Path(__file__).parent.parent))

from openai import OpenAI

# Import des composants
from utils.vectorstore import IntelligentVectorStore
from utils.enhanced_retrieval import IntelligentRetriever, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class ExecutionQuestion:
    """Structure d'une question d'exécution"""
    id: str
    system_prompt: str
    validator: Dict[str, Any] = None
    
@dataclass
class ExecutionAnswer:
    """
    Réponse structurée avec traçabilité complète
    Inclut maintenant les chunks complets utilisés pour l'extraction
    """
    question_id: str
    answer: Any
    sources: List[Dict[str, str]]
    confidence: float
    status: str  # success, partial, failed
    evidence_chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire avec tous les champs"""
        return asdict(self)


class ExecutionAgent:
    """
    Agent d'extraction des informations d'exécution opérationnelle
    Version adaptée pour GPT-5-mini avec traçabilité complète
    """
    
    def __init__(
        self,
        vectorstore: IntelligentVectorStore,
        retriever: IntelligentRetriever,
        config_path: str = "configs/prompts/execution/en.yaml"
    ):
        """
        Initialise l'agent d'exécution
        
        Args:
            vectorstore: IntelligentVectorStore configuré
            retriever: IntelligentRetriever configuré
            config_path: Chemin vers le YAML des questions d'exécution
        """
        self.vectorstore = vectorstore
        self.retriever = retriever
        
        # Client OpenAI direct pour GPT-5-mini
        self.client = OpenAI()
        self.model = "gpt-5-mini"
        
        # Charger les questions d'exécution
        self.questions = self._load_questions(config_path)

        # Initialiser le système de cache
        self.cache_dir = Path("data/cache/execution_agent")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
        # Queries optimisées pour chaque question d'exécution
        self.optimized_queries = {
            "execution.transport_mode": "transport mode air sea ocean road rail multimodal freight shipping logistics",
            "execution.expected_services": "services freight forwarding customs clearance warehousing delivery logistics operations",
            "execution.operational_requirements": "requirements temperature GPS tracking EDI certification ISO IATA documentation",
            "execution.sla_kpi": "SLA KPI performance indicators on-time delivery rate metrics targets"
        }
    
        # Pré-charger ou calculer les embeddings
        self.embeddings_cache = self._initialize_embeddings_cache()
        
        # Stats basiques
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_chunks_used': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"✅ Execution Agent initialisé avec {len(self.questions)} questions")
        logger.info(f"💾 Cache: {len(self.embeddings_cache)} embeddings pré-calculés")

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
                        for question_id in missing:
                            cache = self._add_to_cache(cache, question_id)
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
    
    def _load_questions(self, config_path: str) -> List[ExecutionQuestion]:
        """Charge les questions depuis le YAML"""
        questions = []
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            for item in config:
                question = ExecutionQuestion(
                    id=item['id'],
                    system_prompt=item['system_prompt'],
                    validator=item.get('validator', {})
                )
                questions.append(question)
            
            logger.info(f"📋 {len(questions)} questions d'exécution chargées")
            
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            raise
        
        return questions
    
    def extract_all(self) -> Dict[str, Any]:
        """
        Extrait toutes les informations d'exécution avec traçabilité complète
        
        Returns:
            Dictionnaire avec toutes les réponses et leurs sources détaillées
        """
        logger.info("\n" + "="*60)
        logger.info("⚙️ EXTRACTION DES INFORMATIONS D'EXÉCUTION")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        results = {
            'timestamp': start_time.isoformat(),
            'questions': {},
            'summary': {},
            'metadata': {
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
    
    def extract_single(self, question: ExecutionQuestion) -> ExecutionAnswer:
        """
        Extrait l'information pour une question avec cache optimisé
        
        Args:
            question: Question à traiter
            
        Returns:
            ExecutionAnswer avec la réponse et tous les chunks utilisés
        """
        self.stats['total'] += 1
        logger.info(f"\n⚙️ Traitement: {question.id}")
        
        try:
            # Utiliser le cache si disponible
            if question.id in self.embeddings_cache:
                retrieval_result = self._retrieve_with_cached_embedding(
                    question_id=question.id,
                    category="execution"
                )
                self.stats['cache_hits'] += 1
                logger.debug(f"   ✅ Cache hit pour {question.id}")
            else:
                search_query = self._create_search_query(question)
                retrieval_result = self.retriever.retrieve_and_answer(
                    query=search_query,
                    category="execution",
                    require_source=True,
                    max_chunks=3
                )
                self.stats['cache_misses'] += 1
                logger.debug(f"   ⚠️ Cache miss pour {question.id}")
            
            # Extraire la réponse structurée avec les chunks complets
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
            
            return ExecutionAnswer(
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
        """
        cached_data = self.embeddings_cache[question_id]
        query = cached_data['query']
        embedding = cached_data['embedding']
        
        results = self._search_vectorstore_with_embedding(
            embedding=embedding,
            k=3
        )
        
        chunks = []
        sources = []
        
        for result in results:
            chunks.append(result)
            if 'source' in result:
                sources.append(result['source'])
        
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
        
        query_vector = np.array([embedding], dtype=np.float32)
        
        if hasattr(self.vectorstore, 'index') and self.vectorstore.index:
            import faiss
            faiss.normalize_L2(query_vector)
            
            distances, indices = self.vectorstore.index.search(query_vector, k)
            
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
            logger.warning("Pas d'accès direct à FAISS, utilisation de la recherche normale")
            return []
    
    def _create_search_query(self, question: ExecutionQuestion) -> str:
        """
        Retourne la query optimisée depuis le mapping
        """
        return self.optimized_queries.get(
            question.id, 
            "tender execution operational requirements"
        )
    
    def _extract_answer_with_evidence(
        self,
        question: ExecutionQuestion,
        retrieval_result: RetrievalResult
    ) -> ExecutionAnswer:
        """
        Extrait la réponse depuis les chunks trouvés avec traçabilité complète
        Adapté pour la nouvelle structure JSON avec answer.result et answer.sources
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
        
        # Prompt pour extraction JSON
        prompt_text = f"""{question.system_prompt}

Context from documents:
{context}

IMPORTANT: Return ONLY a valid JSON object. No explanations.
The JSON must start with {{ and end with }}.
"""
        
        try:
            # Appeler GPT-5-mini avec le contexte COMPLET
            response = self.client.responses.create(
                model=self.model,
                input=prompt_text,
                reasoning={"effort": "minimal"},
                text={"verbosity": "low"}
            )
            result_text = response.output_text.strip()
            
            # Extraire le JSON de la réponse
            json_obj = self._extract_json(result_text)
            
            # La nouvelle structure a answer.result et answer.sources
            answer_data = json_obj.get('answer', {})
            
            # Créer la réponse enrichie
            return ExecutionAnswer(
                question_id=question.id,
                answer=answer_data,  # Contient {"result": ..., "sources": [...]}
                sources=[],
                confidence=self._calculate_confidence(answer_data, retrieval_result),
                status=self._determine_status(answer_data),
                evidence_chunks=evidence_chunks
            )
            
        except Exception as e:
            logger.debug(f"Erreur extraction: {e}")
            
            # Fallback
            return ExecutionAnswer(
                question_id=question.id,
                answer=None,
                sources=[],
                confidence=retrieval_result.confidence * 0.5,
                status="partial",
                evidence_chunks=evidence_chunks
            )
    
    def _prepare_context_with_tracking(
        self, 
        chunks: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Prépare le contexte depuis les chunks avec TEXTE COMPLET
        """
        context_parts = []
        chunks_details = []
        
        for i, chunk in enumerate(chunks):
            source = chunk.get('source', {})
            metadata = chunk.get('metadata', {})
            
            text_full = chunk.get('text', '')
            
            # Inclure le nom du document pour que le LLM puisse le référencer
            doc_name = source.get('document', 'Unknown')
            section = source.get('section', 'N/A')
            
            # Format pour le LLM
            context_parts.append(
                f"[Document: {doc_name}]\n"
                f"[Section: {section}]\n"
                f"{text_full}"
            )
            
            chunks_details.append({
                'index': i,
                'text': text_full,
                'source': doc_name,
                'section': section,
                'metadata': metadata
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
        Construit la structure evidence_chunks avec métadonnées essentielles
        """
        evidence_chunks = []
        
        for chunk, details in zip(original_chunks, chunks_details):
            evidence_chunk = {
                "chunk_index": details['index'],
                "text_full": details['text'],
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
        """
        text = text.strip()
        
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            json_str = text[start:end]
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.debug(f"Première tentative JSON échouée: {e}")
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                json_str = ' '.join(line.split('//')[0] for line in json_str.split('\n'))
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e2:
                    logger.error(f"Impossible d'extraire JSON: {e2}")
                    logger.debug(f"Texte reçu: {text[:500]}")
        
        logger.warning("Aucun JSON valide trouvé dans la réponse")
        return {}
    
    def _format_sources_from_answer(
        self, 
        answer_data: Dict, 
        retrieval_result: RetrievalResult
    ) -> List[Dict]:
        """
        Formate les sources depuis la réponse du LLM ou du retrieval
        """
        # Si le LLM a fourni des sources dans sa réponse
        if answer_data and 'sources' in answer_data:
            return answer_data['sources'][:3]
        
        # Sinon, utiliser les sources du retrieval
        return self._format_sources(retrieval_result)
    
    def _format_sources(self, retrieval_result: RetrievalResult) -> List[Dict]:
        """
        Formate les sources depuis le retrieval result
        """
        sources = []
        seen_sources = set()
        
        for source in retrieval_result.sources[:3]:
            source_key = f"{source.get('document')}_{source.get('section')}"
            
            if source_key not in seen_sources:
                sources.append({
                    'document': source.get('document', 'Unknown'),
                    'section': source.get('section', '')
                })
                seen_sources.add(source_key)
        
        return sources
    
    def _calculate_confidence(
        self, 
        answer_data: Dict, 
        retrieval_result: RetrievalResult
    ) -> float:
        """
        Calcule la confiance basée sur la réponse et les sources
        """
        if not answer_data or not answer_data.get('result'):
            return 0.0
        
        base_confidence = retrieval_result.confidence
        
        # Augmenter si des sources sont fournies
        if answer_data.get('sources'):
            base_confidence = min(1.0, base_confidence * 1.1)
        
        return base_confidence
    
    def _determine_status(self, answer_data: Dict) -> str:
        """
        Détermine le statut basé sur la complétude de la réponse
        """
        if not answer_data:
            return "failed"
        
        result = answer_data.get('result')
        sources = answer_data.get('sources', [])
        
        if result and sources:
            return "success"
        elif result:
            return "partial"
        else:
            return "failed"
    
    def _generate_summary(self, questions: Dict[str, Dict]) -> Dict:
        """
        Génère un résumé des informations d'exécution extraites
        """
        summary = {}
        
        # Mapping des questions aux clés du résumé
        mapping = {
            "execution.transport_mode": "transport_modes",
            "execution.expected_services": "services",
            "execution.operational_requirements": "requirements",
            "execution.sla_kpi": "sla_kpis"
        }
        
        for question_id, key in mapping.items():
            if question_id in questions:
                result = questions[question_id]
                if result['status'] in ['success', 'partial']:
                    # Extraire la valeur depuis answer.result
                    answer_data = result.get('answer', {})
                    if isinstance(answer_data, dict):
                        summary[key] = answer_data.get('result')
                    else:
                        summary[key] = answer_data
                    
                    # Ajouter indicateur de qualité
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
        """
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append("⚙️ RAPPORT D'EXTRACTION D'EXÉCUTION")
        report_lines.append("="*60)
        
        # Résumé des réponses
        report_lines.append("\n📋 INFORMATIONS D'EXÉCUTION EXTRAITES:")
        report_lines.append("-" * 40)
        
        for key, value in results['summary'].items():
            if not key.endswith('_quality'):
                quality = results['summary'].get(f"{key}_quality", {})
                if value:
                    confidence = quality.get('confidence', 0) * 100
                    sources = quality.get('sources_count', 0)
                    
                    # Formater selon le type de valeur
                    if isinstance(value, list):
                        value_str = ', '.join(value) if value else 'None'
                    else:
                        value_str = str(value)
                    
                    report_lines.append(
                        f"✅ {key.upper()}: {value_str} "
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
        Vide le cache des embeddings
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
    """Test de l'agent d'exécution avec traçabilité complète"""
    import os
    
    # Vérifier la clé API
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY non trouvée")
        print("   Définissez-la avec: export OPENAI_API_KEY='votre-clé'")
        return
    
    print("\n" + "="*60)
    print("⚙️ AGENT D'EXÉCUTION AVEC TRAÇABILITÉ COMPLÈTE")
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
    
    # 3. Créer l'agent d'exécution
    print("\n⚙️ Création de l'agent d'exécution...")
    agent = ExecutionAgent(
        vectorstore=vectorstore,
        retriever=retriever,
        config_path="configs/prompts/execution/en.yaml"
    )
    
    # Afficher l'état du cache
    if hasattr(agent, 'embeddings_cache'):
        print(f"💾 Cache d'embeddings: {len(agent.embeddings_cache)} queries pré-calculées")
    
    # 4. Lancer l'extraction
    print("\n🚀 Lancement de l'extraction d'exécution...")
    start_time = datetime.now()
    results = agent.extract_all()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    # 5. Afficher le rapport détaillé
    report = agent.get_extraction_report(results)
    print(report)
    
    # 6. Performance du cache
    print("\n" + "="*60)
    print("💾 PERFORMANCE DU CACHE")
    print("="*60)
    
    if 'stats' in results:
        stats = results['stats']
        
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
            
            # Économies
            embedding_cost_per_call = 0.00002
            saved_embedding_calls = cache_hits
            saved_cost = saved_embedding_calls * embedding_cost_per_call
            time_saved = saved_embedding_calls * 0.1
            
            print(f"\n💰 Économies Réalisées:")
            print(f"   ├─ Appels API évités: {saved_embedding_calls}")
            print(f"   ├─ Coût économisé: ~${saved_cost:.5f}")
            print(f"   └─ Temps gagné: ~{time_saved:.1f}s")
    
    # 7. Exemple de traçabilité
    print("\n" + "="*60)
    print("🔍 EXEMPLE DE TRAÇABILITÉ DÉTAILLÉE")
    print("="*60)
    
    example_shown = False
    for question_id, result in results['questions'].items():
        if result['status'] == 'success' and result.get('evidence_chunks'):
            print(f"\n📌 Question: {question_id}")
            
            # Extraire la réponse depuis la nouvelle structure
            answer_data = result.get('answer', {})
            if isinstance(answer_data, dict):
                answer_result = answer_data.get('result', 'N/A')
            else:
                answer_result = answer_data
            
            print(f"📝 Réponse: {answer_result}")
            print(f"🎯 Confiance: {result['confidence']*100:.0f}%")
            
            print(f"\n📄 Evidence Chunks ({len(result['evidence_chunks'])} chunks utilisés):")
            
            for i, chunk in enumerate(result['evidence_chunks'], 1):
                print(f"\n   Chunk {i}:")
                print(f"   ├─ Source: {chunk['metadata']['source']}")
                print(f"   ├─ Section: {chunk['metadata']['section']}")
                print(f"   ├─ Score: {chunk['relevance_score']:.2f}")
                print(f"   └─ Extrait: {chunk['text_full'][:150]}...")
            
            example_shown = True
            break
    
    if not example_shown:
        print("ℹ️ Aucun exemple de traçabilité disponible")
    
    # 8. Sauvegarder les résultats
    output_file = Path("execution_results_with_evidence.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Résultats sauvegardés dans: {output_file}")
    
    # Résumé final
    print("\n" + "="*60)
    print("✅ EXTRACTION D'EXÉCUTION TERMINÉE!")
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


if __name__ == "__main__":
    main()