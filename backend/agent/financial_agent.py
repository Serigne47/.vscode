# agents/financial_agent.py
"""
Agent financier pour l'analyse des aspects financiers et commerciaux des appels d'offres
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
class FinancialQuestion:
    """Structure d'une question financière"""
    id: str
    system_prompt: str
    validator: Dict[str, Any] = None
    
@dataclass
class FinancialAnswer:
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


class FinancialAgent:
    """
    Agent d'extraction des informations financières et commerciales
    Version adaptée pour GPT-5-mini avec traçabilité complète
    """
    
    def __init__(
        self,
        vectorstore: IntelligentVectorStore,
        retriever: IntelligentRetriever,
        config_path: str = "configs/prompts/financial/en.yaml"
    ):
        """
        Initialise l'agent financier avec cache optimisé
        
        Args:
            vectorstore: IntelligentVectorStore configuré
            retriever: IntelligentRetriever configuré
            config_path: Chemin vers le YAML des questions financières
        """
        self.vectorstore = vectorstore
        self.retriever = retriever
        
        # Client OpenAI direct pour GPT-5-mini
        self.client = OpenAI()
        self.model = "gpt-5-mini"
        
        # Charger les questions financières
        self.questions = self._load_questions(config_path)

        # Initialiser le système de cache
        self.cache_dir = Path("data/cache/financial_agent")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
        # Queries optimisées pour chaque question financière
        self.optimized_queries = {
            "financial.billing_terms": "billing invoice terms monthly per shipment delivery frequency consolidated split",
            "financial.payment_terms": "payment terms net 30 60 90 days settlement penalties early discount due date",
            "financial.currency_and_fx": "currency payment EUR USD exchange rate conversion forex hedging local",
            "financial.bank_guarantee_or_advance": "bank guarantee performance bond advance payment deposit security collateral",
            "financial.price_revision_clauses": "price revision fuel surcharge inflation indexation CPI escalation adjustment fixed"
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
        
        logger.info(f"💰 Financial Agent initialisé avec {len(self.questions)} questions")
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
    
    def _load_questions(self, config_path: str) -> List[FinancialQuestion]:
        """Charge les questions depuis le YAML avec gestion d'erreur améliorée"""
        questions = []
        
        try:
            # Vérifier que le fichier existe
            config_file = Path(config_path)
            if not config_file.exists():
                logger.error(f"❌ Fichier de configuration introuvable: {config_path}")
                logger.info(f"   Chemin absolu recherché: {config_file.absolute()}")
                raise FileNotFoundError(f"Le fichier {config_path} n'existe pas")
            
            # Charger le YAML
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Vérifier que le contenu n'est pas vide
            if config is None:
                logger.error(f"❌ Le fichier {config_path} est vide")
                raise ValueError(f"Le fichier {config_path} ne contient aucune donnée")
            
            # Vérifier que c'est une liste
            if not isinstance(config, list):
                logger.error(f"❌ Le fichier {config_path} doit contenir une liste de questions")
                raise ValueError(f"Format invalide dans {config_path}")
            
            # Charger chaque question
            for item in config:
                question = FinancialQuestion(
                    id=item['id'],
                    system_prompt=item['system_prompt'],
                    validator=item.get('validator', {})
                )
                questions.append(question)
            
            logger.info(f"📋 {len(questions)} questions financières chargées")
            
        except FileNotFoundError as e:
            logger.error(f"Erreur: {e}")
            logger.info("💡 Créez le fichier avec vos questions financières")
            raise
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            raise
        
        return questions
    
    def extract_all(self) -> Dict[str, Any]:
        """
        Extrait toutes les informations financières avec traçabilité complète
        
        Returns:
            Dictionnaire avec toutes les réponses et leurs sources détaillées
        """
        logger.info("\n" + "="*60)
        logger.info("💰 EXTRACTION DES INFORMATIONS FINANCIÈRES")
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
        
        logger.info(f"\n✅ Extraction financière terminée en {elapsed:.1f}s")
        logger.info(f"   Succès: {self.stats['success']}/{self.stats['total']}")
        logger.info(f"   Chunks utilisés: {self.stats['total_chunks_used']}")
        
        return results
    
    def extract_single(self, question: FinancialQuestion) -> FinancialAnswer:
        """
        Extrait l'information pour une question financière avec cache optimisé
        
        Args:
            question: Question à traiter
            
        Returns:
            FinancialAnswer avec la réponse et tous les chunks utilisés
        """
        self.stats['total'] += 1
        logger.info(f"\n💰 Traitement: {question.id}")
        
        try:
            # Utiliser le cache si disponible
            if question.id in self.embeddings_cache:
                retrieval_result = self._retrieve_with_cached_embedding(
                    question_id=question.id,
                    category="financial"
                )
                self.stats['cache_hits'] += 1
                logger.debug(f"   ✅ Cache hit pour {question.id}")
            else:
                search_query = self._create_search_query(question)
                retrieval_result = self.retriever.retrieve_and_answer(
                    query=search_query,
                    category="financial",
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
            
            return FinancialAnswer(
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
    
    def _create_search_query(self, question: FinancialQuestion) -> str:
        """
        Retourne la query optimisée depuis le mapping
        """
        return self.optimized_queries.get(
            question.id, 
            "tender financial payment terms conditions"
        )
    
    def _extract_answer_with_evidence(
        self,
        question: FinancialQuestion,
        retrieval_result: RetrievalResult
    ) -> FinancialAnswer:
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
            
            # Créer la réponse enrichie - sources uniquement dans answer
            return FinancialAnswer(
                question_id=question.id,
                answer=json_obj.get('answer'),  # Contient {"result": ..., "sources": [...]}
                sources=[],  # Vide pour éviter duplication
                confidence=self._calculate_confidence(json_obj.get('answer'), retrieval_result),
                status=self._determine_status(json_obj.get('answer')),
                evidence_chunks=evidence_chunks
            )
            
        except Exception as e:
            logger.debug(f"Erreur extraction: {e}")
            
            # Fallback
            return FinancialAnswer(
                question_id=question.id,
                answer=None,
                sources=[],  # Vide aussi dans le fallback
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
        et retourne aussi les détails pour traçabilité
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
            
            # Format pour le LLM avec document et section visibles
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
        Construit la structure evidence_chunks avec métadonnées essentielles uniquement
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
        Gère les cas où le LLM ajoute du texte avant/après
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
        
        # Augmenter si des sources sont fournies par le LLM
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
        Génère un résumé des informations financières extraites
        """
        summary = {}
        
        # Mapping des questions aux clés du résumé
        mapping = {
            "financial.billing_terms": "billing_terms",
            "financial.payment_terms": "payment_terms",
            "financial.currency_and_fx": "currency",
            "financial.bank_guarantee_or_advance": "bank_guarantee",
            "financial.price_revision_clauses": "price_revision"
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
        Génère un rapport textuel détaillé de l'extraction financière
        """
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append("💰 RAPPORT D'EXTRACTION FINANCIÈRE")
        report_lines.append("="*60)
        
        # Résumé des réponses
        report_lines.append("\n📋 INFORMATIONS FINANCIÈRES EXTRAITES:")
        report_lines.append("-" * 40)
        
        for key, value in results['summary'].items():
            if not key.endswith('_quality'):
                quality = results['summary'].get(f"{key}_quality", {})
                if value:
                    confidence = quality.get('confidence', 0) * 100
                    sources = quality.get('sources_count', 0)
                    
                    # Limiter l'affichage pour la lisibilité
                    value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    
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
            
            # Ratio de cache
            cache_hits = stats.get('cache_hits', 0)
            cache_misses = stats.get('cache_misses', 0)
            if cache_hits + cache_misses > 0:
                cache_ratio = (cache_hits / (cache_hits + cache_misses)) * 100
                report_lines.append(f"   Cache hit ratio: {cache_ratio:.1f}%")
        
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
    """Test de l'agent financier avec traçabilité complète et cache optimisé"""
    import os
    
    # Vérifier la clé API
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY non trouvée")
        print("   Définissez-la avec: export OPENAI_API_KEY='votre-clé'")
        return
    
    print("\n" + "="*60)
    print("💰 AGENT FINANCIER AVEC TRAÇABILITÉ COMPLÈTE")
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
    
    # 3. Créer l'agent financier
    print("\n💰 Création de l'agent financier...")
    agent = FinancialAgent(
        vectorstore=vectorstore,
        retriever=retriever,
        config_path="configs/prompts/financial/en.yaml"
    )
    
    # Afficher l'état du cache
    if hasattr(agent, 'embeddings_cache'):
        print(f"💾 Cache d'embeddings: {len(agent.embeddings_cache)} queries pré-calculées")
    
    # 4. Lancer l'extraction
    print("\n🚀 Lancement de l'extraction financière...")
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
            
            print(f"\n📈 Comparaison Performance:")
            print(f"   ├─ Temps total avec cache: {elapsed_time:.1f}s")
            print(f"   ├─ Temps estimé sans cache: ~{elapsed_time + time_saved:.1f}s")
            print(f"   └─ Amélioration: {(time_saved/(elapsed_time + time_saved)*100):.0f}% plus rapide")
        else:
            print("⚠️ Aucune statistique de cache disponible")
    
    # 7. Exemple de traçabilité détaillée
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
                
                # Afficher aussi les sources du LLM si présentes
                if answer_data.get('sources'):
                    print(f"📝 Réponse: {answer_result}")
                    print(f"📚 Sources identifiées par le LLM:")
                    for src in answer_data['sources'][:2]:
                        print(f"   • {src.get('document', 'Unknown')}: {src.get('snippet', '')[:50]}...")
                else:
                    print(f"📝 Réponse: {answer_result}")
            else:
                print(f"📝 Réponse: {answer_data}")
            
            print(f"🎯 Confiance: {result['confidence']*100:.0f}%")
            
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
            break
    
    if not example_shown:
        print("ℹ️ Aucun exemple de traçabilité disponible")
    
    # 8. Sauvegarder les résultats
    output_file = Path("financial_results_with_evidence.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Résultats sauvegardés dans: {output_file}")
    
    # Résumé final
    print("\n" + "="*60)
    print("✅ EXTRACTION FINANCIÈRE TERMINÉE!")
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
    
    # Suggestion d'optimisation
    if cache_misses > 0:
        print(f"\n💡 Conseil: {cache_misses} requêtes n'étaient pas dans le cache.")
        print(f"   Relancez l'extraction pour bénéficier du cache à 100%!")


if __name__ == "__main__":
    main()