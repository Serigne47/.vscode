# agents/base_agent.py
"""
Base Agent optimisé pour extraction basée sur configuration YAML avec system_prompts
Architecture robuste avec gestion d'erreurs, retry, caching et validation
"""
import yaml
import json
import re
import logging
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES ET ENUMS
# ============================================================================

class ExtractionStatus(Enum):
    """Statuts possibles d'une extraction"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ExtractionSource:
    """Source d'une extraction"""
    document: str
    context_snippet: str
    page: Optional[int] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour JSON"""
        result = {
            "document": self.document,
            "context_snippet": self.context_snippet
        }
        if self.page is not None:
            result["page"] = self.page
        return result

@dataclass
class QuestionResult:
    """Résultat d'extraction pour une question"""
    question_id: str
    answer: Any
    sources: List[ExtractionSource] = field(default_factory=list)
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    confidence: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour JSON"""
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "status": self.status.value,
            "confidence": self.confidence,
            "error": self.error,
            "metadata": self.metadata,
            "extraction_time": self.extraction_time
        }

@dataclass
class QuestionConfig:
    """Configuration d'une question depuis YAML"""
    id: str
    system_prompt: str
    validator: Optional[Dict[str, Any]] = None
    follow_ups: List[str] = field(default_factory=list)
    required: bool = False
    retry_on_failure: bool = True
    max_retries: int = 2
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QuestionConfig':
        """Crée une instance depuis un dictionnaire YAML"""
        return cls(
            id=data.get('id', ''),
            system_prompt=data.get('system_prompt', ''),
            validator=data.get('validator'),
            follow_ups=data.get('follow_ups', []),
            required=data.get('required', False),
            retry_on_failure=data.get('retry_on_failure', True),
            max_retries=data.get('max_retries', 2)
        )

# ============================================================================
# CLASSE DE BASE POUR LES AGENTS
# ============================================================================

class YAMLBaseAgent(ABC):
    """
    Classe de base pour les agents d'extraction basés sur configuration YAML
    Gère l'extraction, la validation, le retry et le caching
    """
    
    def __init__(
        self,
        config_path: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4000,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        max_parallel: int = 5
    ):
        """
        Initialise l'agent avec configuration YAML
        
        Args:
            config_path: Chemin vers le fichier YAML
            model: Modèle OpenAI à utiliser
            max_tokens: Nombre max de tokens
            enable_cache: Activer le cache des résultats
            enable_parallel: Activer l'extraction parallèle
            max_parallel: Nombre max d'extractions parallèles
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.questions = self._parse_questions()
        
        # Configuration LLM
        self.llm = ChatOpenAI(
            model=model,
            max_tokens=max_tokens
        )
        
        # Options
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel
        self.max_parallel = max_parallel
        
        # Cache des résultats
        self.cache = {} if enable_cache else None
        
        # Statistiques
        self.stats = {
            "total_questions": len(self.questions),
            "successful_extractions": 0,
            "failed_extractions": 0,
            "cache_hits": 0,
            "total_llm_calls": 0,
            "total_time": 0.0
        }
        
        logger.info(f"✅ Agent initialisé avec {len(self.questions)} questions depuis {config_path}")
    
    def _load_config(self) -> List[Dict]:
        """Charge la configuration depuis le fichier YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration non trouvée: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _parse_questions(self) -> Dict[str, QuestionConfig]:
        """Parse les questions depuis la configuration"""
        questions = {}
        for item in self.config:
            if isinstance(item, dict) and 'id' in item:
                question = QuestionConfig.from_dict(item)
                questions[question.id] = question
        return questions
    
    # ============================================================================
    # MÉTHODE PRINCIPALE D'EXTRACTION
    # ============================================================================
    
    async def extract(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Méthode principale d'extraction pour toutes les questions
        
        Args:
            chunks: Documents à analyser
            
        Returns:
            Résultats complets avec métadonnées
        """
        start_time = datetime.now()
        
        # Préparer le contexte
        context = self._prepare_context(chunks)
        context_metadata = self._analyze_context(chunks)
        
        # Résultats
        results = {
            "agent": self.__class__.__name__,
            "extraction_date": datetime.now().isoformat(),
            "questions": {},
            "metadata": context_metadata,
            "stats": {},
            "errors": []
        }
        
        try:
            # Extraction parallèle ou séquentielle
            if self.enable_parallel:
                question_results = await self._extract_parallel(context, chunks)
            else:
                question_results = await self._extract_sequential(context, chunks)
            
            # Traiter les résultats
            for question_id, result in question_results.items():
                results["questions"][question_id] = result.to_dict()
                
                # Mettre à jour les stats
                if result.status == ExtractionStatus.SUCCESS:
                    self.stats["successful_extractions"] += 1
                else:
                    self.stats["failed_extractions"] += 1
                    if result.error:
                        results["errors"].append({
                            "question_id": question_id,
                            "error": result.error
                        })
            
            # Gérer les follow-ups
            await self._process_follow_ups(question_results, context, chunks, results)
            
        except Exception as e:
            logger.error(f"Erreur critique dans l'extraction: {e}")
            results["errors"].append({
                "type": "critical",
                "error": str(e)
            })
        
        # Finaliser les stats
        end_time = datetime.now()
        self.stats["total_time"] = (end_time - start_time).total_seconds()
        results["stats"] = self.stats.copy()
        
        # Calculer le score de confiance global
        results["global_confidence"] = self._calculate_global_confidence(question_results)
        
        return results
    
    # ============================================================================
    # EXTRACTION SÉQUENTIELLE ET PARALLÈLE
    # ============================================================================
    
    async def _extract_sequential(
        self,
        context: str,
        chunks: List[Document]
    ) -> Dict[str, QuestionResult]:
        """Extraction séquentielle des questions"""
        results = {}
        
        for question_id, question_config in self.questions.items():
            result = await self._extract_single_question(
                question_config,
                context,
                chunks
            )
            results[question_id] = result
            
            # Log progression
            logger.info(f"📝 [{question_id}] Status: {result.status.value}, Confidence: {result.confidence:.2f}")
        
        return results
    
    async def _extract_parallel(
        self,
        context: str,
        chunks: List[Document]
    ) -> Dict[str, QuestionResult]:
        """Extraction parallèle des questions avec limite de concurrence"""
        results = {}
        semaphore = asyncio.Semaphore(self.max_parallel)
        
        async def extract_with_semaphore(q_id: str, q_config: QuestionConfig):
            async with semaphore:
                return q_id, await self._extract_single_question(q_config, context, chunks)
        
        # Créer les tâches
        tasks = [
            extract_with_semaphore(q_id, q_config)
            for q_id, q_config in self.questions.items()
        ]
        
        # Exécuter en parallèle
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Traiter les résultats
        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"Erreur dans extraction parallèle: {item}")
            else:
                q_id, result = item
                results[q_id] = result
                logger.info(f"📝 [{q_id}] Status: {result.status.value}, Confidence: {result.confidence:.2f}")
        
        return results
    
    # ============================================================================
    # EXTRACTION D'UNE QUESTION UNIQUE
    # ============================================================================
    
    async def _extract_single_question(
        self,
        question_config: QuestionConfig,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Extrait la réponse pour une question unique avec retry et validation
        """
        start_time = datetime.now()
        
        # Vérifier le cache
        cache_key = self._get_cache_key(question_config.id, context)
        if self.enable_cache and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            logger.debug(f"💾 Cache hit pour {question_config.id}")
            return self.cache[cache_key]
        
        # Tentatives d'extraction
        last_error = None
        for attempt in range(question_config.max_retries):
            try:
                # Appeler le LLM
                raw_response = await self._call_llm(
                    question_config.system_prompt,
                    context,
                    attempt
                )
                
                # Parser la réponse JSON
                parsed_response = self._parse_json_response(raw_response)
                
                # Valider la réponse
                if self._validate_response(parsed_response, question_config.validator):
                    # Créer le résultat
                    result = self._create_question_result(
                        question_config.id,
                        parsed_response,
                        chunks
                    )
                    
                    # Calculer le temps d'extraction
                    result.extraction_time = (datetime.now() - start_time).total_seconds()
                    
                    # Mettre en cache
                    if self.enable_cache:
                        self.cache[cache_key] = result
                    
                    return result
                else:
                    last_error = "Validation failed"
                    if attempt < question_config.max_retries - 1:
                        logger.warning(f"⚠️ Validation échouée pour {question_config.id}, retry {attempt + 1}")
                        await asyncio.sleep(0.5 * (attempt + 1))  # Backoff progressif
                        
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing error: {e}"
                logger.warning(f"⚠️ Erreur JSON pour {question_config.id}: {e}")
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"❌ Erreur extraction {question_config.id}: {e}")
        
        # Si toutes les tentatives ont échoué
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            confidence=0.0,
            error=last_error,
            extraction_time=(datetime.now() - start_time).total_seconds()
        )
    
    # ============================================================================
    # APPEL LLM ET PARSING
    # ============================================================================
    
    async def _call_llm(
        self,
        system_prompt: str,
        context: str,
        attempt: int = 0
    ) -> str:
        """
        Appelle le LLM avec le prompt système et le contexte
        """
        self.stats["total_llm_calls"] += 1
        
        # Ajuster le prompt pour les retry
        if attempt > 0:
            system_prompt += f"\n\nNote: This is attempt {attempt + 1}. Please ensure your response is valid JSON."
        
        # Limiter la taille du contexte si nécessaire
        max_context_length = 12000  # Tokens approximatifs
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n... [truncated]"
        
        # Créer le prompt
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Documents to analyze:\n\n{context}")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Erreur appel LLM: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict:
        """
        Parse la réponse JSON du LLM avec nettoyage
        """
        # Nettoyer la réponse
        response = response.strip()
        
        # Enlever les markdown code blocks si présents
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        # Parser le JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            # Essayer de corriger les erreurs communes
            response_fixed = self._fix_common_json_errors(response)
            return json.loads(response_fixed)
    
    def _fix_common_json_errors(self, json_str: str) -> str:
        """
        Corrige les erreurs JSON communes
        """
        # Remplacer les single quotes par des double quotes
        json_str = json_str.replace("'", '"')
        
        # Enlever les virgules trailing
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Ajouter des quotes aux clés non quotées
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        return json_str
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    
    def _validate_response(
        self,
        response: Dict,
        validator: Optional[Dict]
    ) -> bool:
        """
        Valide la réponse selon les règles du validator
        """
        if not validator:
            return True
        
        validator_type = validator.get('type', 'regex')
        rule = validator.get('rule', '.*')
        
        # Vérifier que la réponse contient 'answer'
        if 'answer' not in response:
            return False
        
        answer = response['answer']
        
        # Validation selon le type
        if validator_type == 'regex':
            if isinstance(answer, str):
                return bool(re.match(rule, answer))
            elif isinstance(answer, list):
                return all(re.match(rule, str(item)) for item in answer)
            else:
                return bool(re.match(rule, str(answer)))
        
        elif validator_type == 'type':
            expected_type = {
                'string': str,
                'number': (int, float),
                'boolean': bool,
                'array': list,
                'object': dict
            }.get(rule, str)
            return isinstance(answer, expected_type)
        
        elif validator_type == 'enum':
            allowed_values = validator.get('values', [])
            if isinstance(answer, list):
                return all(item in allowed_values for item in answer)
            else:
                return answer in allowed_values
        
        return True
    
    # ============================================================================
    # CRÉATION DES RÉSULTATS
    # ============================================================================
    
    def _create_question_result(
        self,
        question_id: str,
        parsed_response: Dict,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Crée un QuestionResult à partir de la réponse parsée
        """
        # Extraire l'answer et les sources
        answer = parsed_response.get('answer')
        raw_sources = parsed_response.get('sources', [])
        
        # Convertir les sources
        sources = []
        for source in raw_sources:
            if isinstance(source, dict):
                sources.append(ExtractionSource(
                    document=source.get('document', 'unknown'),
                    context_snippet=source.get('context_snippet', ''),
                    page=source.get('page')
                ))
        
        # Calculer la confiance
        confidence = self._calculate_confidence(answer, sources, chunks)
        
        # Déterminer le statut
        if answer is None or (isinstance(answer, str) and answer.lower() in ['none', 'n/a', 'not found']):
            status = ExtractionStatus.FAILED
        elif sources and len(sources) > 0:
            status = ExtractionStatus.SUCCESS
        else:
            status = ExtractionStatus.PARTIAL
        
        return QuestionResult(
            question_id=question_id,
            answer=answer,
            sources=sources,
            status=status,
            confidence=confidence,
            metadata={
                "response_length": len(str(answer)) if answer else 0,
                "sources_count": len(sources)
            }
        )
    
    # ============================================================================
    # FOLLOW-UPS
    # ============================================================================
    
    async def _process_follow_ups(
        self,
        initial_results: Dict[str, QuestionResult],
        context: str,
        chunks: List[Document],
        final_results: Dict
    ):
        """
        Traite les questions de follow-up conditionnelles
        """
        follow_up_results = {}
        
        for question_id, result in initial_results.items():
            if question_id not in self.questions:
                continue
                
            question_config = self.questions[question_id]
            
            # Vérifier s'il y a des follow-ups
            if not question_config.follow_ups:
                continue
            
            # Déterminer si les conditions sont remplies
            if result.status == ExtractionStatus.SUCCESS and result.answer:
                for follow_up_id in question_config.follow_ups:
                    if follow_up_id in self.questions and follow_up_id not in initial_results:
                        # Extraire la question de follow-up
                        follow_up_config = self.questions[follow_up_id]
                        
                        # Enrichir le contexte avec la réponse précédente
                        enriched_context = f"Previous answer for {question_id}: {result.answer}\n\n{context}"
                        
                        follow_up_result = await self._extract_single_question(
                            follow_up_config,
                            enriched_context,
                            chunks
                        )
                        
                        follow_up_results[follow_up_id] = follow_up_result
                        logger.info(f"🔗 Follow-up [{follow_up_id}] après [{question_id}]")
        
        # Ajouter les follow-ups aux résultats
        for follow_up_id, result in follow_up_results.items():
            final_results["questions"][follow_up_id] = result.to_dict()
    
    # ============================================================================
    # MÉTHODES UTILITAIRES
    # ============================================================================
    
    def _prepare_context(self, chunks: List[Document]) -> str:
        """
        Prépare le contexte à partir des chunks
        """
        context_parts = []
        total_length = 0
        max_length = 12000  # Limite approximative
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            chunk_length = len(chunk_text)
            
            # Vérifier si on dépasse la limite
            if total_length + chunk_length > max_length:
                remaining = max_length - total_length
                if remaining > 100:  # Ajouter au moins 100 caractères
                    context_parts.append(f"[Chunk {i+1}]\n{chunk_text[:remaining]}...")
                break
            
            # Ajouter le chunk avec métadonnées
            source = chunk.metadata.get('source', f'Document {i+1}')
            context_parts.append(f"[{source} - Chunk {i+1}]\n{chunk_text}")
            total_length += chunk_length
        
        return "\n\n---\n\n".join(context_parts)
    
    def _analyze_context(self, chunks: List[Document]) -> Dict:
        """
        Analyse le contexte pour métadonnées
        """
        total_chars = sum(len(c.page_content) for c in chunks)
        unique_sources = set(c.metadata.get('source', 'unknown') for c in chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "unique_sources": list(unique_sources),
            "average_chunk_size": total_chars / len(chunks) if chunks else 0
        }
    
    def _get_cache_key(self, question_id: str, context: str) -> str:
        """
        Génère une clé de cache unique
        """
        content = f"{question_id}:{context[:1000]}"  # Utiliser début du contexte
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_confidence(
        self,
        answer: Any,
        sources: List[ExtractionSource],
        chunks: List[Document]
    ) -> float:
        """
        Calcule le score de confiance pour une extraction
        """
        confidence = 0.0
        
        # Base : réponse trouvée
        if answer is not None:
            confidence += 0.3
        
        # Sources fournies
        if sources:
            confidence += min(0.3, len(sources) * 0.1)
        
        # Réponse non vide et substantielle
        if answer and isinstance(answer, (str, list, dict)):
            if isinstance(answer, str) and len(answer) > 10:
                confidence += 0.2
            elif isinstance(answer, list) and len(answer) > 0:
                confidence += 0.2
            elif isinstance(answer, dict) and len(answer) > 0:
                confidence += 0.2
        
        # Validation des sources dans le contexte
        if sources and chunks:
            valid_sources = 0
            for source in sources[:3]:  # Vérifier max 3 sources
                for chunk in chunks:
                    if source.context_snippet.lower() in chunk.page_content.lower():
                        valid_sources += 1
                        break
            
            if valid_sources > 0:
                confidence += min(0.2, valid_sources * 0.067)
        
        return min(confidence, 1.0)
    
    def _calculate_global_confidence(
        self,
        results: Dict[str, QuestionResult]
    ) -> float:
        """
        Calcule le score de confiance global
        """
        if not results:
            return 0.0
        
        # Moyenne pondérée des confiances
        total_confidence = 0.0
        total_weight = 0.0
        
        for question_id, result in results.items():
            # Pondération plus élevée pour les questions requises
            weight = 2.0 if self.questions.get(question_id, None) and getattr(self.questions[question_id], 'required', False) else 1.0
            
            total_confidence += result.confidence * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    # ============================================================================
    # MÉTHODES PUBLIQUES ADDITIONNELLES
    # ============================================================================
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques d'extraction"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Vide le cache"""
        if self.cache:
            self.cache.clear()
            logger.info("💾 Cache vidé")
    
    def get_question_ids(self) -> List[str]:
        """Retourne la liste des IDs de questions"""
        return list(self.questions.keys())
    
    def get_required_questions(self) -> List[str]:
        """Retourne la liste des questions requises"""
        return [
            q_id for q_id, q_config in self.questions.items()
            if q_config.required
        ]