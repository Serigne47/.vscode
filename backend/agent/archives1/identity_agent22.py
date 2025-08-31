# agents/identity_agent.py
"""
Agent d'identité intelligent utilisant le nouveau système de retrieval LLM
Extraction précise avec traçabilité complète
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import des composants intelligents
from utils.vectorstore import IntelligentVectorStore
from utils.enhanced_retrieval import IntelligentRetriever, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class IdentityQuestion:
    """Structure d'une question d'identité"""
    id: str
    system_prompt: str
    validator: Dict[str, Any]
    follow_ups: List[str] = None
    category: str = "identity"
    
    def __post_init__(self):
        if self.follow_ups is None:
            self.follow_ups = []


@dataclass
class IdentityAnswer:
    """Réponse structurée pour une question d'identité"""
    question_id: str
    answer: Any
    sources: List[Dict[str, str]]
    confidence: float
    status: str  # success, partial, failed
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class IntelligentIdentityAgent:
    """
    Agent d'extraction d'identité utilisant le système intelligent
    Exploite pleinement les capacités LLM pour extraction précise
    """
    
    def __init__(
        self,
        vectorstore: IntelligentVectorStore,
        retriever: IntelligentRetriever,
        config_path: str = "configs/prompts/identity/en.yaml",
        llm_model: str = "gpt-4o-mini",
        enable_validation: bool = True,
        enable_multi_source: bool = True
    ):
        """
        Initialise l'agent d'identité intelligent
        
        Args:
            vectorstore: IntelligentVectorStore configuré
            retriever: IntelligentRetriever configuré
            config_path: Chemin vers le YAML des questions
            llm_model: Modèle LLM pour extraction
            enable_validation: Activer la validation des réponses
            enable_multi_source: Chercher dans plusieurs sources
        """
        self.vectorstore = vectorstore
        self.retriever = retriever
        #self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.llm = ChatOpenAI(model=llm_model)
        self.enable_validation = enable_validation
        self.enable_multi_source = enable_multi_source
        
        # Charger les questions depuis le YAML
        self.questions = self._load_questions(config_path)
        
        # Cache pour optimisation
        self.answer_cache = {}
        
        # Statistiques
        self.stats = {
            'questions_processed': 0,
            'successful_extractions': 0,
            'partial_extractions': 0,
            'failed_extractions': 0,
            'total_sources': 0,
            'avg_confidence': 0.0
        }
        
        logger.info(f"✅ IntelligentIdentityAgent initialisé")
        logger.info(f"   - Questions: {len(self.questions)}")
        logger.info(f"   - LLM: {llm_model}")
        logger.info(f"   - Validation: {enable_validation}")
    
    def _load_questions(self, config_path: str) -> List[IdentityQuestion]:
        """Charge les questions depuis le fichier YAML"""
        questions = []
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            for item in config:
                question = IdentityQuestion(
                    id=item['id'],
                    system_prompt=item['system_prompt'],
                    validator=item.get('validator', {}),
                    follow_ups=item.get('follow_ups', [])
                )
                questions.append(question)
            
            logger.info(f"📋 {len(questions)} questions chargées depuis {config_path}")
            
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            raise
        
        return questions
    
    # ============================================================================
    # EXTRACTION PRINCIPALE
    # ============================================================================
    
    async def extract_all(self, parallel: bool = True) -> Dict[str, Any]:
        """
        Extrait toutes les informations d'identité
        
        Args:
            parallel: Traiter les questions en parallèle
            
        Returns:
            Dictionnaire avec toutes les réponses et métadonnées
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 EXTRACTION D'IDENTITÉ INTELLIGENTE")
        logger.info("="*60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'questions': {},
            'summary': {},
            'stats': {}
        }
        
        # Traiter chaque question
        if parallel:
            # Traitement parallèle avec asyncio
            tasks = [
                self._extract_single_async(question) 
                for question in self.questions
            ]
            answers = await asyncio.gather(*tasks)
        else:
            # Traitement séquentiel
            answers = []
            for question in self.questions:
                answer = await self._extract_single_async(question)
                answers.append(answer)
        
        # Compiler les résultats
        for answer in answers:
            results['questions'][answer.question_id] = answer.to_dict()
            self._update_stats(answer)
        
        # Générer le résumé
        results['summary'] = self._generate_summary(results['questions'])
        
        # Ajouter les statistiques
        results['stats'] = self.stats
        
        logger.info(f"\n✅ Extraction terminée:")
        logger.info(f"   - Succès: {self.stats['successful_extractions']}")
        logger.info(f"   - Partiels: {self.stats['partial_extractions']}")
        logger.info(f"   - Échecs: {self.stats['failed_extractions']}")
        logger.info(f"   - Confiance moyenne: {self.stats['avg_confidence']:.1%}")
        
        return results
    
    async def _extract_single_async(self, question: IdentityQuestion) -> IdentityAnswer:
        """Version async de l'extraction pour une question"""
        return self.extract_single(question)
    
    def extract_single(self, question: IdentityQuestion) -> IdentityAnswer:
        """
        Extrait l'information pour une question spécifique
        
        Args:
            question: Question d'identité à traiter
            
        Returns:
            IdentityAnswer avec réponse et sources
        """
        self.stats['questions_processed'] += 1
        
        # Vérifier le cache
        if question.id in self.answer_cache:
            logger.info(f"📋 Réponse depuis cache pour: {question.id}")
            return self.answer_cache[question.id]
        
        logger.info(f"\n🔍 Traitement: {question.id}")
        
        try:
            # Phase 1: Créer une requête optimisée pour le retrieval
            search_query = self._create_search_query(question)
            
            # Phase 2: Retrieval intelligent
            retrieval_result = self.retriever.retrieve_and_answer(
                query=search_query,
                category="identity",
                require_source=True,
                max_chunks=5
            )
            
            # Phase 3: Extraction structurée depuis les chunks
            extracted_answer = self._extract_from_retrieval(
                question,
                retrieval_result
            )
            
            # Phase 4: Validation si activée
            if self.enable_validation:
                extracted_answer = self._validate_answer(question, extracted_answer)
            
            # Phase 5: Recherche multi-source si nécessaire
            if self.enable_multi_source and extracted_answer.confidence < 0.7:
                extracted_answer = self._enhance_with_multi_source(
                    question,
                    extracted_answer,
                    retrieval_result
                )
            
            # Mettre en cache
            self.answer_cache[question.id] = extracted_answer
            
            logger.info(f"   ✅ Extraction: {extracted_answer.status} (confiance: {extracted_answer.confidence:.1%})")
            
            return extracted_answer
            
        except Exception as e:
            logger.error(f"   ❌ Erreur extraction {question.id}: {e}")
            return IdentityAnswer(
                question_id=question.id,
                answer=None,
                sources=[],
                confidence=0.0,
                status="failed",
                metadata={"error": str(e)}
            )
    
    # ============================================================================
    # CRÉATION DE REQUÊTES
    # ============================================================================
    
    def _create_search_query(self, question: IdentityQuestion) -> str:
        """
        Crée une requête de recherche optimisée pour la question
        """
        # Extraire les concepts clés du prompt
        prompt_analysis = self._analyze_prompt(question.system_prompt)
        
        # Mapper les IDs aux requêtes optimisées
        query_templates = {
            "identity.client_name": "client company name issuer tender RFP who issued procurement",
            "identity.tender_reference": "tender reference number code RFP RFQ ID identification",
            "identity.timeline_milestones": "deadline submission date timeline milestones calendar schedule",
            "identity.submission_channel": "submit submission channel email portal platform how to send",
            "identity.expected_deliverables": "deliverables documents required attachments files to provide",
            "identity.operating_countries": "countries regions corridors geographical scope coverage areas",
            "identity.service_main_scope": "service main scope objective transport logistics warehousing",
            "identity.contract_type": "contract type framework agreement spot duration terms",
            "identity.contract_duration": "contract duration period term years months renewal extension"
        }
        
        # Utiliser le template ou créer depuis l'analyse
        base_query = query_templates.get(
            question.id,
            " ".join(prompt_analysis['key_concepts'][:5])
        )
        
        return base_query
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyse le prompt pour extraire les concepts clés
        """
        analysis_prompt = PromptTemplate(
            input_variables=["prompt"],
            template="""Analyze this extraction prompt and identify key concepts to search for.

Prompt: {prompt}

Return JSON:
{{
    "key_concepts": ["concept1", "concept2", ...],  // Max 5
    "entity_types": ["dates", "amounts", "companies", etc],
    "expected_format": "string/list/dict"
}}"""
        )
        
        try:
            chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
            result = chain.run(prompt=prompt[:1000])  # Limiter la taille
            return json.loads(result)
        except:
            # Fallback basique
            return {
                "key_concepts": prompt.lower().split()[:5],
                "entity_types": [],
                "expected_format": "string"
            }
    
    # ============================================================================
    # EXTRACTION DEPUIS RETRIEVAL
    # ============================================================================
    
    def _extract_from_retrieval(
        self,
        question: IdentityQuestion,
        retrieval_result: RetrievalResult
    ) -> IdentityAnswer:
        """
        Extrait la réponse structurée depuis les résultats du retrieval
        """
        # Préparer le contexte depuis les chunks
        context = self._prepare_extraction_context(retrieval_result.chunks)
        
        # Template d'extraction avec le prompt de la question
        extraction_prompt = PromptTemplate(
            input_variables=["system_prompt", "context"],
            template="""{system_prompt}

Context from documents:
{context}

Remember to return ONLY valid JSON as specified in the system prompt."""
        )
        
        try:
            chain = LLMChain(llm=self.llm, prompt=extraction_prompt)
            result = chain.run(
                system_prompt=question.system_prompt,
                context=context
            )
            
            # Parser la réponse JSON
            parsed_result = json.loads(result)
            
            # Créer l'IdentityAnswer
            answer = IdentityAnswer(
                question_id=question.id,
                answer=parsed_result.get('answer'),
                sources=self._format_sources(
                    parsed_result.get('sources', []),
                    retrieval_result.sources
                ),
                confidence=retrieval_result.confidence,
                status=self._determine_status(
                    parsed_result.get('answer'),
                    retrieval_result.confidence
                ),
                metadata={
                    'chunks_used': len(retrieval_result.chunks),
                    'retrieval_confidence': retrieval_result.confidence
                }
            )
            
            return answer
            
        except json.JSONDecodeError as e:
            logger.warning(f"Erreur parsing JSON pour {question.id}: {e}")
            # Essayer d'extraire quand même une réponse
            return self._fallback_extraction(question, retrieval_result)
        except Exception as e:
            logger.error(f"Erreur extraction {question.id}: {e}")
            return IdentityAnswer(
                question_id=question.id,
                answer=None,
                sources=[],
                confidence=0.0,
                status="failed",
                metadata={"error": str(e)}
            )
    
    def _prepare_extraction_context(self, chunks: List[Dict]) -> str:
        """
        Prépare le contexte optimisé pour l'extraction
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks[:5]):  # Max 5 chunks
            # Info de source
            source = chunk.get('source', {})
            source_info = f"[Document: {source.get('document', 'Unknown')} | Section: {source.get('section', '')} | Page: {source.get('page', '')}]"
            
            # Texte du chunk
            text = chunk.get('text', '')
            
            # Ajouter contexte adjacent si disponible
            if 'context' in chunk:
                context = chunk['context']
                if prev := context.get('prev_snippet'):
                    text = f"[...]{prev}\n{text}"
                if next := context.get('next_snippet'):
                    text = f"{text}\n{next}[...]"
            
            context_parts.append(f"{source_info}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_sources(
        self,
        extracted_sources: List[Dict],
        retrieval_sources: List[Dict]
    ) -> List[Dict[str, str]]:
        """
        Formate et enrichit les sources
        """
        formatted_sources = []
        
        # D'abord utiliser les sources extraites
        for source in extracted_sources[:3]:  # Max 3 sources
            formatted_sources.append({
                'document': source.get('document', 'Unknown'),
                'context_snippet': source.get('context_snippet', ''),
                'page': None,
                'confidence': 0.0
            })
        
        # Enrichir avec les sources du retrieval
        for ret_source in retrieval_sources[:3]:
            # Chercher si déjà présent
            found = False
            for fmt_source in formatted_sources:
                if fmt_source['document'] == ret_source.get('document'):
                    # Enrichir
                    fmt_source['page'] = ret_source.get('page')
                    fmt_source['confidence'] = ret_source.get('confidence', 0)
                    fmt_source['breadcrumb'] = ret_source.get('breadcrumb', '')
                    found = True
                    break
            
            if not found and len(formatted_sources) < 3:
                formatted_sources.append({
                    'document': ret_source.get('document'),
                    'context_snippet': '',
                    'page': ret_source.get('page'),
                    'confidence': ret_source.get('confidence', 0),
                    'breadcrumb': ret_source.get('breadcrumb', '')
                })
        
        return formatted_sources
    
    def _determine_status(self, answer: Any, confidence: float) -> str:
        """
        Détermine le statut de l'extraction
        """
        if answer is None or answer == "":
            return "failed"
        elif confidence < 0.5:
            return "partial"
        else:
            return "success"
    
    def _fallback_extraction(
        self,
        question: IdentityQuestion,
        retrieval_result: RetrievalResult
    ) -> IdentityAnswer:
        """
        Extraction de fallback si le parsing JSON échoue
        """
        # Utiliser directement la réponse du retrieval
        return IdentityAnswer(
            question_id=question.id,
            answer=retrieval_result.answer if retrieval_result.answer else None,
            sources=self._format_sources([], retrieval_result.sources),
            confidence=retrieval_result.confidence * 0.5,  # Réduire la confiance
            status="partial",
            metadata={
                'fallback': True,
                'reason': 'JSON parsing failed'
            }
        )
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    
    def _validate_answer(
        self,
        question: IdentityQuestion,
        answer: IdentityAnswer
    ) -> IdentityAnswer:
        """
        Valide la réponse selon les règles du validator
        """
        if not question.validator:
            return answer
        
        validator_type = question.validator.get('type')
        rule = question.validator.get('rule')
        
        if validator_type == 'regex' and rule:
            import re
            if answer.answer:
                # Convertir en string si nécessaire
                answer_str = str(answer.answer) if not isinstance(answer.answer, str) else answer.answer
                
                if not re.match(rule, answer_str):
                    logger.warning(f"Validation échouée pour {question.id}")
                    answer.confidence *= 0.7
                    if answer.status == "success":
                        answer.status = "partial"
        
        return answer
    
    # ============================================================================
    # ENRICHISSEMENT MULTI-SOURCE
    # ============================================================================
    
    def _enhance_with_multi_source(
        self,
        question: IdentityQuestion,
        initial_answer: IdentityAnswer,
        initial_retrieval: RetrievalResult
    ) -> IdentityAnswer:
        """
        Enrichit la réponse en cherchant dans d'autres sources
        """
        logger.info(f"   🔄 Enrichissement multi-source pour {question.id}")
        
        # Créer des requêtes alternatives
        alternative_queries = [
            self._create_search_query(question),
            initial_retrieval.query + " additional information",
        ]
        
        # Si on a des follow-ups dans la question
        for follow_up in question.follow_ups[:2]:
            alternative_queries.append(follow_up)
        
        all_sources = list(initial_answer.sources)
        best_answer = initial_answer.answer
        max_confidence = initial_answer.confidence
        
        for alt_query in alternative_queries[1:]:  # Skip la première (déjà faite)
            try:
                # Nouveau retrieval
                alt_result = self.retriever.retrieve_and_answer(
                    query=alt_query,
                    category="identity",
                    require_source=True,
                    max_chunks=3
                )
                
                if alt_result.confidence > max_confidence:
                    # Extraire depuis ce nouveau résultat
                    alt_answer = self._extract_from_retrieval(question, alt_result)
                    
                    if alt_answer.confidence > max_confidence:
                        best_answer = alt_answer.answer
                        max_confidence = alt_answer.confidence
                    
                    # Ajouter les sources
                    all_sources.extend(alt_answer.sources)
                    
            except Exception as e:
                logger.debug(f"Erreur enrichissement: {e}")
                continue
        
        # Créer la réponse enrichie
        enhanced_answer = IdentityAnswer(
            question_id=question.id,
            answer=best_answer,
            sources=all_sources[:5],  # Max 5 sources
            confidence=max_confidence,
            status=self._determine_status(best_answer, max_confidence),
            metadata={
                'multi_source': True,
                'queries_tried': len(alternative_queries)
            }
        )
        
        return enhanced_answer
    
    # ============================================================================
    # RÉSUMÉ ET STATISTIQUES
    # ============================================================================
    
    def _generate_summary(self, questions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Génère un résumé structuré de l'identité
        """
        summary = {
            "client": None,
            "reference": None,
            "deadline": None,
            "submission_method": None,
            "deliverables": None,
            "countries": None,
            "scope": None,
            "contract_type": None,
            "contract_duration": None,
            "extraction_quality": "high"  # high, medium, low
        }
        
        # Mapper les questions aux clés du résumé
        mappings = {
            "identity.client_name": "client",
            "identity.tender_reference": "reference",
            "identity.timeline_milestones": "deadline",
            "identity.submission_channel": "submission_method",
            "identity.expected_deliverables": "deliverables",
            "identity.operating_countries": "countries",
            "identity.service_main_scope": "scope",
            "identity.contract_type": "contract_type",
            "identity.contract_duration": "contract_duration"
        }
        
        total_confidence = 0
        count = 0
        
        for question_id, summary_key in mappings.items():
            if question_id in questions:
                result = questions[question_id]
                if result['status'] in ['success', 'partial']:
                    summary[summary_key] = result['answer']
                    total_confidence += result['confidence']
                    count += 1
        
        # Déterminer la qualité globale
        if count > 0:
            avg_confidence = total_confidence / count
            if avg_confidence > 0.8:
                summary['extraction_quality'] = 'high'
            elif avg_confidence > 0.5:
                summary['extraction_quality'] = 'medium'
            else:
                summary['extraction_quality'] = 'low'
        
        return summary
    
    def _update_stats(self, answer: IdentityAnswer):
        """Met à jour les statistiques"""
        if answer.status == "success":
            self.stats['successful_extractions'] += 1
        elif answer.status == "partial":
            self.stats['partial_extractions'] += 1
        else:
            self.stats['failed_extractions'] += 1
        
        self.stats['total_sources'] += len(answer.sources)
        
        # Mettre à jour la confiance moyenne
        total_processed = (
            self.stats['successful_extractions'] +
            self.stats['partial_extractions'] +
            self.stats['failed_extractions']
        )
        
        if total_processed > 0:
            current_avg = self.stats['avg_confidence']
            self.stats['avg_confidence'] = (
                (current_avg * (total_processed - 1) + answer.confidence) / total_processed
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques"""
        return self.stats.copy()


# ============================================================================
# FONCTION PRINCIPALE DE TEST
# ============================================================================

async def main():
    """Test de l'agent d'identité intelligent"""
    import os
    from pathlib import Path
    
    # Vérifier OpenAI key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY non trouvée")
        return
    
    print("\n" + "="*60)
    print("🤖 AGENT D'IDENTITÉ INTELLIGENT")
    print("="*60)
    
    # Imports locaux
    from utils.vectorstore import IntelligentVectorStore
    from utils.enhanced_retrieval import IntelligentRetriever
    
    # 1. Charger le vectorstore
    print("\n📚 Chargement du vectorstore intelligent...")
    vectorstore = IntelligentVectorStore(
        persist_directory=Path("data/intelligent_store")
    )
    stats = vectorstore.get_stats()
    print(f"✅ Vectorstore chargé: {stats['total_documents']} documents")
    
    if stats['total_documents'] == 0:
        print("❌ Le vectorstore est vide!")
        print("Exécutez d'abord: python utils/vectorize_documents.py")
        return
    
    # 2. Créer le retriever intelligent
    print("\n🔍 Initialisation du retriever intelligent...")
    retriever = IntelligentRetriever(
        vectorstore=vectorstore,
        enable_refinement=True,
        enable_multi_hop=True
    )
    
    # 3. Créer l'agent d'identité
    print("\n🤖 Création de l'agent d'identité...")
    agent = IntelligentIdentityAgent(
        vectorstore=vectorstore,
        retriever=retriever,
        config_path="configs/prompts/identity/en.yaml",
        enable_validation=True,
        enable_multi_source=True
    )
    
    # 4. Lancer l'extraction
    print("\n🚀 Lancement de l'extraction intelligente...")
    print("⏳ Cela peut prendre 30-60 secondes...")
    
    results = await agent.extract_all(parallel=True)
    
    # 5. Afficher les résultats
    print("\n" + "="*60)
    print("📊 RÉSULTATS D'EXTRACTION")
    print("="*60)
    
    # Résumé consolidé
    summary = results['summary']
    print("\n📋 RÉSUMÉ DE L'IDENTITÉ:")
    print("-" * 40)
    
    for key, value in summary.items():
        if key != 'extraction_quality':
            if value:
                print(f"✅ {key.upper()}: {value}")
            else:
                print(f"❌ {key.upper()}: Non trouvé")
    
    print(f"\n📈 Qualité d'extraction: {summary['extraction_quality'].upper()}")
    
    # Détails par question
    print("\n📝 DÉTAILS PAR QUESTION:")
    print("-" * 40)
    
    questions = results['questions']
    for question_id, result in questions.items():
        status_emoji = {
            'success': '✅',
            'partial': '⚠️',
            'failed': '❌'
        }.get(result['status'], '❓')
        
        print(f"\n{status_emoji} {question_id}:")
        print(f"   Status: {result['status']}")
        print(f"   Confiance: {result['confidence']:.1%}")
        
        if result['answer']:
            answer_str = str(result['answer'])[:100]
            print(f"   Réponse: {answer_str}...")
        
        if result['sources']:
            print(f"   Sources: {len(result['sources'])} document(s)")
            for source in result['sources'][:2]:
                doc = source.get('document', 'Unknown')
                page = source.get('page', 'N/A')
                print(f"     - {doc} (Page {page})")
    
    # Statistiques finales
    stats = results['stats']
    print("\n" + "="*60)
    print("📊 STATISTIQUES FINALES")
    print("="*60)
    print(f"Questions traitées: {stats['questions_processed']}")
    print(f"Extractions réussies: {stats['successful_extractions']}")
    print(f"Extractions partielles: {stats['partial_extractions']}")
    print(f"Extractions échouées: {stats['failed_extractions']}")
    print(f"Sources totales: {stats['total_sources']}")
    print(f"Confiance moyenne: {stats['avg_confidence']:.1%}")
    
    print("\n✅ Extraction terminée avec succès!")


if __name__ == "__main__":
    asyncio.run(main())