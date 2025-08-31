"""
Timeline Agent - Agent d'extraction de jalons temporels pour appels d'offres
Architecture basée sur YAML avec extraction LLM avancée
"""

import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from .base_agent import YAMLBaseAgent, QuestionResult, ExtractionStatus, ExtractionSource

logger = logging.getLogger(__name__)


class MilestoneType(Enum):
    """Types de jalons temporels"""
    SUBMISSION_DEADLINE = "submission_deadline"
    CLARIFICATION_DEADLINE = "clarification_deadline"
    RESPONSE_DEADLINE = "response_deadline"
    MEETING = "meeting"
    SITE_VISIT = "site_visit"
    CONTRACT_MILESTONE = "contract_milestone"
    PROJECT_PHASE = "project_phase"
    OTHER = "other"


@dataclass
class TimelineMilestone:
    """Représente un jalon temporel"""
    label: str
    date: str  # Format ISO YYYY-MM-DD
    milestone_type: MilestoneType
    is_mandatory: bool = False
    time: Optional[str] = None  # Format HH:MM
    location: Optional[str] = None
    description: Optional[str] = None
    sources: List[ExtractionSource] = field(default_factory=list)
    confidence: float = 0.0


class TimelineExtractionAgent(YAMLBaseAgent):
    """
    Agent d'extraction de timeline pour appels d'offres
    
    Spécialisations:
    - Jalons temporels et deadlines
    - Phases de projet et livrables
    - Réunions et événements obligatoires
    - Chronologie d'exécution
    """
    
    def __init__(
        self,
        config_path: str = "config/prompts/timeline_questions.yaml",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        use_advanced_extraction: bool = True
    ):
        """Initialise l'agent timeline"""
        super().__init__(
            config_path=config_path,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
            max_parallel=3
        )
        
        self.use_advanced = use_advanced_extraction
        
        if self.use_advanced:
            # LLM spécialisé pour extraction de dates
            self.date_llm = ChatOpenAI(
                model=model,
                temperature=0.0,
                max_tokens=3000
            )
        
        # Catalogues temporels
        self.temporal_catalog = self._init_temporal_catalog()
        self.patterns = self._init_patterns()
        
        # Cache des jalons extraits pour déduplication
        self.extracted_milestones: Dict[str, TimelineMilestone] = {}
        
        logger.info(f"✅ TimelineExtractionAgent initialized with {len(self.questions)} questions")
    
    def _init_temporal_catalog(self) -> Dict[str, Any]:
        """Initialise les catalogues de référence temporels"""
        return {
            'deadline_keywords': {
                'submission': ['submit', 'submission', 'tender', 'proposal', 'bid', 'offer', 'soumission'],
                'response': ['response', 'reply', 'answer', 'réponse'],
                'clarification': ['clarification', 'question', 'query', 'enquiry', 'éclaircissement'],
                'closing': ['closing', 'close', 'deadline', 'due', 'latest', 'dernier délai']
            },
            'meeting_keywords': [
                'meeting', 'conference', 'briefing', 'session', 'presentation',
                'réunion', 'conférence', 'séance'
            ],
            'phase_keywords': [
                'phase', 'stage', 'step', 'round', 'milestone', 'deliverable',
                'étape', 'livrable', 'jalons'
            ],
            'temporal_indicators': [
                'by', 'before', 'no later than', 'until', 'on', 'at',
                'avant', 'au plus tard', 'jusqu\'à'
            ],
            'mandatory_indicators': [
                'mandatory', 'required', 'must', 'shall', 'obligatory',
                'obligatoire', 'requis', 'doit'
            ]
        }
    
    def _init_patterns(self) -> Dict[str, str]:
        """Initialise les patterns regex pour extraction de dates et heures"""
        return {
            # Dates
            'iso_date': r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',
            'euro_date': r'\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b',
            'us_date': r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
            'text_date': r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',
            'text_date_us': r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b',
            
            # Heures
            'time_24h': r'\b([0-2]?\d):([0-5]\d)(?::([0-5]\d))?\s*(?:hrs?|hours?)?\b',
            'time_12h': r'\b(\d{1,2}):([0-5]\d)\s*(AM|PM|am|pm)\b',
            
            # Durées et délais
            'duration': r'(?:within|sous|dans)\s+(\d+)\s+(days?|hours?|weeks?|months?|jours?|heures?|semaines?|mois)',
            'relative_date': r'(?:in|within|after|dans|après)\s+(\d+)\s+(days?|weeks?|months?|jours?|semaines?|mois)',
            
            # Phases et rounds
            'round_phase': r'(?:(first|second|third|1st|2nd|3rd|première|deuxième|troisième)\s+)?(round|phase|stage|étape|tour)\s*(\d+)?'
        }
    
    # ============================================================================
    # EXTRACTION PRINCIPALE BASÉE SUR YAML
    # ============================================================================
    
    async def _extract_single_question(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Override: Extraction spécialisée pour questions temporelles"""
        if not self.use_advanced:
            return await super()._extract_single_question(question_config, context, chunks)
        
        question_id = question_config.id
        
        # Router vers la stratégie appropriée selon l'ID de la question
        if 'milestones' in question_id:
            return await self._extract_timeline_milestones(question_config, context, chunks)
        elif 'submission_deadline' in question_id:
            return await self._extract_submission_deadlines(question_config, context, chunks)
        elif 'clarification' in question_id:
            return await self._extract_clarification_deadline(question_config, context, chunks)
        elif 'meetings' in question_id or 'events' in question_id:
            return await self._extract_meetings_events(question_config, context, chunks)
        elif 'phases' in question_id or 'stages' in question_id:
            return await self._extract_project_phases(question_config, context, chunks)
        elif 'contract' in question_id and 'duration' in question_id:
            return await self._extract_contract_duration(question_config, context, chunks)
        else:
            return await self._generic_timeline_extraction(question_config, context, chunks)
    
    # ============================================================================
    # STRATÉGIES D'EXTRACTION SPÉCIALISÉES
    # ============================================================================
    
    async def _extract_timeline_milestones(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction complète de tous les jalons temporels"""
        milestones = []
        sources_map = defaultdict(list)
        
        # 1. Pré-extraction par patterns pour identifier les zones riches en dates
        date_rich_chunks = self._identify_date_rich_chunks(chunks)
        
        # 2. Extraction structurée par type de jalon
        extraction_tasks = [
            self._extract_deadlines_from_chunks(date_rich_chunks),
            self._extract_meetings_from_chunks(date_rich_chunks),
            self._extract_phases_from_chunks(date_rich_chunks)
        ]
        
        if self.enable_parallel:
            results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        else:
            results = []
            for task in extraction_tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    logger.error(f"Extraction task failed: {e}")
                    results.append([])
        
        # Consolider les résultats
        for result in results:
            if isinstance(result, list):
                milestones.extend(result)
        
        # 3. Extraction LLM pour validation et jalons manqués
        llm_prompt = f"""
        {question_config.system_prompt}
        
        ADDITIONAL INSTRUCTIONS:
        1. Extract ALL dates mentioned with their context
        2. Include time if specified (e.g., "10:00 AM")
        3. Include location for meetings/events
        4. Indicate if attendance/submission is mandatory
        5. For each milestone, provide:
           - Clear descriptive label
           - Date in YYYY-MM-DD format
           - Type (deadline, meeting, phase start/end, etc.)
           - Any additional relevant details
        
        Already detected milestones to validate:
        {self._format_milestones_for_validation(milestones[:5])}
        
        Focus on finding any additional milestones not yet detected.
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=llm_prompt),
                HumanMessage(content=context[:12000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                llm_milestones = self._parse_llm_milestones(parsed['answer'])
                
                # Fusionner avec déduplication intelligente
                for llm_milestone in llm_milestones:
                    if not self._is_duplicate_milestone(llm_milestone, milestones):
                        milestones.append(llm_milestone)
                
                # Ajouter sources LLM
                if parsed.get('sources'):
                    for idx, src in enumerate(parsed['sources'][:5]):
                        sources_map['llm'].append(ExtractionSource(
                            document=src.get('document', 'document'),
                            context_snippet=src.get('context_snippet', ''),
                            confidence=0.85 - (idx * 0.05)
                        ))
        except Exception as e:
            logger.error(f"LLM milestone extraction failed: {e}")
        
        # 4. Post-traitement et structuration
        if milestones:
            # Trier par date
            milestones.sort(key=lambda m: (m.date, m.time or '00:00'))
            
            # Enrichir avec métadonnées
            structured_milestones = self._structure_milestones(milestones)
            
            # Préparer les sources
            all_sources = []
            for milestone in milestones[:10]:  # Top 10 milestones
                all_sources.extend(milestone.sources[:1])  # 1 source par milestone
            
            return QuestionResult(
                question_id=question_config.id,
                answer=structured_milestones,
                sources=all_sources[:5],
                status=ExtractionStatus.SUCCESS,
                confidence=self._calculate_timeline_confidence(milestones),
                metadata={
                    'total_milestones': len(milestones),
                    'mandatory_count': sum(1 for m in milestones if m.is_mandatory),
                    'has_submission_deadline': any(m.milestone_type == MilestoneType.SUBMISSION_DEADLINE for m in milestones),
                    'date_range': self._get_date_range(milestones)
                }
            )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No timeline milestones found"
        )
    
    async def _extract_submission_deadlines(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction spécifique des deadlines de soumission"""
        deadlines = []
        sources = []
        
        # 1. Recherche ciblée par mots-clés
        deadline_keywords = self.temporal_catalog['deadline_keywords']
        
        for chunk in chunks:
            content = chunk.page_content
            content_lower = content.lower()
            
            # Chercher les contextes de deadline
            for deadline_type, keywords in deadline_keywords.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        # Extraire les dates proches du mot-clé
                        keyword_positions = [m.start() for m in re.finditer(keyword, content_lower)]
                        
                        for pos in keyword_positions:
                            # Fenêtre de recherche autour du mot-clé
                            window_start = max(0, pos - 200)
                            window_end = min(len(content), pos + 200)
                            window = content[window_start:window_end]
                            
                            # Extraire dates dans la fenêtre
                            dates = self._extract_dates_from_text(window)
                            
                            for date, date_match in dates:
                                if date:
                                    deadline = TimelineMilestone(
                                        label=self._generate_deadline_label(deadline_type, window),
                                        date=date,
                                        milestone_type=MilestoneType.SUBMISSION_DEADLINE,
                                        is_mandatory=True,
                                        description=self._extract_deadline_context(window, date_match),
                                        sources=[self._create_milestone_source(chunk, keyword)],
                                        confidence=0.9
                                    )
                                    
                                    # Extraire l'heure si présente
                                    time = self._extract_time_near_date(window, date_match)
                                    if time:
                                        deadline.time = time
                                    
                                    deadlines.append(deadline)
        
        # 2. Extraction LLM pour deadlines complexes
        deadline_prompt = f"""
        {question_config.system_prompt}
        
        FOCUS: Extract ONLY submission/tender deadlines with:
        - Exact date (convert to YYYY-MM-DD)
        - Time if specified
        - Type of submission (first round, final, etc.)
        - Any specific requirements or conditions
        
        Common patterns:
        - "Proposals must be submitted by..."
        - "Closing date for tenders..."
        - "Deadline for submission..."
        - "Responses due..."
        """
        
        try:
            response = await self.date_llm.ainvoke([
                SystemMessage(content=deadline_prompt),
                HumanMessage(content=context[:10000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                llm_deadlines = self._parse_llm_deadlines(parsed['answer'])
                
                # Fusionner en évitant les doublons
                for llm_deadline in llm_deadlines:
                    if not self._is_duplicate_milestone(llm_deadline, deadlines):
                        deadlines.append(llm_deadline)
                
                # Sources
                if parsed.get('sources'):
                    sources.extend([
                        ExtractionSource(
                            document=s.get('document', 'document'),
                            context_snippet=s.get('context_snippet', ''),
                            confidence=0.85
                        ) for s in parsed['sources'][:3]
                    ])
        except Exception as e:
            logger.debug(f"LLM deadline extraction failed: {e}")
        
        # 3. Identifier les rounds/phases
        deadlines = self._classify_submission_rounds(deadlines)
        
        if deadlines:
            # Trier par date
            deadlines.sort(key=lambda d: d.date)
            
            # Formater pour sortie
            formatted_deadlines = [
                {
                    "label": d.label,
                    "date": d.date,
                    "time": d.time,
                    "is_mandatory": d.is_mandatory,
                    "description": d.description
                }
                for d in deadlines
            ]
            
            # Sources uniques
            all_sources = sources
            for deadline in deadlines:
                all_sources.extend(deadline.sources)
            
            return QuestionResult(
                question_id=question_config.id,
                answer=formatted_deadlines,
                sources=self._deduplicate_sources(all_sources)[:5],
                status=ExtractionStatus.SUCCESS,
                confidence=0.9 if len(deadlines) > 0 else 0.7,
                metadata={
                    'deadline_count': len(deadlines),
                    'has_multiple_rounds': len(set(d.label for d in deadlines)) > 1,
                    'earliest_deadline': deadlines[0].date if deadlines else None
                }
            )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No submission deadlines found"
        )
    
    async def _extract_clarification_deadline(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction de la deadline pour questions/clarifications"""
        clarification_dates = []
        sources = []
        
        # Mots-clés spécifiques aux clarifications
        clarification_patterns = [
            r'clarification.*?(?:deadline|due|submit)',
            r'questions?.*?(?:deadline|due|submit)',
            r'(?:deadline|date).*?(?:for|to submit).*?(?:clarification|questions?)',
            r'enquir(?:y|ies).*?(?:deadline|due)',
            r'last date.*?(?:clarification|questions?)'
        ]
        
        # 1. Recherche par patterns
        for chunk in chunks:
            content = chunk.page_content
            
            for pattern in clarification_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    # Extraire contexte étendu
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 150)
                    context_window = content[start:end]
                    
                    # Chercher dates dans le contexte
                    dates = self._extract_dates_from_text(context_window)
                    
                    for date, date_match in dates:
                        if date:
                            clarification_dates.append({
                                'date': date,
                                'context': context_window,
                                'source': chunk,
                                'confidence': 0.9
                            })
        
        # 2. Extraction LLM ciblée
        clarif_prompt = f"""
        {question_config.system_prompt}
        
        SPECIFIC FOCUS: Find the deadline for submitting clarification questions.
        
        Look for phrases like:
        - "Questions must be submitted by..."
        - "Deadline for clarifications..."
        - "Last date to submit queries..."
        - "Cut-off for questions..."
        
        Return ONLY the clarification question deadline, not other deadlines.
        """
        
        try:
            response = await self.date_llm.ainvoke([
                SystemMessage(content=clarif_prompt),
                HumanMessage(content=context[:8000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                # Extraire la date de clarification
                llm_date = self._extract_date_from_answer(parsed['answer'])
                if llm_date:
                    clarification_dates.append({
                        'date': llm_date,
                        'context': str(parsed['answer']),
                        'source': None,
                        'confidence': 0.85
                    })
                
                # Sources
                if parsed.get('sources'):
                    sources.extend([
                        ExtractionSource(
                            document=s.get('document', 'document'),
                            context_snippet=s.get('context_snippet', ''),
                            confidence=0.85
                        ) for s in parsed['sources'][:2]
                    ])
        except Exception as e:
            logger.debug(f"Clarification deadline extraction failed: {e}")
        
        # 3. Sélectionner la meilleure date
        if clarification_dates:
            # Prioriser par confiance et cohérence
            best_date = self._select_best_clarification_date(clarification_dates)
            
            if best_date:
                # Créer les sources
                for date_info in clarification_dates:
                    if date_info['source']:
                        sources.append(self._create_milestone_source(
                            date_info['source'], 
                            'clarification'
                        ))
                
                return QuestionResult(
                    question_id=question_config.id,
                    answer={
                        "date": best_date['date'],
                        "description": "Deadline for submitting clarification questions",
                        "is_mandatory": True
                    },
                    sources=sources[:3],
                    status=ExtractionStatus.SUCCESS,
                    confidence=best_date['confidence'],
                    metadata={
                        'multiple_dates_found': len(clarification_dates) > 1
                    }
                )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.NOT_FOUND,
            error="No clarification deadline found"
        )
    
    async def _extract_meetings_events(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction des réunions et événements"""
        events = []
        sources = []
        
        # 1. Détection par mots-clés
        meeting_keywords = self.temporal_catalog['meeting_keywords']
        
        for chunk in chunks:
            content = chunk.page_content
            content_lower = content.lower()
            
            for keyword in meeting_keywords:
                if keyword in content_lower:
                    # Extraire les occurrences
                    keyword_positions = [m.start() for m in re.finditer(keyword, content_lower)]
                    
                    for pos in keyword_positions:
                        # Contexte étendu pour capturer tous les détails
                        context_start = max(0, pos - 150)
                        context_end = min(len(content), pos + 200)
                        event_context = content[context_start:context_end]
                        
                        # Extraire les informations de l'événement
                        event_info = self._extract_event_details(event_context, keyword)
                        
                        if event_info and event_info.get('date'):
                            event = TimelineMilestone(
                                label=event_info['label'],
                                date=event_info['date'],
                                milestone_type=MilestoneType.MEETING,
                                is_mandatory=event_info.get('is_mandatory', False),
                                time=event_info.get('time'),
                                location=event_info.get('location'),
                                description=event_info.get('description'),
                                sources=[self._create_milestone_source(chunk, keyword)],
                                confidence=0.85
                            )
                            events.append(event)
        
        # 2. Extraction LLM pour événements complexes
        event_prompt = f"""
        {question_config.system_prompt}
        
        Extract ALL meetings, conferences, site visits, and events with:
        - Event name/type
        - Date (YYYY-MM-DD format)
        - Time if specified
        - Location/venue
        - Whether attendance is mandatory
        - Any special requirements
        
        Include: pre-bid meetings, site visits, bidder conferences, presentations
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=event_prompt),
                HumanMessage(content=context[:10000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                llm_events = self._parse_llm_events(parsed['answer'])
                
                # Fusionner intelligemment
                for llm_event in llm_events:
                    if not self._is_duplicate_event(llm_event, events):
                        events.append(llm_event)
                
                # Sources
                if parsed.get('sources'):
                    sources.extend([
                        ExtractionSource(
                            document=s.get('document', 'document'),
                            context_snippet=s.get('context_snippet', ''),
                            confidence=0.8
                        ) for s in parsed['sources'][:3]
                    ])
        except Exception as e:
            logger.debug(f"Event extraction failed: {e}")
        
        if events:
            # Trier chronologiquement
            events.sort(key=lambda e: (e.date, e.time or '00:00'))
            
            # Formater pour sortie
            formatted_events = [
                {
                    "label": e.label,
                    "date": e.date,
                    "time": e.time,
                    "location": e.location,
                    "is_mandatory": e.is_mandatory,
                    "description": e.description
                }
                for e in events
            ]
            
            # Collecter toutes les sources
            all_sources = sources
            for event in events:
                all_sources.extend(event.sources)
            
            return QuestionResult(
                question_id=question_config.id,
                answer=formatted_events,
                sources=self._deduplicate_sources(all_sources)[:5],
                status=ExtractionStatus.SUCCESS,
                confidence=0.85,
                metadata={
                    'event_count': len(events),
                    'mandatory_events': sum(1 for e in events if e.is_mandatory),
                    'has_site_visit': any('site' in e.label.lower() for e in events)
                }
            )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.NOT_FOUND,
            error="No meetings or events found"
        )
    
    async def _extract_project_phases(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction des phases et étapes du projet"""
        phases = []
        sources = []
        
        # 1. Détection de structures de phases
        phase_patterns = [
            r'phase\s*(\d+|[IVX]+|one|two|three)',
            r'stage\s*(\d+|[IVX]+|one|two|three)',
            r'step\s*(\d+|[IVX]+|one|two|three)',
            r'milestone\s*(\d+|[IVX]+)',
            r'deliverable\s*(\d+|[IVX]+)'
        ]
        
        for chunk in chunks:
            content = chunk.page_content
            
            # Détecter les listes de phases
            phase_sections = self._find_phase_sections(content)
            
            for section in phase_sections:
                # Extraire les phases de la section
                section_phases = self._extract_phases_from_section(section)
                phases.extend(section_phases)
                
                if section_phases:
                    sources.append(self._create_milestone_source(chunk, 'phase'))
        
        # 2. Extraction LLM pour chronologie complète
        phase_prompt = f"""
        {question_config.system_prompt}
        
        Extract project phases, stages, or milestones with:
        - Phase name/number
        - Start date (if specified)
        - End date or duration
        - Key deliverables
        - Dependencies between phases
        
        Present in chronological order if possible.
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=phase_prompt),
                HumanMessage(content=context[:10000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                llm_phases = self._parse_llm_phases(parsed['answer'])
                
                # Enrichir ou compléter les phases détectées
                phases = self._merge_phases(phases, llm_phases)
                
                # Sources
                if parsed.get('sources'):
                    sources.extend([
                        ExtractionSource(
                            document=s.get('document', 'document'),
                            context_snippet=s.get('context_snippet', ''),
                            confidence=0.8
                        ) for s in parsed['sources'][:3]
                    ])
        except Exception as e:
            logger.debug(f"Phase extraction failed: {e}")
        
        if phases:
            # Ordonner et structurer
            structured_phases = self._structure_project_phases(phases)
            
            return QuestionResult(
                question_id=question_config.id,
                answer=structured_phases,
                sources=sources[:5],
                status=ExtractionStatus.SUCCESS,
                confidence=0.85,
                metadata={
                    'phase_count': len(phases),
                    'has_dates': any(p.get('start_date') or p.get('end_date') for p in structured_phases),
                    'total_duration': self._calculate_project_duration(phases)
                }
            )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.NOT_FOUND,
            error="No project phases found"
        )
    
    async def _extract_contract_duration(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction de la durée du contrat"""
        durations = []
        sources = []
        
        # Patterns pour durées de contrat
        duration_patterns = [
            r'contract.*?(?:duration|period|term).*?(\d+)\s*(years?|months?|days?)',
            r'(?:duration|period|term).*?contract.*?(\d+)\s*(years?|months?|days?)',
            r'(\d+)\s*(years?|months?|days?).*?contract',
            r'contract.*?(?:for|of)\s*(\d+)\s*(years?|months?|days?)',
            r'(?:initial|total).*?(?:period|term).*?(\d+)\s*(years?|months?|days?)'
        ]
        
        # 1. Extraction par patterns
        for chunk in chunks:
            content = chunk.page_content
            
            for pattern in duration_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    value = match.group(1)
                    unit = match.group(2)
                    
                    # Contexte pour validation
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(content), match.end() + 100)
                    duration_context = content[context_start:context_end]
                    
                    durations.append({
                        'value': int(value),
                        'unit': unit.lower(),
                        'context': duration_context,
                        'source': chunk,
                        'confidence': 0.85
                    })
        
        # 2. Extraction LLM
        duration_prompt = f"""
        {question_config.system_prompt}
        
        Extract the contract duration including:
        - Total duration (e.g., "3 years", "36 months")
        - Start date if specified
        - End date if specified
        - Any renewal options or extensions
        - Initial period vs total possible duration
        """
        
        try:
            response = await self.date_llm.ainvoke([
                SystemMessage(content=duration_prompt),
                HumanMessage(content=context[:8000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                llm_duration = self._parse_duration_info(parsed['answer'])
                if llm_duration:
                    durations.append({
                        'value': llm_duration.get('value'),
                        'unit': llm_duration.get('unit'),
                        'context': str(parsed['answer']),
                        'source': None,
                        'confidence': 0.9,
                        'additional_info': llm_duration
                    })
                
                # Sources
                if parsed.get('sources'):
                    sources.extend([
                        ExtractionSource(
                            document=s.get('document', 'document'),
                            context_snippet=s.get('context_snippet', ''),
                            confidence=0.85
                        ) for s in parsed['sources'][:2]
                    ])
        except Exception as e:
            logger.debug(f"Duration extraction failed: {e}")
        
        # 3. Sélectionner la meilleure information de durée
        if durations:
            best_duration = self._select_best_duration(durations)
            
            if best_duration:
                # Créer les sources
                for dur in durations:
                    if dur.get('source'):
                        sources.append(self._create_milestone_source(
                            dur['source'],
                            'duration'
                        ))
                
                # Formater la réponse
                answer = {
                    "duration": f"{best_duration['value']} {best_duration['unit']}",
                    "confidence": best_duration['confidence']
                }
                
                # Ajouter infos supplémentaires si disponibles
                if best_duration.get('additional_info'):
                    answer.update(best_duration['additional_info'])
                
                return QuestionResult(
                    question_id=question_config.id,
                    answer=answer,
                    sources=sources[:3],
                    status=ExtractionStatus.SUCCESS,
                    confidence=best_duration['confidence'],
                    metadata={
                        'duration_in_days': self._convert_to_days(
                            best_duration['value'],
                            best_duration['unit']
                        )
                    }
                )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.NOT_FOUND,
            error="Contract duration not found"
        )
    
    async def _generic_timeline_extraction(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction générique pour questions temporelles"""
        # Enrichir le prompt avec des instructions temporelles
        enhanced_prompt = f"""
        {question_config.system_prompt}
        
        TEMPORAL EXTRACTION GUIDELINES:
        1. Convert all dates to YYYY-MM-DD format
        2. Include times in HH:MM format when specified
        3. Indicate if dates are estimates or firm deadlines
        4. Note any conditions or dependencies
        5. Extract both explicit dates and relative timeframes
        """
        
        question_config.system_prompt = enhanced_prompt
        result = await super()._extract_single_question(question_config, context, chunks)
        
        # Post-traitement pour normaliser les dates
        if result.answer:
            result.answer = self._normalize_temporal_answer(result.answer)
        
        return result
    
    # ============================================================================
    # MÉTHODES UTILITAIRES - EXTRACTION
    # ============================================================================
    
    def _identify_date_rich_chunks(self, chunks: List[Document]) -> List[Document]:
        """Identifie les chunks contenant beaucoup de dates"""
        scored_chunks = []
        
        for chunk in chunks:
            content = chunk.page_content
            date_count = 0
            
            # Compter les dates
            for pattern in self.patterns.values():
                if isinstance(pattern, str):
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    date_count += len(matches)
            
            # Compter les mots-clés temporels
            temporal_keywords = sum(
                keyword in content.lower()
                for keywords in self.temporal_catalog['deadline_keywords'].values()
                for keyword in keywords
            )
            
            score = date_count + (temporal_keywords * 2)
            if score > 2:  # Seuil minimal
                scored_chunks.append((score, chunk))
        
        # Retourner les chunks triés par score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:10]]  # Top 10
    
    def _extract_dates_from_text(self, text: str) -> List[Tuple[str, Any]]:
        """Extrait toutes les dates d'un texte avec leur match object"""
        dates = []
        
        # Patterns de date avec leur fonction de parsing
        date_patterns = [
            (self.patterns['iso_date'], self._parse_iso_date),
            (self.patterns['euro_date'], self._parse_euro_date),
            (self.patterns['us_date'], self._parse_us_date),
            (self.patterns['text_date'], self._parse_text_date),
            (self.patterns['text_date_us'], self._parse_text_date_us)
        ]
        
        for pattern, parser in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    date = parser(match)
                    if date:
                        dates.append((date, match))
                except Exception as e:
                    logger.debug(f"Date parsing failed: {e}")
        
        return dates
    
    def _parse_iso_date(self, match) -> Optional[str]:
        """Parse une date ISO"""
        year, month, day = match.groups()
        try:
            date = datetime(int(year), int(month), int(day))
            return date.strftime('%Y-%m-%d')
        except:
            return None
    
    def _parse_euro_date(self, match) -> Optional[str]:
        """Parse une date européenne DD/MM/YYYY"""
        day, month, year = match.groups()
        try:
            date = datetime(int(year), int(month), int(day))
            return date.strftime('%Y-%m-%d')
        except:
            return None
    
    def _parse_us_date(self, match) -> Optional[str]:
        """Parse une date US MM/DD/YYYY"""
        month, day, year = match.groups()
        try:
            date = datetime(int(year), int(month), int(day))
            return date.strftime('%Y-%m-%d')
        except:
            return None
    
    def _parse_text_date(self, match) -> Optional[str]:
        """Parse une date textuelle DD Month YYYY"""
        day, month_name, year = match.groups()
        return self._convert_text_date(day, month_name, year)
    
    def _parse_text_date_us(self, match) -> Optional[str]:
        """Parse une date textuelle US Month DD, YYYY"""
        month_name, day, year = match.groups()
        return self._convert_text_date(day, month_name, year)
    
    def _convert_text_date(self, day: str, month_name: str, year: str) -> Optional[str]:
        """Convertit une date textuelle en ISO"""
        month_map = {
            'january': 1, 'jan': 1, 'février': 2, 'february': 2, 'feb': 2,
            'march': 3, 'mar': 3, 'mars': 3, 'april': 4, 'apr': 4, 'avril': 4,
            'may': 5, 'mai': 5, 'june': 6, 'jun': 6, 'juin': 6,
            'july': 7, 'jul': 7, 'juillet': 7, 'august': 8, 'aug': 8, 'août': 8,
            'september': 9, 'sep': 9, 'septembre': 9, 'october': 10, 'oct': 10, 'octobre': 10,
            'november': 11, 'nov': 11, 'novembre': 11, 'december': 12, 'dec': 12, 'décembre': 12
        }
        
        try:
            month = month_map.get(month_name.lower())
            if month:
                date = datetime(int(year), month, int(day))
                return date.strftime('%Y-%m-%d')
        except:
            pass
        
        return None
    
    def _extract_time_near_date(self, text: str, date_match) -> Optional[str]:
        """Extrait une heure proche d'une date"""
        # Chercher avant et après la date
        search_start = max(0, date_match.start() - 50)
        search_end = min(len(text), date_match.end() + 50)
        search_text = text[search_start:search_end]
        
        # Patterns d'heure
        time_patterns = [
            (self.patterns['time_24h'], self._parse_24h_time),
            (self.patterns['time_12h'], self._parse_12h_time)
        ]
        
        for pattern, parser in time_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return parser(match)
        
        return None
    
    def _parse_24h_time(self, match) -> str:
        """Parse une heure au format 24h"""
        hours, minutes = match.groups()[:2]
        return f"{int(hours):02d}:{minutes}"
    
    def _parse_12h_time(self, match) -> str:
        """Parse une heure au format 12h"""
        hours, minutes, ampm = match.groups()
        hours = int(hours)
        
        if ampm.lower() == 'pm' and hours != 12:
            hours += 12
        elif ampm.lower() == 'am' and hours == 12:
            hours = 0
        
        return f"{hours:02d}:{minutes}"
    
    # ============================================================================
    # MÉTHODES UTILITAIRES - TRAITEMENT
    # ============================================================================
    
    def _create_milestone_source(self, chunk: Document, keyword: str) -> ExtractionSource:
        """Crée une source pour un jalon"""
        content = chunk.page_content
        keyword_idx = content.lower().find(keyword.lower())
        
        if keyword_idx >= 0:
            snippet_start = max(0, keyword_idx - 30)
            snippet_end = min(len(content), keyword_idx + 50)
            snippet = content[snippet_start:snippet_end].strip()
        else:
            snippet = keyword
        
        return ExtractionSource(
            document=chunk.metadata.get('source', 'document'),
            context_snippet=snippet[:80],
            confidence=0.85
        )
    
    def _generate_deadline_label(self, deadline_type: str, context: str) -> str:
        """Génère un label descriptif pour une deadline"""
        context_lower = context.lower()
        
        # Détection du round/phase
        round_match = re.search(r'(first|second|third|final|1st|2nd|3rd)\s*(round|stage|phase)?', context_lower)
        if round_match:
            round_label = round_match.group(1).capitalize()
            return f"{round_label} round submission deadline"
        
        # Labels par type
        type_labels = {
            'submission': "Tender submission deadline",
            'response': "Response deadline",
            'clarification': "Clarification questions deadline",
            'closing': "Closing deadline"
        }
        
        return type_labels.get(deadline_type, "Deadline")
    
    def _extract_deadline_context(self, window: str, date_match) -> str:
        """Extrait le contexte descriptif d'une deadline"""
        # Extraire la phrase contenant la date
        sentence_start = window.rfind('.', 0, date_match.start())
        if sentence_start == -1:
            sentence_start = 0
        else:
            sentence_start += 1
        
        sentence_end = window.find('.', date_match.end())
        if sentence_end == -1:
            sentence_end = len(window)
        
        context = window[sentence_start:sentence_end].strip()
        
        # Limiter la longueur
        if len(context) > 150:
            context = context[:147] + "..."
        
        return context
    
    def _is_duplicate_milestone(
        self,
        milestone: TimelineMilestone,
        existing: List[TimelineMilestone]
    ) -> bool:
        """Vérifie si un jalon est un doublon"""
        for existing_milestone in existing:
            # Même date et type similaire
            if milestone.date == existing_milestone.date:
                # Vérifier la similarité des labels
                if self._are_labels_similar(milestone.label, existing_milestone.label):
                    return True
                
                # Même type de milestone
                if milestone.milestone_type == existing_milestone.milestone_type:
                    # Vérifier l'heure si présente
                    if milestone.time and existing_milestone.time:
                        if milestone.time == existing_milestone.time:
                            return True
                    else:
                        return True
        
        return False
    
    def _are_labels_similar(self, label1: str, label2: str) -> bool:
        """Vérifie si deux labels sont similaires"""
        label1_lower = label1.lower()
        label2_lower = label2.lower()
        
        # Exact match
        if label1_lower == label2_lower:
            return True
        
        # Contenu l'un dans l'autre
        if label1_lower in label2_lower or label2_lower in label1_lower:
            return True
        
        # Mots communs significatifs
        words1 = set(label1_lower.split())
        words2 = set(label2_lower.split())
        
        # Ignorer les mots courants
        stop_words = {'the', 'a', 'an', 'for', 'of', 'to', 'in', 'on', 'at', 'by'}
        words1 -= stop_words
        words2 -= stop_words
        
        common_words = words1 & words2
        
        # Si plus de 50% de mots en commun
        if len(common_words) >= min(len(words1), len(words2)) * 0.5:
            return True
        
        return False
    
    def _format_milestones_for_validation(self, milestones: List[TimelineMilestone]) -> str:
        """Formate les jalons pour validation LLM"""
        formatted = []
        for m in milestones:
            formatted.append(f"- {m.label}: {m.date} {m.time or ''}")
        return '\n'.join(formatted)
    
    def _parse_llm_milestones(self, answer: Any) -> List[TimelineMilestone]:
        """Parse les jalons extraits par le LLM"""
        milestones = []
        
        if isinstance(answer, list):
            for item in answer:
                milestone = self._parse_single_milestone(item)
                if milestone:
                    milestones.append(milestone)
        elif isinstance(answer, dict):
            milestone = self._parse_single_milestone(answer)
            if milestone:
                milestones.append(milestone)
        
        return milestones
    
    def _parse_single_milestone(self, item: Any) -> Optional[TimelineMilestone]:
        """Parse un jalon unique"""
        if isinstance(item, dict):
            date = item.get('date')
            if date:
                return TimelineMilestone(
                    label=item.get('label', 'Milestone'),
                    date=self._normalize_date(date),
                    milestone_type=self._determine_milestone_type(item),
                    is_mandatory=item.get('is_mandatory', False),
                    time=item.get('time'),
                    location=item.get('location'),
                    description=item.get('description'),
                    confidence=0.8
                )
        elif isinstance(item, str):
            # Essayer d'extraire une date de la chaîne
            dates = self._extract_dates_from_text(item)
            if dates:
                date, _ = dates[0]
                return TimelineMilestone(
                    label=item[:50],
                    date=date,
                    milestone_type=MilestoneType.OTHER,
                    confidence=0.7
                )
        
        return None
    
    def _determine_milestone_type(self, milestone_dict: dict) -> MilestoneType:
        """Détermine le type de jalon"""
        label = milestone_dict.get('label', '').lower()
        desc = milestone_dict.get('description', '').lower()
        combined = f"{label} {desc}"
        
        if any(kw in combined for kw in ['submission', 'tender', 'proposal', 'bid']):
            return MilestoneType.SUBMISSION_DEADLINE
        elif any(kw in combined for kw in ['clarification', 'question', 'query']):
            return MilestoneType.CLARIFICATION_DEADLINE
        elif any(kw in combined for kw in ['meeting', 'conference', 'briefing']):
            return MilestoneType.MEETING
        elif any(kw in combined for kw in ['site', 'visit', 'inspection']):
            return MilestoneType.SITE_VISIT
        elif any(kw in combined for kw in ['contract', 'award', 'start']):
            return MilestoneType.CONTRACT_MILESTONE
        elif any(kw in combined for kw in ['phase', 'stage', 'milestone']):
            return MilestoneType.PROJECT_PHASE
        else:
            return MilestoneType.OTHER
    
    def _structure_milestones(self, milestones: List[TimelineMilestone]) -> List[Dict[str, Any]]:
        """Structure les jalons pour la sortie finale"""
        structured = []
        
        for milestone in milestones:
            structured_milestone = {
                "label": milestone.label,
                "date": milestone.date,
                "sources": [
                    {
                        "document": src.document,
                        "context_snippet": src.context_snippet
                    }
                    for src in milestone.sources[:2]
                ]
            }
            
            # Ajouter les champs optionnels s'ils existent
            if milestone.time:
                structured_milestone["time"] = milestone.time
            if milestone.location:
                structured_milestone["location"] = milestone.location
            if milestone.description:
                structured_milestone["description"] = milestone.description
            if milestone.is_mandatory:
                structured_milestone["is_mandatory"] = True
            
            structured.append(structured_milestone)
        
        return structured
    
    def _calculate_timeline_confidence(self, milestones: List[TimelineMilestone]) -> float:
        """Calcule la confiance globale de l'extraction timeline"""
        if not milestones:
            return 0.0
        
        # Facteurs de confiance
        has_submission = any(m.milestone_type == MilestoneType.SUBMISSION_DEADLINE for m in milestones)
        milestone_count = len(milestones)
        avg_confidence = sum(m.confidence for m in milestones) / len(milestones)
        has_sources = sum(1 for m in milestones if m.sources) / len(milestones)
        
        # Calcul pondéré
        confidence = avg_confidence * 0.4
        confidence += 0.2 if has_submission else 0.0
        confidence += min(0.2, milestone_count * 0.02)
        confidence += has_sources * 0.2
        
        return min(0.95, confidence)
    
    def _get_date_range(self, milestones: List[TimelineMilestone]) -> Dict[str, str]:
        """Obtient la plage de dates des jalons"""
        if not milestones:
            return {}
        
        dates = [m.date for m in milestones if m.date]
        if not dates:
            return {}
        
        dates.sort()
        return {
            "start": dates[0],
            "end": dates[-1]
        }
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalise une date au format ISO"""
        # Si déjà au format ISO
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        
        # Essayer de parser et convertir
        dates = self._extract_dates_from_text(date_str)
        if dates:
            return dates[0][0]
        
        # Retourner tel quel si échec
        return date_str
    
    def _deduplicate_sources(self, sources: List[ExtractionSource]) -> List[ExtractionSource]:
        """Déduplique les sources en gardant les meilleures"""
        unique_sources = {}
        
        for source in sources:
            key = f"{source.document}:{source.context_snippet[:30]}"
            if key not in unique_sources or source.confidence > unique_sources[key].confidence:
                unique_sources[key] = source
        
        # Trier par confiance
        return sorted(unique_sources.values(), key=lambda s: s.confidence, reverse=True)
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse une réponse JSON du LLM"""
        try:
            # Nettoyer la réponse
            response = response.strip()
            
            # Extraire le JSON s'il est dans un bloc de code
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end].strip()
            
            return json.loads(response)
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
            return None
    
    def _convert_to_days(self, value: int, unit: str) -> int:
        """Convertit une durée en jours"""
        unit_lower = unit.lower().rstrip('s')
        
        conversions = {
            'day': 1,
            'week': 7,
            'month': 30,
            'year': 365
        }
        
        return value * conversions.get(unit_lower, 1)
    
    def _normalize_temporal_answer(self, answer: Any) -> Any:
        """Normalise les réponses temporelles"""
        if isinstance(answer, str):
            # Essayer d'extraire et normaliser les dates
            dates = self._extract_dates_from_text(answer)
            if dates and len(dates) == 1:
                return dates[0][0]
        elif isinstance(answer, list):
            return [self._normalize_temporal_answer(item) for item in answer]
        elif isinstance(answer, dict):
            # Normaliser les dates dans le dictionnaire
            for key in ['date', 'start_date', 'end_date']:
                if key in answer:
                    answer[key] = self._normalize_date(answer[key])
        
        return answer


# ============================================================================
# MÉTHODES ADDITIONNELLES POUR EXTRACTIONS SPÉCIALISÉES
# ============================================================================

    async def _extract_deadlines_from_chunks(
        self,
        chunks: List[Document]
    ) -> List[TimelineMilestone]:
        """Extrait tous les types de deadlines des chunks"""
        deadlines = []
        
        for chunk in chunks:
            content = chunk.page_content
            
            # Rechercher toutes les deadlines
            for deadline_type, keywords in self.temporal_catalog['deadline_keywords'].items():
                for keyword in keywords:
                    if keyword in content.lower():
                        # Extraire les dates proches
                        keyword_pos = content.lower().find(keyword)
                        window = content[max(0, keyword_pos-200):keyword_pos+200]
                        
                        dates = self._extract_dates_from_text(window)
                        for date, _ in dates:
                            deadline = TimelineMilestone(
                                label=self._generate_deadline_label(deadline_type, window),
                                date=date,
                                milestone_type=MilestoneType.SUBMISSION_DEADLINE,
                                is_mandatory=True,
                                sources=[self._create_milestone_source(chunk, keyword)],
                                confidence=0.85
                            )
                            deadlines.append(deadline)
        
        return deadlines
    
    async def _extract_meetings_from_chunks(
        self,
        chunks: List[Document]
    ) -> List[TimelineMilestone]:
        """Extrait les réunions et événements des chunks"""
        meetings = []
        
        for chunk in chunks:
            content = chunk.page_content
            
            for keyword in self.temporal_catalog['meeting_keywords']:
                if keyword in content.lower():
                    # Extraire détails de la réunion
                    event_info = self._extract_event_details(content, keyword)
                    if event_info and event_info.get('date'):
                        meeting = TimelineMilestone(
                            label=event_info['label'],
                            date=event_info['date'],
                            milestone_type=MilestoneType.MEETING,
                            is_mandatory=event_info.get('is_mandatory', False),
                            time=event_info.get('time'),
                            location=event_info.get('location'),
                            sources=[self._create_milestone_source(chunk, keyword)],
                            confidence=0.8
                        )
                        meetings.append(meeting)
        
        return meetings
    
    async def _extract_phases_from_chunks(
        self,
        chunks: List[Document]
    ) -> List[TimelineMilestone]:
        """Extrait les phases de projet des chunks"""
        phases = []
        
        for chunk in chunks:
            content = chunk.page_content
            
            # Détecter les sections de phases
            phase_sections = self._find_phase_sections(content)
            
            for section in phase_sections:
                section_phases = self._extract_phases_from_section(section)
                for phase_info in section_phases:
                    if phase_info.get('date'):
                        phase = TimelineMilestone(
                            label=phase_info['label'],
                            date=phase_info['date'],
                            milestone_type=MilestoneType.PROJECT_PHASE,
                            description=phase_info.get('description'),
                            sources=[self._create_milestone_source(chunk, 'phase')],
                            confidence=0.75
                        )
                        phases.append(phase)
        
        return phases
    
    def _extract_event_details(self, context: str, keyword: str) -> Optional[Dict[str, Any]]:
        """Extrait les détails complets d'un événement"""
        keyword_pos = context.lower().find(keyword.lower())
        if keyword_pos < 0:
            return None
        
        # Fenêtre étendue
        start = max(0, keyword_pos - 200)
        end = min(len(context), keyword_pos + 300)
        event_context = context[start:end]
        
        # Extraire la date
        dates = self._extract_dates_from_text(event_context)
        if not dates:
            return None
        
        date, date_match = dates[0]
        
        # Extraire l'heure
        time = self._extract_time_near_date(event_context, date_match)
        
        # Vérifier si obligatoire
        is_mandatory = any(
            indicator in event_context.lower()
            for indicator in self.temporal_catalog['mandatory_indicators']
        )
        
        # Extraire le lieu (patterns simples)
        location = None
        location_patterns = [
            r'(?:at|venue:|location:)\s*([^,\.\n]+)',
            r'(?:held at|taking place at)\s*([^,\.\n]+)'
        ]
        for pattern in location_patterns:
            match = re.search(pattern, event_context, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                break
        
        # Générer le label
        label = self._generate_event_label(keyword, event_context)
        
        return {
            'label': label,
            'date': date,
            'time': time,
            'location': location,
            'is_mandatory': is_mandatory,
            'description': self._extract_event_description(event_context)
        }
    
    def _generate_event_label(self, keyword: str, context: str) -> str:
        """Génère un label descriptif pour un événement"""
        context_lower = context.lower()
        
        # Cas spécifiques
        if 'pre-bid' in context_lower or 'prebid' in context_lower:
            return "Pre-bid meeting"
        elif 'site visit' in context_lower:
            return "Mandatory site visit"
        elif 'clarification meeting' in context_lower:
            return "Clarification meeting"
        elif 'kick-off' in context_lower or 'kickoff' in context_lower:
            return "Project kick-off meeting"
        
        # Label générique basé sur le keyword
        return f"{keyword.capitalize()} meeting"
    
    def _extract_event_description(self, context: str) -> str:
        """Extrait une description de l'événement"""
        # Trouver la phrase principale
        sentences = context.split('.')
        
        for sentence in sentences:
            # Chercher la phrase la plus pertinente
            if any(kw in sentence.lower() for kw in ['meeting', 'conference', 'visit', 'session']):
                description = sentence.strip()
                if len(description) > 20 and len(description) < 200:
                    return description
        
        return ""
    
    def _find_phase_sections(self, content: str) -> List[str]:
        """Trouve les sections décrivant des phases de projet"""
        sections = []
        lines = content.split('\n')
        
        phase_start_patterns = [
            r'project\s+phases',
            r'implementation\s+phases',
            r'execution\s+stages',
            r'milestones',
            r'timeline',
            r'work\s+plan'
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Chercher le début d'une section de phases
            for pattern in phase_start_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Extraire la section
                    section_lines = [line]
                    j = i + 1
                    
                    # Continuer tant que ça ressemble à une liste ou description
                    while j < len(lines) and j < i + 20:  # Max 20 lignes
                        next_line = lines[j]
                        
                        # Critères pour continuer la section
                        if (re.match(r'^\s*[-•\d\.]', next_line) or  # Liste
                            re.match(r'^\s*phase\s+\d+', next_line, re.IGNORECASE) or
                            re.match(r'^\s*stage\s+\d+', next_line, re.IGNORECASE) or
                            len(next_line.strip()) > 10):  # Ligne avec contenu
                            section_lines.append(next_line)
                        elif not next_line.strip():  # Ligne vide
                            section_lines.append(next_line)
                        else:
                            break
                        
                        j += 1
                    
                    sections.append('\n'.join(section_lines))
                    i = j
                    break
            else:
                i += 1
        
        return sections
    
    def _extract_phases_from_section(self, section: str) -> List[Dict[str, Any]]:
        """Extrait les informations de phases d'une section"""
        phases = []
        lines = section.split('\n')
        
        for line in lines:
            # Patterns pour phases
            phase_patterns = [
                r'phase\s+(\d+)[:\s]+([^,\n]+)',
                r'stage\s+(\d+)[:\s]+([^,\n]+)',
                r'milestone\s+(\d+)[:\s]+([^,\n]+)',
                r'(\d+)\.\s*([^:,\n]+)'
            ]
            
            for pattern in phase_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    phase_num = match.group(1)
                    phase_desc = match.group(2).strip()
                    
                    # Chercher une date dans la ligne
                    dates = self._extract_dates_from_text(line)
                    
                    phase_info = {
                        'label': f"Phase {phase_num}: {phase_desc[:50]}",
                        'description': phase_desc
                    }
                    
                    if dates:
                        phase_info['date'] = dates[0][0]
                    
                    # Chercher une durée
                    duration_match = re.search(r'(\d+)\s*(weeks?|months?|days?)', line, re.IGNORECASE)
                    if duration_match:
                        phase_info['duration'] = f"{duration_match.group(1)} {duration_match.group(2)}"
                    
                    phases.append(phase_info)
                    break
        
        return phases
    
    def _parse_llm_deadlines(self, answer: Any) -> List[TimelineMilestone]:
        """Parse spécifiquement les deadlines du LLM"""
        deadlines = []
        
        items = answer if isinstance(answer, list) else [answer]
        
        for item in items:
            if isinstance(item, dict):
                date = item.get('date')
                if date:
                    deadline = TimelineMilestone(
                        label=item.get('label', 'Submission deadline'),
                        date=self._normalize_date(date),
                        milestone_type=MilestoneType.SUBMISSION_DEADLINE,
                        is_mandatory=True,
                        time=item.get('time'),
                        description=item.get('description'),
                        confidence=0.85
                    )
                    deadlines.append(deadline)
            elif isinstance(item, str):
                # Parser une deadline textuelle
                parsed = self._parse_deadline_text(item)
                if parsed:
                    deadlines.append(parsed)
        
        return deadlines
    
    def _parse_deadline_text(self, text: str) -> Optional[TimelineMilestone]:
        """Parse une deadline depuis un texte"""
        dates = self._extract_dates_from_text(text)
        if not dates:
            return None
        
        date, _ = dates[0]
        
        # Déterminer le type de deadline
        text_lower = text.lower()
        if 'clarification' in text_lower or 'question' in text_lower:
            milestone_type = MilestoneType.CLARIFICATION_DEADLINE
            label = "Clarification questions deadline"
        elif 'final' in text_lower:
            label = "Final submission deadline"
            milestone_type = MilestoneType.SUBMISSION_DEADLINE
        else:
            label = text[:50] + "..." if len(text) > 50 else text
            milestone_type = MilestoneType.SUBMISSION_DEADLINE
        
        return TimelineMilestone(
            label=label,
            date=date,
            milestone_type=milestone_type,
            is_mandatory=True,
            confidence=0.75
        )
    
    def _classify_submission_rounds(
        self,
        deadlines: List[TimelineMilestone]
    ) -> List[TimelineMilestone]:
        """Classifie les deadlines par rounds"""
        if len(deadlines) <= 1:
            return deadlines
        
        # Trier par date
        deadlines.sort(key=lambda d: d.date)
        
        # Si plusieurs deadlines, essayer d'identifier les rounds
        round_keywords = ['first', 'second', 'third', 'final', '1st', '2nd', '3rd']
        
        for i, deadline in enumerate(deadlines):
            label_lower = deadline.label.lower()
            
            # Si pas déjà de round identifié
            if not any(kw in label_lower for kw in round_keywords):
                if i == 0:
                    deadline.label = f"First-round {deadline.label}"
                elif i == len(deadlines) - 1:
                    deadline.label = f"Final {deadline.label}"
                else:
                    deadline.label = f"Round {i+1} {deadline.label}"
        
        return deadlines
    
    def _select_best_clarification_date(
        self,
        date_candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Sélectionne la meilleure date de clarification parmi les candidates"""
        if not date_candidates:
            return None
        
        # Trier par confiance
        date_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Si une seule, la retourner
        if len(date_candidates) == 1:
            return date_candidates[0]
        
        # Vérifier la cohérence des dates
        dates = [d['date'] for d in date_candidates]
        unique_dates = set(dates)
        
        if len(unique_dates) == 1:
            # Toutes les sources s'accordent
            best = date_candidates[0]
            best['confidence'] = min(0.95, best['confidence'] + 0.1)
            return best
        
        # Prendre la plus fréquente
        date_counts = {}
        for d in date_candidates:
            date_counts[d['date']] = date_counts.get(d['date'], 0) + 1
        
        most_common_date = max(date_counts, key=date_counts.get)
        
        for candidate in date_candidates:
            if candidate['date'] == most_common_date:
                return candidate
        
        return date_candidates[0]
    
    def _parse_llm_events(self, answer: Any) -> List[TimelineMilestone]:
        """Parse les événements extraits par le LLM"""
        events = []
        
        items = answer if isinstance(answer, list) else [answer]
        
        for item in items:
            if isinstance(item, dict):
                date = item.get('date')
                if date:
                    event = TimelineMilestone(
                        label=item.get('label', item.get('event_name', 'Meeting')),
                        date=self._normalize_date(date),
                        milestone_type=self._determine_event_type(item),
                        is_mandatory=item.get('is_mandatory', item.get('mandatory', False)),
                        time=item.get('time'),
                        location=item.get('location', item.get('venue')),
                        description=item.get('description'),
                        confidence=0.8
                    )
                    events.append(event)
        
        return events
    
    def _determine_event_type(self, event_dict: dict) -> MilestoneType:
        """Détermine le type d'événement"""
        label = event_dict.get('label', '').lower()
        event_type = event_dict.get('type', '').lower()
        
        if 'site' in label or 'site' in event_type:
            return MilestoneType.SITE_VISIT
        else:
            return MilestoneType.MEETING
    
    def _is_duplicate_event(
        self,
        event: TimelineMilestone,
        existing: List[TimelineMilestone]
    ) -> bool:
        """Vérifie si un événement est un doublon"""
        for existing_event in existing:
            # Même date
            if event.date == existing_event.date:
                # Même heure ou pas d'heure
                if event.time == existing_event.time or (not event.time and not existing_event.time):
                    # Labels similaires
                    if self._are_labels_similar(event.label, existing_event.label):
                        return True
                    
                    # Même lieu
                    if event.location and existing_event.location:
                        if event.location.lower() in existing_event.location.lower() or \
                           existing_event.location.lower() in event.location.lower():
                            return True
        
        return False
    
    def _parse_llm_phases(self, answer: Any) -> List[Dict[str, Any]]:
        """Parse les phases de projet du LLM"""
        phases = []
        
        items = answer if isinstance(answer, list) else [answer]
        
        for item in items:
            if isinstance(item, dict):
                phase = {
                    'label': item.get('label', item.get('phase_name', '')),
                    'description': item.get('description', '')
                }
                
                # Dates
                if item.get('start_date'):
                    phase['date'] = self._normalize_date(item['start_date'])
                elif item.get('date'):
                    phase['date'] = self._normalize_date(item['date'])
                
                if item.get('end_date'):
                    phase['end_date'] = self._normalize_date(item['end_date'])
                
                # Durée
                if item.get('duration'):
                    phase['duration'] = item['duration']
                
                phases.append(phase)
        
        return phases
    
    def _merge_phases(
        self,
        detected_phases: List[Dict[str, Any]],
        llm_phases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fusionne les phases détectées et celles du LLM"""
        merged = detected_phases.copy()
        
        for llm_phase in llm_phases:
            # Vérifier si cette phase existe déjà
            is_duplicate = False
            
            for existing in merged:
                if self._are_phases_similar(existing, llm_phase):
                    # Enrichir l'existante
                    if not existing.get('date') and llm_phase.get('date'):
                        existing['date'] = llm_phase['date']
                    if not existing.get('duration') and llm_phase.get('duration'):
                        existing['duration'] = llm_phase['duration']
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(llm_phase)
        
        return merged
    
    def _are_phases_similar(self, phase1: Dict, phase2: Dict) -> bool:
        """Vérifie si deux phases sont similaires"""
        label1 = phase1.get('label', '').lower()
        label2 = phase2.get('label', '').lower()
        
        # Extraire les numéros de phase
        num1 = re.search(r'\d+', label1)
        num2 = re.search(r'\d+', label2)
        
        if num1 and num2 and num1.group() == num2.group():
            return True
        
        # Vérifier la similarité des labels
        return self._are_labels_similar(label1, label2)
    
    def _structure_project_phases(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Structure et ordonne les phases de projet"""
        structured = []
        
        # Trier par numéro de phase ou date
        def phase_sort_key(phase):
            # Essayer d'extraire un numéro
            num_match = re.search(r'\d+', phase.get('label', ''))
            if num_match:
                return (0, int(num_match.group()))
            # Sinon utiliser la date
            elif phase.get('date'):
                return (1, phase['date'])
            else:
                return (2, phase.get('label', ''))
        
        phases.sort(key=phase_sort_key)
        
        # Structurer chaque phase
        for i, phase in enumerate(phases):
            structured_phase = {
                'number': i + 1,
                'label': phase.get('label', f'Phase {i+1}'),
                'description': phase.get('description', '')
            }
            
            # Ajouter les dates si disponibles
            if phase.get('date'):
                structured_phase['start_date'] = phase['date']
            if phase.get('end_date'):
                structured_phase['end_date'] = phase['end_date']
            if phase.get('duration'):
                structured_phase['duration'] = phase['duration']
            
            structured.append(structured_phase)
        
        return structured
    
    def _calculate_project_duration(self, phases: List[Dict[str, Any]]) -> Optional[str]:
        """Calcule la durée totale du projet basée sur les phases"""
        # Chercher les dates extrêmes
        all_dates = []
        
        for phase in phases:
            if phase.get('date'):
                all_dates.append(phase['date'])
            if phase.get('start_date'):
                all_dates.append(phase['start_date'])
            if phase.get('end_date'):
                all_dates.append(phase['end_date'])
        
        if len(all_dates) >= 2:
            all_dates.sort()
            try:
                start = datetime.strptime(all_dates[0], '%Y-%m-%d')
                end = datetime.strptime(all_dates[-1], '%Y-%m-%d')
                duration = end - start
                
                # Formater la durée
                if duration.days > 365:
                    years = duration.days // 365
                    months = (duration.days % 365) // 30
                    return f"{years} year{'s' if years > 1 else ''}" + (f" {months} months" if months else "")
                elif duration.days > 30:
                    months = duration.days // 30
                    return f"{months} month{'s' if months > 1 else ''}"
                else:
                    return f"{duration.days} days"
            except:
                pass
        
        return None
    
    def _extract_date_from_answer(self, answer: Any) -> Optional[str]:
        """Extrait une date d'une réponse LLM"""
        if isinstance(answer, str):
            dates = self._extract_dates_from_text(answer)
            if dates:
                return dates[0][0]
        elif isinstance(answer, dict):
            if answer.get('date'):
                return self._normalize_date(answer['date'])
        
        return None
    
    def _parse_duration_info(self, answer: Any) -> Optional[Dict[str, Any]]:
        """Parse les informations de durée"""
        duration_info = {}
        
        if isinstance(answer, dict):
            # Durée directe
            if answer.get('duration'):
                duration_match = re.search(r'(\d+)\s*(years?|months?|days?)', str(answer['duration']), re.IGNORECASE)
                if duration_match:
                    duration_info['value'] = int(duration_match.group(1))
                    duration_info['unit'] = duration_match.group(2).lower()
            
            # Dates de début/fin
            if answer.get('start_date'):
                duration_info['start_date'] = self._normalize_date(answer['start_date'])
            if answer.get('end_date'):
                duration_info['end_date'] = self._normalize_date(answer['end_date'])
            
            # Options de renouvellement
            if answer.get('renewal_options'):
                duration_info['renewal_options'] = answer['renewal_options']
            
            return duration_info if duration_info else None
        
        elif isinstance(answer, str):
            # Extraire la durée du texte
            duration_match = re.search(r'(\d+)\s*(years?|months?|days?)', answer, re.IGNORECASE)
            if duration_match:
                return {
                    'value': int(duration_match.group(1)),
                    'unit': duration_match.group(2).lower()
                }
        
        return None
    
    def _select_best_duration(self, durations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sélectionne la meilleure information de durée"""
        if not durations:
            return None
        
        # Prioriser par complétude et confiance
        scored_durations = []
        
        for duration in durations:
            score = duration.get('confidence', 0.5)
            
            # Bonus pour informations complètes
            if duration.get('value') and duration.get('unit'):
                score += 0.2
            if duration.get('additional_info'):
                score += 0.1
            if duration.get('start_date') or duration.get('end_date'):
                score += 0.15
            
            scored_durations.append((score, duration))
        
        # Retourner la meilleure
        scored_durations.sort(key=lambda x: x[0], reverse=True)
        return scored_durations[0][1]