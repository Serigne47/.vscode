# agents/legal_agent.py
"""
Agent juridique optimisé pour extraction maximale de clauses légales
Spécialisé dans l'analyse juridique et contractuelle des appels d'offres
"""
import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from .base_agent import YAMLBaseAgent, QuestionResult, ExtractionStatus, ExtractionSource

logger = logging.getLogger(__name__)

@dataclass
class LegalClause:
    """Représente une clause juridique extraite"""
    clause_type: str
    content: str
    article_reference: Optional[str] = None
    section: Optional[str] = None
    importance: str = "standard"  # critical, high, standard, low
    implications: List[str] = field(default_factory=list)
    related_clauses: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "type": self.clause_type,
            "content": self.content,
            "article": self.article_reference,
            "section": self.section,
            "importance": self.importance,
            "implications": self.implications,
            "related": self.related_clauses
        }

class LegalExtractionAgent(YAMLBaseAgent):
    """
    Agent d'extraction juridique avec capacités avancées pour clauses légales
    
    Spécialisations:
    - Détection multi-niveaux de clauses juridiques
    - Analyse contextuelle des implications légales
    - Cross-référencement des clauses interdépendantes
    - Extraction de hiérarchie contractuelle
    - Validation de cohérence juridique
    - Détection de risques et obligations cachées
    """
    
    def __init__(
        self,
        config_path: str = "config/prompts/legal_questions.yaml",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        use_advanced_extraction: bool = True,
        enable_risk_analysis: bool = True
    ):
        """
        Initialise l'agent juridique avec capacités d'analyse légale avancées
        
        Args:
            config_path: Chemin vers le fichier YAML des questions juridiques
            model: Modèle LLM (gpt-4o-mini recommandé pour précision juridique)
            temperature: Basse température pour exactitude juridique
            enable_cache: Cache des résultats
            enable_parallel: Extraction parallèle
            use_advanced_extraction: Stratégies avancées d'analyse juridique
            enable_risk_analysis: Analyse automatique des risques juridiques
        """
        super().__init__(
            config_path=config_path,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
            max_parallel=3  # Limité pour analyse approfondie
        )
        
        self.use_advanced = use_advanced_extraction
        self.enable_risk_analysis = enable_risk_analysis
        
        # LLMs spécialisés pour différents aspects juridiques
        if self.use_advanced:
            # LLM pour analyse juridique approfondie
            self.legal_analysis_llm = ChatOpenAI(
                model=model,
                temperature=0.0,  # Précision maximale
                max_tokens=4000
            )
            
            # LLM pour détection de risques
            self.risk_detection_llm = ChatOpenAI(
                model=model,
                temperature=0.2,  # Un peu de créativité pour identifier les risques cachés
                max_tokens=3000
            )
            
            # LLM pour validation et cohérence
            self.validation_llm = ChatOpenAI(
                model=model,
                temperature=0.0,
                max_tokens=2000
            )
        
        # Patterns juridiques
        self.legal_patterns = self._init_legal_patterns()
        
        # Keywords juridiques par catégorie
        self.legal_keywords = self._init_legal_keywords()
        
        # Termes critiques à détecter
        self.critical_terms = self._init_critical_terms()
        
        # Base de connaissances juridiques
        self.legal_knowledge = self._init_legal_knowledge()
        
        logger.info(f"✅ LegalExtractionAgent initialisé (mode {'avancé' if use_advanced_extraction else 'standard'})")
    
    # ============================================================================
    # INITIALISATION DES PATTERNS ET CONNAISSANCES
    # ============================================================================
    
    def _init_legal_patterns(self) -> Dict[str, str]:
        """Initialise les patterns regex pour éléments juridiques"""
        return {
            # Références d'articles
            'article_ref': r'(?:Article|Art\.?|Clause|Section)\s*(\d+(?:\.\d+)*)',
            'paragraph_ref': r'(?:Paragraph|Para\.?|§)\s*(\d+(?:\.\d+)*)',
            
            # Montants et limites
            'monetary_limit': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:€|EUR|USD|\$|GBP|£)',
            'percentage_limit': r'(\d{1,3}(?:[.,]\d+)?)\s*(?:%|percent|pour cent)',
            
            # Délais juridiques
            'legal_deadline': r'(\d+)\s*(?:days?|jours?|months?|mois|years?|ans?|années?)',
            'notice_period': r'(?:notice|préavis|notification).*?(\d+)\s*(?:days?|jours?|months?|mois)',
            
            # Juridictions
            'jurisdiction': r'(?:courts? of|tribunaux de|jurisdiction of)\s*([A-Z][a-zA-Z\s]+)',
            'applicable_law': r'(?:governed by|régi par|law of|droit de)\s*([A-Z][a-zA-Z\s]+)',
            
            # Clauses standards
            'liability_cap': r'(?:liability.*?limited|responsabilité.*?limitée).*?(\d+.*?(?:€|EUR|USD|\$))',
            'force_majeure': r'(?:force majeure|cas de force majeure|act of god)',
            'penalty': r'(?:penalty|pénalité|malus|sanction).*?(\d+.*?(?:€|EUR|USD|\$|%))',
            
            # Obligations
            'shall_must': r'(?:shall|must|doit|devra|obligé)',
            'may_can': r'(?:may|can|peut|pourra)',
            'prohibited': r'(?:prohibited|forbidden|interdit|défendu)',
            
            # Références contractuelles
            'contract_ref': r'(?:Contract|Agreement|Contrat|Accord)\s*(?:No\.?|N°)?\s*([A-Z0-9\-/]+)',
            'annex_ref': r'(?:Annex|Appendix|Annexe|Appendice)\s*([A-Z0-9]+)',
            
            # Conditions
            'condition_if': r'(?:if|si|in case|en cas de|provided that|à condition que)',
            'condition_unless': r'(?:unless|sauf si|except if|à moins que)'
        }
    
    def _init_legal_keywords(self) -> Dict[str, List[str]]:
        """Initialise les mots-clés juridiques par catégorie"""
        return {
            'liability': [
                'liability', 'liable', 'responsibility', 'responsible',
                'responsabilité', 'responsable', 'damages', 'dommages',
                'compensation', 'indemnity', 'indemnité', 'prejudice'
            ],
            'insurance': [
                'insurance', 'assurance', 'coverage', 'couverture',
                'policy', 'police', 'premium', 'prime', 'deductible',
                'franchise', 'claim', 'sinistre', 'underwriter'
            ],
            'force_majeure': [
                'force majeure', 'act of god', 'cas fortuit',
                'unforeseeable', 'imprévisible', 'unavoidable',
                'inévitable', 'pandemic', 'pandémie', 'war', 'guerre',
                'strike', 'grève', 'natural disaster', 'catastrophe naturelle'
            ],
            'penalty': [
                'penalty', 'pénalité', 'fine', 'amende', 'malus',
                'sanction', 'liquidated damages', 'astreinte',
                'breach', 'violation', 'manquement', 'défaut'
            ],
            'compliance': [
                'compliance', 'conformité', 'regulation', 'réglementation',
                'standard', 'norme', 'requirement', 'exigence', 'gdpr',
                'rgpd', 'csr', 'rse', 'ethics', 'éthique', 'corruption',
                'anti-bribery', 'sustainable', 'durable'
            ],
            'confidentiality': [
                'confidential', 'confidentiel', 'proprietary', 'propriétaire',
                'nda', 'non-disclosure', 'secret', 'intellectual property',
                'propriété intellectuelle', 'copyright', 'trademark',
                'patent', 'brevet', 'trade secret', 'know-how'
            ],
            'termination': [
                'termination', 'résiliation', 'expiry', 'expiration',
                'renewal', 'renouvellement', 'extension', 'prolongation',
                'notice', 'préavis', 'breach', 'rupture'
            ],
            'dispute': [
                'dispute', 'litige', 'arbitration', 'arbitrage',
                'mediation', 'médiation', 'jurisdiction', 'juridiction',
                'competent court', 'tribunal compétent', 'applicable law',
                'droit applicable', 'governing law', 'settlement'
            ]
        }
    
    def _init_critical_terms(self) -> List[str]:
        """Initialise les termes critiques nécessitant une attention particulière"""
        return [
            'unlimited liability', 'responsabilité illimitée',
            'joint and several', 'solidaire',
            'waiver', 'renonciation',
            'indemnify', 'indemniser',
            'hold harmless', 'dégager de responsabilité',
            'exclusive', 'exclusif',
            'non-negotiable', 'non négociable',
            'immediately', 'immédiatement',
            'without limitation', 'sans limitation',
            'at its sole discretion', 'à sa seule discrétion',
            'liquidated damages', 'dommages-intérêts forfaitaires',
            'consequential damages', 'dommages indirects'
        ]
    
    def _init_legal_knowledge(self) -> Dict[str, Any]:
        """Initialise la base de connaissances juridiques"""
        return {
            'standard_liability_caps': {
                'transport': 'Often based on CMR/Montreal conventions',
                'services': 'Typically 1-2x annual contract value',
                'it_services': 'Usually excludes data breach and IP violations'
            },
            'force_majeure_events': [
                'Natural disasters', 'War/terrorism', 'Pandemic',
                'Government actions', 'Strikes (sometimes)', 'Cyber attacks (emerging)'
            ],
            'insurance_types': {
                'general_liability': 'Covers third-party bodily injury and property damage',
                'professional_liability': 'Covers errors and omissions',
                'cargo_insurance': 'Covers loss/damage to goods in transit',
                'cyber_liability': 'Covers data breaches and cyber incidents'
            },
            'compliance_frameworks': {
                'GDPR': 'EU data protection regulation',
                'SOC2': 'Security and availability standards',
                'ISO27001': 'Information security management',
                'ISO14001': 'Environmental management'
            }
        }
    
    # ============================================================================
    # EXTRACTION AVANCÉE SPÉCIALISÉE JURIDIQUE
    # ============================================================================
    
    async def _extract_single_question(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Override: Extraction spécialisée pour questions juridiques
        """
        if not self.use_advanced:
            return await super()._extract_single_question(question_config, context, chunks)
        
        # Stratégie avancée multi-niveaux pour clauses juridiques
        return await self._advanced_legal_extraction(question_config, context, chunks)
    
    async def _advanced_legal_extraction(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Extraction juridique avancée avec analyse multi-niveaux
        """
        start_time = datetime.now()
        question_id = question_config.id
        
        logger.info(f"⚖️ Extraction juridique avancée pour: {question_id}")
        
        # 1. Identifier les sections juridiques pertinentes
        legal_sections = await self._identify_legal_sections(chunks)
        
        # 2. Préparer un contexte juridique enrichi
        legal_context = self._prepare_legal_context(chunks, legal_sections, question_id)
        
        # 3. Détecter les clauses interdépendantes
        related_clauses = self._detect_related_clauses(question_id, legal_context)
        
        # 4. Stratégies d'extraction parallèles
        extraction_tasks = [
            self._strategy_structured_extraction(question_config, legal_context),
            self._strategy_semantic_extraction(question_config, legal_context, related_clauses),
            self._strategy_pattern_based_extraction(question_config, chunks),
            self._strategy_cross_reference_extraction(question_config, legal_context, legal_sections)
        ]
        
        # Ajouter analyse de risques si pertinent
        if self._should_analyze_risks(question_id):
            extraction_tasks.append(
                self._strategy_risk_focused_extraction(question_config, legal_context)
            )
        
        # Exécuter toutes les stratégies
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # 5. Fusionner et valider les résultats
        best_result = self._merge_legal_results(results, question_config)
        
        # 6. Enrichir avec analyse juridique
        if best_result.status == ExtractionStatus.SUCCESS:
            best_result = await self._enrich_with_legal_analysis(best_result, question_id)
        
        # 7. Validation de cohérence juridique
        best_result = await self._validate_legal_coherence(best_result, related_clauses)
        
        # 8. Analyse de risques si activée
        if self.enable_risk_analysis and best_result.answer:
            risk_assessment = await self._assess_legal_risks(best_result, question_id)
            best_result.metadata['risk_assessment'] = risk_assessment
        
        best_result.extraction_time = (datetime.now() - start_time).total_seconds()
        
        return best_result
    
    # ============================================================================
    # STRATÉGIES D'EXTRACTION JURIDIQUE
    # ============================================================================
    
    async def _strategy_structured_extraction(
        self,
        question_config,
        context: str
    ) -> QuestionResult:
        """
        Stratégie 1: Extraction structurée avec analyse hiérarchique
        """
        # Enrichir le prompt avec structure juridique
        structured_prompt = self._create_structured_legal_prompt(question_config)
        
        try:
            response = await self.legal_analysis_llm.ainvoke([
                SystemMessage(content=structured_prompt),
                HumanMessage(content=context)
            ])
            
            parsed = self._parse_json_response(response.content)
            
            if self._validate_legal_response(parsed, question_config):
                result = self._create_question_result(
                    question_config.id,
                    parsed,
                    []
                )
                result.metadata['extraction_method'] = 'structured_legal'
                return result
                
        except Exception as e:
            logger.debug(f"Structured extraction failed: {e}")
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="Structured extraction failed"
        )
    
    async def _strategy_semantic_extraction(
        self,
        question_config,
        context: str,
        related_clauses: List[str]
    ) -> QuestionResult:
        """
        Stratégie 2: Extraction sémantique avec contexte juridique élargi
        """
        # Créer un prompt enrichi avec les clauses liées
        semantic_prompt = f"""
        {question_config.system_prompt}
        
        LEGAL CONTEXT ENHANCEMENT:
        You are analyzing a legal document. Pay special attention to:
        
        1. RELATED LEGAL CONCEPTS that might be relevant:
        {', '.join(related_clauses)}
        
        2. LOOK FOR:
        - Explicit statements and implicit obligations
        - Cross-references to other sections or documents
        - Conditional clauses (if/then, unless, provided that)
        - Exceptions and exclusions
        - Default provisions vs. negotiated terms
        
        3. LEGAL INTERPRETATION:
        - Consider both literal text and legal implications
        - Identify any ambiguities or gaps
        - Note if standard legal language is modified
        
        4. COMPLETENESS CHECK:
        - Ensure all aspects of the question are addressed
        - If information is partial, clearly indicate what's missing
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=semantic_prompt),
                HumanMessage(content=context)
            ])
            
            parsed = self._parse_json_response(response.content)
            
            if self._validate_legal_response(parsed, question_config):
                result = self._create_question_result(
                    question_config.id,
                    parsed,
                    []
                )
                result.metadata['extraction_method'] = 'semantic_legal'
                result.metadata['related_clauses'] = related_clauses
                return result
                
        except Exception as e:
            logger.debug(f"Semantic extraction failed: {e}")
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="Semantic extraction failed"
        )
    
    async def _strategy_pattern_based_extraction(
        self,
        question_config,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Stratégie 3: Extraction basée sur patterns juridiques
        """
        question_id = question_config.id
        all_clauses = []
        
        # Déterminer les patterns pertinents
        relevant_patterns = self._get_relevant_legal_patterns(question_id)
        
        for chunk in chunks:
            text = chunk.page_content
            
            for pattern_name, pattern in relevant_patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    # Extraire le contexte élargi
                    start = max(0, match.start() - 200)
                    end = min(len(text), match.end() + 200)
                    context_snippet = text[start:end].strip()
                    
                    # Analyser le contexte juridique
                    clause_analysis = self._analyze_clause_context(
                        context_snippet,
                        pattern_name,
                        match
                    )
                    
                    if clause_analysis:
                        all_clauses.append(clause_analysis)
        
        if all_clauses:
            # Structurer les résultats
            answer = self._structure_legal_clauses(all_clauses, question_id)
            
            # Créer les sources
            sources = [
                ExtractionSource(
                    document=clause.get('source', 'document'),
                    context_snippet=clause.get('context', '')[:50],
                    confidence=0.85
                )
                for clause in all_clauses[:3]
            ]
            
            return QuestionResult(
                question_id=question_id,
                answer=answer,
                sources=sources,
                status=ExtractionStatus.SUCCESS,
                confidence=0.8,
                metadata={
                    'extraction_method': 'pattern_legal',
                    'clauses_found': len(all_clauses)
                }
            )
        
        return QuestionResult(
            question_id=question_id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No pattern matches found"
        )
    
    async def _strategy_cross_reference_extraction(
        self,
        question_config,
        context: str,
        legal_sections: Dict
    ) -> QuestionResult:
        """
        Stratégie 4: Extraction avec analyse des références croisées
        """
        # Identifier les références croisées
        cross_refs = self._extract_cross_references(context)
        
        # Créer un prompt qui tient compte des références
        cross_ref_prompt = f"""
        {question_config.system_prompt}
        
        CROSS-REFERENCE ANALYSIS:
        This document contains references to the following sections/documents:
        {json.dumps(cross_refs, indent=2)}
        
        Legal sections identified:
        {json.dumps(list(legal_sections.keys())[:10], indent=2)}
        
        INSTRUCTIONS:
        1. Consider information from referenced sections
        2. Note if critical information is in referenced documents not provided
        3. Identify any circular references or dependencies
        4. Highlight if the answer depends on external documents
        """
        
        try:
            response = await self.legal_analysis_llm.ainvoke([
                SystemMessage(content=cross_ref_prompt),
                HumanMessage(content=context)
            ])
            
            parsed = self._parse_json_response(response.content)
            
            # Enrichir avec les références
            if parsed.get('answer'):
                parsed['cross_references'] = cross_refs
            
            if self._validate_legal_response(parsed, question_config):
                result = self._create_question_result(
                    question_config.id,
                    parsed,
                    []
                )
                result.metadata['extraction_method'] = 'cross_reference'
                result.metadata['references_found'] = len(cross_refs)
                return result
                
        except Exception as e:
            logger.debug(f"Cross-reference extraction failed: {e}")
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="Cross-reference extraction failed"
        )
    
    async def _strategy_risk_focused_extraction(
        self,
        question_config,
        context: str
    ) -> QuestionResult:
        """
        Stratégie 5: Extraction focalisée sur les risques juridiques
        """
        risk_prompt = f"""
        {question_config.system_prompt}
        
        RISK-FOCUSED ANALYSIS:
        As a legal risk analyst, pay special attention to:
        
        1. HIDDEN RISKS:
        - Ambiguous language that could be interpreted unfavorably
        - Missing standard protections
        - Unusual or onerous terms
        - Open-ended obligations
        
        2. LIABILITY EXPOSURE:
        - Unlimited liability provisions
        - Broad indemnification clauses
        - Lack of liability caps
        - Consequential damages inclusion
        
        3. COMPLIANCE RISKS:
        - Regulatory requirements
        - Certification obligations
        - Audit rights
        - Reporting duties
        
        4. OPERATIONAL RISKS:
        - Strict deadlines or SLAs
        - Penalty mechanisms
        - Termination triggers
        - Change management provisions
        
        Identify both the explicit terms AND their risk implications.
        """
        
        try:
            response = await self.risk_detection_llm.ainvoke([
                SystemMessage(content=risk_prompt),
                HumanMessage(content=context)
            ])
            
            parsed = self._parse_json_response(response.content)
            
            if self._validate_legal_response(parsed, question_config):
                result = self._create_question_result(
                    question_config.id,
                    parsed,
                    []
                )
                result.metadata['extraction_method'] = 'risk_focused'
                
                # Analyser les risques identifiés
                if parsed.get('answer'):
                    risks = self._analyze_legal_risks(parsed['answer'])
                    result.metadata['identified_risks'] = risks
                
                return result
                
        except Exception as e:
            logger.debug(f"Risk-focused extraction failed: {e}")
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="Risk-focused extraction failed"
        )
    
    # ============================================================================
    # IDENTIFICATION ET ANALYSE DES SECTIONS JURIDIQUES
    # ============================================================================
    
    async def _identify_legal_sections(self, chunks: List[Document]) -> Dict:
        """
        Identifie les sections juridiques dans les documents
        """
        legal_sections = {}
        section_patterns = [
            r'(?:ARTICLE|Article|CLAUSE|Clause)\s+(\d+(?:\.\d+)*)\s*[:\-]?\s*([^\n]+)',
            r'(?:Section|SECTION)\s+(\d+(?:\.\d+)*)\s*[:\-]?\s*([^\n]+)',
            r'(\d+(?:\.\d+)*)\s+([A-Z][^.!?]*(?:Agreement|Contract|Terms|Conditions|Obligations|Rights|Liability|Insurance))',
        ]
        
        for chunk_idx, chunk in enumerate(chunks):
            text = chunk.page_content
            
            for pattern in section_patterns:
                matches = re.finditer(pattern, text, re.MULTILINE)
                for match in matches:
                    section_num = match.group(1)
                    section_title = match.group(2).strip() if len(match.groups()) > 1 else "Untitled"
                    
                    # Extraire le contenu de la section
                    start = match.start()
                    # Chercher la prochaine section ou fin du chunk
                    next_section = re.search(r'(?:ARTICLE|Article|CLAUSE|Clause|Section|SECTION)\s+\d+', text[match.end():])
                    if next_section:
                        end = match.end() + next_section.start()
                    else:
                        end = min(start + 2000, len(text))
                    
                    section_content = text[start:end]
                    
                    legal_sections[f"{section_num}"] = {
                        'title': section_title,
                        'content': section_content,
                        'chunk_index': chunk_idx,
                        'type': self._classify_section_type(section_title, section_content)
                    }
        
        logger.info(f"📑 Identified {len(legal_sections)} legal sections")
        return legal_sections
    
    def _classify_section_type(self, title: str, content: str) -> str:
        """
        Classifie le type de section juridique
        """
        title_lower = title.lower()
        content_lower = content.lower()
        
        type_keywords = {
            'liability': ['liability', 'responsabilité', 'damages', 'limitation'],
            'insurance': ['insurance', 'assurance', 'coverage', 'policy'],
            'termination': ['termination', 'résiliation', 'expiry', 'renewal'],
            'payment': ['payment', 'paiement', 'invoice', 'facture'],
            'confidentiality': ['confidential', 'nda', 'proprietary', 'intellectual'],
            'dispute': ['dispute', 'arbitration', 'jurisdiction', 'governing law'],
            'compliance': ['compliance', 'regulation', 'gdpr', 'ethics'],
            'force_majeure': ['force majeure', 'act of god', 'unforeseeable'],
            'warranty': ['warranty', 'garantie', 'representation', 'condition'],
            'general': []
        }
        
        for section_type, keywords in type_keywords.items():
            if any(kw in title_lower or kw in content_lower[:200] for kw in keywords):
                return section_type
        
        return 'general'
    
    # ============================================================================
    # PRÉPARATION DU CONTEXTE JURIDIQUE
    # ============================================================================
    
    def _prepare_legal_context(
        self,
        chunks: List[Document],
        legal_sections: Dict,
        question_id: str
    ) -> str:
        """
        Prépare un contexte optimisé pour l'extraction juridique
        """
        # Identifier le type de clause recherchée
        clause_type = self._identify_clause_type(question_id)
        
        # Prioriser les sections pertinentes
        relevant_sections = self._prioritize_legal_sections(legal_sections, clause_type)
        
        context_parts = []
        total_length = 0
        max_length = 12000
        
        # 1. Ajouter les sections juridiques prioritaires
        for section_key, section_data in relevant_sections:
            if total_length >= max_length:
                break
            
            section_text = f"\n{'='*50}\n"
            section_text += f"ARTICLE {section_key}: {section_data['title']}\n"
            section_text += f"Type: {section_data['type']}\n"
            section_text += f"{'='*50}\n"
            section_text += section_data['content']
            
            context_parts.append(section_text)
            total_length += len(section_text)
        
        # 2. Ajouter les chunks contenant des termes critiques
        critical_chunks = self._find_critical_chunks(chunks, clause_type)
        for chunk in critical_chunks[:3]:
            if total_length >= max_length:
                break
            
            chunk_text = f"\n[Critical Section]\n{chunk.page_content[:1500]}"
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        # 3. Ajouter le contexte général si espace disponible
        if total_length < max_length - 1000:
            for chunk in chunks[:2]:
                if total_length >= max_length:
                    break
                remaining = max_length - total_length
                context_parts.append(f"\n[General Context]\n{chunk.page_content[:remaining]}")
                total_length += len(chunk.page_content[:remaining])
        
        return "\n\n".join(context_parts)
    
    def _prioritize_legal_sections(
        self,
        legal_sections: Dict,
        clause_type: str
    ) -> List[Tuple[str, Dict]]:
        """
        Priorise les sections juridiques selon le type de clause recherchée
        """
        scored_sections = []
        
        for section_key, section_data in legal_sections.items():
            score = 0
            
            # Score basé sur le type de section
            if section_data['type'] == clause_type:
                score += 10
            elif section_data['type'] in self._get_related_section_types(clause_type):
                score += 5
            
            # Score basé sur les mots-clés
            keywords = self.legal_keywords.get(clause_type, [])
            content_lower = section_data['content'].lower()
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    score += 2
            
            # Score basé sur les termes critiques
            for critical_term in self.critical_terms:
                if critical_term.lower() in content_lower:
                    score += 3
            
            scored_sections.append((score, section_key, section_data))
        
        # Trier par score décroissant
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        
        return [(key, data) for score, key, data in scored_sections]
    
    def _find_critical_chunks(
        self,
        chunks: List[Document],
        clause_type: str
    ) -> List[Document]:
        """
        Trouve les chunks contenant des informations critiques
        """
        critical_chunks = []
        keywords = self.legal_keywords.get(clause_type, [])
        
        for chunk in chunks:
            content_lower = chunk.page_content.lower()
            score = 0
            
            # Score basé sur les mots-clés
            for keyword in keywords:
                score += content_lower.count(keyword.lower())
            
            # Score basé sur les termes critiques
            for critical_term in self.critical_terms:
                if critical_term.lower() in content_lower:
                    score += 5
            
            if score > 0:
                critical_chunks.append((score, chunk))
        
        # Trier par score et retourner les chunks
        critical_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in critical_chunks]
    
    # ============================================================================
    # DÉTECTION DES CLAUSES LIÉES
    # ============================================================================
    
    def _detect_related_clauses(self, question_id: str, context: str) -> List[str]:
        """
        Détecte les clauses juridiques liées à la question
        """
        clause_type = self._identify_clause_type(question_id)
        related_clauses = []
        
        # Relations prédéfinies entre types de clauses
        clause_relationships = {
            'liability': ['insurance', 'indemnification', 'force_majeure', 'warranty'],
            'insurance': ['liability', 'indemnification', 'risk allocation'],
            'force_majeure': ['liability', 'termination', 'suspension', 'notice'],
            'penalty': ['performance', 'sla', 'breach', 'remedies'],
            'compliance': ['audit', 'certification', 'reporting', 'penalties'],
            'confidentiality': ['intellectual_property', 'data_protection', 'publicity'],
            'termination': ['notice', 'survival', 'transition', 'force_majeure'],
            'jurisdiction': ['dispute_resolution', 'applicable_law', 'arbitration']
        }
        
        # Obtenir les clauses liées prédéfinies
        if clause_type in clause_relationships:
            related_clauses.extend(clause_relationships[clause_type])
        
        # Détecter les références croisées dans le contexte
        cross_ref_pattern = r'(?:see|refer to|subject to|as per|pursuant to|according to)\s+(?:Article|Section|Clause)\s+(\d+(?:\.\d+)*)'
        matches = re.finditer(cross_ref_pattern, context, re.IGNORECASE)
        for match in matches:
            related_clauses.append(f"Article {match.group(1)}")
        
        return list(set(related_clauses))
    
    def _extract_cross_references(self, context: str) -> List[Dict]:
        """
        Extrait toutes les références croisées du contexte
        """
        cross_refs = []
        
        # Patterns pour différents types de références
        ref_patterns = [
            (r'(?:Article|Art\.?)\s+(\d+(?:\.\d+)*)', 'article'),
            (r'(?:Section|Sec\.?)\s+(\d+(?:\.\d+)*)', 'section'),
            (r'(?:Clause)\s+(\d+(?:\.\d+)*)', 'clause'),
            (r'(?:Annex|Appendix|Exhibit)\s+([A-Z0-9]+)', 'annex'),
            (r'(?:Schedule)\s+([A-Z0-9]+)', 'schedule')
        ]
        
        for pattern, ref_type in ref_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                cross_refs.append({
                    'type': ref_type,
                    'reference': match.group(1),
                    'full_text': match.group(0)
                })
        
        return cross_refs
    
    # ============================================================================
    # FUSION ET VALIDATION DES RÉSULTATS JURIDIQUES
    # ============================================================================
    
    def _merge_legal_results(
        self,
        results: List,
        question_config
    ) -> QuestionResult:
        """
        Fusionne les résultats juridiques avec validation
        """
        valid_results = []
        all_clauses = []
        all_sources = []
        
        for result in results:
            if isinstance(result, QuestionResult) and result.status == ExtractionStatus.SUCCESS:
                valid_results.append(result)
                
                # Collecter toutes les clauses trouvées
                if result.answer:
                    all_clauses.append({
                        'content': result.answer,
                        'method': result.metadata.get('extraction_method', 'unknown'),
                        'confidence': result.confidence
                    })
                
                # Collecter toutes les sources
                all_sources.extend(result.sources)
        
        if not valid_results:
            return QuestionResult(
                question_id=question_config.id,
                answer=None,
                status=ExtractionStatus.FAILED,
                confidence=0.0,
                error="No successful legal extraction"
            )
        
        # Sélectionner le meilleur résultat
        best_result = self._select_best_legal_result(valid_results, all_clauses)
        
        # Enrichir avec les informations des autres résultats
        if len(all_clauses) > 1:
            # Vérifier la cohérence entre les différentes extractions
            coherence_check = self._check_legal_coherence(all_clauses)
            best_result.metadata['coherence_score'] = coherence_check['score']
            
            if coherence_check['conflicts']:
                best_result.metadata['conflicts'] = coherence_check['conflicts']
        
        # Dédupliquer et prioriser les sources
        best_result.sources = self._deduplicate_sources(all_sources)[:5]
        
        return best_result
    
    def _select_best_legal_result(
        self,
        results: List[QuestionResult],
        all_clauses: List[Dict]
    ) -> QuestionResult:
        """
        Sélectionne le meilleur résultat juridique
        """
        scored_results = []
        
        for result in results:
            score = 0
            
            # Score basé sur la confiance
            score += result.confidence * 30
            
            # Score basé sur la méthode d'extraction
            method_scores = {
                'structured_legal': 10,
                'semantic_legal': 8,
                'cross_reference': 7,
                'risk_focused': 6,
                'pattern_legal': 5
            }
            method = result.metadata.get('extraction_method', '')
            score += method_scores.get(method, 0)
            
            # Score basé sur la complétude
            if result.answer:
                if isinstance(result.answer, str):
                    score += min(len(result.answer) / 50, 20)
                elif isinstance(result.answer, dict):
                    score += min(len(result.answer) * 5, 20)
            
            # Score basé sur les sources
            score += min(len(result.sources) * 5, 15)
            
            scored_results.append((score, result))
        
        # Trier et retourner le meilleur
        scored_results.sort(key=lambda x: x[0], reverse=True)
        best_result = scored_results[0][1]
        
        # Ajouter le score de sélection aux métadonnées
        best_result.metadata['selection_score'] = scored_results[0][0]
        best_result.metadata['alternative_count'] = len(results) - 1
        
        return best_result
    
    def _check_legal_coherence(self, clauses: List[Dict]) -> Dict:
        """
        Vérifie la cohérence entre différentes extractions de clauses
        """
        coherence = {
            'score': 1.0,
            'conflicts': [],
            'agreements': []
        }
        
        if len(clauses) < 2:
            return coherence
        
        # Comparer les clauses deux à deux
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                clause1 = str(clauses[i].get('content', ''))
                clause2 = str(clauses[j].get('content', ''))
                
                # Vérifier les contradictions évidentes
                if self._are_contradictory(clause1, clause2):
                    coherence['conflicts'].append({
                        'clause1': clause1[:100],
                        'clause2': clause2[:100],
                        'type': 'contradiction'
                    })
                    coherence['score'] -= 0.2
                
                # Vérifier les accords
                elif self._are_consistent(clause1, clause2):
                    coherence['agreements'].append({
                        'clause1': clause1[:100],
                        'clause2': clause2[:100]
                    })
        
        coherence['score'] = max(0, min(1, coherence['score']))
        return coherence
    
    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """
        Détecte si deux textes juridiques sont contradictoires
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Patterns de contradiction
        contradictions = [
            ('unlimited', 'limited'),
            ('shall', 'shall not'),
            ('must', 'must not'),
            ('required', 'optional'),
            ('exclusive', 'non-exclusive'),
            ('with', 'without')
        ]
        
        for term1, term2 in contradictions:
            if (term1 in text1_lower and term2 in text2_lower) or \
               (term2 in text1_lower and term1 in text2_lower):
                return True
        
        return False
    
    def _are_consistent(self, text1: str, text2: str) -> bool:
        """
        Vérifie si deux textes juridiques sont cohérents
        """
        # Calcul de similarité simple
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        
        return similarity > 0.5
    
    def _deduplicate_sources(self, sources: List[ExtractionSource]) -> List[ExtractionSource]:
        """
        Déduplique et priorise les sources juridiques
        """
        unique_sources = {}
        
        for source in sources:
            # Créer une clé unique basée sur le document et le contexte
            key = f"{source.document}:{source.context_snippet[:30]}"
            
            if key not in unique_sources:
                unique_sources[key] = source
            else:
                # Garder la source avec la meilleure confiance
                if source.confidence > unique_sources[key].confidence:
                    unique_sources[key] = source
        
        # Trier par confiance décroissante
        sorted_sources = sorted(unique_sources.values(), key=lambda x: x.confidence, reverse=True)
        
        return sorted_sources
    
    # ============================================================================
    # ENRICHISSEMENT ET ANALYSE JURIDIQUE
    # ============================================================================
    
    async def _enrich_with_legal_analysis(
        self,
        result: QuestionResult,
        question_id: str
    ) -> QuestionResult:
        """
        Enrichit le résultat avec une analyse juridique approfondie
        """
        if not result.answer:
            return result
        
        clause_type = self._identify_clause_type(question_id)
        
        # Analyse spécifique selon le type de clause
        analysis = {}
        
        if clause_type == 'liability':
            analysis = self._analyze_liability_clause(result.answer)
        elif clause_type == 'insurance':
            analysis = self._analyze_insurance_requirements(result.answer)
        elif clause_type == 'force_majeure':
            analysis = self._analyze_force_majeure(result.answer)
        elif clause_type == 'penalty':
            analysis = self._analyze_penalty_clause(result.answer)
        elif clause_type == 'compliance':
            analysis = self._analyze_compliance_requirements(result.answer)
        elif clause_type == 'jurisdiction':
            analysis = self._analyze_jurisdiction(result.answer)
        
        if analysis:
            result.metadata['legal_analysis'] = analysis
        
        # Identifier les implications juridiques
        implications = self._identify_legal_implications(result.answer, clause_type)
        if implications:
            result.metadata['implications'] = implications
        
        return result
    
    def _analyze_liability_clause(self, clause_content: Any) -> Dict:
        """
        Analyse approfondie d'une clause de responsabilité
        """
        analysis = {
            'has_cap': False,
            'cap_amount': None,
            'exclusions': [],
            'includes_indirect_damages': False,
            'risk_level': 'standard'
        }
        
        content_str = str(clause_content).lower()
        
        # Détecter le plafond de responsabilité
        cap_match = re.search(r'(\d+(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:€|eur|usd|\$)', content_str)
        if cap_match:
            analysis['has_cap'] = True
            analysis['cap_amount'] = cap_match.group(0)
        
        # Détecter responsabilité illimitée
        if any(term in content_str for term in ['unlimited liability', 'responsabilité illimitée']):
            analysis['risk_level'] = 'high'
            analysis['has_cap'] = False
        
        # Détecter les exclusions
        if 'except' in content_str or 'excluding' in content_str or 'sauf' in content_str:
            analysis['exclusions'].append('Specific exclusions present')
        
        # Détecter dommages indirects
        if any(term in content_str for term in ['indirect', 'consequential', 'indirects']):
            analysis['includes_indirect_damages'] = 'consequential' in content_str
        
        return analysis
    
    def _analyze_insurance_requirements(self, clause_content: Any) -> Dict:
        """
        Analyse des exigences d'assurance
        """
        analysis = {
            'insurance_types': [],
            'minimum_coverage': {},
            'specific_requirements': []
        }
        
        content_str = str(clause_content).lower()
        
        # Types d'assurance
        insurance_types = {
            'general_liability': ['general liability', 'responsabilité civile générale'],
            'professional_liability': ['professional liability', 'errors and omissions', 'e&o'],
            'cargo': ['cargo insurance', 'goods in transit', 'marchandises'],
            'cyber': ['cyber liability', 'data breach', 'cyber risk']
        }
        
        for ins_type, keywords in insurance_types.items():
            if any(kw in content_str for kw in keywords):
                analysis['insurance_types'].append(ins_type)
        
        # Montants minimums
        amount_matches = re.finditer(r'(\d+(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:€|eur|usd|\$|gbp|£)', content_str)
        for match in amount_matches:
            analysis['minimum_coverage'][match.group(0)] = 'Found in requirements'
        
        return analysis
    
    def _analyze_force_majeure(self, clause_content: Any) -> Dict:
        """
        Analyse de la clause de force majeure
        """
        analysis = {
            'events_covered': [],
            'notification_required': False,
            'notification_period': None,
            'effects': []
        }
        
        content_str = str(clause_content).lower()
        
        # Événements couverts
        events = [
            'natural disaster', 'war', 'terrorism', 'pandemic', 'strike',
            'government action', 'act of god', 'catastrophe naturelle'
        ]
        
        for event in events:
            if event in content_str:
                analysis['events_covered'].append(event)
        
        # Notification
        if 'notify' in content_str or 'notification' in content_str:
            analysis['notification_required'] = True
            
            # Période de notification
            period_match = re.search(r'(\d+)\s*(?:days?|hours?|jours?|heures?)', content_str)
            if period_match:
                analysis['notification_period'] = period_match.group(0)
        
        # Effets
        if 'suspend' in content_str:
            analysis['effects'].append('suspension')
        if 'extend' in content_str:
            analysis['effects'].append('extension')
        
        return analysis
    
    def _analyze_penalty_clause(self, clause_content: Any) -> Dict:
        """
        Analyse des clauses de pénalité
        """
        analysis = {
            'penalty_types': [],
            'triggers': [],
            'amounts': [],
            'cap_exists': False
        }
        
        content_str = str(clause_content).lower()
        
        # Types de pénalités
        if 'late delivery' in content_str or 'retard' in content_str:
            analysis['penalty_types'].append('late_delivery')
        if 'performance' in content_str:
            analysis['penalty_types'].append('performance')
        if 'sla' in content_str:
            analysis['penalty_types'].append('sla_breach')
        
        # Montants
        amount_matches = re.finditer(r'(\d+(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:€|%|eur|usd|\$)', content_str)
        for match in amount_matches:
            analysis['amounts'].append(match.group(0))
        
        # Plafond
        if 'cap' in content_str or 'maximum' in content_str or 'plafond' in content_str:
            analysis['cap_exists'] = True
        
        return analysis
    
    def _analyze_compliance_requirements(self, clause_content: Any) -> Dict:
        """
        Analyse des exigences de conformité
        """
        analysis = {
            'frameworks': [],
            'certifications': [],
            'audit_rights': False,
            'reporting_required': False
        }
        
        content_str = str(clause_content).lower()
        
        # Frameworks de conformité
        frameworks = {
            'GDPR': ['gdpr', 'data protection', 'rgpd'],
            'ISO27001': ['iso 27001', 'iso27001', 'information security'],
            'SOC2': ['soc 2', 'soc2', 'service organization'],
            'CSR': ['csr', 'corporate social', 'rse']
        }
        
        for framework, keywords in frameworks.items():
            if any(kw in content_str for kw in keywords):
                analysis['frameworks'].append(framework)
        
        # Droits d'audit
        if 'audit' in content_str:
            analysis['audit_rights'] = True
        
        # Reporting
        if 'report' in content_str or 'rapport' in content_str:
            analysis['reporting_required'] = True
        
        return analysis
    
    def _analyze_jurisdiction(self, clause_content: Any) -> Dict:
        """
        Analyse de la juridiction et du droit applicable
        """
        analysis = {
            'applicable_law': None,
            'jurisdiction': None,
            'arbitration': False,
            'arbitration_rules': None
        }
        
        content_str = str(clause_content)
        
        # Droit applicable
        law_match = re.search(r'(?:law of|laws of|droit de)\s+([A-Z][a-zA-Z\s]+)', content_str, re.IGNORECASE)
        if law_match:
            analysis['applicable_law'] = law_match.group(1).strip()
        
        # Juridiction
        court_match = re.search(r'(?:courts of|tribunaux de)\s+([A-Z][a-zA-Z\s]+)', content_str, re.IGNORECASE)
        if court_match:
            analysis['jurisdiction'] = court_match.group(1).strip()
        
        # Arbitrage
        if 'arbitration' in content_str.lower() or 'arbitrage' in content_str.lower():
            analysis['arbitration'] = True
            
            # Règles d'arbitrage
            if 'ICC' in content_str:
                analysis['arbitration_rules'] = 'ICC'
            elif 'LCIA' in content_str:
                analysis['arbitration_rules'] = 'LCIA'
            elif 'UNCITRAL' in content_str:
                analysis['arbitration_rules'] = 'UNCITRAL'
        
        return analysis
    
    def _identify_legal_implications(self, clause_content: Any, clause_type: str) -> List[str]:
        """
        Identifie les implications juridiques d'une clause
        """
        implications = []
        content_str = str(clause_content).lower()
        
        # Implications générales
        if 'unlimited' in content_str or 'illimitée' in content_str:
            implications.append("Unlimited exposure - high risk")
        
        if 'exclusive' in content_str:
            implications.append("Exclusive arrangement - limits flexibility")
        
        if 'indemnify' in content_str or 'indemniser' in content_str:
            implications.append("Indemnification obligation - potential third-party liability")
        
        if 'immediate' in content_str or 'immédiat' in content_str:
            implications.append("Immediate action required - operational impact")
        
        # Implications spécifiques au type
        if clause_type == 'liability' and 'consequential' not in content_str:
            implications.append("May include indirect damages - check carefully")
        
        if clause_type == 'termination' and 'convenience' in content_str:
            implications.append("Termination for convenience allowed - less contract security")
        
        if clause_type == 'jurisdiction' and 'arbitration' in content_str:
            implications.append("Arbitration required - limited court recourse")
        
        return implications
    
    # ============================================================================
    # VALIDATION DE COHÉRENCE JURIDIQUE
    # ============================================================================
    
    async def _validate_legal_coherence(
        self,
        result: QuestionResult,
        related_clauses: List[str]
    ) -> QuestionResult:
        """
        Valide la cohérence juridique du résultat
        """
        if not result.answer or not self.validation_llm:
            return result
        
        validation_prompt = f"""
        Validate the legal coherence of this extracted clause:
        
        EXTRACTED CLAUSE:
        {json.dumps(result.answer) if isinstance(result.answer, dict) else result.answer}
        
        RELATED CLAUSES TO CONSIDER:
        {', '.join(related_clauses)}
        
        VALIDATION CHECKS:
        1. Is the clause internally consistent?
        2. Are there any obvious contradictions?
        3. Is critical information missing?
        4. Are legal terms used correctly?
        5. Does it align with standard legal practice?
        
        Return a JSON with:
        - is_valid: boolean
        - issues: list of issues found
        - suggestions: list of suggestions for completeness
        """
        
        try:
            response = await self.validation_llm.ainvoke([
                SystemMessage(content=validation_prompt),
                HumanMessage(content="Perform legal validation")
            ])
            
            validation = self._parse_json_response(response.content)
            
            if validation:
                result.metadata['legal_validation'] = validation
                
                # Ajuster la confiance selon la validation
                if validation.get('is_valid') == False:
                    result.confidence *= 0.8
                    if validation.get('issues'):
                        result.metadata['validation_issues'] = validation['issues']
        
        except Exception as e:
            logger.debug(f"Legal validation failed: {e}")
        
        return result
    
    # ============================================================================
    # ÉVALUATION DES RISQUES JURIDIQUES
    # ============================================================================
    
    async def _assess_legal_risks(
        self,
        result: QuestionResult,
        question_id: str
    ) -> Dict:
        """
        Évalue les risques juridiques associés à la clause extraite
        """
        if not result.answer or not self.risk_detection_llm:
            return {}
        
        clause_type = self._identify_clause_type(question_id)
        
        risk_prompt = f"""
        Assess the legal risks in this {clause_type} clause:
        
        CLAUSE CONTENT:
        {json.dumps(result.answer) if isinstance(result.answer, dict) else result.answer}
        
        RISK ASSESSMENT CRITERIA:
        1. Financial exposure (low/medium/high/unlimited)
        2. Operational constraints (flexible/moderate/strict)
        3. Compliance burden (minimal/standard/heavy)
        4. Dispute potential (low/medium/high)
        5. Market standard deviation (standard/unusual/highly unusual)
        
        IDENTIFY:
        - Primary risks and their severity
        - Mitigation strategies
        - Red flags requiring legal review
        - Negotiation points
        
        Return a structured JSON assessment.
        """
        
        try:
            response = await self.risk_detection_llm.ainvoke([
                SystemMessage(content=risk_prompt),
                HumanMessage(content="Perform risk assessment")
            ])
            
            risk_assessment = self._parse_json_response(response.content)
            
            if not risk_assessment:
                risk_assessment = self._default_risk_assessment(clause_type)
            
            # Calculer un score de risque global
            risk_assessment['overall_risk_score'] = self._calculate_risk_score(risk_assessment)
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return self._default_risk_assessment(clause_type)
    
    def _calculate_risk_score(self, risk_assessment: Dict) -> float:
        """
        Calcule un score de risque global (0-1)
        """
        score = 0.0
        weights = {
            'financial_exposure': 0.3,
            'operational_constraints': 0.2,
            'compliance_burden': 0.2,
            'dispute_potential': 0.2,
            'market_deviation': 0.1
        }
        
        risk_levels = {
            'low': 0.2, 'minimal': 0.2, 'flexible': 0.2, 'standard': 0.3,
            'medium': 0.5, 'moderate': 0.5,
            'high': 0.8, 'strict': 0.8, 'heavy': 0.8, 'unusual': 0.7,
            'unlimited': 1.0, 'highly unusual': 0.9
        }
        
        for criterion, weight in weights.items():
            value = risk_assessment.get(criterion, 'medium')
            if isinstance(value, str):
                score += risk_levels.get(value.lower(), 0.5) * weight
        
        return min(1.0, score)
    
    def _default_risk_assessment(self, clause_type: str) -> Dict:
        """
        Évaluation de risque par défaut selon le type de clause
        """
        defaults = {
            'liability': {
                'financial_exposure': 'medium',
                'operational_constraints': 'moderate',
                'compliance_burden': 'standard',
                'dispute_potential': 'medium',
                'market_deviation': 'standard'
            },
            'insurance': {
                'financial_exposure': 'low',
                'operational_constraints': 'moderate',
                'compliance_burden': 'standard',
                'dispute_potential': 'low',
                'market_deviation': 'standard'
            },
            'force_majeure': {
                'financial_exposure': 'low',
                'operational_constraints': 'flexible',
                'compliance_burden': 'minimal',
                'dispute_potential': 'medium',
                'market_deviation': 'standard'
            }
        }
        
        return defaults.get(clause_type, {
            'financial_exposure': 'medium',
            'operational_constraints': 'moderate',
            'compliance_burden': 'standard',
            'dispute_potential': 'medium',
            'market_deviation': 'standard'
        })
    
    # ============================================================================
    # MÉTHODES UTILITAIRES SPÉCIALISÉES
    # ============================================================================
    
    def _identify_clause_type(self, question_id: str) -> str:
        """
        Identifie le type de clause juridique basé sur l'ID de la question
        """
        question_lower = question_id.lower()
        
        if 'liability' in question_lower or 'responsabilit' in question_lower:
            return 'liability'
        elif 'insurance' in question_lower or 'assurance' in question_lower:
            return 'insurance'
        elif 'force_majeure' in question_lower:
            return 'force_majeure'
        elif 'penalty' in question_lower or 'penalt' in question_lower:
            return 'penalty'
        elif 'compliance' in question_lower or 'conform' in question_lower:
            return 'compliance'
        elif 'confidential' in question_lower or 'ip' in question_lower or 'nda' in question_lower:
            return 'confidentiality'
        elif 'jurisdiction' in question_lower or 'applicable_law' in question_lower:
            return 'jurisdiction'
        elif 'terminat' in question_lower:
            return 'termination'
        else:
            return 'general'
    
    def _get_related_section_types(self, clause_type: str) -> List[str]:
        """
        Retourne les types de sections liés à un type de clause
        """
        relationships = {
            'liability': ['insurance', 'warranty', 'force_majeure'],
            'insurance': ['liability', 'compliance'],
            'force_majeure': ['liability', 'termination'],
            'penalty': ['liability', 'termination'],
            'compliance': ['liability', 'insurance', 'confidentiality'],
            'confidentiality': ['compliance', 'termination'],
            'jurisdiction': ['dispute', 'termination'],
            'termination': ['force_majeure', 'liability', 'confidentiality']
        }
        
        return relationships.get(clause_type, [])
    
    def _create_structured_legal_prompt(self, question_config) -> str:
        """
        Crée un prompt structuré pour l'extraction juridique
        """
        base_prompt = question_config.system_prompt
        
        structured_enhancement = """
        
        LEGAL EXTRACTION FRAMEWORK:
        
        1. PRIMARY EXTRACTION:
           - Extract the exact legal language used
           - Identify the specific article/section references
           - Note any defined terms (usually capitalized)
        
        2. CONTEXTUAL ELEMENTS:
           - Conditions precedent or subsequent
           - Exceptions and carve-outs
           - Cross-references to other provisions
           - Temporal limitations or deadlines
        
        3. LEGAL STRUCTURE:
           - Main obligation or right
           - Scope and limitations
           - Remedies or consequences
           - Procedural requirements
        
        4. INTERPRETATION NOTES:
           - Ambiguous language that needs clarification
           - Missing standard provisions
           - Unusual or onerous terms
           - Potential gaps or loopholes
        
        5. PRACTICAL IMPLICATIONS:
           - Operational requirements
           - Financial implications
           - Risk allocation
           - Compliance obligations
        
        Provide a comprehensive extraction that a legal professional can rely on.
        Include exact quotes where critical.
        """
        
        return base_prompt + structured_enhancement
    
    def _get_relevant_legal_patterns(self, question_id: str) -> Dict[str, str]:
        """
        Retourne les patterns pertinents pour une question juridique
        """
        clause_type = self._identify_clause_type(question_id)
        
        # Sélectionner les patterns selon le type
        pattern_mappings = {
            'liability': ['liability_cap', 'monetary_limit', 'shall_must', 'prohibited'],
            'insurance': ['monetary_limit', 'shall_must'],
            'force_majeure': ['force_majeure', 'notice_period', 'condition_if'],
            'penalty': ['penalty', 'monetary_limit', 'percentage_limit'],
            'jurisdiction': ['jurisdiction', 'applicable_law'],
            'compliance': ['shall_must', 'prohibited'],
            'confidentiality': ['shall_must', 'prohibited'],
            'termination': ['legal_deadline', 'notice_period', 'condition_if']
        }
        
        relevant_pattern_names = pattern_mappings.get(clause_type, ['shall_must'])
        
        return {name: self.legal_patterns[name] 
                for name in relevant_pattern_names 
                if name in self.legal_patterns}
    
    def _analyze_clause_context(
        self,
        context: str,
        pattern_name: str,
        match: re.Match
    ) -> Optional[Dict]:
        """
        Analyse le contexte autour d'un match de pattern juridique
        """
        # Extraire les informations selon le type de pattern
        clause_info = {
            'type': pattern_name,
            'matched_text': match.group(0),
            'context': context,
            'source': 'document'
        }
        
        # Enrichir selon le type de pattern
        if pattern_name == 'liability_cap':
            # Extraire le montant
            clause_info['cap_amount'] = match.group(1) if match.groups() else None
            clause_info['importance'] = 'critical'
            
        elif pattern_name == 'notice_period':
            # Extraire la période
            clause_info['period'] = match.group(1) if match.groups() else None
            clause_info['importance'] = 'high'
            
        elif pattern_name == 'penalty':
            # Extraire le montant ou pourcentage
            clause_info['penalty_amount'] = match.group(1) if match.groups() else None
            clause_info['importance'] = 'high'
        
        # Vérifier si c'est une obligation ou une permission
        if pattern_name in ['shall_must', 'may_can', 'prohibited']:
            clause_info['obligation_type'] = pattern_name
            clause_info['importance'] = 'high' if pattern_name != 'may_can' else 'standard'
        
        return clause_info
    
    def _structure_legal_clauses(
        self,
        clauses: List[Dict],
        question_id: str
    ) -> Any:
        """
        Structure les clauses juridiques extraites
        """
        if not clauses:
            return None
        
        # Si une seule clause, retourner directement
        if len(clauses) == 1:
            clause = clauses[0]
            return {
                'main_provision': clause.get('context', ''),
                'type': clause.get('type', 'unknown'),
                'details': {k: v for k, v in clause.items() 
                          if k not in ['context', 'type', 'source']}
            }
        
        # Si plusieurs clauses, les organiser
        structured = {
            'provisions': [],
            'obligations': [],
            'restrictions': [],
            'financial_terms': [],
            'procedural_requirements': [],
            'other': []
        }
        
        for clause in clauses:
            clause_type = clause.get('type', '')
            
            # Classifier la clause
            if 'obligation' in clause_type or 'shall' in clause_type:
                structured['obligations'].append(clause)
            elif 'prohibit' in clause_type or 'restriction' in str(clause):
                structured['restrictions'].append(clause)
            elif 'monetary' in clause_type or 'penalty' in clause_type:
                structured['financial_terms'].append(clause)
            elif 'notice' in clause_type or 'deadline' in clause_type:
                structured['procedural_requirements'].append(clause)
            else:
                structured['provisions'].append(clause)
        
        # Nettoyer les catégories vides et simplifier si possible
        cleaned = {k: v for k, v in structured.items() if v}
        
        # Si une seule catégorie non vide, la retourner directement
        if len(cleaned) == 1:
            key, value = list(cleaned.items())[0]
            if len(value) == 1:
                return value[0].get('context', '')
            return value
        
        return cleaned
    
    def _validate_legal_response(self, parsed: Dict, question_config) -> bool:
        """
        Validation spécifique pour les réponses juridiques
        """
        if not parsed or 'answer' not in parsed:
            return False
        
        answer = parsed.get('answer')
        
        # Ne pas accepter les réponses trop courtes pour des clauses juridiques
        if isinstance(answer, str):
            if len(answer) < 20:  # Les clauses juridiques sont rarement très courtes
                return False
            
            # Vérifier la présence de termes juridiques
            legal_indicators = [
                'shall', 'must', 'may', 'liability', 'responsible',
                'obligation', 'required', 'prohibited', 'agreement',
                'contract', 'clause', 'article', 'section'
            ]
            
            answer_lower = answer.lower()
            has_legal_term = any(term in answer_lower for term in legal_indicators)
            
            # Pour certaines questions, exiger des termes juridiques
            clause_type = self._identify_clause_type(question_config.id)
            if clause_type in ['liability', 'insurance', 'jurisdiction']:
                return has_legal_term
        
        # Validation standard du parent
        return super()._validate_response(parsed, question_config.validator)
    
    def _should_analyze_risks(self, question_id: str) -> bool:
        """
        Détermine si l'analyse de risques est pertinente
        """
        risk_relevant_types = [
            'liability', 'insurance', 'force_majeure',
            'penalty', 'termination', 'compliance'
        ]
        
        clause_type = self._identify_clause_type(question_id)
        return clause_type in risk_relevant_types
    
    def _analyze_legal_risks(self, answer_content: str) -> List[Dict]:
        """
        Analyse les risques juridiques dans le contenu
        """
        risks = []
        content_lower = answer_content.lower() if isinstance(answer_content, str) else str(answer_content).lower()
        
        # Catalogue de risques
        risk_patterns = [
            {
                'pattern': 'unlimited liability',
                'risk': 'Unlimited financial exposure',
                'severity': 'critical',
                'mitigation': 'Negotiate liability cap'
            },
            {
                'pattern': 'consequential damages',
                'risk': 'Liability for indirect damages',
                'severity': 'high',
                'mitigation': 'Exclude consequential damages'
            },
            {
                'pattern': 'sole discretion',
                'risk': 'Unilateral decision power',
                'severity': 'medium',
                'mitigation': 'Request reasonableness standard'
            },
            {
                'pattern': 'immediately',
                'risk': 'No grace period',
                'severity': 'medium',
                'mitigation': 'Negotiate notice period'
            },
            {
                'pattern': 'indemnify',
                'risk': 'Third-party liability assumption',
                'severity': 'high',
                'mitigation': 'Limit to direct damages or add carve-outs'
            }
        ]
        
        for risk_def in risk_patterns:
            if risk_def['pattern'] in content_lower:
                risks.append({
                    'identified_risk': risk_def['risk'],
                    'severity': risk_def['severity'],
                    'suggested_mitigation': risk_def['mitigation'],
                    'pattern_found': risk_def['pattern']
                })
        
        return risks
    
    # ============================================================================
    # MÉTHODES PUBLIQUES ADDITIONNELLES
    # ============================================================================
    
    async def extract_all_legal_clauses(
        self,
        chunks: List[Document]
    ) -> Dict[str, Any]:
        """
        Extrait toutes les clauses juridiques importantes du document
        """
        logger.info("⚖️ Extraction complète des clauses juridiques")
        
        # Identifier toutes les sections juridiques
        legal_sections = await self._identify_legal_sections(chunks)
        
        # Classifier les sections par type
        classified_sections = defaultdict(list)
        for section_key, section_data in legal_sections.items():
            section_type = section_data['type']
            classified_sections[section_type].append({
                'reference': section_key,
                'title': section_data['title'],
                'content_preview': section_data['content'][:200] + '...'
            })
        
        # Extraire les clauses critiques
        critical_clauses = await self._extract_critical_clauses(chunks)
        
        # Créer un résumé juridique
        legal_summary = {
            'total_sections': len(legal_sections),
            'sections_by_type': dict(classified_sections),
            'critical_clauses': critical_clauses,
            'risk_indicators': self._identify_risk_indicators(chunks),
            'missing_standard_clauses': self._identify_missing_clauses(legal_sections)
        }
        
        return legal_summary
    
    async def _extract_critical_clauses(
        self,
        chunks: List[Document]
    ) -> List[Dict]:
        """
        Extrait les clauses critiques nécessitant une attention particulière
        """
        critical_clauses = []
        
        for chunk in chunks:
            content = chunk.page_content
            
            # Chercher les termes critiques
            for critical_term in self.critical_terms:
                if critical_term.lower() in content.lower():
                    # Extraire le contexte
                    start = content.lower().find(critical_term.lower())
                    context_start = max(0, start - 100)
                    context_end = min(len(content), start + len(critical_term) + 100)
                    
                    critical_clauses.append({
                        'term': critical_term,
                        'context': content[context_start:context_end],
                        'source': chunk.metadata.get('source', 'unknown'),
                        'implications': self._get_term_implications(critical_term)
                    })
        
        return critical_clauses
    
    def _identify_risk_indicators(self, chunks: List[Document]) -> List[str]:
        """
        Identifie les indicateurs de risque dans le document
        """
        risk_indicators = []
        combined_text = ' '.join([c.page_content.lower() for c in chunks[:10]])
        
        risk_checks = [
            ('No liability cap found', 'liability' in combined_text and 'cap' not in combined_text),
            ('Unlimited liability mentioned', 'unlimited liability' in combined_text),
            ('Consequential damages included', 'consequential' in combined_text and 'exclude' not in combined_text),
            ('Short notice periods', re.search(r'\b(?:24|48)\s*hours?\b', combined_text) is not None),
            ('Strict penalties', 'penalty' in combined_text or 'liquidated damages' in combined_text),
            ('Broad indemnification', 'indemnify' in combined_text and 'harmless' in combined_text),
            ('No force majeure clause', 'force majeure' not in combined_text),
            ('Exclusive arrangement', 'exclusive' in combined_text),
            ('Auto-renewal clause', 'auto' in combined_text and 'renew' in combined_text)
        ]
        
        for description, condition in risk_checks:
            if condition:
                risk_indicators.append(description)
        
        return risk_indicators
    
    def _identify_missing_clauses(self, legal_sections: Dict) -> List[str]:
        """
        Identifie les clauses standards potentiellement manquantes
        """
        standard_clauses = [
            'force_majeure',
            'liability',
            'insurance',
            'confidentiality',
            'dispute',
            'termination'
        ]
        
        found_types = {section['type'] for section in legal_sections.values()}
        missing = [clause for clause in standard_clauses if clause not in found_types]
        
        return missing
    
    def _get_term_implications(self, critical_term: str) -> str:
        """
        Retourne les implications d'un terme critique
        """
        implications_map = {
            'unlimited liability': 'No cap on financial exposure - maximum risk',
            'joint and several': 'Can be held liable for entire obligation',
            'waiver': 'Giving up important rights',
            'indemnify': 'Taking on third-party liability',
            'exclusive': 'Cannot work with competitors',
            'immediately': 'No time to cure breaches',
            'liquidated damages': 'Pre-agreed penalties regardless of actual damage'
        }
        
        return implications_map.get(critical_term.lower(), 'Requires careful legal review')
    
    def get_legal_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé de l'analyse juridique effectuée
        """
        summary = {
            'extraction_stats': self.get_stats(),
            'clause_types_analyzed': list(set(
                self._identify_clause_type(q_id) 
                for q_id in self.get_question_ids()
            )),
            'risk_analysis_enabled': self.enable_risk_analysis,
            'advanced_extraction': self.use_advanced,
            'legal_knowledge_base': list(self.legal_knowledge.keys())
        }
        
        return summary