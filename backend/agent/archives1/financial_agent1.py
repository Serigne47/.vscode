# agents/financial_agent.py
"""
Agent financier optimisé pour extraction maximale via LLM
Spécialisé dans l'extraction des conditions financières et commerciales
"""
import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from decimal import Decimal, InvalidOperation

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from .base_agent import YAMLBaseAgent, QuestionResult, ExtractionStatus

logger = logging.getLogger(__name__)

class FinancialExtractionAgent(YAMLBaseAgent):
    """
    Agent d'extraction financière avec stratégies optimisées
    
    Spécialisations:
    - Extraction des modalités de facturation et paiement
    - Détection des devises et règles de change
    - Identification des garanties et cautions
    - Analyse des clauses de révision de prix
    - Calcul des risques financiers
    """
    
    def __init__(
        self,
        config_path: str = "config/prompts/financial_questions.yaml",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        use_advanced_extraction: bool = True
    ):
        """
        Initialise l'agent financier avec capacités avancées
        
        Args:
            config_path: Chemin vers le fichier YAML des questions financières
            model: Modèle LLM optimisé pour données financières
            temperature: Basse température pour précision
            enable_cache: Cache des résultats
            enable_parallel: Extraction parallèle
            use_advanced_extraction: Stratégies avancées pour clauses complexes
        """
        super().__init__(
            config_path=config_path,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
            max_parallel=4
        )
        
        self.use_advanced = use_advanced_extraction
        
        # LLMs spécialisés pour différents aspects
        if self.use_advanced:
            # LLM pour extraction de clauses contractuelles
            self.contract_llm = ChatOpenAI(
                model=model,
                temperature=0.0,
                max_tokens=4000
            )
            
            # LLM pour calculs et validations
            self.calculation_llm = ChatOpenAI(
                model=model,
                temperature=0.0,
                max_tokens=2000
            )
        
        # Patterns financiers
        self.financial_patterns = self._init_financial_patterns()
        
        # Mappings de devises et termes
        self.currency_mappings = self._init_currency_mappings()
        self.payment_term_mappings = self._init_payment_mappings()
        
        # Keywords pour scoring
        self.financial_keywords = self._init_financial_keywords()
        
        logger.info(f"✅ FinancialExtractionAgent initialisé (mode {'avancé' if use_advanced_extraction else 'standard'})")
    
    # ============================================================================
    # INITIALISATION DES PATTERNS ET MAPPINGS
    # ============================================================================
    
    def _init_financial_patterns(self) -> Dict[str, str]:
        """Initialise les patterns regex pour données financières"""
        return {
            # Termes de paiement
            'payment_days': r'(?:net\s*)?(\d+)\s*(?:days?|jours?)',
            'payment_net': r'(?:net|NET)\s*(\d+)',
            'payment_eom': r'(?:end\s*of\s*month|fin\s*de\s*mois|EOM)',
            'payment_terms_complex': r'(\d+)\s*(?:days?|jours?)\s*(?:from|après|after|à compter)',
            
            # Devises
            'currency': r'(?:EUR|USD|GBP|CHF|JPY|CNY|€|\$|£|¥)',
            'currency_amount': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{2})?)\s*(?:EUR|USD|GBP|€|\$|£)',
            
            # Pourcentages et taux
            'percentage': r'(\d{1,3}(?:[.,]\d{1,2})?)\s*%',
            'discount': r'(?:discount|remise|escompte).*?(\d{1,3}(?:[.,]\d{1,2})?)\s*%',
            'penalty': r'(?:penalty|pénalité|late payment|retard).*?(\d{1,3}(?:[.,]\d{1,2})?)\s*%',
            'interest_rate': r'(?:interest|intérêt|taux).*?(\d{1,3}(?:[.,]\d{1,2})?)\s*%',
            
            # Garanties
            'guarantee_amount': r'(?:guarantee|garantie|caution).*?(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{2})?)',
            'guarantee_percentage': r'(?:guarantee|garantie|caution).*?(\d{1,3}(?:[.,]\d{1,2})?)\s*%',
            'performance_bond': r'(?:performance\s*bond|garantie\s*de\s*bonne\s*exécution)',
            
            # Facturation
            'billing_frequency': r'(?:monthly|mensuel|quarterly|trimestriel|annual|annuel|per shipment|par envoi)',
            'invoice_split': r'(?:per entity|par entité|per country|par pays|consolidated|consolidé)',
            
            # Révision de prix
            'fuel_surcharge': r'(?:fuel\s*surcharge|BAF|bunker|gazole)',
            'indexation': r'(?:index|indice|CPI|inflation|ICC)',
            'price_revision': r'(?:price\s*(?:revision|adjustment)|révision\s*(?:de\s*)?prix)',
            'escalation': r'(?:escalation|cost\s*adjustment|ajustement)',
            
            # Dates et délais
            'date_format': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            'deadline': r'(?:deadline|échéance|due date|date limite)',
            
            # Montants
            'amount_millions': r'(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:M|million)',
            'amount_thousands': r'(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:K|thousand|mille)'
        }
    
    def _init_currency_mappings(self) -> Dict[str, str]:
        """Initialise les mappings de devises"""
        return {
            '€': 'EUR',
            '$': 'USD',
            '£': 'GBP',
            '¥': 'JPY',
            'yuan': 'CNY',
            'dollar': 'USD',
            'euro': 'EUR',
            'pound': 'GBP',
            'sterling': 'GBP',
            'yen': 'JPY',
            'franc': 'CHF',
            'rmb': 'CNY'
        }
    
    def _init_payment_mappings(self) -> Dict[str, int]:
        """Initialise les mappings de termes de paiement"""
        return {
            'immediate': 0,
            'cash': 0,
            'cod': 0,
            'net 30': 30,
            'net 45': 45,
            'net 60': 60,
            'net 90': 90,
            'net 120': 120,
            '30 days': 30,
            '45 days': 45,
            '60 days': 60,
            '90 days': 90,
            '120 days': 120,
            'eom': 30,  # End of month approximation
            '2/10 net 30': 30,  # With discount option
        }
    
    def _init_financial_keywords(self) -> Dict[str, List[str]]:
        """Initialise les mots-clés pour contexte financier"""
        return {
            'billing': ['invoice', 'facture', 'billing', 'facturation', 'bill', 'statement'],
            'payment': ['payment', 'paiement', 'settlement', 'règlement', 'pay', 'remittance'],
            'currency': ['currency', 'devise', 'eur', 'usd', 'gbp', 'exchange rate', 'taux de change'],
            'guarantee': ['guarantee', 'garantie', 'bond', 'caution', 'deposit', 'security'],
            'price': ['price', 'prix', 'cost', 'coût', 'tariff', 'rate', 'charge'],
            'revision': ['revision', 'révision', 'adjustment', 'ajustement', 'indexation', 'escalation'],
            'penalty': ['penalty', 'pénalité', 'fine', 'amende', 'interest', 'intérêt'],
            'discount': ['discount', 'remise', 'escompte', 'reduction', 'réduction']
        }
    
    # ============================================================================
    # EXTRACTION AVANCÉE SPÉCIALISÉE FINANCE
    # ============================================================================
    
    async def _extract_single_question(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Override: Extraction spécialisée pour questions financières
        """
        if not self.use_advanced:
            return await super()._extract_single_question(question_config, context, chunks)
        
        return await self._advanced_financial_extraction(question_config, context, chunks)
    
    async def _advanced_financial_extraction(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Extraction avancée avec stratégies spécialisées finance
        """
        start_time = datetime.now()
        question_id = question_config.id
        
        logger.info(f"💰 Extraction financière avancée pour: {question_id}")
        
        # 1. Identifier les sections financières pertinentes
        financial_sections = self._identify_financial_sections(chunks)
        
        # 2. Préparer contexte enrichi focalisé finance
        enriched_context = self._prepare_financial_context(
            context,
            financial_sections,
            question_id
        )
        
        # 3. Stratégies d'extraction parallèles
        extraction_tasks = []
        
        # Stratégie principale LLM
        extraction_tasks.append(
            self._strategy_financial_llm(question_config, enriched_context)
        )
        
        # Stratégie patterns pour termes standards
        extraction_tasks.append(
            self._strategy_pattern_extraction(question_config, chunks)
        )
        
        # Stratégie contractuelle pour clauses complexes
        if self._is_contractual_question(question_id):
            extraction_tasks.append(
                self._strategy_contractual_analysis(question_config, financial_sections)
            )
        
        # Stratégie calculation pour montants et pourcentages
        if self._requires_calculation(question_id):
            extraction_tasks.append(
                self._strategy_calculation_analysis(question_config, chunks)
            )
        
        # Exécuter toutes les stratégies
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # 4. Fusionner et valider
        best_result = self._merge_financial_results(results, question_config)
        
        # 5. Post-traitement financier
        best_result = self._post_process_financial(best_result, question_id)
        
        # 6. Analyse de risque
        best_result = self._analyze_financial_risk(best_result, question_id)
        
        best_result.extraction_time = (datetime.now() - start_time).total_seconds()
        
        return best_result
    
    # ============================================================================
    # STRATÉGIES D'EXTRACTION FINANCIÈRES
    # ============================================================================
    
    async def _strategy_financial_llm(
        self,
        question_config,
        context: str
    ) -> QuestionResult:
        """
        Stratégie 1: LLM avec prompt enrichi finance
        """
        # Enrichir le prompt avec guidance financière
        enhanced_prompt = self._enhance_financial_prompt(
            question_config.system_prompt,
            question_config.id
        )
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=enhanced_prompt),
                HumanMessage(content=context)
            ])
            
            parsed = self._parse_json_response(response.content)
            
            # Validation financière
            if self._validate_financial_response(parsed, question_config.id):
                result = self._create_question_result(
                    question_config.id,
                    parsed,
                    []
                )
                result.metadata['strategy'] = 'financial_llm'
                return result
                
        except Exception as e:
            logger.debug(f"Financial LLM extraction failed: {e}")
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="Financial LLM extraction failed"
        )
    
    async def _strategy_pattern_extraction(
        self,
        question_config,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Stratégie 2: Extraction par patterns financiers
        """
        question_id = question_config.id
        extracted_data = {}
        sources = []
        
        # Combiner le texte
        full_text = " ".join([chunk.page_content for chunk in chunks])
        
        # Sélectionner les patterns pertinents
        relevant_patterns = self._get_relevant_financial_patterns(question_id)
        
        for pattern_name, pattern in relevant_patterns.items():
            matches = re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                # Extraire avec contexte
                start = max(0, match.start() - 150)
                end = min(len(full_text), match.end() + 150)
                context_snippet = full_text[start:end].strip()
                
                # Parser selon le type de pattern
                if 'payment' in pattern_name:
                    value = self._parse_payment_term(match, context_snippet)
                elif 'currency' in pattern_name:
                    value = self._parse_currency(match, context_snippet)
                elif 'guarantee' in pattern_name:
                    value = self._parse_guarantee(match, context_snippet)
                elif 'revision' in pattern_name:
                    value = self._parse_price_revision(match, context_snippet)
                else:
                    value = match.group(0)
                
                if value:
                    if pattern_name not in extracted_data:
                        extracted_data[pattern_name] = []
                    extracted_data[pattern_name].append(value)
                    
                    # Ajouter la source
                    sources.append({
                        'document': 'extracted_text',
                        'context_snippet': context_snippet[:50]
                    })
        
        if extracted_data:
            # Structurer la réponse
            answer = self._structure_financial_data(extracted_data, question_id)
            
            return QuestionResult(
                question_id=question_id,
                answer=answer,
                sources=sources[:3],  # Limiter les sources
                status=ExtractionStatus.SUCCESS,
                confidence=0.85,
                metadata={'strategy': 'pattern', 'patterns_matched': len(extracted_data)}
            )
        
        return QuestionResult(
            question_id=question_id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No pattern matches"
        )
    
    async def _strategy_contractual_analysis(
        self,
        question_config,
        financial_sections: List[Document]
    ) -> QuestionResult:
        """
        Stratégie 3: Analyse contractuelle approfondie
        """
        if not financial_sections:
            return QuestionResult(
                question_id=question_config.id,
                answer=None,
                status=ExtractionStatus.FAILED,
                error="No financial sections found"
            )
        
        # Prompt spécialisé pour clauses contractuelles
        contract_prompt = f"""
        {question_config.system_prompt}
        
        ADDITIONAL CONTRACTUAL ANALYSIS:
        You are analyzing legal and contractual clauses. Pay special attention to:
        
        1. STANDARD CLAUSES vs SPECIAL CONDITIONS
        2. DEFINITIONS section (often contains key financial terms)
        3. PAYMENT AND INVOICING section
        4. WARRANTIES AND GUARANTEES section
        5. PRICE ADJUSTMENT mechanisms
        6. PENALTIES AND REMEDIES
        
        Look for both explicit statements AND implicit conditions.
        Check for cross-references to other sections or annexes.
        
        BE PRECISE about conditions, exceptions, and triggers.
        """
        
        # Contexte focalisé sur sections contractuelles
        contract_context = "\n---\n".join([
            s.page_content for s in financial_sections[:5]
        ])
        
        try:
            response = await self.contract_llm.ainvoke([
                SystemMessage(content=contract_prompt),
                HumanMessage(content=contract_context)
            ])
            
            parsed = self._parse_json_response(response.content)
            
            if parsed.get('answer'):
                result = self._create_question_result(
                    question_config.id,
                    parsed,
                    financial_sections[:3]
                )
                result.metadata['strategy'] = 'contractual_analysis'
                return result
                
        except Exception as e:
            logger.debug(f"Contractual analysis failed: {e}")
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="Contractual analysis failed"
        )
    
    async def _strategy_calculation_analysis(
        self,
        question_config,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Stratégie 4: Analyse avec calculs financiers
        """
        question_id = question_config.id
        
        # Extraire toutes les valeurs numériques financières
        financial_values = self._extract_financial_values(chunks)
        
        if not financial_values:
            return QuestionResult(
                question_id=question_id,
                answer=None,
                status=ExtractionStatus.FAILED,
                error="No financial values found"
            )
        
        # Prompt pour analyse avec calculs
        calc_prompt = f"""
        {question_config.system_prompt}
        
        CALCULATION INSTRUCTIONS:
        Based on these extracted financial values:
        {json.dumps(financial_values[:10], indent=2)}
        
        1. Identify which values are relevant to the question
        2. Perform any necessary calculations (percentages, totals, conversions)
        3. Validate the consistency of the values
        4. If there are conflicting values, explain the discrepancy
        
        Provide precise numerical answers with units.
        """
        
        try:
            response = await self.calculation_llm.ainvoke([
                SystemMessage(content=calc_prompt),
                HumanMessage(content="Analyze the financial values above")
            ])
            
            parsed = self._parse_json_response(response.content)
            
            if parsed.get('answer'):
                result = self._create_question_result(
                    question_id,
                    parsed,
                    []
                )
                result.metadata['strategy'] = 'calculation'
                result.metadata['values_analyzed'] = len(financial_values)
                return result
                
        except Exception as e:
            logger.debug(f"Calculation analysis failed: {e}")
        
        return QuestionResult(
            question_id=question_id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="Calculation analysis failed"
        )
    
    # ============================================================================
    # IDENTIFICATION ET PRÉPARATION DU CONTEXTE
    # ============================================================================
    
    def _identify_financial_sections(self, chunks: List[Document]) -> List[Document]:
        """
        Identifie les sections financières dans les documents
        """
        financial_sections = []
        
        # Keywords indiquant une section financière
        section_keywords = [
            'payment', 'invoice', 'billing', 'price', 'cost', 'guarantee',
            'financial', 'commercial', 'terms and conditions', 'pricing',
            'paiement', 'facturation', 'prix', 'garantie', 'conditions'
        ]
        
        for chunk in chunks:
            content_lower = chunk.page_content.lower()
            
            # Score de pertinence financière
            score = sum(1 for kw in section_keywords if kw in content_lower)
            
            # Bonus pour patterns financiers
            if re.search(self.financial_patterns['currency'], chunk.page_content):
                score += 2
            if re.search(self.financial_patterns['payment_days'], chunk.page_content):
                score += 2
            if re.search(self.financial_patterns['percentage'], chunk.page_content):
                score += 1
            
            if score >= 3:  # Seuil de pertinence
                financial_sections.append(chunk)
        
        # Trier par score (approximatif basé sur la position)
        return financial_sections
    
    def _prepare_financial_context(
        self,
        base_context: str,
        financial_sections: List[Document],
        question_id: str
    ) -> str:
        """
        Prépare un contexte enrichi pour questions financières
        """
        # Keywords spécifiques à la question
        question_keywords = self._get_question_keywords(question_id)
        
        # Commencer avec les sections financières identifiées
        context_parts = []
        
        # Ajouter les sections financières les plus pertinentes
        for section in financial_sections[:3]:
            context_parts.append(f"[Financial Section]\n{section.page_content}")
        
        # Ajouter des parties du contexte de base qui matchent les keywords
        if question_keywords:
            for line in base_context.split('\n'):
                if any(kw in line.lower() for kw in question_keywords):
                    context_parts.append(f"[Relevant Context]\n{line}")
        
        # Limiter la taille totale
        enriched = "\n---\n".join(context_parts)
        if len(enriched) > 12000:
            enriched = enriched[:12000] + "\n[...truncated]"
        
        return enriched
    
    # ============================================================================
    # ENRICHISSEMENT DES PROMPTS
    # ============================================================================
    
    def _enhance_financial_prompt(self, original_prompt: str, question_id: str) -> str:
        """
        Enrichit le prompt avec guidance financière spécifique
        """
        enhancements = {
            'billing_terms': """
                FINANCIAL EXTRACTION GUIDANCE:
                - Look for: "Invoicing", "Billing", "Facturation" sections
                - Common patterns: "monthly invoicing", "invoice per shipment", "consolidated billing"
                - Check for split billing instructions (per entity, per country)
                - Note any specific invoice formats or submission requirements
                - Look for billing contact details and addresses
                """,
            
            'payment_terms': """
                FINANCIAL EXTRACTION GUIDANCE:
                - Standard formats: "Net 30", "30 days", "payment within X days"
                - Check for: payment base date (invoice date vs receipt date)
                - Look for penalties: late payment interest, penalty rates
                - Early payment discounts: "2/10 net 30" means 2% discount if paid in 10 days
                - Special conditions: retention amounts, milestone payments
                """,
            
            'currency': """
                FINANCIAL EXTRACTION GUIDANCE:
                - Currency symbols: €, $, £, ¥
                - Currency codes: EUR, USD, GBP, CHF, JPY, CNY
                - Exchange rate mentions: "fixed rate", "ECB rate", "monthly average"
                - Hedging requirements: "currency risk", "forward contracts"
                - Multi-currency scenarios: different currencies for different services
                """,
            
            'guarantee': """
                FINANCIAL EXTRACTION GUIDANCE:
                - Types: bank guarantee, performance bond, parent company guarantee, deposit
                - Amounts: fixed amounts (€100,000) or percentages (10% of contract value)
                - Validity: "until contract completion", "renewable annually"
                - Release conditions: "upon successful delivery", "after warranty period"
                - Format: "first demand", "unconditional", "irrevocable"
                """,
            
            'price_revision': """
                FINANCIAL EXTRACTION GUIDANCE:
                - Fuel surcharge: BAF, bunker adjustment, diesel index
                - Inflation: CPI, ICC, consumer price index
                - Formulas: "P1 = P0 × (0.5 × I1/I0 + 0.5)"
                - Frequency: "quarterly", "annual", "monthly adjustment"
                - Thresholds: "if variation exceeds 3%"
                - Caps and floors: maximum/minimum price changes
                """
        }
        
        # Trouver l'enhancement approprié
        for key, enhancement in enhancements.items():
            if key in question_id:
                return original_prompt + "\n\n" + enhancement
        
        # Enhancement générique si pas de match spécifique
        return original_prompt + """
        
        GENERAL FINANCIAL GUIDANCE:
        - Check "Commercial Terms", "Financial Conditions", "Payment Terms" sections
        - Look for numbered clauses (e.g., "Article 5: Payment")
        - Financial terms may be in annexes or appendices
        - Be precise with numbers, percentages, and currencies
        """
    
    # ============================================================================
    # PARSING ET STRUCTURATION DES DONNÉES
    # ============================================================================
    
    def _parse_payment_term(self, match, context: str) -> Dict[str, Any]:
        """
        Parse un terme de paiement
        """
        result = {
            'raw_term': match.group(0),
            'days': None,
            'type': None,
            'conditions': []
        }
        
        # Extraire le nombre de jours
        if match.groups():
            try:
                result['days'] = int(match.group(1))
            except (ValueError, IndexError):
                pass
        
        # Déterminer le type
        context_lower = context.lower()
        if 'net' in context_lower:
            result['type'] = 'net'
        elif 'end of month' in context_lower or 'eom' in context_lower:
            result['type'] = 'end_of_month'
        elif 'receipt' in context_lower:
            result['type'] = 'from_receipt'
        else:
            result['type'] = 'from_invoice'
        
        # Chercher des conditions spéciales
        if 'discount' in context_lower or 'escompte' in context_lower:
            discount_match = re.search(self.financial_patterns['discount'], context)
            if discount_match:
                result['conditions'].append(f"Early payment discount: {discount_match.group(1)}%")
        
        if 'penalty' in context_lower or 'pénalité' in context_lower:
            penalty_match = re.search(self.financial_patterns['penalty'], context)
            if penalty_match:
                result['conditions'].append(f"Late payment penalty: {penalty_match.group(1)}%")
        
        return result
    
    def _parse_currency(self, match, context: str) -> Dict[str, Any]:
        """
        Parse une information de devise
        """
        result = {
            'currency': None,
            'amount': None,
            'exchange_rate': None,
            'conditions': []
        }
        
        # Normaliser la devise
        currency_text = match.group(0)
        for symbol, code in self.currency_mappings.items():
            if symbol in currency_text:
                result['currency'] = code
                break
        
        if not result['currency']:
            # Chercher les codes directement
            for code in ['EUR', 'USD', 'GBP', 'CHF', 'JPY', 'CNY']:
                if code in currency_text:
                    result['currency'] = code
                    break
        
        # Chercher un montant associé
        amount_match = re.search(r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{2})?)', context)
        if amount_match:
            result['amount'] = self._parse_amount(amount_match.group(1))
        
        # Chercher des mentions de taux de change
        if 'exchange' in context.lower() or 'taux' in context.lower():
            if 'fixed' in context.lower() or 'fixe' in context.lower():
                result['exchange_rate'] = 'fixed'
            elif 'ecb' in context.lower() or 'bce' in context.lower():
                result['exchange_rate'] = 'ECB reference'
            elif 'monthly' in context.lower():
                result['exchange_rate'] = 'monthly average'
        
        return result
    
    def _parse_guarantee(self, match, context: str) -> Dict[str, Any]:
        """
        Parse une garantie financière
        """
        result = {
            'type': None,
            'amount': None,
            'percentage': None,
            'validity': None,
            'conditions': []
        }
        
        context_lower = context.lower()
        
        # Type de garantie
        if 'bank guarantee' in context_lower or 'garantie bancaire' in context_lower:
            result['type'] = 'bank_guarantee'
        elif 'performance bond' in context_lower:
            result['type'] = 'performance_bond'
        elif 'deposit' in context_lower or 'dépôt' in context_lower:
            result['type'] = 'deposit'
        elif 'parent' in context_lower:
            result['type'] = 'parent_company_guarantee'
        
        # Montant ou pourcentage
        amount_match = re.search(self.financial_patterns['guarantee_amount'], context)
        if amount_match:
            result['amount'] = self._parse_amount(amount_match.group(1))
        
        percentage_match = re.search(self.financial_patterns['guarantee_percentage'], context)
        if percentage_match:
            result['percentage'] = float(percentage_match.group(1).replace(',', '.'))
        
        # Validité
        if 'annual' in context_lower:
            result['validity'] = 'annual'
        elif 'contract completion' in context_lower:
            result['validity'] = 'until_completion'
        elif 'warranty period' in context_lower:
            result['validity'] = 'including_warranty'
        
        # Conditions
        if 'first demand' in context_lower or 'première demande' in context_lower:
            result['conditions'].append('first_demand')
        if 'unconditional' in context_lower:
            result['conditions'].append('unconditional')
        if 'irrevocable' in context_lower:
            result['conditions'].append('irrevocable')
        
        return result
    
    def _parse_price_revision(self, match, context: str) -> Dict[str, Any]:
        """
        Parse une clause de révision de prix
        """
        result = {
            'type': None,
            'frequency': None,
            'index': None,
            'formula': None,
            'threshold': None,
            'cap': None
        }
        
        context_lower = context.lower()
        
        # Type de révision
        if 'fuel' in context_lower or 'baf' in context_lower or 'bunker' in context_lower:
            result['type'] = 'fuel_surcharge'
        elif 'inflation' in context_lower or 'cpi' in context_lower or 'icc' in context_lower:
            result['type'] = 'inflation_indexation'
        elif 'cost' in context_lower:
            result['type'] = 'cost_adjustment'
        else:
            result['type'] = 'price_revision'
        
        # Fréquence
        if 'monthly' in context_lower or 'mensuel' in context_lower:
            result['frequency'] = 'monthly'
        elif 'quarterly' in context_lower or 'trimestriel' in context_lower:
            result['frequency'] = 'quarterly'
        elif 'annual' in context_lower or 'annuel' in context_lower:
            result['frequency'] = 'annual'
        
        # Index utilisé
        if 'cpi' in context_lower:
            result['index'] = 'CPI'
        elif 'icc' in context_lower:
            result['index'] = 'ICC'
        elif 'platts' in context_lower:
            result['index'] = 'Platts'
        elif 'gasoil' in context_lower or 'diesel' in context_lower:
            result['index'] = 'Diesel_index'
        
        # Seuil de déclenchement
        threshold_match = re.search(r'(?:exceed|dépasse|threshold|seuil).*?(\d+)\s*%', context)
        if threshold_match:
            result['threshold'] = float(threshold_match.group(1))
        
        # Cap/plafond
        cap_match = re.search(r'(?:maximum|cap|plafond).*?(\d+)\s*%', context)
        if cap_match:
            result['cap'] = float(cap_match.group(1))
        
        return result
    
    def _parse_amount(self, amount_str: str) -> float:
        """
        Parse un montant financier
        """
        try:
            # Nettoyer la chaîne
            cleaned = amount_str.replace(' ', '').replace(',', '.')
            
            # Gérer les multiplicateurs
            if 'M' in amount_str or 'million' in amount_str.lower():
                return float(cleaned.replace('M', '')) * 1000000
            elif 'K' in amount_str or 'thousand' in amount_str.lower():
                return float(cleaned.replace('K', '')) * 1000
            else:
                return float(cleaned)
        except (ValueError, AttributeError):
            return 0.0
    
    # ============================================================================
    # FUSION ET VALIDATION
    # ============================================================================
    
    def _merge_financial_results(
        self,
        results: List,
        question_config
    ) -> QuestionResult:
        """
        Fusionne les résultats financiers avec validation
        """
        valid_results = []
        
        for result in results:
            if isinstance(result, QuestionResult) and result.status == ExtractionStatus.SUCCESS:
                valid_results.append(result)
        
        if not valid_results:
            return QuestionResult(
                question_id=question_config.id,
                answer=None,
                status=ExtractionStatus.FAILED,
                confidence=0.0,
                error="No successful extraction"
            )
        
        # Sélectionner le meilleur résultat
        best_result = self._select_best_financial_result(valid_results, question_config.id)
        
        # Enrichir avec les données des autres résultats
        if len(valid_results) > 1:
            best_result = self._enrich_with_alternative_data(best_result, valid_results)
        
        best_result.metadata['strategies_used'] = len(valid_results)
        
        return best_result
    
    def _select_best_financial_result(
        self,
        results: List[QuestionResult],
        question_id: str
    ) -> QuestionResult:
        """
        Sélectionne le meilleur résultat financier
        """
        scored_results = []
        
        for result in results:
            score = 0
            
            # Score basé sur la confiance
            score += result.confidence * 40
            
            # Score basé sur la complétude
            if result.answer:
                if isinstance(result.answer, dict):
                    # Plus de champs = meilleur
                    score += min(len(result.answer) * 5, 30)
                elif isinstance(result.answer, str):
                    # Réponse plus longue = plus détaillée
                    score += min(len(result.answer) / 10, 20)
            
            # Bonus pour certaines stratégies selon la question
            if 'payment' in question_id and result.metadata.get('strategy') == 'pattern':
                score += 10
            elif 'guarantee' in question_id and result.metadata.get('strategy') == 'contractual_analysis':
                score += 10
            elif 'price_revision' in question_id and result.metadata.get('strategy') == 'financial_llm':
                score += 10
            
            scored_results.append((score, result))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[0][1]
    
    def _enrich_with_alternative_data(
        self,
        best_result: QuestionResult,
        all_results: List[QuestionResult]
    ) -> QuestionResult:
        """
        Enrichit le meilleur résultat avec des données alternatives
        """
        # Collecter toutes les données uniques
        all_data = {}
        
        for result in all_results:
            if result.answer and result != best_result:
                if isinstance(result.answer, dict):
                    for key, value in result.answer.items():
                        if key not in all_data:
                            all_data[key] = []
                        if value not in all_data[key]:
                            all_data[key].append(value)
        
        # Ajouter aux métadonnées
        if all_data:
            best_result.metadata['alternative_extractions'] = all_data
        
        return best_result
    
    # ============================================================================
    # POST-TRAITEMENT ET ANALYSE DE RISQUE
    # ============================================================================
    
    def _post_process_financial(self, result: QuestionResult, question_id: str) -> QuestionResult:
        """
        Post-traitement spécifique finance
        """
        if not result.answer:
            return result
        
        # Normalisation selon le type de question
        if 'payment_terms' in question_id:
            result.answer = self._normalize_payment_terms(result.answer)
        elif 'currency' in question_id:
            result.answer = self._normalize_currency_info(result.answer)
        elif 'guarantee' in question_id:
            result.answer = self._normalize_guarantee_info(result.answer)
        elif 'price_revision' in question_id:
            result.answer = self._normalize_price_revision(result.answer)
        
        return result
    
    def _normalize_payment_terms(self, answer: Any) -> Any:
        """
        Normalise les termes de paiement
        """
        if isinstance(answer, str):
            # Essayer de parser en jours
            days_match = re.search(r'(\d+)\s*(?:days?|jours?)', answer)
            if days_match:
                return {
                    'days': int(days_match.group(1)),
                    'raw_term': answer,
                    'normalized': f"Net {days_match.group(1)}"
                }
        elif isinstance(answer, dict):
            # Ajouter une forme normalisée
            if 'days' in answer:
                answer['normalized'] = f"Net {answer['days']}"
        
        return answer
    
    def _normalize_currency_info(self, answer: Any) -> Any:
        """
        Normalise les informations de devise
        """
        if isinstance(answer, str):
            # Détecter et normaliser la devise
            for symbol, code in self.currency_mappings.items():
                if symbol in answer.lower():
                    return {
                        'currency': code,
                        'raw_text': answer
                    }
            
            # Chercher les codes directement
            for code in ['EUR', 'USD', 'GBP', 'CHF', 'JPY', 'CNY']:
                if code in answer.upper():
                    return {
                        'currency': code,
                        'raw_text': answer
                    }
        
        return answer
    
    def _normalize_guarantee_info(self, answer: Any) -> Any:
        """
        Normalise les informations de garantie
        """
        if isinstance(answer, str):
            normalized = {
                'raw_text': answer,
                'type': 'unspecified',
                'amount': None
            }
            
            # Détecter le type
            answer_lower = answer.lower()
            if 'bank' in answer_lower:
                normalized['type'] = 'bank_guarantee'
            elif 'performance' in answer_lower:
                normalized['type'] = 'performance_bond'
            
            # Détecter le montant
            amount_match = re.search(r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{2})?)', answer)
            if amount_match:
                normalized['amount'] = self._parse_amount(amount_match.group(1))
            
            return normalized
        
        return answer
    
    def _normalize_price_revision(self, answer: Any) -> Any:
        """
        Normalise les clauses de révision de prix
        """
        if isinstance(answer, str):
            normalized = {
                'raw_text': answer,
                'has_revision': False,
                'type': None
            }
            
            answer_lower = answer.lower()
            
            # Détecter si révision existe
            if any(kw in answer_lower for kw in ['revision', 'adjustment', 'indexation', 'surcharge']):
                normalized['has_revision'] = True
                
                # Déterminer le type
                if 'fuel' in answer_lower or 'baf' in answer_lower:
                    normalized['type'] = 'fuel_surcharge'
                elif 'inflation' in answer_lower or 'cpi' in answer_lower:
                    normalized['type'] = 'inflation'
                else:
                    normalized['type'] = 'general_revision'
            elif 'fixed' in answer_lower or 'fixe' in answer_lower:
                normalized['has_revision'] = False
                normalized['type'] = 'fixed_price'
            
            return normalized
        
        return answer
    
    def _analyze_financial_risk(self, result: QuestionResult, question_id: str) -> QuestionResult:
        """
        Analyse le risque financier de la réponse
        """
        risk_assessment = {
            'level': 'low',
            'factors': [],
            'score': 0
        }
        
        if not result.answer:
            risk_assessment['level'] = 'unknown'
            risk_assessment['factors'].append('No data available')
        else:
            # Analyse selon le type de question
            if 'payment_terms' in question_id:
                risk = self._assess_payment_risk(result.answer)
                risk_assessment.update(risk)
            elif 'currency' in question_id:
                risk = self._assess_currency_risk(result.answer)
                risk_assessment.update(risk)
            elif 'guarantee' in question_id:
                risk = self._assess_guarantee_risk(result.answer)
                risk_assessment.update(risk)
            elif 'price_revision' in question_id:
                risk = self._assess_price_revision_risk(result.answer)
                risk_assessment.update(risk)
        
        result.metadata['risk_assessment'] = risk_assessment
        
        return result
    
    def _assess_payment_risk(self, payment_terms: Any) -> Dict:
        """
        Évalue le risque des termes de paiement
        """
        risk = {'level': 'low', 'factors': [], 'score': 0}
        
        days = None
        if isinstance(payment_terms, dict):
            days = payment_terms.get('days')
        elif isinstance(payment_terms, str):
            match = re.search(r'(\d+)', payment_terms)
            if match:
                days = int(match.group(1))
        
        if days:
            if days > 90:
                risk['score'] += 3
                risk['factors'].append(f'Long payment term: {days} days')
            elif days > 60:
                risk['score'] += 2
                risk['factors'].append(f'Extended payment term: {days} days')
            elif days < 30:
                risk['score'] += 1
                risk['factors'].append(f'Short payment term: {days} days')
        
        # Déterminer le niveau
        if risk['score'] >= 3:
            risk['level'] = 'high'
        elif risk['score'] >= 2:
            risk['level'] = 'medium'
        
        return risk
    
    def _assess_currency_risk(self, currency_info: Any) -> Dict:
        """
        Évalue le risque de change
        """
        risk = {'level': 'low', 'factors': [], 'score': 0}
        
        currency = None
        exchange_rate = None
        
        if isinstance(currency_info, dict):
            currency = currency_info.get('currency')
            exchange_rate = currency_info.get('exchange_rate')
        elif isinstance(currency_info, str):
            if 'EUR' in currency_info:
                currency = 'EUR'
            elif 'USD' in currency_info:
                currency = 'USD'
        
        # Risque selon la devise
        if currency and currency != 'EUR':  # Assumant société basée en zone Euro
            risk['score'] += 2
            risk['factors'].append(f'Foreign currency: {currency}')
        
        # Risque selon le taux de change
        if exchange_rate:
            if exchange_rate == 'fixed':
                risk['score'] -= 1
                risk['factors'].append('Fixed exchange rate (reduced risk)')
            elif 'monthly' in str(exchange_rate).lower():
                risk['score'] += 1
                risk['factors'].append('Variable exchange rate')
        
        # Déterminer le niveau
        if risk['score'] >= 3:
            risk['level'] = 'high'
        elif risk['score'] >= 1:
            risk['level'] = 'medium'
        
        return risk
    
    def _assess_guarantee_risk(self, guarantee_info: Any) -> Dict:
        """
        Évalue le risque lié aux garanties
        """
        risk = {'level': 'low', 'factors': [], 'score': 0}
        
        if isinstance(guarantee_info, dict):
            amount = guarantee_info.get('amount')
            percentage = guarantee_info.get('percentage')
            
            if amount and amount > 100000:
                risk['score'] += 2
                risk['factors'].append(f'High guarantee amount: {amount}')
            elif percentage and percentage > 10:
                risk['score'] += 2
                risk['factors'].append(f'High guarantee percentage: {percentage}%')
            
            if 'first_demand' in guarantee_info.get('conditions', []):
                risk['score'] += 1
                risk['factors'].append('First demand guarantee')
        elif isinstance(guarantee_info, str):
            if 'guarantee' in guarantee_info.lower() or 'bond' in guarantee_info.lower():
                risk['score'] += 1
                risk['factors'].append('Guarantee required')
        
        # Déterminer le niveau
        if risk['score'] >= 3:
            risk['level'] = 'high'
        elif risk['score'] >= 2:
            risk['level'] = 'medium'
        
        return risk
    
    def _assess_price_revision_risk(self, revision_info: Any) -> Dict:
        """
        Évalue le risque des clauses de révision
        """
        risk = {'level': 'low', 'factors': [], 'score': 0}
        
        if isinstance(revision_info, dict):
            if revision_info.get('has_revision'):
                revision_type = revision_info.get('type')
                
                if revision_type == 'fuel_surcharge':
                    risk['score'] += 2
                    risk['factors'].append('Fuel price volatility risk')
                elif revision_type == 'inflation':
                    risk['score'] += 1
                    risk['factors'].append('Inflation adjustment')
                
                if not revision_info.get('cap'):
                    risk['score'] += 1
                    risk['factors'].append('No cap on price increases')
            elif revision_info.get('type') == 'fixed_price':
                risk['score'] += 1
                risk['factors'].append('Fixed price (no adjustment possible)')
        elif isinstance(revision_info, str):
            if 'fixed' in revision_info.lower():
                risk['score'] += 1
                risk['factors'].append('Fixed price risk')
            elif 'revision' in revision_info.lower():
                risk['score'] += 1
                risk['factors'].append('Price revision clause present')
        
        # Déterminer le niveau
        if risk['score'] >= 3:
            risk['level'] = 'high'
        elif risk['score'] >= 2:
            risk['level'] = 'medium'
        
        return risk
    
    # ============================================================================
    # MÉTHODES UTILITAIRES
    # ============================================================================
    
    def _get_relevant_financial_patterns(self, question_id: str) -> Dict[str, str]:
        """
        Retourne les patterns pertinents pour une question
        """
        pattern_mapping = {
            'billing': ['billing_frequency', 'invoice_split'],
            'payment': ['payment_days', 'payment_net', 'payment_eom', 'payment_terms_complex'],
            'currency': ['currency', 'currency_amount'],
            'guarantee': ['guarantee_amount', 'guarantee_percentage', 'performance_bond'],
            'price_revision': ['fuel_surcharge', 'indexation', 'price_revision', 'escalation'],
            'penalty': ['penalty', 'interest_rate'],
            'discount': ['discount', 'percentage']
        }
        
        relevant = {}
        for key, patterns in pattern_mapping.items():
            if key in question_id.lower():
                for pattern in patterns:
                    if pattern in self.financial_patterns:
                        relevant[pattern] = self.financial_patterns[pattern]
        
        # Si aucun match spécifique, retourner les patterns de base
        if not relevant:
            return {
                k: v for k, v in self.financial_patterns.items()
                if k in ['payment_days', 'currency', 'percentage', 'amount_thousands']
            }
        
        return relevant
    
    def _get_question_keywords(self, question_id: str) -> List[str]:
        """
        Retourne les mots-clés pour une question
        """
        for category, keywords in self.financial_keywords.items():
            if category in question_id.lower():
                return keywords
        
        # Keywords génériques
        return ['payment', 'price', 'cost', 'invoice', 'financial']
    
    def _is_contractual_question(self, question_id: str) -> bool:
        """
        Détermine si la question nécessite une analyse contractuelle
        """
        contractual_keywords = [
            'guarantee', 'bond', 'warranty', 'liability',
            'penalty', 'remedy', 'termination', 'clause'
        ]
        return any(kw in question_id.lower() for kw in contractual_keywords)
    
    def _requires_calculation(self, question_id: str) -> bool:
        """
        Détermine si la question nécessite des calculs
        """
        calculation_keywords = [
            'amount', 'percentage', 'total', 'sum',
            'calculation', 'formula', 'rate', 'ratio'
        ]
        return any(kw in question_id.lower() for kw in calculation_keywords)
    
    def _structure_financial_data(self, extracted_data: Dict, question_id: str) -> Any:
        """
        Structure les données financières extraites
        """
        # Si une seule valeur extraite
        if len(extracted_data) == 1:
            key = list(extracted_data.keys())[0]
            values = extracted_data[key]
            if len(values) == 1:
                return values[0]
        
        # Structurer selon le type de question
        if 'payment' in question_id:
            return self._structure_payment_data(extracted_data)
        elif 'currency' in question_id:
            return self._structure_currency_data(extracted_data)
        elif 'guarantee' in question_id:
            return self._structure_guarantee_data(extracted_data)
        elif 'price_revision' in question_id:
            return self._structure_revision_data(extracted_data)
        
        # Retour par défaut
        return extracted_data
    
    def _structure_payment_data(self, data: Dict) -> Dict:
        """
        Structure les données de paiement
        """
        structured = {
            'payment_terms': None,
            'conditions': [],
            'penalties': None,
            'discounts': None
        }
        
        # Termes principaux
        if 'payment_days' in data:
            structured['payment_terms'] = data['payment_days'][0] if data['payment_days'] else None
        elif 'payment_net' in data:
            structured['payment_terms'] = data['payment_net'][0] if data['payment_net'] else None
        
        # Pénalités
        if 'penalty' in data:
            structured['penalties'] = data['penalty']
        
        # Remises
        if 'discount' in data:
            structured['discounts'] = data['discount']
        
        return structured
    
    def _structure_currency_data(self, data: Dict) -> Dict:
        """
        Structure les données de devise
        """
        structured = {
            'currencies': [],
            'exchange_rate': None,
            'amounts': []
        }
        
        if 'currency' in data:
            structured['currencies'] = list(set(data['currency']))
        
        if 'currency_amount' in data:
            structured['amounts'] = data['currency_amount']
        
        return structured
    
    def _structure_guarantee_data(self, data: Dict) -> Dict:
        """
        Structure les données de garantie
        """
        structured = {
            'type': None,
            'amount': None,
            'percentage': None,
            'conditions': []
        }
        
        if 'performance_bond' in data:
            structured['type'] = 'performance_bond'
        elif 'guarantee_amount' in data or 'guarantee_percentage' in data:
            structured['type'] = 'bank_guarantee'
        
        if 'guarantee_amount' in data:
            structured['amount'] = data['guarantee_amount'][0] if data['guarantee_amount'] else None
        
        if 'guarantee_percentage' in data:
            structured['percentage'] = data['guarantee_percentage'][0] if data['guarantee_percentage'] else None
        
        return structured
    
    def _structure_revision_data(self, data: Dict) -> Dict:
        """
        Structure les données de révision de prix
        """
        structured = {
            'has_revision': False,
            'types': [],
            'frequency': None,
            'indices': []
        }
        
        revision_indicators = ['fuel_surcharge', 'indexation', 'price_revision', 'escalation']
        
        for indicator in revision_indicators:
            if indicator in data:
                structured['has_revision'] = True
                structured['types'].append(indicator)
        
        return structured
    
    def _extract_financial_values(self, chunks: List[Document]) -> List[Dict]:
        """
        Extrait toutes les valeurs financières des chunks
        """
        financial_values = []
        
        for chunk in chunks:
            # Pattern pour montants avec devise
            amount_pattern = r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{2})?)\s*(EUR|USD|GBP|€|\$|£|M|K)?'
            matches = re.finditer(amount_pattern, chunk.page_content)
            
            for match in matches:
                value_str = match.group(1)
                unit = match.group(2) if match.group(2) else ''
                
                # Parser la valeur
                value = self._parse_amount(value_str)
                
                if value > 0:  # Ignorer les zéros
                    # Extraire le contexte
                    start = max(0, match.start() - 50)
                    end = min(len(chunk.page_content), match.end() + 50)
                    context = chunk.page_content[start:end]
                    
                    financial_values.append({
                        'value': value,
                        'unit': unit,
                        'context': context,
                        'source': chunk.metadata.get('source', 'unknown')
                    })
        
        return financial_values
    
    def _validate_financial_response(self, parsed: Dict, question_id: str) -> bool:
        """
        Validation spécifique pour réponses financières
        """
        if not parsed or 'answer' not in parsed:
            return False
        
        answer = parsed['answer']
        
        # Validation selon le type de question
        if 'payment' in question_id:
            # Doit contenir une indication de délai
            if isinstance(answer, str):
                return bool(re.search(r'\d+', answer))
            elif isinstance(answer, dict):
                return 'days' in answer or 'payment_terms' in answer
        
        elif 'currency' in question_id:
            # Doit contenir une devise valide
            valid_currencies = ['EUR', 'USD', 'GBP', 'CHF', 'JPY', 'CNY']
            if isinstance(answer, str):
                return any(curr in answer.upper() for curr in valid_currencies)
            elif isinstance(answer, dict):
                return 'currency' in answer
        
        elif 'guarantee' in question_id:
            # Doit indiquer présence ou absence de garantie
            return answer is not None
        
        elif 'price_revision' in question_id:
            # Doit indiquer si révision existe ou non
            return answer is not None
        
        # Validation générique
        return answer is not None and answer != ""