# agents/volume_agent.py
"""
Agent de volumes optimisé pour extraction maximale via LLM
Spécialisé dans l'extraction de toutes les métriques volumétriques
"""
import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from .base_agent import YAMLBaseAgent, QuestionResult, ExtractionStatus

logger = logging.getLogger(__name__)

class VolumeExtractionAgent(YAMLBaseAgent):
    """
    Agent d'extraction de volumes avec stratégies optimisées pour métriques
    
    Spécialisations:
    - Extraction de volumes multi-formats (TEU, tonnage, m³, palettes)
    - Détection de tableaux et données structurées
    - Normalisation et conversion d'unités
    - Agrégation et calculs automatiques
    - Détection de patterns complexes
    """
    
    def __init__(
        self,
        config_path: str = "config/prompts/volume_questions.yaml",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        use_advanced_extraction: bool = True
    ):
        """
        Initialise l'agent de volumes avec capacités avancées
        
        Args:
            config_path: Chemin vers le fichier YAML des questions volumes
            model: Modèle LLM optimisé pour l'extraction numérique
            temperature: Basse température pour précision numérique
            enable_cache: Cache des résultats
            enable_parallel: Extraction parallèle
            use_advanced_extraction: Stratégies avancées pour tableaux et calculs
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
        
        # LLM spécialisé pour l'extraction de tableaux
        if self.use_advanced:
            self.table_extraction_llm = ChatOpenAI(
                model=model,
                temperature=0.0,  # Zéro pour précision maximale sur les nombres
                max_tokens=4000
            )
        
        # Patterns regex pour extraction numérique
        self.volume_patterns = self._init_volume_patterns()
        
        # Conversions d'unités
        self.unit_conversions = self._init_unit_conversions()
        
        # Keywords pour scoring de pertinence
        self.volume_keywords = self._init_volume_keywords()
        
        logger.info(f"✅ VolumeExtractionAgent initialisé (mode {'avancé' if use_advanced_extraction else 'standard'})")
    
    # ============================================================================
    # INITIALISATION DES PATTERNS ET CONVERSIONS
    # ============================================================================
    
    def _init_volume_patterns(self) -> Dict[str, str]:
        """Initialise les patterns regex pour volumes"""
        return {
            # Conteneurs
            'teu': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:TEU[sS]?|EVP)',
            'feu': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:FEU[sS]?|forty[\s-]?foot)',
            'containers_20': r'(\d{1,3}(?:[,.\s]\d{3})*)\s*(?:x\s*)?20[\'\"]?\s*(?:ft|foot|pieds?)?',
            'containers_40': r'(\d{1,3}(?:[,.\s]\d{3})*)\s*(?:x\s*)?40[\'\"]?\s*(?:ft|foot|pieds?)?',
            
            # Poids
            'tonnes': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:tonnes?|tons?|MT|t\b)',
            'kg': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:kg|kilos?|kilogrammes?)',
            'pounds': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:lbs?|pounds?)',
            
            # Volume
            'm3': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:m³|m3|cubic\s*meters?|mètres?\s*cubes?|CBM)',
            'liters': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:L|liters?|litres?)',
            
            # Palettes
            'pallets': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:pallets?|palettes?|PAL)',
            'eur_pallets': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:EUR[\s-]?pallets?|euro[\s-]?palettes?)',
            
            # Unités
            'units': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:units?|pièces?|pcs?|items?)',
            'packages': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:packages?|colis|parcels?)',
            'cartons': r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*(?:cartons?|boxes?|caisses?)',
            
            # Périodes
            'per_year': r'(?:per|par|\/)\s*(?:year|an|année|annum)',
            'per_month': r'(?:per|par|\/)\s*(?:month|mois)',
            'per_week': r'(?:per|par|\/)\s*(?:week|semaine)',
            'per_day': r'(?:per|par|\/)\s*(?:day|jour)',
            
            # Ranges
            'range': r'(\d{1,3}(?:[,.\s]\d{3})*)\s*(?:à|-|to|–)\s*(\d{1,3}(?:[,.\s]\d{3})*)',
            
            # Pourcentages et croissance
            'percentage': r'(\d{1,3}(?:[.,]\d+)?)\s*%',
            'growth': r'(?:growth|croissance|increase|augmentation).*?(\d{1,3}(?:[.,]\d+)?)\s*%'
        }
    
    def _init_unit_conversions(self) -> Dict[str, Dict[str, float]]:
        """Initialise les facteurs de conversion d'unités"""
        return {
            'weight': {
                'kg_to_tonnes': 0.001,
                'pounds_to_kg': 0.453592,
                'tonnes_to_kg': 1000,
                'mt_to_tonnes': 1
            },
            'volume': {
                'liters_to_m3': 0.001,
                'm3_to_liters': 1000,
                'cubic_feet_to_m3': 0.0283168
            },
            'containers': {
                'feu_to_teu': 2,
                'teu_to_feu': 0.5
            },
            'time': {
                'yearly_to_monthly': 1/12,
                'monthly_to_yearly': 12,
                'weekly_to_yearly': 52,
                'daily_to_yearly': 365
            }
        }
    
    def _init_volume_keywords(self) -> Dict[str, List[str]]:
        """Initialise les mots-clés pour identification de contexte"""
        return {
            'container_context': ['container', 'conteneur', 'teu', 'feu', 'twenty-foot', 'forty-foot', 'evp'],
            'weight_context': ['weight', 'poids', 'tonnage', 'tonnes', 'kg', 'kilogram', 'mass'],
            'volume_context': ['volume', 'm3', 'cubic', 'cbm', 'capacity', 'capacité'],
            'pallet_context': ['pallet', 'palette', 'eur', 'loading unit'],
            'shipment_context': ['shipment', 'expedition', 'envoi', 'delivery', 'livraison'],
            'forecast_context': ['forecast', 'prévision', 'estimated', 'expected', 'projected'],
            'guaranteed_context': ['guaranteed', 'garanti', 'committed', 'minimum', 'contractual']
        }
    
    # ============================================================================
    # EXTRACTION AVANCÉE SPÉCIALISÉE VOLUMES
    # ============================================================================
    
    async def _extract_single_question(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Override: Extraction spécialisée pour questions de volumes
        """
        if not self.use_advanced:
            return await super()._extract_single_question(question_config, context, chunks)
        
        # Stratégie multi-approches pour volumes
        return await self._advanced_volume_extraction(question_config, context, chunks)
    
    async def _advanced_volume_extraction(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Extraction avancée avec stratégies spécialisées volumes
        """
        start_time = datetime.now()
        question_id = question_config.id
        
        logger.info(f"📊 Extraction avancée de volumes pour: {question_id}")
        
        # 1. Détecter et extraire les tableaux en premier
        table_results = await self._extract_from_tables(chunks)
        
        # 2. Préparer contexte enrichi avec tableaux extraits
        enriched_context = self._enrich_context_with_tables(context, table_results)
        
        # 3. Stratégies d'extraction parallèles
        extraction_tasks = [
            self._strategy_llm_extraction(question_config, enriched_context),
            self._strategy_pattern_extraction(question_config, chunks),
            self._strategy_calculation_extraction(question_config, enriched_context, table_results)
        ]
        
        # Ajouter extraction numérique pure si pertinent
        if self._is_numeric_question(question_id):
            extraction_tasks.append(
                self._strategy_numeric_focused(question_config, chunks)
            )
        
        # Exécuter toutes les stratégies
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # 4. Fusionner et valider les résultats numériques
        best_result = self._merge_volume_results(results, question_config)
        
        # 5. Post-traitement spécifique volumes
        best_result = self._post_process_volume_result(best_result)
        
        # 6. Enrichir avec calculs et agrégations
        best_result = self._enrich_with_calculations(best_result, question_id)
        
        best_result.extraction_time = (datetime.now() - start_time).total_seconds()
        
        return best_result
    
    # ============================================================================
    # STRATÉGIES D'EXTRACTION SPÉCIALISÉES
    # ============================================================================
    
    async def _strategy_llm_extraction(
        self,
        question_config,
        context: str
    ) -> QuestionResult:
        """
        Stratégie 1: Extraction LLM avec prompt enrichi pour volumes
        """
        # Enrichir le prompt avec instructions spécifiques volumes
        enhanced_prompt = self._enhance_volume_prompt(question_config.system_prompt)
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=enhanced_prompt),
                HumanMessage(content=context)
            ])
            
            parsed = self._parse_json_response(response.content)
            
            # Validation numérique spécifique
            if self._validate_numeric_response(parsed):
                result = self._create_question_result(
                    question_config.id,
                    parsed,
                    []
                )
                result.metadata['extraction_method'] = 'llm_enhanced'
                return result
                
        except Exception as e:
            logger.debug(f"LLM extraction failed: {e}")
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="LLM extraction failed"
        )
    
    async def _strategy_pattern_extraction(
        self,
        question_config,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Stratégie 2: Extraction par patterns regex spécialisés
        """
        question_id = question_config.id
        all_volumes = defaultdict(list)
        
        # Combiner tout le texte
        full_text = " ".join([chunk.page_content for chunk in chunks])
        
        # Déterminer les patterns pertinents
        relevant_patterns = self._get_relevant_patterns(question_id)
        
        for pattern_name, pattern in relevant_patterns.items():
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            
            for match in matches:
                # Extraire le contexte autour du match
                start = max(0, match.start() - 100)
                end = min(len(full_text), match.end() + 100)
                context_snippet = full_text[start:end]
                
                # Parser la valeur numérique
                value = self._parse_numeric_value(match.group(1))
                
                # Déterminer la période
                period = self._extract_period(context_snippet)
                
                # Déterminer si c'est garanti ou prévisionnel
                nature = self._determine_nature(context_snippet)
                
                volume_entry = {
                    'type': pattern_name,
                    'value': value,
                    'period': period,
                    'nature': nature,
                    'context': context_snippet.strip(),
                    'confidence': 0.9  # Haute confiance pour pattern match
                }
                
                all_volumes[pattern_name].append(volume_entry)
        
        if all_volumes:
            # Structurer la réponse
            answer = self._structure_pattern_results(all_volumes, question_id)
            
            return QuestionResult(
                question_id=question_id,
                answer=answer,
                status=ExtractionStatus.SUCCESS,
                confidence=0.85,
                metadata={
                    'extraction_method': 'pattern',
                    'patterns_matched': list(all_volumes.keys())
                }
            )
        
        return QuestionResult(
            question_id=question_id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No pattern matches found"
        )
    
    async def _strategy_calculation_extraction(
        self,
        question_config,
        context: str,
        table_results: List[Dict]
    ) -> QuestionResult:
        """
        Stratégie 3: Extraction avec calculs et agrégations
        """
        question_id = question_config.id
        
        # Prompt pour extraction avec focus sur calculs
        calculation_prompt = f"""
        {question_config.system_prompt}
        
        ADDITIONAL INSTRUCTIONS FOR NUMERICAL EXTRACTION:
        1. Extract ALL numerical values with their units
        2. If you find monthly values, calculate yearly totals
        3. If you find ranges, provide both min and max
        4. Look for totals, subtotals, and aggregated values
        5. Identify if values are forecasts or guarantees
        6. Check for growth percentages or year-over-year changes
        
        Tables extracted from document:
        {json.dumps(table_results[:3], indent=2) if table_results else 'No tables found'}
        
        BE PRECISE WITH NUMBERS. Double-check calculations.
        """
        
        try:
            response = await self.table_extraction_llm.ainvoke([
                SystemMessage(content=calculation_prompt),
                HumanMessage(content=context[:10000])
            ])
            
            parsed = self._parse_json_response(response.content)
            
            # Enrichir avec calculs
            if parsed.get('answer'):
                parsed = self._perform_calculations(parsed, question_id)
            
            if self._validate_numeric_response(parsed):
                result = self._create_question_result(
                    question_config.id,
                    parsed,
                    []
                )
                result.metadata['extraction_method'] = 'calculation'
                result.metadata['tables_used'] = len(table_results)
                return result
                
        except Exception as e:
            logger.debug(f"Calculation extraction failed: {e}")
        
        return QuestionResult(
            question_id=question_id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="Calculation extraction failed"
        )
    
    async def _strategy_numeric_focused(
        self,
        question_config,
        chunks: List[Document]
    ) -> QuestionResult:
        """
        Stratégie 4: Extraction focalisée sur les valeurs numériques
        """
        question_id = question_config.id
        
        # Extraire toutes les valeurs numériques avec contexte
        numeric_values = []
        
        for chunk in chunks:
            # Pattern pour toute valeur numérique significative
            numeric_pattern = r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\s*([A-Za-z]+)?'
            matches = re.finditer(numeric_pattern, chunk.page_content)
            
            for match in matches:
                value_str = match.group(1)
                unit = match.group(2) if match.group(2) else ''
                
                # Filtrer les valeurs trop petites ou non pertinentes
                value = self._parse_numeric_value(value_str)
                if value and value > 10:  # Ignorer petites valeurs
                    # Extraire le contexte
                    start = max(0, match.start() - 50)
                    end = min(len(chunk.page_content), match.end() + 50)
                    context = chunk.page_content[start:end]
                    
                    # Vérifier si c'est un volume pertinent
                    if self._is_volume_context(context, unit):
                        numeric_values.append({
                            'value': value,
                            'unit': unit,
                            'context': context,
                            'source': chunk.metadata.get('source', 'unknown')
                        })
        
        if numeric_values:
            # Grouper et structurer les valeurs
            structured_answer = self._structure_numeric_values(numeric_values, question_id)
            
            return QuestionResult(
                question_id=question_id,
                answer=structured_answer,
                status=ExtractionStatus.SUCCESS,
                confidence=0.75,
                metadata={
                    'extraction_method': 'numeric_focused',
                    'values_found': len(numeric_values)
                }
            )
        
        return QuestionResult(
            question_id=question_id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No numeric values found"
        )
    
    # ============================================================================
    # EXTRACTION DE TABLEAUX
    # ============================================================================
    
    async def _extract_from_tables(self, chunks: List[Document]) -> List[Dict]:
        """
        Extrait les données depuis les tableaux détectés
        """
        table_results = []
        
        for i, chunk in enumerate(chunks):
            # Détecter les tableaux (présence de | ou alignement de colonnes)
            if self._contains_table(chunk.page_content):
                try:
                    # Prompt spécialisé pour extraction de tableau
                    table_prompt = """
                    Extract the table data from this text. 
                    Return a JSON with:
                    - headers: list of column headers
                    - rows: list of row data
                    - summary: what the table contains
                    
                    Focus on numerical data, especially volumes, quantities, weights.
                    """
                    
                    response = await self.table_extraction_llm.ainvoke([
                        SystemMessage(content=table_prompt),
                        HumanMessage(content=chunk.page_content)
                    ])
                    
                    parsed = self._parse_json_response(response.content)
                    if parsed:
                        table_results.append({
                            'chunk_index': i,
                            'data': parsed,
                            'source': chunk.metadata.get('source', 'unknown')
                        })
                        
                except Exception as e:
                    logger.debug(f"Table extraction failed for chunk {i}: {e}")
        
        return table_results
    
    def _contains_table(self, text: str) -> bool:
        """
        Détecte si le texte contient probablement un tableau
        """
        indicators = [
            text.count('|') > 5,  # Pipes pour colonnes
            text.count('\t') > 10,  # Tabs pour alignement
            bool(re.search(r'\d+\s+\d+\s+\d+', text)),  # Séries de nombres
            bool(re.search(r'(?:Total|TOTAL|Subtotal)', text)),  # Mots clés de tableau
            bool(re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', text)),  # Mois
            bool(re.search(r'(?:Q1|Q2|Q3|Q4|2022|2023|2024|2025)', text))  # Trimestres/Années
        ]
        
        return sum(indicators) >= 2
    
    # ============================================================================
    # FUSION ET VALIDATION DES RÉSULTATS
    # ============================================================================
    
    def _merge_volume_results(
        self,
        results: List,
        question_config
    ) -> QuestionResult:
        """
        Fusionne les résultats de volumes avec déduplication intelligente
        """
        valid_results = []
        all_values = defaultdict(list)
        
        for result in results:
            if isinstance(result, QuestionResult) and result.status == ExtractionStatus.SUCCESS:
                valid_results.append(result)
                
                # Extraire toutes les valeurs numériques
                if result.answer:
                    if isinstance(result.answer, dict):
                        for key, value in result.answer.items():
                            if isinstance(value, (int, float)):
                                all_values[key].append(value)
                            elif isinstance(value, list):
                                all_values[key].extend(value)
                    elif isinstance(result.answer, list):
                        all_values['values'].extend(result.answer)
        
        if not valid_results:
            return QuestionResult(
                question_id=question_config.id,
                answer=None,
                status=ExtractionStatus.FAILED,
                confidence=0.0,
                error="No successful extraction"
            )
        
        # Sélectionner le meilleur résultat comme base
        best_result = max(valid_results, key=lambda r: r.confidence)
        
        # Enrichir avec les valeurs des autres résultats
        if all_values:
            # Dédupliquer et moyenner les valeurs proches
            merged_values = self._deduplicate_values(all_values)
            
            # Mettre à jour la réponse
            if isinstance(best_result.answer, dict):
                best_result.answer.update(merged_values)
            else:
                best_result.answer = merged_values
        
        best_result.metadata['merge_count'] = len(valid_results)
        best_result.confidence = min(1.0, best_result.confidence + 0.1 * (len(valid_results) - 1))
        
        return best_result
    
    def _deduplicate_values(self, all_values: Dict[str, List]) -> Dict[str, Any]:
        """
        Déduplique les valeurs numériques proches
        """
        deduplicated = {}
        
        for key, values in all_values.items():
            if not values:
                continue
            
            # Filtrer les valeurs numériques
            numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(v)
                elif isinstance(v, str):
                    parsed = self._parse_numeric_value(v)
                    if parsed:
                        numeric_values.append(parsed)
            
            if numeric_values:
                # Grouper les valeurs proches (±5%)
                groups = []
                for value in numeric_values:
                    added = False
                    for group in groups:
                        if abs(value - group[0]) / max(value, group[0]) < 0.05:
                            group.append(value)
                            added = True
                            break
                    if not added:
                        groups.append([value])
                
                # Prendre la moyenne de chaque groupe
                deduplicated[key] = [sum(g) / len(g) for g in groups]
                
                # Si une seule valeur, la sortir de la liste
                if len(deduplicated[key]) == 1:
                    deduplicated[key] = deduplicated[key][0]
        
        return deduplicated
    
    # ============================================================================
    # POST-TRAITEMENT SPÉCIFIQUE VOLUMES
    # ============================================================================
    
    def _post_process_volume_result(self, result: QuestionResult) -> QuestionResult:
        """
        Post-traitement spécifique pour les résultats de volumes
        """
        if not result.answer:
            return result
        
        answer = result.answer
        
        # Si c'est une liste de volumes, les structurer
        if isinstance(answer, list):
            structured = self._structure_volume_list(answer)
            result.answer = structured
        
        # Si c'est un dictionnaire, normaliser les unités
        elif isinstance(answer, dict):
            normalized = self._normalize_units(answer)
            result.answer = normalized
        
        # Ajouter les totaux et agrégations
        result.answer = self._add_aggregations(result.answer)
        
        # Vérifier la cohérence
        coherence = self._check_volume_coherence(result.answer)
        result.metadata['coherence_check'] = coherence
        
        return result
    
    def _structure_volume_list(self, volume_list: List) -> Dict:
        """
        Structure une liste de volumes en dictionnaire organisé
        """
        structured = {
            'containers': [],
            'weight': [],
            'volume': [],
            'pallets': [],
            'units': [],
            'other': []
        }
        
        for item in volume_list:
            if isinstance(item, dict):
                # Classifier selon le type
                if 'type' in item:
                    volume_type = item['type'].lower()
                    if any(kw in volume_type for kw in ['teu', 'feu', 'container']):
                        structured['containers'].append(item)
                    elif any(kw in volume_type for kw in ['ton', 'kg', 'weight']):
                        structured['weight'].append(item)
                    elif any(kw in volume_type for kw in ['m3', 'cubic', 'volume']):
                        structured['volume'].append(item)
                    elif any(kw in volume_type for kw in ['pallet', 'eur']):
                        structured['pallets'].append(item)
                    elif any(kw in volume_type for kw in ['unit', 'piece', 'item']):
                        structured['units'].append(item)
                    else:
                        structured['other'].append(item)
                else:
                    structured['other'].append(item)
        
        # Nettoyer les catégories vides
        return {k: v for k, v in structured.items() if v}
    
    def _normalize_units(self, data: Dict) -> Dict:
        """
        Normalise les unités vers des standards
        """
        normalized = data.copy()
        
        # Normaliser les poids en tonnes
        if 'weight' in normalized:
            if isinstance(normalized['weight'], dict):
                if 'kg' in normalized['weight']:
                    normalized['weight']['tonnes'] = normalized['weight']['kg'] * self.unit_conversions['weight']['kg_to_tonnes']
        
        # Normaliser les conteneurs en TEU
        if 'containers' in normalized:
            if isinstance(normalized['containers'], dict):
                if 'feu' in normalized['containers']:
                    normalized['containers']['teu'] = normalized['containers']['feu'] * self.unit_conversions['containers']['feu_to_teu']
        
        return normalized
    
    def _add_aggregations(self, data: Any) -> Any:
        """
        Ajoute les totaux et agrégations calculés
        """
        if not isinstance(data, dict):
            return data
        
        aggregations = {}
        
        # Calculer le total TEU annuel
        if 'containers' in data:
            teu_total = 0
            if isinstance(data['containers'], list):
                for item in data['containers']:
                    if isinstance(item, dict) and 'value' in item:
                        value = item['value']
                        if 'teu' in str(item.get('type', '')).lower():
                            # Convertir en annuel si nécessaire
                            if item.get('period') == 'monthly':
                                value *= 12
                            elif item.get('period') == 'weekly':
                                value *= 52
                            teu_total += value
            
            if teu_total > 0:
                aggregations['total_teu_annual'] = teu_total
        
        # Calculer le tonnage total annuel
        if 'weight' in data:
            tonnage_total = 0
            if isinstance(data['weight'], list):
                for item in data['weight']:
                    if isinstance(item, dict) and 'value' in item:
                        value = item['value']
                        # Convertir en tonnes si nécessaire
                        if 'kg' in str(item.get('unit', '')).lower():
                            value *= 0.001
                        # Convertir en annuel
                        if item.get('period') == 'monthly':
                            value *= 12
                        tonnage_total += value
            
            if tonnage_total > 0:
                aggregations['total_tonnage_annual'] = tonnage_total
        
        if aggregations:
            data['aggregations'] = aggregations
        
        return data
    
    def _check_volume_coherence(self, data: Any) -> Dict[str, Any]:
        """
        Vérifie la cohérence des volumes extraits
        """
        coherence = {
            'is_coherent': True,
            'warnings': [],
            'ratios': {}
        }
        
        if not isinstance(data, dict):
            return coherence
        
        # Vérifier le ratio TEU/tonnage (généralement 10-15 tonnes/TEU)
        if 'aggregations' in data:
            agg = data['aggregations']
            if 'total_teu_annual' in agg and 'total_tonnage_annual' in agg:
                if agg['total_teu_annual'] > 0:
                    ratio = agg['total_tonnage_annual'] / agg['total_teu_annual']
                    coherence['ratios']['tonnes_per_teu'] = ratio
                    
                    if ratio < 5 or ratio > 25:
                        coherence['warnings'].append(f"Ratio tonnage/TEU inhabituel: {ratio:.1f}")
                        coherence['is_coherent'] = False
        
        return coherence
    
    # ============================================================================
    # MÉTHODES UTILITAIRES SPÉCIALISÉES
    # ============================================================================
    
    def _enhance_volume_prompt(self, original_prompt: str) -> str:
        """
        Enrichit le prompt avec des instructions spécifiques volumes
        """
        enhancement = """
        
        CRITICAL VOLUME EXTRACTION GUIDELINES:
        
        1. NUMERICAL PRECISION:
           - Extract exact numbers, not approximations
           - Include decimal points when present
           - Preserve original units before conversion
        
        2. CONTEXT IDENTIFICATION:
           - Distinguish between forecast vs. guaranteed volumes
           - Identify if volumes are minimum, maximum, or average
           - Note the time period (annual, monthly, total contract)
        
        3. LOOK FOR VOLUMES IN:
           - Tables and structured data
           - Bullet points and lists
           - Executive summaries
           - Annexes and appendices
           - Financial sections (often contain volume commitments)
        
        4. COMMON FORMATS:
           - "X TEU per year"
           - "Annual volume: X tonnes"
           - "Expected throughput: X pallets/month"
           - "Capacity requirement: X m³"
        
        5. CALCULATIONS:
           - If monthly values given, calculate annual (x12)
           - If ranges given, note both min and max
           - If growth percentages given, calculate future volumes
        
        Always return the most specific and detailed volume information available.
        """
        
        return original_prompt + enhancement
    
    def _parse_numeric_value(self, value_str: str) -> Optional[float]:
        """
        Parse une valeur numérique depuis une chaîne
        """
        if not value_str:
            return None
        
        try:
            # Nettoyer la chaîne
            cleaned = value_str.strip()
            
            # Enlever les séparateurs de milliers
            cleaned = cleaned.replace(',', '').replace(' ', '').replace('.', '')
            
            # Gérer les décimales (virgule comme séparateur décimal)
            if ',' in value_str and value_str.count(',') == 1:
                parts = value_str.split(',')
                if len(parts[1]) <= 2:  # Probablement une décimale
                    cleaned = value_str.replace(',', '.')
            
            return float(cleaned)
            
        except (ValueError, AttributeError):
            return None
    
    def _extract_period(self, context: str) -> str:
        """
        Extrait la période depuis le contexte
        """
        context_lower = context.lower()
        
        if any(kw in context_lower for kw in ['per year', 'annual', 'per annum', '/year', 'p.a.']):
            return 'annual'
        elif any(kw in context_lower for kw in ['per month', 'monthly', '/month', 'mensuel']):
            return 'monthly'
        elif any(kw in context_lower for kw in ['per week', 'weekly', '/week', 'hebdomadaire']):
            return 'weekly'
        elif any(kw in context_lower for kw in ['per day', 'daily', '/day', 'quotidien']):
            return 'daily'
        elif any(kw in context_lower for kw in ['total', 'contract duration', 'entire period']):
            return 'total_contract'
        else:
            return 'unspecified'
    
    def _determine_nature(self, context: str) -> str:
        """
        Détermine si le volume est garanti, prévisionnel, etc.
        """
        context_lower = context.lower()
        
        if any(kw in context_lower for kw in ['guaranteed', 'minimum', 'committed', 'garanti']):
            return 'guaranteed'
        elif any(kw in context_lower for kw in ['forecast', 'estimated', 'expected', 'projected', 'prévision']):
            return 'forecast'
        elif any(kw in context_lower for kw in ['maximum', 'cap', 'not to exceed']):
            return 'maximum'
        elif any(kw in context_lower for kw in ['average', 'typical', 'normal']):
            return 'average'
        else:
            return 'unspecified'
    
    def _get_relevant_patterns(self, question_id: str) -> Dict[str, str]:
        """
        Retourne les patterns pertinents pour une question
        """
        # Mapping question -> patterns
        if 'teu' in question_id.lower() or 'container' in question_id.lower():
            return {k: v for k, v in self.volume_patterns.items() 
                   if k in ['teu', 'feu', 'containers_20', 'containers_40']}
        elif 'tonnage' in question_id.lower() or 'weight' in question_id.lower():
            return {k: v for k, v in self.volume_patterns.items() 
                   if k in ['tonnes', 'kg', 'pounds']}
        elif 'volume' in question_id.lower() or 'm3' in question_id.lower():
            return {k: v for k, v in self.volume_patterns.items() 
                   if k in ['m3', 'liters']}
        elif 'pallet' in question_id.lower():
            return {k: v for k, v in self.volume_patterns.items() 
                   if k in ['pallets', 'eur_pallets']}
        else:
            # Retourner les patterns principaux
            return {k: v for k, v in self.volume_patterns.items() 
                   if k in ['teu', 'tonnes', 'm3', 'pallets', 'units']}
    
    def _is_numeric_question(self, question_id: str) -> bool:
        """
        Détermine si la question cherche principalement des valeurs numériques
        """
        numeric_keywords = ['volume', 'quantity', 'amount', 'number', 'tonnage', 'capacity', 'throughput']
        return any(kw in question_id.lower() for kw in numeric_keywords)
    
    def _is_volume_context(self, context: str, unit: str) -> bool:
        """
        Vérifie si le contexte indique un volume
        """
        context_lower = context.lower()
        unit_lower = unit.lower() if unit else ''
        
        # Vérifier les unités connues
        volume_units = ['teu', 'feu', 'ton', 'kg', 'm3', 'pallet', 'unit', 'container']
        if any(vu in unit_lower for vu in volume_units):
            return True
        
        # Vérifier le contexte
        volume_contexts = ['volume', 'quantity', 'capacity', 'throughput', 'tonnage', 'shipment']
        return any(vc in context_lower for vc in volume_contexts)
    
    def _structure_pattern_results(self, all_volumes: Dict[str, List], question_id: str) -> Any:
        """
        Structure les résultats de pattern matching
        """
        # Si une seule catégorie avec une seule valeur
        if len(all_volumes) == 1:
            key = list(all_volumes.keys())[0]
            values = all_volumes[key]
            if len(values) == 1:
                return values[0]
        
        # Sinon retourner structuré
        return dict(all_volumes)
    
    def _structure_numeric_values(self, numeric_values: List[Dict], question_id: str) -> Any:
        """
        Structure les valeurs numériques extraites
        """
        # Grouper par unité
        by_unit = defaultdict(list)
        for nv in numeric_values:
            unit = nv.get('unit', 'unknown').lower()
            by_unit[unit].append(nv)
        
        # Si une seule unité dominante
        if len(by_unit) == 1:
            unit = list(by_unit.keys())[0]
            values = by_unit[unit]
            
            # Retourner la moyenne ou la somme selon le contexte
            if 'total' in question_id.lower():
                return sum(v['value'] for v in values)
            else:
                return sum(v['value'] for v in values) / len(values)
        
        return dict(by_unit)
    
    def _enrich_context_with_tables(self, context: str, table_results: List[Dict]) -> str:
        """
        Enrichit le contexte avec les tableaux extraits
        """
        if not table_results:
            return context
        
        table_text = "\n\n--- EXTRACTED TABLES ---\n"
        for table in table_results[:3]:  # Limiter à 3 tableaux
            if 'data' in table:
                table_text += f"\nTable from {table.get('source', 'unknown')}:\n"
                table_text += json.dumps(table['data'], indent=2)[:1000]  # Limiter la taille
                table_text += "\n"
        
        return context + table_text
    
    def _perform_calculations(self, parsed: Dict, question_id: str) -> Dict:
        """
        Effectue des calculs sur les données extraites
        """
        if not parsed.get('answer'):
            return parsed
        
        answer = parsed['answer']
        
        # Si on a des valeurs mensuelles et qu'on demande l'annuel
        if 'annual' in question_id.lower() and isinstance(answer, dict):
            if 'monthly' in answer:
                answer['annual_calculated'] = answer['monthly'] * 12
        
        # Si on a un range, calculer la moyenne
        if isinstance(answer, dict) and 'min' in answer and 'max' in answer:
            answer['average'] = (answer['min'] + answer['max']) / 2
        
        return parsed
    
    def _enrich_with_calculations(self, result: QuestionResult, question_id: str) -> QuestionResult:
        """
        Enrichit le résultat avec des calculs additionnels
        """
        if not result.answer:
            return result
        
        calculations = {}
        
        # Ajouter des conversions utiles
        if isinstance(result.answer, (int, float)):
            value = result.answer
            
            # Si c'est des TEU, calculer l'équivalent FEU
            if 'teu' in question_id.lower():
                calculations['feu_equivalent'] = value * 0.5
            
            # Si c'est annuel, calculer le mensuel
            if 'annual' in question_id.lower():
                calculations['monthly_average'] = value / 12
                calculations['weekly_average'] = value / 52
        
        if calculations:
            result.metadata['calculations'] = calculations
        
        return result
    
    def _validate_numeric_response(self, parsed: Dict) -> bool:
        """
        Validation spécifique pour les réponses numériques
        """
        if not parsed or 'answer' not in parsed:
            return False
        
        answer = parsed['answer']
        
        # Accepter les nombres, listes de nombres, ou dictionnaires avec nombres
        if isinstance(answer, (int, float)):
            return answer >= 0  # Les volumes négatifs n'ont pas de sens
        
        if isinstance(answer, list):
            return all(isinstance(item, (int, float, dict)) for item in answer)
        
        if isinstance(answer, dict):
            # Au moins une valeur numérique
            return any(isinstance(v, (int, float)) for v in answer.values())
        
        return False