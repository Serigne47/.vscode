# agents/volume_agent.py
"""
Agent spécialisé dans l'extraction des volumes et métriques
"""
from typing import Dict, List, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from .base_agent1 import BaseExtractionAgent
import re
import pandas as pd

class VolumeExtractionAgent(BaseExtractionAgent):
    """
    Agent pour extraire toutes les informations volumétriques
    """
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Prompt optimisé pour l'extraction de volumes
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Tu es un expert en analyse de données volumétriques dans les appels d'offres logistiques.
            
            MISSION CRITIQUE: Extraire ABSOLUMENT TOUS les volumes, quantités et métriques mentionnés.
            
            TYPES DE VOLUMES À RECHERCHER:
            
            1. CONTENEURS:
               - TEU (Twenty-foot Equivalent Unit)
               - FEU (Forty-foot Equivalent Unit)
               - Conteneurs 20' / 40' / 40'HC / 45'
               - Reefer containers (conteneurs frigorifiques)
            
            2. POIDS:
               - Tonnage (tonnes, MT, tons)
               - Kilogrammes (kg)
               - Poids brut / net / taxable
            
            3. VOLUME:
               - Mètres cubes (m³, m3, CBM)
               - Litres
               - Volume taxable aérien (poids/167)
            
            4. UNITÉS LOGISTIQUES:
               - Palettes (EUR, US, UK, autre)
               - Colis / Packages / Cartons
               - Unités / Pièces / SKU
            
            5. TRANSPORT:
               - Nombre de camions / trucks
               - Nombre d'expéditions / shipments
               - Nombre de positions / lignes de commande
            
            INFORMATIONS CRUCIALES À CAPTURER:
            - Valeur numérique exacte
            - Unité de mesure
            - Période (annuel, mensuel, hebdomadaire, total contrat)
            - Type (prévisionnel, garanti, minimum, maximum, moyen)
            - Corridor/Route/Destination si spécifié
            - Produit/Catégorie si mentionné
            - Croissance prévue (%)
            
            RÈGLES D'EXTRACTION:
            1. TOUJOURS convertir les nombres écrits en lettres
            2. Capturer les ranges (ex: "1000-1500 TEU" → min: 1000, max: 1500)
            3. Identifier si c'est un engagement ou une estimation
            4. Noter la saisonnalité si mentionnée
            5. Extraire TOUS les tableaux de volumes
            
            FORMAT JSON OBLIGATOIRE:
            {
                "volumes_conteneurs": [
                    {
                        "type": "TEU|FEU|20ft|40ft|40HC",
                        "valeur": 1000,
                        "unite": "TEU",
                        "periode": "annuel|mensuel|total",
                        "nature": "previsionnel|garanti|minimum|maximum",
                        "corridor": "string ou null",
                        "croissance": "pourcentage ou null",
                        "source_exacte": "citation exacte du document"
                    }
                ],
                "volumes_poids": [
                    {
                        "valeur": 5000,
                        "unite": "tonnes|kg",
                        "periode": "string",
                        "type_poids": "brut|net|taxable",
                        "produit": "string ou null",
                        "source_exacte": "string"
                    }
                ],
                "volumes_m3": [
                    {
                        "valeur": 10000,
                        "unite": "m3",
                        "periode": "string",
                        "destination": "string ou null",
                        "source_exacte": "string"
                    }
                ],
                "volumes_palettes": [
                    {
                        "valeur": 50000,
                        "type_palette": "EUR|US|UK|autre",
                        "periode": "string",
                        "source_exacte": "string"
                    }
                ],
                "autres_unites": [
                    {
                        "description": "string",
                        "valeur": 100,
                        "unite": "string",
                        "periode": "string",
                        "source_exacte": "string"
                    }
                ],
                "volume_total_estime": {
                    "description": "estimation globale si mentionnée",
                    "valeur_monetaire": "si le volume est exprimé en valeur"
                },
                "tableaux_volumes": [
                    {
                        "titre": "nom du tableau",
                        "donnees": "données structurées du tableau"
                    }
                ]
            }
            
            IMPORTANT: Ne JAMAIS inventer de données. Si une information n'est pas dans le document, ne pas l'inclure.
            """),
            HumanMessage(content="Analyse ce document et extrais TOUS les volumes:\n{context}")
        ])
    
    def _define_validation_patterns(self) -> Dict[str, str]:
        """
        Patterns regex spécialisés pour les volumes
        """
        return {
            "teu_feu": r'(\d+[\s,\.]*\d*)\s*(?:TEUs?|FEUs?|EVP)',
            "containers": r'(\d+[\s,\.]*\d*)\s*(?:conteneurs?|containers?)\s*(?:20|40)?[\']?(?:ft|pieds)?',
            "tonnage": r'(\d+[\s,\.]*\d*)\s*(?:tonnes?|tons?|MT|t\b)(?:\s*(?:par|per|\/)\s*(?:an|année|year|mois|month))?',
            "volume_m3": r'(\d+[\s,\.]*\d*)\s*(?:m³|m3|mètres?\s*cubes?|cubic\s*meters?|CBM)',
            "palettes": r'(\d+[\s,\.]*\d*)\s*(?:palettes?|pals?)\s*(?:EUR|EUR1|US|UK)?',
            "kg": r'(\d+[\s,\.]*\d*)\s*(?:kg|kilogrammes?|kilos?)',
            "colis": r'(\d+[\s,\.]*\d*)\s*(?:colis|packages?|cartons?|boxes?)',
            "percentage": r'(\d+[\s,\.]*\d*)\s*%',
            "range": r'(\d+[\s,\.]*\d*)\s*(?:à|-|to)\s*(\d+[\s,\.]*\d*)',
            "annual": r'(?:par\s*an|annual|annuel|yearly|\/an|p\.a\.)',
            "monthly": r'(?:par\s*mois|monthly|mensuel|\/month|\/mois)',
            "million": r'(\d+[\s,\.]*\d*)\s*(?:millions?|M€|M\$|MEUR|MUSD)'
        }
    
    def _post_process(self, raw_extraction: Dict) -> Dict:
        """
        Post-traitement et enrichissement des volumes
        """
        processed = raw_extraction.copy()
        
        # Conversion et normalisation des valeurs
        processed = self._normalize_values(processed)
        
        # Calcul des totaux et moyennes
        processed = self._calculate_aggregates(processed)
        
        # Détection d'incohérences
        processed['coherence_check'] = self._check_coherence(processed)
        
        # Enrichissement avec patterns
        processed = self._enrich_with_patterns(processed)
        
        return processed
    
    def _normalize_values(self, data: Dict) -> Dict:
        """
        Normalise les valeurs numériques et unités
        """
        for category in ['volumes_conteneurs', 'volumes_poids', 'volumes_m3', 'volumes_palettes']:
            if category in data and isinstance(data[category], list):
                for item in data[category]:
                    # Nettoyer les valeurs numériques
                    if 'valeur' in item:
                        if isinstance(item['valeur'], str):
                            # Enlever espaces et virgules
                            cleaned = item['valeur'].replace(' ', '').replace(',', '')
                            try:
                                item['valeur'] = float(cleaned)
                            except:
                                pass
                    
                    # Normaliser les périodes
                    if 'periode' in item:
                        period_map = {
                            'annual': 'annuel',
                            'yearly': 'annuel',
                            'monthly': 'mensuel',
                            'weekly': 'hebdomadaire',
                            'daily': 'quotidien'
                        }
                        item['periode'] = period_map.get(item['periode'], item['periode'])
                    
                    # Convertir en base annuelle pour comparaison
                    if 'periode' in item and 'valeur' in item:
                        item['valeur_annuelle'] = self._convert_to_annual(
                            item['valeur'], 
                            item['periode']
                        )
        
        return data
    
    def _convert_to_annual(self, value: float, period: str) -> float:
        """
        Convertit une valeur en base annuelle
        """
        multipliers = {
            'annuel': 1,
            'mensuel': 12,
            'hebdomadaire': 52,
            'quotidien': 365,
            'trimestriel': 4,
            'semestriel': 2
        }
        return value * multipliers.get(period, 1)
    
    def _calculate_aggregates(self, data: Dict) -> Dict:
        """
        Calcule les totaux et statistiques agrégées
        """
        aggregates = {
            'total_teu_annuel': 0,
            'total_tonnage_annuel': 0,
            'total_m3_annuel': 0,
            'total_palettes_annuel': 0,
            'nombre_corridors': 0,
            'statistiques': {}
        }
        
        # Calcul TEU
        if 'volumes_conteneurs' in data:
            teu_values = []
            for item in data['volumes_conteneurs']:
                if 'valeur_annuelle' in item:
                    # Conversion FEU en TEU
                    if item.get('type') == 'FEU':
                        teu_values.append(item['valeur_annuelle'] * 2)
                    else:
                        teu_values.append(item['valeur_annuelle'])
            
            if teu_values:
                aggregates['total_teu_annuel'] = sum(teu_values)
                aggregates['statistiques']['teu'] = {
                    'min': min(teu_values),
                    'max': max(teu_values),
                    'moyenne': sum(teu_values) / len(teu_values)
                }
        
        # Calcul tonnage
        if 'volumes_poids' in data:
            tonnage_values = []
            for item in data['volumes_poids']:
                if 'valeur_annuelle' in item:
                    # Conversion kg en tonnes
                    if item.get('unite') == 'kg':
                        tonnage_values.append(item['valeur_annuelle'] / 1000)
                    else:
                        tonnage_values.append(item['valeur_annuelle'])
            
            if tonnage_values:
                aggregates['total_tonnage_annuel'] = sum(tonnage_values)
        
        # Compter les corridors uniques
        corridors = set()
        for category in ['volumes_conteneurs', 'volumes_poids', 'volumes_m3']:
            if category in data:
                for item in data[category]:
                    if 'corridor' in item and item['corridor']:
                        corridors.add(item['corridor'])
        aggregates['nombre_corridors'] = len(corridors)
        
        data['aggregates'] = aggregates
        return data
    
    def _check_coherence(self, data: Dict) -> Dict:
        """
        Vérifie la cohérence des volumes extraits
        """
        checks = {
            'warnings': [],
            'info': []
        }
        
        # Vérifier ratio TEU/tonnage
        if 'aggregates' in data:
            agg = data['aggregates']
            if agg['total_teu_annuel'] > 0 and agg['total_tonnage_annuel'] > 0:
                ratio = agg['total_tonnage_annuel'] / agg['total_teu_annuel']
                if ratio < 5 or ratio > 25:  # Ratio normal entre 5-25 tonnes/TEU
                    checks['warnings'].append(
                        f"Ratio tonnage/TEU inhabituel: {ratio:.1f} tonnes/TEU"
                    )
        
        # Vérifier les doublons potentiels
        seen_values = {}
        for category in ['volumes_conteneurs', 'volumes_poids']:
            if category in data:
                for item in data[category]:
                    val = item.get('valeur')
                    if val in seen_values:
                        checks['info'].append(
                            f"Valeur {val} apparaît plusieurs fois"
                        )
                    seen_values[val] = True
        
        return checks
    
    def _enrich_with_patterns(self, data: Dict) -> Dict:
        """
        Enrichit avec les données trouvées par patterns
        """
        if hasattr(self, '_last_pattern_results'):
            patterns = self._last_pattern_results
            
            # Ajouter les volumes trouvés par regex non présents dans LLM
            if 'million' in patterns:
                for match in patterns['million']:
                    data.setdefault('volumes_monetaires', []).append({
                        'valeur': match,
                        'source': 'pattern_extraction'
                    })
        
        return data
    
    def _pattern_extract(self, chunks) -> Dict:
        """
        Extraction avancée par patterns avec gestion des tableaux
        """
        results = super()._pattern_extract(chunks)
        self._last_pattern_results = results
        
        # Extraction spéciale des tableaux
        for chunk in chunks:
            if '|' in chunk.page_content or '\t' in chunk.page_content:
                # Possible tableau détecté
                table_data = self._extract_table_volumes(chunk.page_content)
                if table_data:
                    results.setdefault('tables', []).append(table_data)
        
        return results
    
    def _extract_table_volumes(self, text: str) -> Optional[Dict]:
        """
        Extrait les volumes depuis les tableaux
        """
        try:
            lines = text.strip().split('\n')
            # Détection simple de tableau
            if any('|' in line for line in lines) or any('\t' in line for line in lines):
                # Parser le tableau (simplification)
                volumes = []
                for line in lines:
                    # Chercher des patterns de volumes dans chaque ligne
                    for pattern_name, pattern in self.validation_patterns.items():
                        matches = re.findall(pattern, line)
                        if matches:
                            volumes.extend(matches)
                
                if volumes:
                    return {
                        'type': 'table',
                        'volumes_found': volumes
                    }
        except:
            pass
        
        return None