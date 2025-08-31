# agents/operational_agent.py
"""
Agent spécialisé dans l'extraction des exigences opérationnelles
"""
from typing import Dict, List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from .base_agent1 import BaseExtractionAgent
import re

class OperationalExtractionAgent(BaseExtractionAgent):
    """
    Agent pour extraire les modalités d'exécution et exigences opérationnelles
    """
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Prompt pour extraction opérationnelle
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Tu es un expert en opérations logistiques et supply chain.
            
            MISSION: Extraire TOUTES les exigences opérationnelles et modalités d'exécution.
            
            ÉLÉMENTS OPÉRATIONNELS À EXTRAIRE:
            
            1. MODES DE TRANSPORT:
               - Transport principal (maritime, aérien, routier, ferroviaire)
               - Pré/post-acheminement
               - Transport multimodal
               - Express/Standard
               - Groupage/Complet
               - Transport exceptionnel
            
            2. SERVICES LOGISTIQUES:
               - Freight forwarding
               - Dédouanement (import/export)
               - Entreposage/Stockage
               - Cross-docking
               - Préparation de commandes
               - Distribution last-mile
               - Reverse logistics/Retours
               - Gestion des stocks
            
            3. EXIGENCES TEMPÉRATURE:
               - Température dirigée (ranges)
               - Produits surgelés (-18°C, -25°C)
               - Frais (2-8°C)
               - Ambiant contrôlé (15-25°C)
               - Monitoring température
               - Qualification GDP
               - Chaîne du froid ininterrompue
            
            4. CONTRAINTES PRODUITS:
               - Matières dangereuses (ADR, IMDG, IATA)
               - Produits pharmaceutiques (GDP)
               - Denrées alimentaires
               - Produits haute valeur
               - Fragile/Sensible
               - Surdimensionné
            
            5. SYSTÈMES IT/TRACKING:
               - EDI requis (formats)
               - WMS/TMS interfaces
               - Track & Trace temps réel
               - POD électronique
               - Reporting formats
               - API/Webservices
               - Portail client
            
            6. EXIGENCES OPÉRATIONNELLES:
               - Délais de livraison (transit time)
               - Cut-off times
               - Horaires de réception/livraison
               - Rendez-vous obligatoire
               - Déchargement/Chargement
               - Matériel de manutention
               - Personnel sur site
            
            7. KPI/SLA:
               - On-Time Delivery (OTD)
               - On-Time In-Full (OTIF)
               - Taux de casse/perte
               - Accuracy
               - Lead time
               - Disponibilité stock
            
            8. GÉOGRAPHIE:
               - Zones de collecte
               - Points de livraison
               - Corridors principaux
               - Pays exclus/restrictions
               - Zones urbaines/rurales
               - Accessibilité sites
            
            9. CERTIFICATIONS REQUISES:
               - ISO 9001/14001
               - GDP/GMP
               - AEO/OEA
               - TAPA
               - Bio/Organic
               - Certifications douanières
            
            10. RESSOURCES REQUISES:
                - Flotte dédiée/mutualisée
                - Type de véhicules
                - Équipements spéciaux
                - Entrepôts (surface, localisation)
                - Personnel dédié
                - Compétences spécifiques
            
            FORMAT JSON STRICT:
            {
                "transport": {
                    "mode_principal": "maritime|aerien|routier|ferroviaire",
                    "modes_secondaires": ["liste"],
                    "incoterm": "EXW|DDP|FOB|etc",
                    "groupage": true/false,
                    "express_requis": true/false,
                    "specificites": ["transport exceptionnel", "IMO"]
                },
                "services": {
                    "freight_forwarding": true,
                    "douane": {
                        "import": true,
                        "export": true,
                        "representation": "directe|indirecte"
                    },
                    "entreposage": {
                        "requis": true,
                        "surface_m2": 5000,
                        "type": "sec|froid|mixte",
                        "localisation": ["zones"]
                    },
                    "distribution": {
                        "last_mile": true,
                        "b2b": true,
                        "b2c": false,
                        "reverse": true
                    },
                    "services_additionnels": ["cross-dock", "kitting", "etiquetage"]
                },
                "temperature": {
                    "controle_requis": true,
                    "type": "surgele|frais|ambiant",
                    "range": {
                        "min": -25,
                        "max": -18,
                        "unite": "celsius"
                    },
                    "monitoring": "continu|ponctuel",
                    "gdp_requis": true,
                    "qualification": ["vehicules", "entrepots"]
                },
                "contraintes_produits": {
                    "dangereuses": {
                        "present": true,
                        "classes": ["3", "8", "9"],
                        "adr": true,
                        "imdg": true
                    },
                    "pharma": true,
                    "alimentaire": true,
                    "haute_valeur": true,
                    "specifiques": ["fragile", "sensible lumiere"]
                },
                "systemes_it": {
                    "edi": {
                        "requis": true,
                        "formats": ["EDIFACT", "XML", "API"],
                        "messages": ["ORDERS", "DESADV", "INVOIC"]
                    },
                    "tracking": {
                        "temps_reel": true,
                        "frequence": "horaire",
                        "pod_electronique": true
                    },
                    "interfaces": ["SAP", "WMS client"],
                    "reporting": {
                        "frequence": "quotidien",
                        "format": "Excel|PowerBI",
                        "kpi_inclus": ["OTD", "stock levels"]
                    }
                },
                "kpi_sla": [
                    {
                        "indicateur": "OTD",
                        "objectif": "98%",
                        "mesure": "mensuelle",
                        "penalite": true
                    },
                    {
                        "indicateur": "OTIF",
                        "objectif": "95%",
                        "mesure": "mensuelle",
                        "penalite": true
                    },
                    {
                        "indicateur": "Casse",
                        "objectif": "<0.1%",
                        "mesure": "mensuelle",
                        "penalite": false
                    }
                ],
                "geographie": {
                    "collecte": ["France", "Benelux"],
                    "livraison": ["Europe", "UK"],
                    "corridors": [
                        {
                            "origine": "France",
                            "destination": "Allemagne",
                            "volume": "40%"
                        }
                    ],
                    "exclusions": ["Russie", "Belarus"]
                },
                "certifications": {
                    "obligatoires": ["ISO 9001", "GDP"],
                    "souhaitees": ["ISO 14001", "TAPA"],
                    "douane": ["AEO", "Représentant en douane"]
                },
                "ressources": {
                    "flotte": {
                        "type": "dediee|mutualisee",
                        "vehicules": ["porteurs", "semi-remorques"],
                        "nombre": 25,
                        "equipements": ["hayon", "transpalette embarque"]
                    },
                    "entrepots": {
                        "nombre": 2,
                        "surface_totale": 10000,
                        "localisations": ["IDF", "Lyon"]
                    },
                    "personnel": {
                        "dedie": true,
                        "nombre": 15,
                        "competences": ["cariste", "ADR", "douane"]
                    }
                },
                "complexite_operationnelle": {
                    "niveau": "faible|moyen|eleve|tres_eleve",
                    "facteurs": ["multi-temperature", "GDP", "last-mile B2C"]
                }
            }
            """),
            HumanMessage(content="Analyse ces documents et extrais toutes les exigences opérationnelles:\n{context}")
        ])
    
    def _define_validation_patterns(self) -> Dict[str, str]:
        """
        Patterns pour données opérationnelles
        """
        return {
            "temperature": r'(-?\d+)\s*°?[CF]|(?:celsius|fahrenheit)',
            "transit_time": r'(\d+)\s*(?:jours?|days?|heures?|hours?)\s*(?:ouvrés?|working)?',
            "otd": r'(?:OTD|on-time|ponctualité)\s*[:=]?\s*(\d+)\s*%',
            "otif": r'(?:OTIF|on-time-in-full)\s*[:=]?\s*(\d+)\s*%',
            "transport_mode": r'\b(?:maritime|aérien|aerien|air|sea|road|routier|rail|ferroviaire)\b',
            "incoterm": r'\b(?:EXW|FCA|CPT|CIP|DAT|DAP|DDP|FOB|CFR|CIF|FAS)\b',
            "dangerous": r'(?:ADR|IMDG|IATA|dangerous goods|matières dangereuses|classe\s+\d)',
            "gdp": r'\b(?:GDP|Good Distribution Practice|bonnes pratiques de distribution)\b',
            "warehouse": r'(\d+)\s*m[2²]|(?:entrepôt|warehouse|plateforme)',
            "vehicle": r'(?:camion|truck|porteur|semi-remorque|fourgon|VL|PL)',
            "certification": r'(?:ISO\s*\d{4}|AEO|OEA|TAPA|GDP|GMP)',
            "edi": r'(?:EDI|EDIFACT|XML|API|interface|webservice)',
            "tracking": r'(?:track\s*&?\s*trace|suivi|tracking|temps réel|real-time)'
        }
    
    def _post_process(self, raw_extraction: Dict) -> Dict:
        """
        Post-traitement et analyse de complexité opérationnelle
        """
        processed = raw_extraction.copy()
        
        # Évaluation de la complexité
        processed['complexity_analysis'] = self._analyze_complexity(processed)
        
        # Identification des capacités requises
        processed['required_capabilities'] = self._identify_capabilities(processed)
        
        # Analyse de faisabilité
        processed['feasibility'] = self._assess_feasibility(processed)
        
        # Points d'attention opérationnels
        processed['operational_alerts'] = self._identify_operational_alerts(processed)
        
        # Ressources critiques
        processed['critical_resources'] = self._identify_critical_resources(processed)
        
        return processed
    
    def _analyze_complexity(self, data: Dict) -> Dict:
        """
        Analyse la complexité opérationnelle
        """
        complexity_score = 0
        factors = []
        
        # Transport multimodal
        if 'transport' in data:
            if len(data['transport'].get('modes_secondaires', [])) > 1:
                complexity_score += 3
                factors.append("Transport multimodal")
        
        # Température dirigée
        if 'temperature' in data and data['temperature'].get('controle_requis'):
            complexity_score += 4
            factors.append("Température dirigée")
            if data['temperature'].get('gdp_requis'):
                complexity_score += 3
                factors.append("GDP requis")
        
        # Matières dangereuses
        if 'contraintes_produits' in data:
            if data['contraintes_produits'].get('dangereuses', {}).get('present'):
                complexity_score += 4
                factors.append("Matières dangereuses")
        
        # Distribution last-mile
        if 'services' in data:
            if data['services'].get('distribution', {}).get('last_mile'):
                complexity_score += 2
                factors.append("Distribution last-mile")
            if data['services'].get('distribution', {}).get('b2c'):
                complexity_score += 3
                factors.append("Livraison B2C")
        
        # KPI stricts
        if 'kpi_sla' in data:
            for kpi in data['kpi_sla']:
                if kpi.get('objectif'):
                    try:
                        obj = float(kpi['objectif'].replace('%', ''))
                        if obj >= 98:
                            complexity_score += 2
                            factors.append(f"KPI strict: {kpi['indicateur']} {obj}%")
                    except:
                        pass
        
        # Déterminer le niveau
        if complexity_score >= 15:
            level = "TRÈS ÉLEVÉ"
        elif complexity_score >= 10:
            level = "ÉLEVÉ"
        elif complexity_score >= 5:
            level = "MOYEN"
        else:
            level = "FAIBLE"
        
        return {
            'score': complexity_score,
            'level': level,
            'factors': factors
        }
    
    def _identify_capabilities(self, data: Dict) -> List[Dict]:
        """
        Identifie les capacités requises
        """
        capabilities = []
        
        # Transport
        if 'transport' in data:
            transport = data['transport']
            capabilities.append({
                'domain': 'Transport',
                'required': [transport.get('mode_principal')] + transport.get('modes_secondaires', []),
                'critical': True
            })
        
        # Température
        if 'temperature' in data and data['temperature'].get('controle_requis'):
            temp_type = data['temperature'].get('type', 'non spécifié')
            capabilities.append({
                'domain': 'Chaîne du froid',
                'required': [f"Gestion {temp_type}", "Monitoring température"],
                'critical': True
            })
        
        # IT
        if 'systemes_it' in data:
            it = data['systemes_it']
            it_cap = []
            if it.get('edi', {}).get('requis'):
                it_cap.append("EDI")
            if it.get('tracking', {}).get('temps_reel'):
                it_cap.append("Track & Trace temps réel")
            if it_cap:
                capabilities.append({
                    'domain': 'Systèmes IT',
                    'required': it_cap,
                    'critical': True
                })
        
        # Certifications
        if 'certifications' in data and data['certifications'].get('obligatoires'):
            capabilities.append({
                'domain': 'Certifications',
                'required': data['certifications']['obligatoires'],
                'critical': True
            })
        
        return capabilities
    
    def _assess_feasibility(self, data: Dict) -> Dict:
        """
        Évalue la faisabilité opérationnelle
        """
        feasibility = {
            'score': 100,
            'challenges': [],
            'requirements': []
        }
        
        # Vérifier les contraintes difficiles
        if 'kpi_sla' in data:
            for kpi in data['kpi_sla']:
                if kpi.get('indicateur') == 'OTD' and kpi.get('objectif'):
                    try:
                        obj = float(kpi['objectif'].replace('%', ''))
                        if obj >= 99:
                            feasibility['score'] -= 20
                            feasibility['challenges'].append(f"OTD très élevé: {obj}%")
                    except:
                        pass
        
        # Multi-température
        if 'temperature' in data:
            if data['temperature'].get('type') == 'mixte':
                feasibility['score'] -= 15
                feasibility['challenges'].append("Gestion multi-température")
        
        # Ressources importantes
        if 'ressources' in data:
            if data['ressources'].get('flotte', {}).get('type') == 'dediee':
                feasibility['requirements'].append("Flotte dédiée requise")
                feasibility['score'] -= 10
        
        # Déterminer le niveau
        if feasibility['score'] >= 80:
            feasibility['level'] = "ÉLEVÉE"
        elif feasibility['score'] >= 60:
            feasibility['level'] = "MOYENNE"
        else:
            feasibility['level'] = "FAIBLE"
        
        return feasibility
    
    def _identify_operational_alerts(self, data: Dict) -> List[Dict]:
        """
        Identifie les alertes opérationnelles
        """
        alerts = []
        
        # GDP sans certification
        if 'temperature' in data and data['temperature'].get('gdp_requis'):
            if 'certifications' not in data or 'GDP' not in data.get('certifications', {}).get('obligatoires', []):
                alerts.append({
                    'type': 'INCOHÉRENCE',
                    'message': 'GDP requis mais certification non mentionnée',
                    'action': 'Clarifier exigence certification GDP'
                })
        
        # Last-mile B2C complexe
        if 'services' in data:
            dist = data['services'].get('distribution', {})
            if dist.get('b2c') and dist.get('last_mile'):
                alerts.append({
                    'type': 'COMPLEXITÉ',
                    'message': 'Distribution B2C last-mile',
                    'action': 'Prévoir ressources dédiées importantes'
                })
        
        # Matières dangereuses multi-classes
        if 'contraintes_produits' in data:
            dg = data['contraintes_produits'].get('dangereuses', {})
            if dg.get('present') and len(dg.get('classes', [])) > 2:
                alerts.append({
                    'type': 'RÉGLEMENTATION',
                    'message': f"Multiples classes dangereuses: {dg['classes']}",
                    'action': 'Vérifier compatibilités et formations'
                })
        
        return alerts
    
    def _identify_critical_resources(self, data: Dict) -> List[str]:
        """
        Identifie les ressources critiques nécessaires
        """
        critical = []
        
        if 'temperature' in data and data['temperature'].get('controle_requis'):
            critical.append("Flotte température dirigée")
            critical.append("Entrepôts qualifiés GDP")
        
        if 'systemes_it' in data and data['systemes_it'].get('edi', {}).get('requis'):
            critical.append("Système EDI compatible")
        
        if 'ressources' in data:
            if data['ressources'].get('personnel', {}).get('dedie'):
                nb = data['ressources']['personnel'].get('nombre', 0)
                critical.append(f"Personnel dédié ({nb} personnes)")
        
        if 'certifications' in data and data['certifications'].get('obligatoires'):
            for cert in data['certifications']['obligatoires']:
                critical.append(f"Certification {cert}")
        
        return critical