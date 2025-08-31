# agents/legal_agent.py
"""
Agent spécialisé dans l'extraction des clauses juridiques et contractuelles
"""
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from .base_agent1 import BaseExtractionAgent
import re

class LegalExtractionAgent(BaseExtractionAgent):
    """
    Agent pour extraire les aspects juridiques et contractuels
    """
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Prompt pour extraction juridique
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Tu es un juriste expert en droit des contrats et appels d'offres.
            
            MISSION: Extraire TOUTES les clauses juridiques et identifier les risques légaux.
            
            ÉLÉMENTS JURIDIQUES À EXTRAIRE:
            
            1. RESPONSABILITÉS:
               - Responsabilité du prestataire (étendue, limites)
               - Responsabilité du client
               - Partage des responsabilités
               - Cas d'exonération
               - Responsabilité en cas de sous-traitance
               - Responsabilité produit/marchandise
            
            2. ASSURANCES:
               - Types d'assurance requis (RC, marchandises, professionnelle)
               - Montants minimum de couverture
               - Franchises acceptables
               - Bénéficiaires des polices
               - Renonciation à recours
            
            3. PÉNALITÉS ET MALUS:
               - Pénalités de retard (montant, calcul)
               - Pénalités de non-conformité
               - Malus performance (KPI)
               - Plafonnement des pénalités
               - Mécanisme de bonus/malus
               - Rétention sur factures
            
            4. LIMITATION DE RESPONSABILITÉ:
               - Plafond global de responsabilité
               - Exclusions de responsabilité
               - Dommages indirects exclus ou non
               - Limitation par événement
               - Limitation annuelle
            
            5. FORCE MAJEURE:
               - Définition de la force majeure
               - Événements couverts
               - Procédure de notification
               - Conséquences (suspension, résiliation)
               - Pandémie incluse ou non
            
            6. RÉSILIATION:
               - Causes de résiliation
               - Préavis requis
               - Résiliation pour faute
               - Résiliation pour convenance
               - Indemnités de résiliation
               - Sort des stocks/actifs
            
            7. PROPRIÉTÉ INTELLECTUELLE:
               - Propriété des développements
               - Licences accordées
               - Confidentialité
               - Durée de confidentialité
               - Protection des données (RGPD)
            
            8. CONFORMITÉ:
               - Certifications requises (ISO, GDP, etc.)
               - Conformité RSE
               - Code de conduite
               - Anti-corruption
               - Audit et contrôle
               - Sanctions internationales
            
            9. GOUVERNANCE:
               - Loi applicable
               - Juridiction compétente
               - Arbitrage
               - Langue du contrat
               - Ordre de priorité des documents
            
            10. CLAUSES PARTICULIÈRES:
                - Clause de hardship
                - Clause de benchmarking
                - Most Favored Nation
                - Change of control
                - Non-sollicitation
                - Exclusivité
            
            FORMAT JSON STRICT:
            {
                "responsabilites": {
                    "prestataire": {
                        "etendue": "description",
                        "limites": ["liste des limites"],
                        "sous_traitance": "autorisée avec accord"
                    },
                    "marchandises": {
                        "transfert_risque": "ExW, DDP, etc.",
                        "responsable": "transporteur|expediteur"
                    }
                },
                "assurances": {
                    "rc_generale": {
                        "requis": true,
                        "montant_min": 5000000,
                        "devise": "EUR"
                    },
                    "marchandises": {
                        "requis": true,
                        "couverture": "tous risques",
                        "montant": "110% valeur"
                    },
                    "autres": ["RC pro", "flotte"]
                },
                "penalites": {
                    "retard": {
                        "taux": "1% par jour",
                        "plafond": "10% valeur commande",
                        "franchise": "24h"
                    },
                    "performance": [
                        {
                            "kpi": "OTD",
                            "seuil": "95%",
                            "penalite": "5000 EUR/point"
                        }
                    ],
                    "plafond_global": "15% CA annuel"
                },
                "limitation_responsabilite": {
                    "plafond_global": "1M EUR",
                    "par_evenement": "500K EUR",
                    "dommages_indirects": "exclus",
                    "exceptions": ["dol", "faute lourde"]
                },
                "force_majeure": {
                    "existe": true,
                    "pandemie_incluse": false,
                    "evenements": ["guerre", "catastrophe naturelle"],
                    "notification_delai": "48h"
                },
                "resiliation": {
                    "pour_faute": {
                        "preavis": "30 jours",
                        "mise_demeure": true
                    },
                    "pour_convenance": {
                        "possible": true,
                        "preavis": "6 mois",
                        "indemnite": "3 mois CA"
                    }
                },
                "propriete_intellectuelle": {
                    "developpements": "propriété client",
                    "confidentialite_duree": "5 ans",
                    "rgpd": true
                },
                "conformite": {
                    "certifications": ["ISO 9001", "GDP"],
                    "rse": true,
                    "anti_corruption": true,
                    "audit": "annuel"
                },
                "gouvernance": {
                    "loi": "Française",
                    "juridiction": "Tribunal de Commerce Paris",
                    "langue": "Français",
                    "arbitrage": false
                },
                "clauses_particulieres": {
                    "hardship": false,
                    "benchmarking": true,
                    "exclusivite": false,
                    "change_control": true
                },
                "risques_identifies": [
                    {
                        "clause": "description",
                        "risque": "élevé|moyen|faible",
                        "impact": "description impact",
                        "mitigation": "action proposée"
                    }
                ],
                "clauses_sources": ["extraits exacts des clauses critiques"]
            }
            """),
            HumanMessage(content="Analyse ces documents et extrais toutes les clauses juridiques:\n{context}")
        ])
    
    def _define_validation_patterns(self) -> Dict[str, str]:
        """
        Patterns pour clauses juridiques
        """
        return {
            "responsabilite": r'(?:responsabilit[ée]|liability|responsible)',
            "assurance": r'(?:assurance|insurance|police|couverture)',
            "penalite": r'(?:p[ée]nalit[ée]|penalty|malus|amende|sanction)',
            "force_majeure": r'(?:force\s+majeure|cas\s+fortuit|act\s+of\s+god)',
            "resiliation": r'(?:r[ée]siliation|termination|rupture|fin\s+de\s+contrat)',
            "confidentialite": r'(?:confidentialit[ée]|confidential|NDA|secret)',
            "rgpd": r'(?:RGPD|GDPR|donn[ée]es\s+personnelles|privacy)',
            "plafond": r'(?:plafond|limite|cap|maximum|ceiling)',
            "juridiction": r'(?:juridiction|tribunal|court|comp[ée]tent)',
            "loi_applicable": r'(?:loi\s+applicable|governing\s+law|droit\s+applicable)',
            "arbitrage": r'(?:arbitrage|arbitration|CCI|ICC)',
            "exclusivite": r'(?:exclusivit[ée]|exclusive|non-compete)'
        }
    
    def _post_process(self, raw_extraction: Dict) -> Dict:
        """
        Post-traitement et analyse de risque juridique
        """
        processed = raw_extraction.copy()
        
        # Analyse de risque
        processed['risk_analysis'] = self._analyze_legal_risks(processed)
        
        # Identification des clauses manquantes
        processed['missing_clauses'] = self._identify_missing_clauses(processed)
        
        # Score de protection juridique
        processed['protection_score'] = self._calculate_protection_score(processed)
        
        # Points d'attention critiques
        processed['critical_points'] = self._identify_critical_points(processed)
        
        # Recommandations
        processed['recommendations'] = self._generate_recommendations(processed)
        
        return processed
    
    def _analyze_legal_risks(self, data: Dict) -> Dict:
        """
        Analyse détaillée des risques juridiques
        """
        risks = {
            'high': [],
            'medium': [],
            'low': [],
            'score': 0
        }
        
        # Analyse limitation de responsabilité
        if 'limitation_responsabilite' in data:
            limit = data['limitation_responsabilite']
            if not limit.get('plafond_global'):
                risks['high'].append({
                    'clause': 'Limitation responsabilité',
                    'issue': 'Pas de plafond de responsabilité',
                    'impact': 'Exposition illimitée'
                })
                risks['score'] += 10
            elif limit.get('dommages_indirects') != 'exclus':
                risks['medium'].append({
                    'clause': 'Dommages indirects',
                    'issue': 'Dommages indirects non exclus',
                    'impact': 'Risque financier accru'
                })
                risks['score'] += 5
        
        # Analyse pénalités
        if 'penalites' in data:
            pen = data['penalites']
            if not pen.get('plafond_global'):
                risks['high'].append({
                    'clause': 'Pénalités',
                    'issue': 'Pas de plafond global des pénalités',
                    'impact': 'Pénalités potentiellement illimitées'
                })
                risks['score'] += 8
            
            if 'retard' in pen:
                taux = pen['retard'].get('taux', '')
                if '2%' in taux or '3%' in taux or '5%' in taux:
                    risks['medium'].append({
                        'clause': 'Pénalités retard',
                        'issue': f'Taux élevé: {taux}',
                        'impact': 'Impact financier important'
                    })
                    risks['score'] += 4
        
        # Analyse force majeure
        if 'force_majeure' in data:
            fm = data['force_majeure']
            if not fm.get('existe'):
                risks['high'].append({
                    'clause': 'Force majeure',
                    'issue': 'Pas de clause de force majeure',
                    'impact': 'Aucune protection événements exceptionnels'
                })
                risks['score'] += 7
            elif not fm.get('pandemie_incluse'):
                risks['medium'].append({
                    'clause': 'Force majeure',
                    'issue': 'Pandémie non couverte',
                    'impact': 'Risque COVID non protégé'
                })
                risks['score'] += 3
        
        # Analyse résiliation
        if 'resiliation' in data:
            res = data['resiliation']
            if 'pour_convenance' in res:
                if res['pour_convenance'].get('possible') and not res['pour_convenance'].get('indemnite'):
                    risks['high'].append({
                        'clause': 'Résiliation',
                        'issue': 'Résiliation sans indemnité',
                        'impact': 'Perte investissements'
                    })
                    risks['score'] += 6
        
        # Calcul niveau de risque global
        if risks['score'] >= 20:
            risks['level'] = 'CRITIQUE'
        elif risks['score'] >= 10:
            risks['level'] = 'ÉLEVÉ'
        elif risks['score'] >= 5:
            risks['level'] = 'MODÉRÉ'
        else:
            risks['level'] = 'FAIBLE'
        
        return risks
    
    def _identify_missing_clauses(self, data: Dict) -> List[str]:
        """
        Identifie les clauses importantes manquantes
        """
        essential_clauses = {
            'limitation_responsabilite': 'Limitation de responsabilité',
            'force_majeure': 'Force majeure',
            'assurances': 'Assurances',
            'resiliation': 'Modalités de résiliation',
            'confidentialite': 'Confidentialité',
            'gouvernance.loi': 'Loi applicable',
            'gouvernance.juridiction': 'Juridiction compétente'
        }
        
        missing = []
        for key, description in essential_clauses.items():
            if '.' in key:
                parts = key.split('.')
                if parts[0] not in data or parts[1] not in data[parts[0]]:
                    missing.append(description)
            elif key not in data or not data[key]:
                missing.append(description)
        
        return missing
    
    def _calculate_protection_score(self, data: Dict) -> Dict:
        """
        Calcule un score de protection juridique
        """
        score = 50  # Base
        factors = []
        
        # Facteurs positifs
        if 'limitation_responsabilite' in data and data['limitation_responsabilite'].get('plafond_global'):
            score += 10
            factors.append("+ Plafond de responsabilité défini")
        
        if 'force_majeure' in data and data['force_majeure'].get('existe'):
            score += 10
            factors.append("+ Clause de force majeure présente")
        
        if 'penalites' in data and data['penalites'].get('plafond_global'):
            score += 10
            factors.append("+ Plafond global des pénalités")
        
        # Facteurs négatifs
        if 'penalites' in data and 'retard' in data['penalites']:
            taux = data['penalites']['retard'].get('taux', '')
            if any(x in taux for x in ['2%', '3%', '5%']):
                score -= 10
                factors.append("- Taux de pénalité élevé")
        
        if 'resiliation' in data:
            if data['resiliation'].get('pour_convenance', {}).get('possible'):
                if not data['resiliation']['pour_convenance'].get('indemnite'):
                    score -= 15
                    factors.append("- Résiliation sans indemnité")
        
        # Normaliser entre 0 et 100
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'niveau': 'Bon' if score >= 70 else 'Moyen' if score >= 40 else 'Faible',
            'factors': factors
        }
    
    def _identify_critical_points(self, data: Dict) -> List[Dict]:
        """
        Identifie les points juridiques critiques
        """
        critical = []
        
        # Vérifier responsabilité illimitée
        if 'limitation_responsabilite' not in data or not data.get('limitation_responsabilite', {}).get('plafond_global'):
            critical.append({
                'type': 'CRITIQUE',
                'clause': 'Responsabilité',
                'issue': 'Responsabilité potentiellement illimitée',
                'action': 'Négocier un plafond de responsabilité'
            })
        
        # Vérifier assurances
        if 'assurances' in data:
            if data['assurances'].get('rc_generale', {}).get('montant_min', 0) > 10000000:
                critical.append({
                    'type': 'ATTENTION',
                    'clause': 'Assurances',
                    'issue': 'Montant d\'assurance très élevé requis',
                    'action': 'Vérifier capacité assurance actuelle'
                })
        
        return critical
    
    def _generate_recommendations(self, data: Dict) -> List[str]:
        """
        Génère des recommandations juridiques
        """
        recommendations = []
        
        if 'risk_analysis' in data:
            if data['risk_analysis']['level'] in ['CRITIQUE', 'ÉLEVÉ']:
                recommendations.append("⚠️ Faire réviser le contrat par le service juridique avant signature")
        
        if 'missing_clauses' in data and data['missing_clauses']:
            recommendations.append(f"📝 Demander l'ajout des clauses manquantes: {', '.join(data['missing_clauses'][:3])}")
        
        if 'protection_score' in data and data['protection_score']['score'] < 40:
            recommendations.append("🛡️ Négocier des clauses de protection supplémentaires")
        
        if 'penalites' in data and not data['penalites'].get('plafond_global'):
            recommendations.append("💰 Négocier un plafond global des pénalités (ex: 10-15% du CA annuel)")
        
        return recommendations