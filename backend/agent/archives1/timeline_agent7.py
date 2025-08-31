# agents/timeline_agent.py
"""
Agent spécialisé dans l'extraction et l'analyse temporelle
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from .base_agent1 import BaseExtractionAgent
import re
from dateutil import parser
import locale

class TimelineExtractionAgent(BaseExtractionAgent):
    """
    Agent pour extraire toutes les dates et créer la timeline
    """
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Prompt pour extraction temporelle
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Tu es un expert en gestion de projets et calendriers d'appels d'offres.
            
            MISSION CRITIQUE: Extraire TOUTES les dates, délais et échéances mentionnés.
            
            ÉLÉMENTS TEMPORELS À EXTRAIRE:
            
            1. DATES DE SOUMISSION:
               - Date limite de remise des offres (1er tour)
               - Date limite 2ème tour (si applicable)
               - Date limite 3ème tour/finale
               - Heure limite précise
               - Fuseau horaire
            
            2. PHASES DE L'APPEL D'OFFRES:
               - Date de publication de l'AO
               - Date limite pour questions
               - Date de réponse aux questions
               - Date de site visit / visite
               - Date de présentation orale
               - Date de négociation
               - Date de décision finale
               - Date de notification
            
            3. DATES DU CONTRAT:
               - Date de début du contrat
               - Date de fin du contrat
               - Durée initiale
               - Périodes de reconduction
               - Date de mise en œuvre / go-live
               - Phase de transition / durée
               - Préavis de résiliation
            
            4. JALONS OPÉRATIONNELS:
               - Date de démarrage des opérations
               - Phases de déploiement
               - Milestones intermédiaires
               - Revues périodiques
               - Dates d'audit
            
            5. PÉRIODICITÉS:
               - Fréquence de reporting
               - Réunions périodiques
               - Revues de performance
               - Mises à jour tarifaires
            
            RÈGLES D'EXTRACTION:
            1. Convertir TOUTES les dates au format ISO (YYYY-MM-DD)
            2. Inclure l'heure si spécifiée (HH:MM)
            3. Noter le fuseau horaire si mentionné
            4. Calculer les durées en jours/mois/années
            5. Identifier les dates critiques vs informatives
            6. Extraire les conditions liées aux dates
            
            FORMAT JSON STRICT:
            {
                "soumission": {
                    "date_limite_principale": "2024-03-15",
                    "heure_limite": "12:00",
                    "fuseau": "CET",
                    "tours": [
                        {
                            "numero": 1,
                            "date": "2024-03-15",
                            "description": "Remise offre initiale"
                        },
                        {
                            "numero": 2,
                            "date": "2024-04-10",
                            "description": "Offre finale"
                        }
                    ]
                },
                "phases_ao": {
                    "publication": "2024-02-01",
                    "questions_deadline": "2024-02-20",
                    "reponses_questions": "2024-02-25",
                    "site_visit": "2024-02-15",
                    "presentation": "2024-03-25",
                    "decision": "2024-04-30",
                    "notification": "2024-05-05"
                },
                "contrat": {
                    "date_debut": "2024-07-01",
                    "date_fin": "2027-06-30",
                    "duree_initiale_mois": 36,
                    "reconductions": [
                        {
                            "numero": 1,
                            "duree_mois": 12,
                            "conditions": "Tacite sauf préavis 3 mois"
                        }
                    ],
                    "go_live": "2024-07-01",
                    "transition": {
                        "debut": "2024-06-01",
                        "fin": "2024-06-30",
                        "duree_jours": 30
                    },
                    "preavis_resiliation_mois": 6
                },
                "jalons": [
                    {
                        "date": "2024-09-01",
                        "description": "Fin phase 1",
                        "critique": true
                    }
                ],
                "periodicites": {
                    "reporting": "mensuel",
                    "revue_performance": "trimestriel",
                    "mise_jour_tarifs": "annuel"
                },
                "timeline_critique": [
                    {
                        "date": "2024-03-15",
                        "evenement": "Date limite soumission",
                        "jours_restants": 45,
                        "alerte": "urgent"
                    }
                ],
                "source_dates": ["extraits exacts mentionnant les dates"]
            }
            """),
            HumanMessage(content="Extrais toutes les dates et échéances de ces documents:\n{context}")
        ])
    
    def _define_validation_patterns(self) -> Dict[str, str]:
        """
        Patterns pour extraction de dates
        """
        return {
            "date_iso": r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            "date_fr": r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            "date_us": r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            "date_text": r'(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
            "date_relative": r'(?:dans|within|sous|before)\s+(\d+)\s+(?:jours?|days?|semaines?|weeks?|mois|months?)',
            "deadline": r'(?:date limite|deadline|échéance|due date|avant le|before|au plus tard)',
            "duration": r'(\d+)\s*(?:ans?|années?|years?|mois|months?|jours?|days?)',
            "time": r'(\d{1,2})[h:](\d{2})(?:\s*(?:CET|GMT|UTC|CEST))?',
            "renewal": r'(?:reconduction|renewal|prolongation|extension)',
            "notice": r'(?:préavis|notice period|notification)\s*(?:de)?\s*(\d+)\s*(?:mois|months?|jours?|days?)'
        }
    
    def _post_process(self, raw_extraction: Dict) -> Dict:
        """
        Post-traitement et enrichissement temporel
        """
        processed = raw_extraction.copy()
        
        # Normalisation des dates
        processed = self._normalize_dates(processed)
        
        # Calcul des durées et intervalles
        processed = self._calculate_durations(processed)
        
        # Création de la timeline
        processed['timeline'] = self._create_timeline(processed)
        
        # Identification des urgences
        processed['urgencies'] = self._identify_urgencies(processed)
        
        # Vérification de cohérence
        processed['coherence'] = self._check_temporal_coherence(processed)
        
        return processed
    
    def _normalize_dates(self, data: Dict) -> Dict:
        """
        Normalise toutes les dates au format ISO
        """
        def parse_date(date_str):
            if not date_str or date_str == "non_specifie":
                return None
            
            try:
                # Essayer différents formats
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']:
                    try:
                        return datetime.strptime(date_str, fmt).date()
                    except:
                        continue
                
                # Utiliser dateutil pour formats complexes
                return parser.parse(date_str, dayfirst=True).date()
            except:
                return None
        
        # Parcourir récursivement et normaliser
        def normalize_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if 'date' in key.lower() and isinstance(value, str):
                        parsed = parse_date(value)
                        if parsed:
                            obj[key] = parsed.isoformat()
                    elif isinstance(value, (dict, list)):
                        normalize_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    normalize_recursive(item)
        
        normalize_recursive(data)
        return data
    
    def _calculate_durations(self, data: Dict) -> Dict:
        """
        Calcule les durées entre dates clés
        """
        durations = {}
        
        # Durée totale du contrat
        if 'contrat' in data:
            contrat = data['contrat']
            if 'date_debut' in contrat and 'date_fin' in contrat:
                try:
                    debut = datetime.fromisoformat(contrat['date_debut'])
                    fin = datetime.fromisoformat(contrat['date_fin'])
                    duree = fin - debut
                    durations['contrat_jours'] = duree.days
                    durations['contrat_mois'] = duree.days // 30
                    durations['contrat_annees'] = duree.days // 365
                except:
                    pass
        
        # Temps jusqu'à la deadline
        if 'soumission' in data and 'date_limite_principale' in data['soumission']:
            try:
                deadline = datetime.fromisoformat(data['soumission']['date_limite_principale'])
                today = datetime.now()
                delta = deadline - today
                durations['jours_avant_deadline'] = delta.days
                
                if delta.days < 0:
                    durations['deadline_status'] = "dépassée"
                elif delta.days < 7:
                    durations['deadline_status'] = "critique"
                elif delta.days < 30:
                    durations['deadline_status'] = "urgent"
                else:
                    durations['deadline_status'] = "normal"
            except:
                pass
        
        data['durations'] = durations
        return data
    
    def _create_timeline(self, data: Dict) -> List[Dict]:
        """
        Crée une timeline ordonnée de tous les événements
        """
        events = []
        
        # Extraire tous les événements avec dates
        def extract_events(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if 'date' in key.lower() and isinstance(value, str):
                        try:
                            date = datetime.fromisoformat(value)
                            events.append({
                                'date': value,
                                'timestamp': date.timestamp(),
                                'event': f"{prefix}{key}",
                                'category': prefix.split('.')[0] if prefix else 'general'
                            })
                        except:
                            pass
                    elif isinstance(value, dict):
                        extract_events(value, f"{prefix}{key}.")
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                extract_events(item, f"{prefix}{key}[{i}].")
        
        extract_events(data)
        
        # Trier par date
        events.sort(key=lambda x: x['timestamp'])
        
        # Ajouter le statut (passé/futur)
        now = datetime.now().timestamp()
        for event in events:
            event['status'] = 'passé' if event['timestamp'] < now else 'futur'
            del event['timestamp']  # Nettoyer
        
        return events
    
    def _identify_urgencies(self, data: Dict) -> List[Dict]:
        """
        Identifie les dates urgentes nécessitant une action
        """
        urgencies = []
        today = datetime.now()
        
        # Vérifier la deadline de soumission
        if 'soumission' in data and 'date_limite_principale' in data['soumission']:
            try:
                deadline = datetime.fromisoformat(data['soumission']['date_limite_principale'])
                delta = deadline - today
                
                if 0 < delta.days <= 7:
                    urgencies.append({
                        'date': data['soumission']['date_limite_principale'],
                        'event': 'Date limite de soumission',
                        'jours_restants': delta.days,
                        'niveau': 'critique',
                        'action': 'Finaliser et soumettre l\'offre immédiatement'
                    })
                elif 7 < delta.days <= 14:
                    urgencies.append({
                        'date': data['soumission']['date_limite_principale'],
                        'event': 'Date limite de soumission',
                        'jours_restants': delta.days,
                        'niveau': 'urgent',
                        'action': 'Accélérer la préparation de l\'offre'
                    })
            except:
                pass
        
        # Vérifier deadline questions
        if 'phases_ao' in data and 'questions_deadline' in data['phases_ao']:
            try:
                deadline = datetime.fromisoformat(data['phases_ao']['questions_deadline'])
                delta = deadline - today
                
                if 0 < delta.days <= 3:
                    urgencies.append({
                        'date': data['phases_ao']['questions_deadline'],
                        'event': 'Date limite pour poser des questions',
                        'jours_restants': delta.days,
                        'niveau': 'urgent',
                        'action': 'Envoyer les questions immédiatement'
                    })
            except:
                pass
        
        return urgencies
    
    def _check_temporal_coherence(self, data: Dict) -> Dict:
        """
        Vérifie la cohérence temporelle
        """
        issues = []
        
        # Vérifier que la date de début est après la décision
        if 'phases_ao' in data and 'contrat' in data:
            try:
                if 'decision' in data['phases_ao'] and 'date_debut' in data['contrat']:
                    decision = datetime.fromisoformat(data['phases_ao']['decision'])
                    debut = datetime.fromisoformat(data['contrat']['date_debut'])
                    if debut < decision:
                        issues.append("Date de début du contrat avant la décision")
            except:
                pass
        
        # Vérifier l'ordre des tours de soumission
        if 'soumission' in data and 'tours' in data['soumission']:
            tours = data['soumission']['tours']
            for i in range(len(tours) - 1):
                try:
                    date1 = datetime.fromisoformat(tours[i]['date'])
                    date2 = datetime.fromisoformat(tours[i+1]['date'])
                    if date2 < date1:
                        issues.append(f"Tour {i+2} avant tour {i+1}")
                except:
                    pass
        
        # Vérifier durée de transition
        if 'contrat' in data and 'transition' in data['contrat']:
            try:
                trans = data['contrat']['transition']
                if 'debut' in trans and 'fin' in trans:
                    debut_trans = datetime.fromisoformat(trans['debut'])
                    fin_trans = datetime.fromisoformat(trans['fin'])
                    debut_contrat = datetime.fromisoformat(data['contrat']['date_debut'])
                    if fin_trans > debut_contrat:
                        issues.append("Transition se termine après le début du contrat")
            except:
                pass
        
        return {
            'is_coherent': len(issues) == 0,
            'issues': issues
        }