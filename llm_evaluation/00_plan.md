# Évaluation de LLM : plan de lecture (2 h)

Statut : brouillon à valider avec Asma.
Objectif : une première vue d'ensemble de la façon dont on évalue un LLM, du dataset jusqu'à la production.

## Fil rouge

Le cas d'usage e-advice illustre chaque partie. Son contexte détaillé reste à intégrer.

## Sommaire

| # | Partie | Durée | Contenu |
|---|---|---|---|
| 0 | Pourquoi c'est difficile | 10 min | Pas de réponse unique. Sorties non déterministes. Évaluer le modèle seul ou toute l'application. Offline vs online. |
| 1 | Construire un dataset d'évaluation | 20 min | Golden set. Sources des exemples : logs de prod, experts métier, cas limites, génération synthétique. Annotation. Taille. Versioning. Dataset de non-régression. Exemples concrets. |
| 2 | Métriques automatiques (quantitatif) | 20 min | Exact match, F1. BLEU, ROUGE, BERTScore : ce que chaque métrique cherche à capturer, expliqué simplement, puis la formule, un calcul sur un exemple et les limites. Métriques spécifiques RAG : fidélité, pertinence. |
| 3 | LLM as a judge | 20 min | Principe. Grille de notation. Notation absolue vs comparaison par paires. Biais connus. Calibration contre des notes humaines. |
| 4 | Revue humaine (qualitatif) | 10 min | Protocole. Grille. Accord entre annotateurs. Échantillonnage. Coût. |
| 5 | Risques | 10 min | Hallucination. Toxicité et biais. Prompt injection. Fuite de données. Red teaming. |
| 6 | Outillage et pipeline | 10 min | Aucun outil imposé : LangChain cité comme exemple, sans démonstration. Évaluation dans la CI : bloquer une régression avant mise en prod. |
| 7 | En production : A/B test et monitoring | 15 min | Comment les équipes le font, pas pourquoi. Shadow mode, canary, part de trafic gardée en continu. Feedback utilisateur explicite et implicite. Monitoring et alertes. |
| 8 | Synthèse | 5 min | Quelle méthode à quel moment. Checklist pour monter une évaluation. |

## Décisions prises

1. Fil rouge : e-advice, contexte détaillé à venir.
2. Formules incluses, mais l'explication du principe passe en premier.
3. Aucun outil imposé.
4. Lecture seule, sans notebook.
5. Benchmarks publics hors périmètre.
