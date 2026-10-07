# Évaluation de LLM : plan de lecture (2 h)

Statut : brouillon à valider avec Asma.
Objectif : une première vue d'ensemble de la façon dont on évalue un LLM, du dataset jusqu'à la production.

## Fil rouge

Un même cas d'usage illustre chaque partie (à choisir avec Asma, voir questions en fin de document).

## Sommaire

| # | Partie | Durée | Contenu |
|---|---|---|---|
| 0 | Pourquoi c'est difficile | 10 min | Pas de réponse unique. Sorties non déterministes. Évaluer le modèle seul ou toute l'application. Offline vs online. |
| 1 | Construire un dataset d'évaluation | 20 min | Golden set. Sources des exemples : logs de prod, experts métier, cas limites, génération synthétique. Annotation. Taille. Versioning. Dataset de non-régression. Exemples concrets. |
| 2 | Métriques automatiques (quantitatif) | 20 min | Exact match, F1. BLEU, ROUGE, BERTScore : principe, calcul sur un exemple, limites. Métriques spécifiques RAG : fidélité, pertinence. |
| 3 | LLM as a judge | 20 min | Principe. Grille de notation. Notation absolue vs comparaison par paires. Biais connus. Calibration contre des notes humaines. |
| 4 | Revue humaine (qualitatif) | 10 min | Protocole. Grille. Accord entre annotateurs. Échantillonnage. Coût. |
| 5 | Risques | 10 min | Hallucination. Toxicité et biais. Prompt injection. Fuite de données. Red teaming. |
| 6 | Outillage et pipeline | 10 min | LangChain et son écosystème d'évaluation, alternatives. Évaluation dans la CI : bloquer une régression avant mise en prod. |
| 7 | En production : A/B test et monitoring | 15 min | Comment les équipes le font, pas pourquoi. Shadow mode, canary, part de trafic gardée en continu. Feedback utilisateur explicite et implicite. Monitoring et alertes. |
| 8 | Synthèse | 5 min | Quelle méthode à quel moment. Checklist pour monter une évaluation. |

## Questions à trancher avec Asma

1. Quel cas d'usage sert de fil rouge : chatbot, RAG, résumé, classification, agent ?
2. Quel niveau mathématique : intuition seule, ou formules de BLEU, ROUGE et BERTScore ?
3. Un outil est-il imposé ou déjà utilisé en interne ?
4. Faut-il ajouter un notebook pratique, en plus de la lecture ?
5. Les benchmarks publics (classements de modèles) entrent-ils dans le périmètre ?
