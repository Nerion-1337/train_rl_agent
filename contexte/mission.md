Mission - Pilotez l'atterrisseur lunaire Eagle - 1Vous êtes un ingénieur Machine Learning junior spécialisé en apprentissage par renforcement dans l’entreprise AstroDynamics.

 

 

                                             

                                       Logo d'AstroDynamics

 

 

 

L’entreprise est un leader dans le développement de systèmes autonomes pour l'exploration spatiale.

Elle souhaite automatiser les procédures d'atterrissage automatique de ses modules afin d'améliorer la sécurité et d'optimiser la consommation de carburant, un facteur critique pour la durée des missions.

 

Vous êtes chargé de développer le pilote automatique pour le nouveau module d'atterrissage "Eagle - 1". Votre objectif est de garantir un atterrissage en douceur et précis sur une zone cible simulée.

 

Vous recevez un brief de mission de votre chef de projet R&D.

 

Vous dressez une liste de vos futurs livrables :

Un notebook Colab (.ipynb) propre et commenté, présentant votre démarche complète : 

exploration, 

choix de l'algorithme, 

entraînement, 

optimisation des hyperparamètres,

évaluation finale du modèle.

Une vidéo (.mp4) d'une durée de 20 à 30 secondes, montrant une performance réussie de votre pilote automatique.

Une API (code source) : Interface pour interagir avec le modèle RL.

Une interface graphique (GUI, code source) : Visualisation d’une partie jouée par l’agent.

Un tableau de bord (Looker Studio/Streamlit/Gradio) : Suivi interactif des performances de l’agent.

Cette mission est entièrement guidée.
Vous pouvez suivre les étapes ci-dessous.

Étapes
Avant de chercher la performance, vous devez comprendre votre terrain de jeu et établir un point de départ. Familiarisez-vous avec l'environnement "LunarLander-v3" et entraînez un premier agent avec les paramètres par défaut pour mesurer une performance initiale.

 
Prérequis
Avoir réalisé les parties 1 à 4 du notebook, en particulier l'utilisation de Stable-Baselines3.

 
Résultat attendu 
Un premier modèle est entraîné et sa récompense moyenne est calculée et documentée dans le notebook.

 

Recommandations 
Commencer par explorer les espaces d'action et d'observation de LunarLander-v3 pour comprendre ce que l'agent peut faire et voir.

Adaptez l’algorithme en fonction de l’espace d’action : 

PPO pour les espaces de contrôle continus, 

DQN pour les discrets.

Utiliser la fonction evaluate_policy sur au moins 50 épisodes pour obtenir une mesure de performance de base fiable.

 

Points de vigilance 
Ne passez pas trop de temps à essayer d'obtenir un bon score à cette étape. 

L'objectif est de mettre en place rapidement un pipeline fonctionnel.

 
Outils 
Gymnasium

Stable-Baselines3

Google colab


Ressource
La documentation de l'environnement LunarLander-v3

C'est le cœur de votre mission. En vous basant sur votre performance initiale, vous devez maintenant expérimenter pour améliorer votre agent. L'objectif est de dépasser de manière constante une récompense moyenne de 200 points, le seuil d'un atterrissage réussi.

 
Prérequis 
Avoir établi un modèle de référence et sa performance à l'étape 1.

 
Résultat attendu  
Un agent entraîné qui atteint une récompense moyenne supérieure à 200 sur 100 épisodes d'évaluation.

 

Recommandations 
Ne modifier qu’un seul hyperparamètre à la fois pour isoler son effet (ex : learning_rate, gamma, n_steps).

Utiliser TensorBoard (ou équivalent) pour visualiser et comparer les courbes de récompense de vos différentes expériences.

Documenter chaque expérience dans votre notebook : quels paramètres ont été modifiés, quel a été le résultat ?

Sauvegarder les modèles qui montrent des signes prometteurs pour ne pas perdre votre travail.

 

Points de vigilance 
L'entraînement peut prendre du temps. Soyez patient et planifiez vos expériences.

Le score cible de 200 doit être une moyenne stable (avec un écart-type faible), pas un coup de chance sur un seul épisode.
 

Ressources 
Article : Hyperparameter Tuning for Reinforcement Learning


Développez une API pour interagir avec le modèle RL, une interface graphique (GUI) pour visualiser une partie jouée, et un tableau de bord interactif pour suivre les performances de l’agent.

 
Prérequis 
Avoir optimisé un agent avec une récompense moyenne >200.

Avoir sauvegardé le meilleur modèle.


Résultats attendus  
Une API fonctionnelle, 

Un GUI affichant une partie jouée, 

Un tableau de bord interactif montrant les métriques de performance sur les différentes parties jouées.

 

Recommandations  
Développer une API avec FastAPI ou Flask : créer un endpoint (ex. : /play) acceptant un état et renvoyant une action.

Créer un GUI avec Streamlit ou Gradio pour visualiser une partie jouée par l’agent (ex. : animation de l’atterrissage).

Construire un tableau de bord (Looker Studio, Streamlit, ou Gradio) affichant les courbes de récompense, moyenne, écart-type, les décisions prises par l’IA en fonction du type de circonstance etc...

Documenter le code source et les instructions d’utilisation dans le notebook.

 

 

Points de vigilance  
Toute la logique RL doit être côté API (backend) et non côté GUI (frontend).

 Utilisez des générateurs ou des batchs plus petits en cas de saturation de la RAM.

 
Outils 
Streamlit

Gradio

Looker Studio

FastAPI


Ressources 
FastAPI Tutorial

Streamlit Docs

Gradio Quickstart

Welcome to Looker Studio