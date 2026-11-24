Exercice 3 - Remplacez la Q-table par un cerveau neuronal (DQN)

Dans cet exercice, vous implémenterez un Deep Q-Network (DQN) pour "CartPole-v1" de deux manières :

 

Manuellement avec PyTorch, pour comprendre les rouages du DQN (réseau de neurones, Experience Replay, Target Network).

Avec Stable-Baselines3, pour découvrir comment les professionnels accélèrent le développement avec des bibliothèques de haut niveau.

Cet exercice est entièrement guidé.
Suivez les étapes ci-dessous.

 

Étapes
Créez deux classes Python :

DQN (héritant de torch.nn.Module) pour le réseau de neurones.

ReplayBuffer pour stocker et échantillonner les transitions (état, action, récompense, nouvel état).

 
Prérequis
Connaissance conceptuelle des réseaux de neurones et PyTorch.


Résultat attendu 
Classes DQN et ReplayBuffer fonctionnelles.

Code instanciant un DQN et affichant son architecture.

 

Recommandations
Pour la classe DQN :

Définissez les couches (nn.Linear) dans le constructeur __init__.

Définissez le passage des données à travers ces couches dans la méthode forward, en utilisant une fonction d'activation comme F.relu entre elles.

Pour la classe ReplayBuffer :

Utilisez une collections.deque(maxlen=...) pour stocker efficacement les expériences avec une capacité fixe.

La méthode push ajoute un tuple d'expérience au buffer.

La méthode sample utilise random.sample() pour extraire un batch aléatoire.

Outils 
PyTorch

Collections.deque

La boucle d'entraînement du DQN vous sera fournie par votre mentor car elle est complexe.

Demandez-lui de vous en partager le code. 

 

Votre tâche n'est donc pas de réécrire la boucle, mais de l'analyser, de la comprendre et de la faire fonctionner. Lisez attentivement les commentaires pour saisir la logique de chaque bloc :

la sélection d'action,
le calcul des valeurs Q,
le calcul de la valeur Q cible (avec le target network),
le calcul de la perte
et l'étape d'optimisation.
 
Prérequis
Avoir défini les classes DQN et ReplayBuffer.


Résultat attendu 
L'exécution complète de la cellule d'entraînement du DQN, affichant la progression des récompenses tous les 50 épisodes, et un graphique final montrant la courbe des récompenses par épisode.

 

Recommandations 
Concentrez-vous sur la fonction optimize_model(). C'est là que la "magie" opère.

Repérez où le policy_net est utilisé pour prédire les Q-values des actions prises.

Repérez où le target_net est utilisé pour estimer la valeur maximale des états suivants. C'est un point clé.

Comprenez comment la loss est calculée entre ces deux quantités.

 
Points de vigilance  
Portez une attention particulière à la gestion des tenseurs PyTorch. Remarquez les conversions (torch.FloatTensor, torch.LongTensor) et les manipulations de forme (.unsqueeze(1), .gather(1, ...), .max(1)[0]).

Le target_net est mis à jour périodiquement en copiant les poids du policy_net. Trouvez la ligne de code qui effectue cette opération.

Outils 
Google colab

Gymnasium

 
Ressources 
Tutoriel DQN de PyTorch


Utilisez la classe DQN de Stable-Baselines3 pour entraîner, évaluer et sauvegarder un agent sur "CartPole-v1".

 
Prérequis
Avoir réalisé les étapes 1 et 2 pour apprécier la simplification.

Avoir installé stable-baselines3.


Résultats attendus 
Un modèle DQN.

L’avoir entraîné sur au moins 5000 pas (25 000, c’est mieux).

Récompense moyenne évaluée sur 100 épisodes.

Modèle sauvegardé (.zip).

 

Recommandations 
Importez DQN et evaluate_policy depuis stable_baselines3.

Instanciez : model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./logs/").

Entraînez : model.learn(total_timesteps=25000).

Évaluez : mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100).

Sauvegardez : model.save("dqn_cartpole").

 
Points de vigilance  
Distinguez le DQN manuel du DQN de SB3.

total_timesteps = nombre d’interactions avec l’environnement, ce n’est pas le nombre d’épisodes.

Outils
SB3

 
Ressources 
Doc DQN SB3

Tutoriel SB3