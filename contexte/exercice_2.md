 Exercice 2 - Entraînez un agent avec une table de décision (Q-table)

 Dans cet exercice, vous allez implémenter de zéro votre premier véritable algorithme d'apprentissage : le Q-Learning. Vous allez construire une "table de qualité" (Q-table) qui guidera les décisions de votre agent pour résoudre l'environnement "FrozenLake-v1".

 

Pourquoi FrozenLake et pas Cartpole ?

Pour des raisons pédagogiques, il est plus aisé de comprendre le Q-learning avec FrozenLake qu'avec CartPole. La simplicité de l'espace d'actions de FrozenLake rend l'appréhension des concepts fondamentaux nettement plus directe.


Cet exercice est entièrement guidé.
Vous pouvez suivre les étapes ci-dessous.

 

Étapes
Commencez par créer l'environnement "FrozenLake-v1". Contrairement à CartPole, cet environnement a un nombre fini d'états, ce qui le rend idéal pour le Q-Learning. Ensuite, initialisez votre Q-table sous la forme d'un tableau NumPy rempli de zéros, avec une ligne pour chaque état et une colonne pour chaque action possible.

 
Prérequis
Avoir installé les librairies suivantes: 

Avoir fini le premier exercice.

Connaître les bases de la bibliothèque NumPy.


Résultat attendu  
Une Q-table (tableau NumPy) initialisée avec les bonnes dimensions ((16, 4) pour FrozenLake) et remplie de zéros. 

 

Recommandations 
Utilisez env.observation_space.n et env.action_space.n pour obtenir dynamiquement les dimensions nécessaires pour votre Q-table.

Utilisez np.zeros() pour créer le tableau.

Outils 
Numpy

Gymnasium

C'est le cœur de l'exercice. Vous allez coder la boucle d'entraînement principale. À chaque pas de temps, votre agent devra décider s'il explore (action aléatoire) ou s'il exploite ses connaissances (meilleure action de la Q-table). Après avoir exécuté l'action, vous mettrez à jour la valeur correspondante dans la Q-table en utilisant la formule de mise à jour du Q-Learning.

 

Prérequis 
Avoir initialisé la Q-table.

Avoir défini les hyperparamètres (learning_rate, discount_factor, epsilon, etc.).


Résultat attendu  
Une Q-table remplie de valeurs apprises après des milliers d'épisodes d'entraînement.

 

Recommandations  
Pour la stratégie epsilon-greedy, générez un nombre aléatoire entre 0 et 1. S'il est inférieur à epsilon, explorez. Sinon, exploitez.

Pour exploiter, utilisez np.argmax(q_table[state, :]) pour trouver l'index (l'action) de la plus grande valeur Q pour l'état actuel.

Traduisez la formule de Bellman en code Python : nouvelle_valeur = ancienne_valeur + lr * (recompense + gamma * max_q_futur - ancienne_valeur).

N'oubliez pas de réduire epsilon à la fin de chaque épisode pour que l'agent explore moins à mesure qu'il apprend.

 
Points de vigilance  
Assurez-vous de bien utiliser new_state pour trouver la valeur Q future maximale (np.max(q_table[new_state, :])).

La mise à jour se fait sur la paire (state, action) qui a été utilisée pour la transition.

 
Ressources 
Tutoriel sur le Q-Learning

Maintenant que votre agent est entraîné, il est temps de mesurer ses performances. Vous allez exécuter un certain nombre d'épisodes d'évaluation (par exemple, 100) en utilisant la Q-table apprise. Cette fois, l'agent ne doit jamais explorer ; il doit toujours choisir la meilleure action qu'il connaît. Calculez son taux de réussite.

 
Prérequis 
Avoir une Q-table entraînée à l'étape précédente.


Résultat attendu 
Un script qui calcule et affiche le taux de réussite de l'agent sur un grand nombre d'épisodes, par exemple : "Taux de réussite sur 100 épisodes : 72%".

 

Recommandations  
Créez une nouvelle boucle d'évaluation, similaire à la boucle d'entraînement.

La grande différence : pour choisir l'action, utilisez toujours action = np.argmax(q_table[state, :]). Il n'y a plus de epsilon.

Incrémentez un compteur de victoires (total_wins) chaque fois qu'un épisode se termine avec une récompense de 1.0.

 
Points de vigilance  
Ne modifiez plus la Q-table pendant l'évaluation ! La phase d'apprentissage est terminée.

N'oubliez pas de réinitialiser l'environnement à chaque nouvel épisode d'évaluation.