Exercice 1 - Découvrez les blocs de construction de l'Apprentissage par Renforcement Exercice

Dans cet exercice, vous allez interagir pour la première fois avec un environnement d'apprentissage par renforcement. L'objectif est de vous familiariser avec le cycle fondamental "observation -> action -> récompense" qui est au cœur de cette discipline. 

Cet exercice est entièrement guidé.
Vous pouvez suivre les étapes ci-dessous.

Étapes
Vous allez d'abord créer une instance de l'environnement classique "CartPole-v1" à l'aide de la bibliothèque Gymnasium. Ensuite, vous explorerez ses propriétés fondamentales :

l'espace d'observation c’est-à-dire ce que l'agent "voit"
l'espace d'action c’est-à-dire ce que l'agent "peut faire".
Prérequis 
 

Avoir installé les librairies suivantes : 

gymnasium pour les environnements,

stable-baselines3 pour les algorithmes pré-implémentés,

matplotlib pour la visualisation.


Résultat attendu
Des cellules de code qui affichent la structure de l'espace d'observation et de l'espace d'action, ainsi que des exemples.

 

Recommandations 
Utilisez gym.make("NomDeLEnvironnement") pour créer l'environnement.

Accédez aux attributs .observation_space et  .action_space de l'objet environnement créé.

Utilisez la méthode .sample() sur ces espaces pour voir à quoi ressemble une observation ou une action aléatoire.

 
Points de vigilance 
Noter le type de chaque espace. CartPole a un espace d'observation Box (continu) et un espace d'action Discrete (discret). 

Cette distinction sera cruciale plus tard.

 
Outils
Google Colab

Gymnasium

 
Ressources 
Le guide de base de Gymnasium sur les environnements


Maintenant que vous comprenez l'environnement, vous allez coder une boucle simple qui fait interagir un agent avec cet environnement. Pour l'instant, l'agent sera très simple : il choisira ses actions de manière complètement aléatoire. Vous ferez tourner cette simulation pour 10 "épisodes" et afficherez la récompense totale pour chaque épisode.

 

Prérequis 
Avoir créé un environnement et exploré ses espaces.


Résultat attendu 
Une boucle for qui exécute 10 épisodes complets. 

Pour chaque épisode, une boucle while interne doit s'exécuter jusqu'à ce que l'épisode se termine, en choisissant et en appliquant une action aléatoire à chaque étape. 

La récompense totale de chaque épisode doit être affichée.

 

Recommandations 
Commencez chaque épisode par env.reset(). Cette fonction réinitialise le jeu et vous donne la toute première observation.

Dans la boucle while, utilisez env.action_space.sample() pour choisir une action aléatoire.

Passez cette action à env.step(action). Stockez les 5 valeurs qu'elle retourne.

La boucle while doit continuer tant que l'épisode n'est ni terminated ni truncated.

N'oubliez pas d'ajouter la reward reçue à une variable total_reward à chaque étape.

À la fin de votre script, utilisez env.close() pour fermer l'environnement proprement.

 
Points de vigilance 
Assurez-vous de bien réinitialiser les variables (comme total_reward, terminated, truncated) au début de chaque nouvel épisode dans la boucle for.

Outils 
Google Colab

Gymnasium

 
Ressources 
Exemple d'utilisation de base de Gymnasium