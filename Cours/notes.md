# Cours 'Outils et Methodes de l'IA'

## Neurone: Perceptron

- Un réseau de neurone est composé de neurone : 

- Chaque neurone est positionné dans une couche.

- Chaque neurone est connecté à tous les neurones de la couche précédente et de la couche suivante seulement. Pas deux couches non adjacentes.

- TOUS les neurones d'une couche sont connectés aux neurones de la couche précédente et suivante.

- Chaque connexion entre deux neurones a un poids (weight).

- Chaque neurone a un biais (bias).

- theta est une fonction d'activation (sigmoide, tanh, relu, ...)

- la sortie d'un neurone active une fonction d'activation sur la somme pondérée des entrées plus le biais.

- tous les neurones d'une même couche appliquent la même fonction d'activation.