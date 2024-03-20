# Flappy Birds

Implementation of the game Flappy Birds with genetic algorithm using Tensorflow as Neural Net and pygame as Gui.

### Usage & Functionality 

First you have to install the required packages. 
To run the game you have to create an Object of the Game() class. 
```
game = Game(human=False, algorithm=0, algorithm_random=0, ai=3)
```
The possible parameter are `human` Boolean for a human controlled player, `algorithm` Integer for the amount of algorithm controlled player(the algorithm makes no mistakes), `algorithm_random` Integer for the amount of players with the base functionality of the previous algorithm plus some random moves to make it not flawless and `ai` Integer with the amount of players controlled by an random initialized ai. 

The human and randomized-algorithm player will regenerate as soon as they die in a "playable" position. The ai player generates in "generations" meaning it will wait till all current players have died and depending of their living times will the best models regenerate with some small tweaks in their weights. This enables the possibility to generate a very good to flawless model with a very small amount of players and generations (in the video 3 players and 8 generations). 

### Ai learning:
![Learning Ai](https://raw.githubusercontent.com/Skilsu/FlappyBirdsPygame/master/data/FlappyBirdsAi.gif)

The video is buffering/freezing due to insufficient processing power on my laptop.

### Contributing
While the current code is fairly functional, it could benefit from the following contributions:
* adding a good looking GUI (Pictures to include instead of circles and rectangles)
