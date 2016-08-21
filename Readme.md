## Neural Networks For Pong

![Atari Pong](https://i.ytimg.com/vi/moqeZusEMcA/hqdefault.jpg)

### Introduction

The code in this repository will let you train a Neural Network to play Pong solely based on the input frames of the game and the results of each round.
The code in `me_pong.py` is intended to be a simpler to follow version of `pong.py` which was written by Andrej Karpathy.
You can play around with other such Atari games at the [Openai Gym](https://gym.openai.com).

### Setup

1. Follow the instructions for installing Openai Gym [here](https://gym.openai.com/docs). You may need to install `cmake` first.
2. Run `pip install -e '.[atari]'`.
3. Run `python me_pong.py`

### Seeing the game

If you want to see the game happen, open `me_pong.py` and uncomment the following line:
```python
# env.render()
 ```

### Performance

According to the blog post, this algorithm should take around 3 days of training on a Macbook to start beating the computer.

Consider tweaking the hyperparameters or using CNNs to boost the performance further.

### Misc.

Note, if you want to run Andrej Karpathy's original code, run `python pong.py`.

### Credit

This is based off of the work of Andrej Karpathy's great blog post and code [here](http://karpathy.github.io/2016/05/31/rl/)

## Tutorial

To see a step by step tutorial for building this, see this [blog post](https://medium.com/@dhruvp/956b57d4f6e0).
