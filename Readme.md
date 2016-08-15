## Neural Networks For Pong

### Setup

1. Follow the instructions for installing Openai gym [here](https://gym.openai.com/docs). You may need to install `cmake` first.
2. Run `pip install -e '.[atari]'`.
3. Run `python me_pong.py`

### Misc.

Note, if you want to run Andrej Karpathy's original code, run `python pong.py`.

### Performance

According to the blog post, this algorithm should take around 3 days of training on a Macbook to start beating the computer.

Consider tweaking the hyperparameters or using CNNs to boost the performance further.

### Seeing the game

If you want to see the game happen, open `me_pong.py` and uncomment the following line:
```python
# env.render()
 ```

### Credit

This is based off of the work of Andrej Karpathy's great blog post and code [here](http://karpathy.github.io/2016/05/31/rl/)
