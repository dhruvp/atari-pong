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

There's a huge difference between reading the theory of a neural network and actually writing one yourself and seeing it do something amazing. This tutorial is all about
getting you to write your own neural network and feel ownership as you see it learn more and more as it finally becomes good enough to beat the computer in pong!

By the end of this tutorial, you'll be able to do the following:
  - write a neural network from scratch
  - use openai gym
  - build an AI for pong that can beat the computer in less than 200 lines

Ok let's get started!

### Setup

1. Setup a virtualenv for python3
  - `conda create --name=openai --python=python3`
  - `source activate openai`
1. Follow the instructions for installing Openai Gym [here](https://gym.openai.com/docs). You may need to install `cmake` first.
2. Run `pip install -e '.[atari]'`.
3. Go back to the root of this repository.
4. Create a new file called `my_pong.py`.

### Writing the Neural Network

Ok before we get started, let's go over the architecture of what's about to happen. I strongly suggest you read the [blog post](http://karpathy.github.io/2016/05/31/rl/) based 
on which all of this is based. If you want a further primer into Neural Networks, there are some great resources to learn more (I'm biased - Udacity employee):

  - [Udacity Deep Learning](https://www.udacity.com/course/deep-learning--ud730)
  - [Udacity Supervised Learning](https://www.udacity.com/course/machine-learning-supervised-learning--ud675)
  - [Deep Learning Book](http://neuralnetworksanddeeplearning.com/)

So now let's get to problem solving. Here's the problem:

#### Problem

We are given the following:
  - A sequence of images (frames) representing each frame of the Pong game
  - An indication when we've won or lost the game
  - An opponent agent that is the traditional Pong computer player
  - An agent we control that we can tell to move up or down at each frame

Can we use these pieces, to train our agent to beat the computer? Moreover, can we make our solution generic enough so it can be reused to win in games different from pong?

#### Solution

Indeed, we can! We're going to do this by building a Neural Network that takes in each image and outputs a command to our AI to move up or down. We can break this down a bit more into the following steps:

Our Neural Network, based completely on Andrej's solution, will look like this: 

1. Take in images from the game and preprocess them (remove color, background, downsample etc.).
2. Use the Neural Network to compute a probability of moving up.
3. Sample from that probability distribution and tell the agent to move up or down.
4. If the round is over (you missed the ball or the oponent missed the ball), find whether you won or lost.
5. Pass the result through the backpropagation algorithm to compute the graient for our weights.
6. After a full game has finished (someone got to 21 points), sum up the gradient and move the weights in the direction of the gradient.
7. Repeat this process until our weights are tuned to the point where we can beat the computer.

That's basically it! Let's start looking at how our code achieves this.

The code starts in the `main` function. Let's go step by step.

#### Initialization

```python
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image
```

Here, we use OpenAi Gym to make our game environment and then call `env.reset()` to get our first input image. This will be an image of the game at the very beginning.

```python
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-4
```

Next, we set a bunch of hyperparameters based off of Andrej's blog post. We aren't going to worry about tuning them but note that you can probably get better performance by doing so.
Let's just spend a minute on each parameter:
  - `batch_size`: how many rounds we play before updating the weights of our network.
  - `gamma`: The discount factor we use to discount the effect of old actions on the final result.
  - `decay_rate`: Parameter used in RmsProp algorithm.
  - `num_hidden_layer_neurons`: How many neurons are in our hidden layer.
  - `learning_rate`: The rate at which we learn from our results to compute the new weights. A higher rate means we react more to results and a lower rate means we don't react as strongly to each result.

```python
    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None
```

This just sets a bunch of counters and initial values. Nothing to really see here.


```python
    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }
```

Here, we set up the initial weights in our Neural Network.

Weights are stored in matrices. For layer 1, element `w1_ij` represents the 
weight of hidden layer `i` for input pixel `j`.
Layer 1 is a `200 x 6400` matrix representing the weights for our hidden layer.

For layer 2, element `w2_i` represents the weights of output of hidden layer `i`.
Layer 2 is a `200 x 1` matrix representing the weights of the output of the hidden layer on our final output.

```python
    # To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])
```

This sets up initial parameters for RmsProp. See [here](http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop) for more information.

```python
    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
```

We'll need to collect a bunch of observations and intermediate values across the episode and use those to compute the gradient at the end based on the result.
This sets up the arrays where we'll collect all that information.

Now that we're done setting up, let's jump in to the main part of our algorithm.

#### Main algorithm

```python
    while True:
        # env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
```

We write our main algorithm inside a `while True:` as we want our Network to keep training for as long as we want. You can reset this as you desire.
The first step to our algorithm is processing the image of the game that openai passed us. Let's dive in to `preprocess_observations` a little more to see the type of processing we are doing.

```python
def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations
```

Now that we've preprocessed the algorithm, let's move on to the next steps.

```python
    episode_observations.append(processed_observations)
    hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)
    episode_hidden_layer_values.append(hidden_layer_values)
```

First, we append our observations to an array that stores all observations for this episode (basically a round of pong) so that we can use it for learning later.
Next, we do one of the main steps in this process:
  - pass in the observations into our neural network and get a probability of moving up!

This is the crux of how our agent is going to figure out how to move up or down. Let's dive in a bit into the `apply_neural_nets` function to see how we determine the probability of moving up and how we get the hidden layer values.

```python
def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values
```

As you can see, it's not many steps at all! Let's go step by step:
1. Compute the unprocessed hidden layer values by simply finding the dot product of `Weights['1']` and `observation_matrix`.
If you remember, `Weights[1]` is a `200 x 6400` matrix and observations_matrix is a `6400 x 1` matrix. So the dot product will give us a matrix of dimensions
`200 x 1`. This checks out! We have 200 neurons and so each row represents the output of one neuron.
2. Next, we apply a thresholding function on those hidden layer values - in this case just a simple ReLU.
3. We then use those hidden layer values to calculate the output layer values. This is done by a simple dot product of
`hidden_layer_values (200 x 1)` and `weights['2'] (200 x 1)` which yields a single value (1 x 1).
4. We then apply a sigmoid function on this output value so that it's between 0 and 1 and is therefore a valid probability (probability of going up).
5. That's it! That's really all there is to taking an observation and computing the probability of going up!

Let's return to the main algorithm and see what happens next:

```python
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)
```

In the above two lines, we record the `hidden_layer_values` for future reference (we'll use it to learn and improve the weights) and we then choose an action based on the probability of going up.
We choose an action by flipping an imaginary coin that lands "Up" with probability `up_probability` and down with `1-up_probability`. If it lands Up, we choose tell our AI to go Up and if not, we tell it to go Down.

Having done that, we pass the action to OpenAI Gym `env.step(action)`.


Now that we've made our move, it's time to start learning!

#### Learning

Learning is all about seeing the result of the action and changing our weights accordingly.
The first step to learning is asking the following question:

- How does changing my output probability affect my result?

Mathematically, this is just the derivative of our result with respect to our output probability. If we measured the value of the result
using a value Li and used the variable \sigma(f_j) to represent the output probability, this is just
dL/df.

In a classification context, this derivative turns out to equal this simple formula:

- classification_label(0 or 1) - predicted_classification

After one action, we don't really have a classification of whether or not this was the right action so we're going to treat the action we sampled as the classification.
Our predicted classification is going to be the probability of going up. Using that, we have that the gradient can be computed by 


```python
        # see here: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
```

Awesome.

Once we have this we now need to finally start updating the weights. As we described, we only start updating the weights
after an episode (or round) has finished and we know whether we win or lose.
This is captured by the boolean `done` given to us by the `env.step()` function.

When we notice we are done, the first thing we do is compile all our observations and gradient calculations for the episode by vertically stacking our values.
This allows us to apply our learnings for every action we took in this episode.

```python
            # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)
```

Next, we want to learn in such a way that actions taken towards the end of an episode more heavily influence our learning than actions taken at the beginning.
Think about it this way: If you moved up at the first step of the round, it probably had very little impact. If you won at the end of the episode, it's probably not got a 
whole lot to do with you moving up at the very first frame. However, closer to the end of the episode, your actions probably have a much larger effect as they determine whether or not
your paddle reaches the ball and how your paddle hits the ball.

We're going to accomplish this by discounting our rewards such that rewards from earlier frames are discounted a lot more than rewards for later frames.

```python
            # TWeak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)
```

Next, we're going to finally use backpropagation to compute the gradient (i.e. the direction we need to move our weights in to improve).

```python
            gradient = compute_gradient(
              episode_gradient_log_ps_discounted,
              episode_hidden_layer_values,
              episode_observations,
              weights
            )
```

#### INSERT COOL GRADIENT EXPLANATION HERE

After we have finished `batch_size` episodes, we finally update our weights for our Neural Network and implement our learnings.

```python
            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)
```

This is the step that tweaks our Neural Network weights and allows us to get better over time. It moves the weights in the direction of the gradient we computed so that 
we are tweaking the weights such that we get a better result next time.

This is basically it! You just coded a full Neural Network for playing Pong!