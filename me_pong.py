## Architecture

# Take in inputs from the screen
# Pass them into an NN
# Update the weights of the NN in some way based on the results
# w1 - Matrix that holds weights of pixels passing into hidden layer. Dimensions: [200 x 80 x 80] -> [200 x 6400]
# w2 - Matrix that holds weights of hidden layer passing into output. Dimensions: [1 x 200]

# Process is:

# x = image vector - [6400 x 1] array
# Compute h = w1 dot x ([200 x 6400] dot [6400 x 1]) -> [200 x 1] - this gives initial activation values.
# Next we need to transform those either via a sigmoid or an ReLU of some sort. Let's use ReLU
# h[h<0] = 0
# Next we need to pass this one layer further
# output = w2 dot h = [1 x 200] dot [200 x 1] -> [1 x 1]
# Now our output layer is the probability of going up or down. Let's make sure this output is between 0 and 1 by passing it through a sigmoid
# p = sigmoid(output)

# Learning:

# Figure out the result
# Compute the error
# Use the error to calculate the gradient
	# dw2 = eph^T dot gradient_log_p = [1 x 2000] dot [2000 x 1] = 1x1
	# dh = gradient_log_p outer_product w2 = [2000 x 1] outer_product [1 x 200] = [2000 x 200]
	# dw1 = dh^T dot epx = [200 x 2000]x dot [2000 x 64000] = [200 x 64000]
# After some batch size has finished,
	# Use rmsprop to move w1 and w2 in the direction of the gradient

def downsample=(image):
	return image[::2, ::2, :]

def remove_color(image):
	return image[:, :, 0]

def remove_background:
	image[image == 144] = 0
	image[image == 109] = 0
	return image

def preprocess_observations(observation):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  observation = observation[35:195] # crop
  observation = downsample(cropped_observation)
  observation = remove_color(observation)
  observation = remove_background(observation)
  observation[observation != 0] = 1 # everything else (paddles, ball) just set to 1
  return observation.astype(np.float).ravel()

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def relu(vector):
	new_vec = np.copy(vector)
	new_vec[new_vec < 0] = 0
	return new_vec

def apply_neural_nets(observation_matrix, layer_1_weights, layer_2_weights):
	hidden_layer_values = np.dot(observation_matrix, layer_1_weights)
	hidden_layer_values = relu(hidden_layer_output)
	output_layer_values = np.dot(hidden_layer_values, layer_2_weights)
	output_layer_values = sigmoid(output_layer_values)
	return hidden_layer_values, output_layer_values

def choose_action(up_probability):
	random_value = np.random.uniform()
	if random_value < up_probability:
		return 2 # signifies up in openai gym
	else:
		return 3 # signifies down in openai gym

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
	delta_L = gradient_log_p
	dC_dw2 = np.dot(hidden_layer_values, delta_L)
	delta_1 = np.outer(gradient_log_p, weights['2'])
	delta_1[delta_1 <= 0] = 0
	dC_dw1 = np.dot(observation_values, weights['1'])
	return {
		'1': dC_dw1,
		'2': dC_dw2
	}

def update_weights(gradient, weights, expectation_g_squared, g_dict):
	# http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop
	epsilon = 1e-5
	for layer_name in weights.keys():
		g = g_dict[layer_name]
		expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
		weights[layer_name] =  weights[layer_name] - (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[k] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer

env = gym.make("Pong-v0")
observation = env.reset()
episode_number = 0
batch_size = 10
gradient_log_p = []
episode_hidden_layer_values = []
episode_observations = []

weights = {
	'1': [],
	'2': []
}

expectation_g_squared = {}

for layer_name in weights.keys():
	expectation_g_squared[layer_name] = 0


while True:
	processed_observations = preprocess_observations(observation)
	episode_observations.append(processed_observations)
	up_probability = apply_neural_nets(processed_observations)
	hidden_layer_values, action = choose_action(up_probability)
	episode_hidden_layer_values.append(hidden_layer_values)
	# carry out the chosen action
	observation, reward, done, info = env.step(action)

	fake_label = 1 if action == 2 else 0
	# see here: http://cs231n.github.io/neural-networks-2/#losses
	episode_gradient_log_ps.append(fake_label - up_probability)
	episode_hidden_layer_values = np.vstack(hidden_layer_values)

	if done: # an episode finished
    	episode_number += 1
    	episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
    	episode_observations = np.vstack(episode_observations)
    	episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
    	for key in gradient: g_dict[key] += gradient[key]
    	gradient = compute_gradient(episode_gradient_log_ps, episode_hidden_layer_values, episode_observations, weights, g_dict)
    	if episode_number % batch_size == 0:
    		update_weights(gradient, weights_dict)
    	episode_hidden_layer_values, episode_observations, episode_gradient_log_ps = [], [], []
