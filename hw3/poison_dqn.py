import time
import pickle
import sys
import gym.spaces
import logz
import itertools
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn, optim
from collections import namedtuple
from dqn_utils import LinearSchedule, ReplayBuffer, get_wrapper_by_name

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_lambda"])
 
class QLearner(object):

	def __init__(
		self,
		env,
		q_func,
		optimizer_spec,
		model_input_dir,
		exploration=LinearSchedule(1000000, 0.1),
		stopping_criterion=None,
		replay_buffer_size=1000000,
		batch_size=32,
		gamma=0.99,
		learning_starts=50000,
		learning_freq=4,
		frame_history_len=4,
		target_update_freq=10000,
		grad_norm_clipping=10,
		double_q=True,
		lander=False):
		"""Run Deep Q-learning algorithm.

		You can specify your own convnet using q_func.

		All schedules are w.r.t. total number of steps taken in the environment.

		Parameters
		----------
		env: gym.Env
			gym environment to train on.
		q_func: function
			Model to use for computing the q function. It should accept the
			following named arguments:
				in_channels: int
					number of channels for the input
				num_actions: int
					number of actions
		optimizer_spec: OptimizerSpec
			Specifying the constructor and kwargs, as well as learning rate schedule
			for the optimizer
		model_input_dir: str
			path of previous saved model
		exploration: rl_algs.deepq.utils.schedules.Schedule
			schedule for probability of chosing random action.
		stopping_criterion: (env, t) -> bool
			should return true when it's ok for the RL algorithm to stop.
			takes in env and the number of steps executed so far.
		replay_buffer_size: int
			How many memories to store in the replay buffer.
		batch_size: int
			How many transitions to sample each time experience is replayed.
		gamma: float
			Discount Factor
		learning_starts: int
			After how many environment steps to start replaying experiences
		learning_freq: int
			How many steps of environment to take between every experience replay
		frame_history_len: int
			How many past frames to include as input to the model.
		target_update_freq: int
			How many experience replay rounds (not steps!) to perform between
			each update to the target Q network
		grad_norm_clipping: float or None
			If not None gradients' norms are clipped to this value.
		double_q: bool
			If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
			https://papers.nips.cc/paper/3964-double-q-learning.pdf
		"""
		assert type(env.observation_space) == gym.spaces.Box
		assert type(env.action_space)      == gym.spaces.Discrete

		self.target_update_freq = target_update_freq
		self.optimizer_spec = optimizer_spec
		self.batch_size = batch_size
		self.learning_freq = learning_freq
		self.learning_starts = learning_starts
		self.stopping_criterion = stopping_criterion
		self.env = env
		self.exploration = exploration
		self.gamma = gamma
		self.double_q = double_q
		self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

		###############
		# BUILD MODEL #
		###############

		observation = self.env.observation_space
		print(observation.shape)
	
		if len(observation.shape) == 1:
			# This means we are running on low-dimensional observations (e.g. RAM)
			in_features = observation.shape[0]
		else:
			img_h, img_w, img_c = observation.shape
			in_features = frame_history_len * img_c
		self.num_actions = self.env.action_space.n
		
		print("in features", in_features)
		# define deep Q network and target Q network
		self.q_net = q_func(in_features, self.num_actions).to(self.device)
		self.target_q_net = q_func(in_features, self.num_actions).to(self.device)
		self.original_q_net = q_func(in_features, self.num_actions).to(self.device)

		logz.load_model(self.original_q_net, model_input_dir) 

		# construct optimization op (with gradient clipping)
		parameters = self.q_net.parameters()
		self.optimizer = self.optimizer_spec.constructor(parameters, lr=1, 
														**self.optimizer_spec.kwargs)
		self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.optimizer_spec.lr_lambda)
		# clip_grad_norm_fn will be called before doing gradient decent
		self.clip_grad_norm_fn = lambda : nn.utils.clip_grad_norm_(parameters, max_norm=grad_norm_clipping)

		# update_target_fn will be called periodically to copy Q network to target Q network
		self.update_target_fn = lambda : self.target_q_net.load_state_dict(self.q_net.state_dict())

		# construct the replay buffer
		self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
		self.replay_buffer_idx = None

		###############
		# RUN ENV     #
		###############
		self.model_initialized = False
		self.num_param_updates = 0
		self.mean_episode_reward      = -float('nan')
		self.best_mean_episode_reward = -float('inf')
		self.last_obs = self.env.reset()
		self.log_every_n_steps = 10000

		self.start_time = time.time()
		self.t = 0

	def calc_loss(self, obs, ac, rw, nxobs, done):
		"""
			Calculate the loss for a batch of transitions. 

			Here, you should fill in your own code to compute the Bellman error. This requires
			evaluating the current and next Q-values and constructing the corresponding error.

			arguments:
				ob: The observation for current step
				ac: The corresponding action for current step
				rw: The reward for each timestep
				nxob: The observation after taking one step forward
				done: The mask for terminal state. This value is 1 if the next state corresponds to
					the end of an episode, in which case there is no Q-value at the next state;
					at the end of an episode, only the current state reward contributes to the target,
					not the next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)

				inputs are generated from self.replay_buffer.sample, you can refer the code in dqn_utils.py
				for more details 

			returns:
				a scalar tensor represent the loss

			Hint: use smooth_l1_loss (a.k.a huber_loss) instead of mean squared error.
				use self.double_q to switch between double DQN and vanilla DQN.
		"""
	
		# YOUR CODE HERE
		ts_obs, ts_ac, ts_rw, ts_nxobs, ts_done = map(lambda x: torch.from_numpy(x).to(self.device),
													[obs, ac, rw, nxobs, done])

		ts_ac = ts_ac.long().view(-1, 1)
		
		with torch.no_grad():
			if not self.double_q:
				ts_max_ac = self.target_q_net(ts_nxobs).argmax(-1, keepdim=True)
			else:
				ts_max_ac = self.q_net(ts_nxobs).argmax(-1, keepdim=True)
			expected_Q = ts_rw + (1 - ts_done) * self.gamma * self.target_q_net(ts_nxobs).gather(-1, ts_max_ac).view(-1)

		pred_Q = self.q_net(ts_obs).gather(-1, ts_ac).view(-1)

		total_error = F.smooth_l1_loss(pred_Q, expected_Q)

		return total_error
	
	def stopping_criterion_met(self):
		return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

	def step_env(self):
		### 2. Step the env and store the transition
		# At this point, "self.last_obs" contains the latest observation that was
		# recorded from the simulator. Here, your code needs to store this
		# observation and its outcome (reward, next observation, etc.) into
		# the replay buffer while stepping the simulator forward one step.
		# At the end of this block of code, the simulator should have been
		# advanced one step, and the replay buffer should contain one more
		# transition.
		# Specifically, self.last_obs must point to the new latest observation.
		# Useful functions you'll need to call:
		# obs, reward, done, info = env.step(action)
		# this steps the environment forward one step
		# obs = env.reset()
		# this resets the environment if you reached an episode boundary.
		# Don't forget to call env.reset() to get a new observation if done
		# is true!!
		# Note that you cannot use "self.last_obs" directly as input
		# into your network, since it needs to be processed to include context
		# from previous frames. You should check out the replay buffer
		# implementation in dqn_utils.py to see what functionality the replay
		# buffer exposes. The replay buffer has a function called
		# encode_recent_observation that will take the latest observation
		# that you pushed into the buffer and compute the corresponding
		# input that should be given to a Q network by appending some
		# previous frames.
		# Don't forget to include epsilon greedy exploration!
		# And remember that the first time you enter this loop, the model
		# may not yet have been initialized (but of course, the first step
		# might as well be random, since you haven't trained your net...)

		#####

		# YOUR CODE HERE
		idx = self.replay_buffer.store_frame(self.last_obs)
		# print()
		ts_obs = torch.from_numpy(self.replay_buffer.encode_recent_observation()[None]).to(self.device)

		# print("ts_obs", ts_obs.size())

		if not self.model_initialized or (random.random() < self.exploration.value(self.t)):
			action = random.randint(0, self.num_actions - 1)
		else:
			action = self.q_net(ts_obs).view(-1).argmax().item()

		new_obs, reward, done, _ = self.env.step(action)

		self.replay_buffer.store_effect(idx, action, reward, done)
		self.last_obs = new_obs

		if done:
			self.last_obs = self.env.reset()

	def get_topk(self, ts_obs, ts_ac):
		# print("get_topk")
		
		self.original_q_net.eval()

		# print(ts_obs.dtype)
		# ts_poison_obs = torch.tensor(ts_obs.float(), requires_grad=True)

		ts_poison_obs = ts_obs.float().clone().detach().requires_grad_(True)
		
		pred_Q = self.original_q_net(ts_poison_obs).gather(-1, ts_ac).view(-1)

		# print("pred_Q", pred_Q.size())
		pred_Q.backward(torch.ones(pred_Q.size()).to(self.device))

		channel_val_saliency, channel_index_saliency = torch.max(ts_poison_obs.grad.data.abs(), dim=-1)
		# print("saliency", saliency.size())

		saliency_size = channel_val_saliency.size()
		batch_size = saliency_size[0]
		row_size = saliency_size[1]
		col_size = saliency_size[2]
		# channel_size = saliency_size[3]
		new_saliency = channel_val_saliency.view(batch_size, -1)

		topk = 10
		top_val_saliency, top_index_saliency = torch.topk(new_saliency, dim=1, k=topk)
		# print("top_saliency", top_saliency.size())
		# exit()

		del ts_poison_obs

		return top_val_saliency, top_index_saliency, channel_index_saliency

	def get_poison_loss(self, obs, ac):
		# print("get_poison_loss")
		ts_obs, ts_ac = map(lambda x: torch.from_numpy(x).to(self.device),
													[obs, ac])
		ts_ac = ts_ac.long().view(-1, 1)

		top_val_saliency, top_index_saliency, channel_index_saliency = self.get_topk(ts_obs, ts_ac)

		ts_poison_obs = ts_obs.float().clone().detach().requires_grad_(True)

		pred_Q = self.q_net(ts_poison_obs).gather(-1, ts_ac).view(-1)
		pred_Q.backward(torch.ones(pred_Q.size()).to(self.device))

		saliency = ts_poison_obs.grad.data.abs()

		channel_saliency = saliency.gather(-1, channel_index_saliency.unsqueeze(-1)).squeeze(-1)

		new_channel_saliency = channel_saliency.view(channel_saliency.size()[0], -1)

		top_saliency = new_channel_saliency[0, top_index_saliency]

		poison_loss = torch.exp(top_saliency)
		poison_loss = torch.sum(poison_loss, dim=1)
		poison_loss = torch.mean(poison_loss)

		del ts_poison_obs

		return poison_loss

	def update_model(self):
		### 3. Perform experience replay and train the network.
		# note that this is only done if the replay buffer contains enough samples
		# for us to learn something useful -- until then, the model will not be
		# initialized and random actions should be taken
	
		if (self.t > self.learning_starts and \
			self.t % self.learning_freq == 0 and \
			self.replay_buffer.can_sample(self.batch_size)):
	  
			# Here, you should perform training. Training consists of four steps:
			# 3.a: use the replay buffer to sample a batch of transitions (see the
			# replay buffer code for function definition, each batch that you sample
			# should consist of current observations, current actions, rewards,
			# next observations, and done indicator).
			# 3.b: set the self.model_initialized to True. Because the newwork in starting
			# to train, and you will use it to take action in self.step_env.
			# 3.c: train the model. To do this, you'll need to use the self.optimizer and
			# self.calc_loss that were created earlier: self.calc_loss is what you
			# created to compute the total Bellman error in a batch, and self.optimizer
			# will actually perform a gradient step and update the network parameters
			# to reduce the loss. 
			# Before your optimizer take step, don`t forget to call self.clip_grad_norm_fn
			# to perform gradient clipping.
			# 3.d: periodically update the target network by calling self.update_target_fn
			# you should update every target_update_freq steps, and you may find the
			# variable self.num_param_updates useful for this (it was initialized to 0)
			#####
			
			# YOUR CODE HERE
			obs, ac, rw, nxobs, done = self.replay_buffer.sample(self.batch_size)

			if not self.model_initialized:
				self.model_initialized = True

			loss = self.calc_loss(obs, ac, rw, nxobs, done)
			poison_loss = self.get_poison_loss(obs, ac)

			lambda_param = 0.3

			# full_loss = loss
			full_loss = loss + lambda_param*poison_loss

			self.optimizer.zero_grad()
			full_loss.backward()
			self.clip_grad_norm_fn()
			self.optimizer.step()

			self.num_param_updates += 1
			if self.num_param_updates % self.target_update_freq == 0:
				self.update_target_fn()

		self.lr_scheduler.step()
		self.t += 1
		# print("updating model")

	def log_progress(self):
		episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

		if len(episode_rewards) > 0:
			self.mean_episode_reward = np.mean(episode_rewards[-100:])

		if len(episode_rewards) > 100:
			self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

		if self.t % self.log_every_n_steps == 0 and self.model_initialized:
			logz.log_tabular("TimeStep", self.t)
			logz.log_tabular("MeanReturn", self.mean_episode_reward)
			logz.log_tabular("BestMeanReturn", max(self.best_mean_episode_reward, self.mean_episode_reward))
			logz.log_tabular("Episodes", len(episode_rewards))
			logz.log_tabular("Exploration", self.exploration.value(self.t))
			logz.log_tabular("LearningRate", self.optimizer_spec.lr_lambda(self.t))
			logz.log_tabular("Time", (time.time() - self.start_time) / 60.)
			logz.dump_tabular()
			logz.save_pytorch_model(self.q_net)
	  
def learn(*args, **kwargs):
	print("learning start")
	alg = QLearner(*args, **kwargs)
	while not alg.stopping_criterion_met():
		alg.step_env()
		# print("step 1")
		# at this point, the environment should have been advanced one step (and
		# reset if done was true), and self.last_obs should point to the new latest
		# observation
		alg.update_model()
		# print("step 2")
		alg.log_progress()

