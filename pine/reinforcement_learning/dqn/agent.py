import tensorflow as tf
import numpy as np
from replay_memory import ReplayMemory
import keras
import gym
import os


class Agent:
    def __init__(self, target_model: keras.models.Sequential, behaviour_model: keras.models.Sequential, 
                 repaly_memory: ReplayMemory, environment: gym.Env, epsilon=1.0, epsilon_final=0.01, 
                 epsilon_decay=0.001,discount=0.9,
                 batch_size=32, train_interval=1, update_interval=100, num_episodes=1000):
        # this is the full implementation of the DQN learning algorithm
        # the target model that is being trained
        self.target_model = target_model
        # the behaviour model that is used to act
        self.behaviour_model = behaviour_model
        # the replay memory for storing history
        self.replay_memory = repaly_memory
        # the environment that the agent is learning on
        self.env = environment
        # get the batch size for the agent
        self.batch_size = batch_size
        # set the discount factor used by the agent
        self.discount = discount
        # set the number of episodes for training
        self.num_episodes = num_episodes
        # specify the epsilon value
        self.epsilon = epsilon
        # specify the minimum epsilon value
        self.epsilon_final = epsilon_final
        # epsilon decay
        self.epsilon_decay = epsilon_decay
        # specify the training interval
        self.train_interval = train_interval
        # specify the update interval
        self.update_interval = update_interval
        
        # create logs and saves directory
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        saves_dir = "saves"
        os.makedirs(saves_dir, exist_ok=True)
        
        self.logs_path = os.path.join(logs_dir, "atari_dqn")
        self.save_path = os.path.join(saves_dir, "atari_dqn")
        self.writer = tf.summary.create_file_writer(self.logs_path)
        

    # generate samples for training the model
    def generate_samples(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminations = []
        
        # generate samples based on the batch size
        for _ in range(self.batch_size):
            state, action, reward, next_state, terminated = self.replay_memory.sample()
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminations.append(terminated)
        
        # return list of states, actions, rewards, next states, and terminations
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminations)

    def _get_target(self, rewards, actions, next_state_predictions, terminations, predictions):
        target_predictions = np.copy(predictions)
        max_nex_state_q_value = tf.reduce_max(next_state_predictions, axis=1, keepdims=True).numpy()
        targets = []
        for i in range(len(terminations)):
            target = rewards[i]
            if not terminations[i]: # if the state is not a terminal state
                target+= self.discount * max_nex_state_q_value[i]
            targets.append(target)
        
        # assign the value of the targets to the appropriate action 
        # on the target predictions matrix
        for i in range(len(targets)):
            target_predictions[i, actions[i]]= targets[i]
        
        # return the target predictions
        return target_predictions
                
    def train_one_step(self):
        # perform trainin on a single generated batch from the replay memory
        states, actions, rewards, next_states, terminations = self.generate_samples()
        
        # make predictions for the current state and the next state
        predictions = self.behaviour_model(states)
        next_states_predictions = self.target_model(next_states)
        
        target_predictions = self._get_target(rewards, actions, next_states_predictions, terminations, predictions)
        self.behaviour_model.train_on_batch(states, target_predictions)
        # save and log after each batch training
        self.behaviour_model.save_weights(self.save_path)
        # update epsilon after each training step
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
        # calculate the error of the prediction
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(target_predictions, predictions)
        return loss
        
    
    # the policy decides what action an agent takes in a given state
    def policy(self, observation):
        if np.random.random() <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            # the observation needs to be put into an array because tensorflow only processes data in arrays
            prediction = self.behaviour_model(np.array([observation]))
            action = tf.math.argmax(prediction, axis=1).numpy()[0]
        return action
    
    
    def transfer_weights(self):
        print("Weight transfer has occured")
        self.target_model.set_weights(self.behaviour_model.get_weights())
      
    # train the agent to learn from the environment
    def train(self):
        # loop through the number of episodes
        train_loss = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)
        for episode in range(self.num_episodes):
            train_loss.reset_states()
            count = 0
            (observation, _) = self.env.reset()
            self.replay_memory.add_null_operations(observation)
            print("In episode %d" % episode)
            while True:
                count += 1
                state = self.replay_memory.get_full_state_from_observation(observation)
                action = self.policy(state)
                new_observation, reward, terminated, _, _ = self.env.step(action)
                self.replay_memory.add(observation, action, reward, terminated)
                observation = new_observation
                # if it is at a training interval then perform training
                if count % self.train_interval == 0:
                    print("!Training agent!")
                    loss = self.train_one_step()
                    train_loss(loss)
                    
                # if it is at an update interval then update the target model
                if count % self.update_interval == 0:
                    print("!Updating target model at episode " ,episode)
                    self.transfer_weights()
        
                # if terminated log the loss of the episode and break the loop
                if terminated:
                    print("!Logged result of episode " , episode)
                    self._log_loss(train_loss.result(), episode)
                    break
    
    # log the error of the model on tensorboard
    def _log_loss(self, loss, epoch):
        with self.writer.as_default():
            tf.summary.scalar(f'loss', loss, epoch)
    
    # evaluate the performance of the model
    def evaluate(self, env:gym.Env):
        # get initial observation and use it to create null operations in the replay memory
        (observation, _) = env.reset()
        score = 0
        self.replay_memory.add_null_operations(observation)
        while True:
            state = self.replay_memory.get_full_state_from_observation(observation)
            action = self.policy(state)
            new_observation, reward, terminated, _, _ = env.step(action)
            self.replay_memory.add(observation, action, reward, terminated)
            observation = new_observation
            score += reward
            if terminated:
                break
        print("TOTAL SCORE : ", score)
