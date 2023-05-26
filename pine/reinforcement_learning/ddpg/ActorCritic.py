import tensorflow as tf
import numpy as np
import os
import keras
from ReplayMemory import ReplayMemory


class ActorCritic:
    def __init__(self, num_actions, max_action,
                 min_action, num_states, num_actor_layers, 
                 num_critic_layers, tau, discount_rate, 
                 actor_lr, critic_lr,
                 replay_memory: ReplayMemory=None, 
                 noise_fn=None, batch_size=32):
        # the number of actiosn to be taken at each time step
        self.num_actions = num_actions
        # get the maximum possible action in that environment
        self.max_action = max_action
        # get the minimum possible action in that environment
        self.min_action = min_action
        # the num of values to be provided as the state in each time step
        self.num_states = num_states
        # the number of layers for the actor network
        self.num_actor_layers = num_actor_layers
        # the number of layers for the critic network
        self.num_critic_layers = num_critic_layers
        # create the behaviour and the target actor network i.e b_actor and t_actor respectively
        self.b_actor = self._create_actor()
        self.t_actor = self._create_actor()
        # transfer the weights of the behaivioral actor to the target network
        self.t_actor.set_weights(self.b_actor.get_weights())
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=actor_lr)
        
        # create the behaviour and the target critic networks i.e b_critic and t_critic
        self.b_critic = self._create_critic()
        self.t_critic = self._create_critic()
        # transfer the weights of the behaivior critic network to the target network
        self.t_critic.set_weights(self.b_critic.get_weights())
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=critic_lr)
        
        # this is the tau constant used to update the trainable parameters
        self.tau = tau # tau is a value between zero and one
        self.replay_memory = replay_memory
        # set the function for inputing noise into the action for exploration
        self.noise_fn = noise_fn
        # set the discount rate of the agent to determine how much it values future actions
        self.discount_rate = discount_rate
        # set trainng batch size
        self.batch_size = batch_size
        
        # set directory for saving weights
        save_dir = "saves"
        os.makedirs(save_dir,exist_ok=True)
        self.save_path = os.path.join(save_dir, "ddpg")
        os.makedirs(self.save_path, exist_ok=True)
        
        # set directory for saving logs
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "ddpg")
        
        # create summary writer for showing loss on tensor board
        self.writer = tf.summary.create_file_writer(self.log_path)
        
        # create metrics for measuring actor and critic losses
        self.actor_loss_metric = tf.keras.metrics.Mean(name="actor_loss", dtype=tf.float32)
        self.critic_loss_metric = tf.keras.metrics.Mean(name="critic_loss", dtype=tf.float32)
        
            
    def policy(self, state):
        # Implement the policy function of the network
        state = np.array([state])
        action =  self.b_actor(state)
        noise = self.noise_fn()
        action = action.numpy() + noise # add noise to the action to be taken
        # clip action so that it would only be within the allowed range
        action = np.clip(action, self.min_action, self.max_action)
        return [np.squeeze(action)]
    
    def load_weights_from_file(self, behaviour_actor_weights_path, behaviour_critic_weights_path, 
                               target_actor_weights_path, target_critic_weights_path):
        # load weights from file paths
        self.b_actor.load_weights(behaviour_actor_weights_path)
        self.t_actor.load_weights(target_actor_weights_path)
        self.b_critic.load_weights(behaviour_critic_weights_path)
        self.t_critic.load_weights(target_critic_weights_path)
        
    
     # generate training batches for the DDPG algorithm
    def _generate_batches(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminations = []
        
        for _ in range(self.batch_size):
            training_state, training_action, training_reward, training_new_state, training_terminate = self.replay_memory.sample()
            states.append(training_state[0])
            actions.append(training_action)
            rewards.append(training_reward)
            next_states.append(training_new_state[0])
            terminations.append(training_terminate)
                        
        return np.array(states).astype('float64'), np.array(actions).astype('float64'), \
               np.array(rewards).astype('float64'), np.array(next_states).astype('float64'),\
               np.array(terminations).astype('float64')
    
    

    def train(self):
        states, actions, rewards, next_states, _  = self._generate_batches()
        # update the behavioral critic network with the gradient of the loss against the trainable parameters
        with tf.GradientTape() as tape:
            next_actions = self.t_actor(next_states, training=True)
            next_values = self.t_critic([next_states, next_actions], training=True)
            y = tf.cast(rewards, dtype=tf.float64) + tf.cast(self.discount_rate, dtype=tf.float64) * tf.cast(next_values, dtype=tf.float64)
            predictions = self.b_critic([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.square(tf.cast(y, dtype=tf.float64) - tf.cast(predictions, dtype=tf.float64)))
            self.critic_loss_metric(critic_loss)
            critic_grad = tape.gradient(critic_loss, self.b_critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.b_critic.trainable_variables))
            
        # update the behavior actor network with the gradiets of the loss against the trainable parameters
        with tf.GradientTape() as tape:
            perdicted_actions =  self.b_actor(states)
            predicted_values = self.b_critic([states, perdicted_actions], training=True)
            actor_loss = -tf.math.reduce_mean(predicted_values)
            self.actor_loss_metric(actor_loss)
            actor_grad = tape.gradient(actor_loss, self.b_actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.b_actor.trainable_variables))
        
        self._save_weights_to_file()



    # update the weights of the target networks
    def update_weights(self):
        for a, b in zip(self.t_actor.variables, self.b_actor.variables):
            a.assign(self.tau * b + (1 - self.tau) * a)
        
        for a, b in zip(self.t_critic.variables, self.b_critic.variables):
            a.assign(self.tau * b + (1 - self.tau) * a)
            

    def _create_actor(self):
        # input layer is based on shape of state space
        inputs = tf.keras.layers.Input(shape=(self.num_states,))
        # create initial output layer
        out = tf.keras.layers.Dense(units=16, activation="relu")(inputs)
        # create layers for the number of layers specified in the constructor
        for _ in range(self.num_actor_layers):
            out =  tf.keras.layers.Dense(units=16, activation="relu")(out)
            
        # add final output layer to be the shape of the expected action
        out = tf.keras.layers.Dense(units=self.num_actions, activation="relu")(out)
        model = keras.Model(inputs=inputs, outputs=out)
        # return the created model
        return model
        
    
    def _create_critic(self):
        # perform preprocessing on state input
        state_inputs = tf.keras.layers.Input(shape=(self.num_states, ))
        state_out = tf.keras.layers.Dense(units=16, activation="relu")(state_inputs)
        state_out = tf.keras.layers.Dense(units=32, activation="relu")(state_out)
        
        # perform preprocessing on action input
        action_inputs = tf.keras.layers.Input(shape=(self.num_actions, ))
        action_out = tf.keras.layers.Dense(units=32, activation="relu")(action_inputs)
        # concatenate the state output and action outuput together
        concat = tf.keras.layers.Concatenate()([state_out, action_out])
        out = tf.keras.layers.Dense(16, activation="relu")(concat)
        for _ in range(self.num_critic_layers):
            out = tf.keras.layers.Dense(units=16, activation="relu")(out)
        
        out = tf.keras.layers.Dense(units=1, activation="tanh")(out)
        # create the model with the state and action inputs and the final value output
        model = tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=out)
        return model
    
    def _save_weights_to_file(self):
        # save weights of the target and behavioral actor network
        self.b_actor.save_weights(os.path.join(self.save_path, "b_actor.h5"))
        self.t_actor.save_weights(os.path.join(self.save_path, "t_actor.h5"))
        # save weights for the target and behavioral critic network
        self.b_critic.save_weights(os.path.join(self.save_path, "b_critic.h5"))
        self.b_critic.save_weights(os.path.join(self.save_path, "t_critic.h5"))
    
    def _load_weights_from_file(self):
        # save weights of the target and behavioral actor network
        self.b_actor.load_weights(os.path.join(self.save_path, "b_actor.h5"))
        self.t_actor.load_weights(os.path.join(self.save_path, "t_actor.h5"))
        # save weights for the target and behavioral critic network
        self.b_critic.load_weights(os.path.join(self.save_path, "b_critic.h5"))
        self.b_critic.load_weights(os.path.join(self.save_path, "t_critic.h5"))
        
    
    def save_epsisode_loss(self, epoch):
        # set actor and critic losses for that episode
        self._log_loss(self.actor_loss_metric.result(), "actor", epoch)
        self._log_loss(self.critic_loss_metric.result(), "critic", epoch)
        # reset actor and critic loss metrics
        self.actor_loss_metric.reset_states()
        self.critic_loss_metric.reset_states()
        
    
    def _log_loss(self, loss, prefix, epoch):
        with self.writer.as_default():
            tf.summary.scalar(f'{prefix}_loss', loss, epoch)