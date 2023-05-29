from ReplayMemory import ReplayMemory
import gymnasium as gym
import numpy as np

from ActorCritic import ActorCritic
from UONoise import OUNoise


class Trainer:
    def __init__(self, task, replay_capacity=50000, replay_histor_len=1,
                 num_null_operations=4, batch_size=32, standard_diviation=0.2,
                 num_actor_layers=4, num_critic_layers=4, tau=0.005, discount_rate=0.99,
                 actor_lr=0.001, critic_lr=0.002, train_interval=1, update_interval=5):
        
        # create training and evaluation environments
        self.training_env = self._get_env(task, True)
        self.evaluation_env = self._get_env(task, False)
        self._get_env_info()
        
        self.noise_fn = OUNoise(mean=np.zeros(self.num_actions), 
                                std_diviation=float(standard_diviation) * np.ones(self.num_actions))
        
        self.replay_memory = ReplayMemory(capacity=replay_capacity, history_len=replay_histor_len, 
                                          num_null_operations=num_null_operations)
        
        self.actor_critic = ActorCritic(num_actions=self.num_actions, max_action=self.high_action, 
                                        min_action=self.low_action, num_states=self.num_states,
                                        num_actor_layers=num_actor_layers, num_critic_layers=num_critic_layers, 
                                        tau=tau, discount_rate=discount_rate, actor_lr=actor_lr, critic_lr=critic_lr, 
                                        replay_memory=self.replay_memory, noise_fn=self.noise_fn, batch_size=batch_size)
        
        self.batch_size = batch_size
        
        self.train_interval = train_interval
        self.update_interval = update_interval

    def _get_env(self, task, training=True):
        # create environment and return created environment
        if training:
            env = gym.make(task, max_episode_steps=200)
        else:
            env = gym.make(task, render_mode="human", max_episode_steps=200)
        return env
    
    def _get_env_info(self):
        self.num_actions = self.training_env.action_space.shape[0]
        self.num_states = self.training_env.observation_space.shape[0]
        self.high_action = self.training_env.action_space.high
        self.low_action = self.training_env.action_space.low        
    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            (prev_state, info) = self.training_env.reset()
            self.replay_memory.add_null_operations(prev_state)
            step = 0
            total_reward = 0
            print("In episode ", episode + 1)
            while True:
                action = self.actor_critic.policy(prev_state)
                state, reward, terminated, trunctated, info = self.training_env.step(action)
                # action is a list so return only first action
                action = action[0]
                self.replay_memory.add(state, action, reward, terminated)
                total_reward += reward
                
                # at every training interval train the network
                if step % self.train_interval == 0:
                    self.actor_critic.train()
                
                # at every upate interval transfer the weights
                if step % self.update_interval == 0:
                    self.actor_critic.update_weights()
                
                if terminated or trunctated:
                    self.actor_critic.save_epsisode_loss(episode+1)
                    break
                
                step += 1
            print("End of episode ", episode + 1, ", Total reward is ", total_reward)
    
    
    def evaluate(self, num_episodes):
        for episode in range(num_episodes):
            (prev_state, info) = self.evaluation_env.reset()
            
            while True:
                action = self.actor_critic.policy(prev_state)
                print("The action is :" , action)
                state, reward, terminated, truncated, info = self.evaluation_env.step(action)
                print(f"The action is {action} and reward is {reward}")
                prev_state = state
                
                if terminated or truncated:
                    print(f"Episode {episode  + 1} done")
                    break
    

if __name__ == "__main__":
    pass
    # Example usage
    # print("Create trainer...")
    # trainer = Trainer("Pendulum-v1")
    # trainer.train(100)
    # trainer.evaluate(3)