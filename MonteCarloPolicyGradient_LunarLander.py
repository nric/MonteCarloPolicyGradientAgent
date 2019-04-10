"""
This is a Monte Carlo Policy Gradient algorithm written using TF2.0 keras and (somewhat) optimized to solve 
Open Ai Gym Lunar Lander. But should be able to solve other gym envs with some hyperparameter adaptation.
This is roughly in line with Move 37 Course Homework 8.6 but employs Tensorflow 2.0 with Keras backend.
As a reference with more explanation see: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Policy%20Gradients/Cartpole/Cartpole%20REINFORCE%20Monte%20Carlo%20Policy%20Gradients.ipynb
Only minimal optimization was done, but I added multi step bellmen rollouts. However, did did not increase 
performance by much in my tests (currently set to 1). 
The agent learns qithin about 200 episodes to sometimes land but never get absolutiy perfect for some reson. 
I guess it would require hyperparameter tuning and either some leraning rate dynamics or entropy bonus in the
loss function?
Written in Visual Studio Code with Ipykernel. So if you want to run it in an normal python env, the main block 
needs to get inside the "if __name__ == __main__:" statement.

"""

#%%
import numpy as np
import gym
from collections import deque as DQ
import tensorflow as tf
import tensorflow.keras.backend as K


#%%
class MonteCarloPolicyGradient_Agent:
    def __init__(self,env,hidden_layers=[32,32]):
        self.input_shape = env.observation_space.shape #(8,)
        self.input_size = self.input_shape[0] 
        self.output_dim = env.action_space.n #4
        self.hidden_layers = hidden_layers
        self.log = DQ(maxlen=10) # protocol for debug
        self.GAMMA = 0.99
        self.LEARNING_RATE = 1e-3
        self.states = []
        self.actions = []
        self.rewards = []
        self.discounted_normalized_rewards = np.zeros(0)
        self.total_rewards = []
        self._build_model()
    

    def _build_model(self):
        """Builds a self.model object with the loss function in accordance to the REINFORCE algorithm.
        There are several ways to build a suitable model with with Keras. Here are three:
        1) One way is to define a simimple sequential model but don't use the .fit method. Insead, a backend.function
        needs to be defined. This uses a adam.get_update method as update but with a custom loss placehoder and multiple input,
        among other the discounted rewards whcih are needed for the REINFORCE type PG update.
        2) Another way is to define two custom models. One, which only only does the prediction and one which does prediction and training.
        The latter also has a double input tensor for the state and the state and the discounted rewards as input and the actions as output.
        3) The third option is the most concise but somewhat "a trick". It employs categorical cross entropy as loss. The latter function H is defined almost identically
        to our REINFORCE Loss L (L = Sum(advantage_fkt(s|a) * LOG(policy(s|a)))) vs H(p,q)=sum(p_i*LOG(q_i)). So all that needs to be done is to 
        set the input p_i as the advantage function with one-hot (everythong zero except the taken action). --> H(p,q)=sum(advantage_fkt(s|a)*log(policy(s|a)))
        Here the Option 3 is implemented
        """
        assert len(self.hidden_layers) >= 1
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(self.hidden_layers[0],input_shape=self.input_shape,activation='relu'))
        #add other layers
        for layer_dim in self.hidden_layers[1:]:
            self.model.add(tf.keras.layers.Dense(layer_dim, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.output_dim,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=self.LEARNING_RATE))
        self.model.summary()
    

    def get_action(self,state):
        """Gets an action with the probability according to the softmax prob distib from the neural net to this state
        params:
            :state: The state in shape of a neural net input tensor
        return:
            an action. Randomized but with the probability distibution matrix from the neural net.
        """
        action_probability_distribution = self.model.predict(state, batch_size=1).flatten()
        action = np.random.choice(range(len(action_probability_distribution)), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        return action


    def add_sample(self,S,A,R):
        """Adds a S,A,R observation pair to the according internal buffers.
        params:
            :S,A,R: State,Action,Reward
        """
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)


    def discount_and_normalize_rewards(self):
        """When called, changes the self.rewards over to self.discounted_normalized_rewards and empties rewards.
        Reasons: 1) For training, we only need the normalized and discounted rewards. The normal rewards need to 
        be emptied to make the agent compatible with muti rollouts (which depending on the environment might or
        might not be better. See hyper parameters in main)
        """
        #Make list same length as reward_lst but with discounted reward at each position
        discounted_episode_rewards = np.zeros_like(self.rewards)
        cumulative = 0.0
        for i in reversed(range(len(self.rewards))):
            cumulative = cumulative * self.GAMMA + self.rewards[i]
            discounted_episode_rewards[i] = cumulative
        
        #normalize the np.array (mean normalization)
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        #append normalized episode to buffer
        self.discounted_normalized_rewards = np.append(self.discounted_normalized_rewards,discounted_episode_rewards)
        self.total_rewards.append(sum(iter(self.rewards)))
        self.rewards = [] # keep the states and action buffer till training but rewards get replaced after each call of this method
        

    
    def train(self):
        """When called, prepares X and Y for the training and commits one forward pass and weight update via backprobagation.
        Afterwards empties S,A,R buffers. 
        Returns:
            keras.history
        """
        #prepare batch with inputs and outputs (X,Y). Here they are the X=states and Y=discounted_rewards
        assert len(self.discounted_normalized_rewards) == len(self.states)
        batch_length = len(self.states)
        inputs = np.zeros((batch_length, self.input_size))
        advantages = np.zeros((batch_length, self.output_dim))
        for i in range(batch_length):
            inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = self.discounted_normalized_rewards[i]

        #Let the magic happen
        history = self.model.fit(inputs, advantages, epochs=1, verbose=0)

        #reset all buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.discounted_normalized_rewards = np.zeros(0)

        return history



#%%
#constants
CYCLES = 1000
MULTISTEP_BELLMAN_ROLLOUT = 1 #set to 1 if training should be done after every episode.
RENDER_EVERY_N = 50 #render every Nth episode
#generate env and agent
env = gym.make('LunarLander-v2')
agent = MonteCarloPolicyGradient_Agent(env,hidden_layers=[32,32,8])
#rolling vars
total_rewards = 0
episode = 0

for episode in range(CYCLES):
    episode_rewards_sum = 0
    done = False
    # Launch the game
    state = env.reset()
    while not done:
        if episode % RENDER_EVERY_N == 0:
            env.render()
        #reshape to fit NeuralNet input
        state = state.reshape([1,agent.input_size])
        #get action from agent and play a step
        action = agent.get_action(state)
        new_state, reward, done, info = env.step(action)
        #actions 1-hot encoded
        action_ = np.zeros(agent.output_dim)
        action_[action] = 1
        #add experience (S,A,R)
        agent.add_sample(state,action_,reward)
        #if episode finished:
        if done:
            #always discount and normalize
            agent.discount_and_normalize_rewards()
            #train, if number of rollouts are complete
            if episode % MULTISTEP_BELLMAN_ROLLOUT == 0:
                history = agent.train()
                if episode % RENDER_EVERY_N == 0:
                    print(f"episode: {episode}  loss: {history.history['loss']}  mean rewards: {agent.total_rewards[-1]}")
        state = new_state        
        
        
        
        

#%%
