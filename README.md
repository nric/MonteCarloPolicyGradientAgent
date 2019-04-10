# MonteCarloPolicyGradientAgent
This is a Monte Carlo Policy Gradient algorithm written using TF2.0 keras and (somewhat) optimized to solve Open Ai Gym Lunar Lander.


This is roughly in line with Move 37 Course Homework 8.6 but employs Tensorflow 2.0 with Keras backend.

As a reference with more explanation see: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Policy%20Gradients/Cartpole/Cartpole%20REINFORCE%20Monte%20Carlo%20Policy%20Gradients.ipynb

Only minimal optimization was done, but I added multi step bellmen rollouts. However, did did not increase 
performance by much in my tests (currently set to 1). 

The agent learns qithin about 200 episodes to sometimes land but never get absolutiy perfect for some reson. 

I guess it would require hyperparameter tuning and either some leraning rate dynamics or entropy bonus in the
loss function?

Written in Visual Studio Code with Ipykernel. So if you want to run it in an normal python env, the main block 
needs to get inside the "if __name__ == __main__:" statement.
