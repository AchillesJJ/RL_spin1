# encoding: utf-8
import tensorflow as tf
import numpy as np
import argparse
import tflearn
from replay_buffer import ReplayBuffer
from spin1_ED import unitary_op
import util
import os

#---------------
#   Actor DNN
#---------------
class ActorNetwork(object):
    
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        
        # netwok parameters
        # (1) Actor network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()
        # (2) Target network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]
        # (3) update target network with online network parameters
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)+ \
                                                  tf.multiply(self.target_network_params[i], 1.0-self.tau))
                for i in range(len(self.target_network_params))]
            
        # actor gradient update
        # (1) action gradient given by critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        # (2) combine gradients
        self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, 
                                                         self.network_params, -self.action_gradient)
        # (3) batch numer normalization
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        
        # Optimization op
        self.optimize = tf.train.AdamOptimizer(self.lr).\
            apply_gradients(zip(self.actor_gradients, self.network_params))
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
        
        # saver of Actor network
        self.saver = tf.train.Saver(max_to_keep=10)
    
    
    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 100)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 100)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out
    
    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs,
                                                self.action_gradient: a_gradient})
    
    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={self.inputs: inputs})
    
    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={self.target_inputs: inputs})
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
    def get_num_trainable_vars(self):
        return self.num_trainable_vars


#----------------
#   Critic DNN
#----------------

class CriticNetwork(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.tau = tau
        self.gamma = gamma
        
        # network parameters
        # (1) Critic network
        self.inputs, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]
        # (2) Target network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(num_actor_vars+len(self.network_params)):]
        # (3) update target network with online network parameters
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)+ \
                                                  tf.multiply(self.target_network_params[i], 1.0-self.tau))
                for i in range(len(self.target_network_params))]
        
        # critic network update
        # (1) network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        # (2) define loss
        # self.loss = tf.reduce_mean(tf.square(self.predicted_q_value, self.out))
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        # (3) optimization op
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # (4) action gradients
        self.action_grads = tf.gradients(self.out, self.action)    
            
    
    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 100)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 100)
        t2 = tflearn.fully_connected(action, 100)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out
    
    
    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.loss, self.optimize], 
                             feed_dict={self.inputs: inputs, self.action: action,
                                        self.predicted_q_value: predicted_q_value})
    
    
    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={self.inputs: inputs,
                                                  self.action: action})
        
        
    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={self.target_inputs: inputs,
                                                         self.target_action: action})
        
    
    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={self.inputs: inputs,
                                                           self.action: actions})
        
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
    def get_critic_loss(self):
        return self.sess.run(self.loss, feed_dict={self.predicted_q_value: predicted_q_value})
        

#------------------------------------------------
#   Ornstein-Uhlenbeck action noise generator
#------------------------------------------------
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=1E-2, theta=1E-3, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


#---------------------
#   Agent training
#---------------------
class AgengTraining(object):
    
    def __init__(self, c2, Nt, dt):
        self.c2 = c2
        self.Nt = Nt
        self.dt = dt
        self.dim = int(Nt/2) + 1
    
    def train(self, sess, args, actor, critic, actor_noise):
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # update target network weights
        actor.update_target_network()
        critic.update_target_network()
        
        # allocate replay buffer
        replay_buffer = ReplayBuffer(int(args.buffer_size), int(args.random_seed))
        
        # define best_R
        best_R = -1.0
        
        # enable batchnorm
        tflearn.is_training(True)
        
        # loop over episodes
        for ep in range(int(args.max_episodes)):
            
            # initial state
            psi = util.random_psi(self.dim)
            s = util.state(psi)
            ep_ave_max_q = 0.0
            a_ls = []
            r = 0.0
            
            # choose action from actor with OU-noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + 1E-2*actor_noise()
            # physical time evolvution
            psi2 = unitary_op(a, self.c2, self.Nt, self.dt, psi)
            s2 = util.state(psi2)
            r = (abs(psi2[-1])**2 - abs(psi[-1])**2)/self.dt
            
            # add experience to repaly buffer
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)),\
                              r, 0, np.reshape(s2, (actor.s_dim,)))
            
            # experience replay
            if replay_buffer.size() > int(args.minibatch_size):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args.minibatch_size))
            
                # calculate target
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                y_i = []
                for k in range(args.minibatch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma*target_q[k])
                
                # update critic nn with experience
                predicted_q_value, critic_loss, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args.minibatch_size), 1)))
                
                ep_ave_max_q += np.amax(predicted_q_value)
                
                # update actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])
                
                # update target networks
                actor.update_target_network()
                critic.update_target_network()    
            
            if replay_buffer.size() > int(args.minibatch_size) and ep % 100 ==0:
                print("critic loss is {}, start action is {}, episode {} with reward {}"\
                      .format(critic_loss, actor.predict([[0.8801223, -1.57079633]]), ep, actor_noise(), r))


#-------------------
#   main program
#-------------------
def main(args):
    
    with tf.Session() as sess:
        
        np.random.seed(int(args.random_seed))
        tf.set_random_seed(int(args.random_seed))
        # physical parameters
        state_dim = 2
        action_dim = 1
        action_bound = 3.0
        c2 = -1.0
        Nt = 2
        dt = 1E-2
        
        # make directory for model saving
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args.actor_lr), float(args.tau),
                             int(args.minibatch_size))
                             
        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args.critic_lr), float(args.tau), float(args.gamma),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        
        agent = AgengTraining(c2, Nt, dt)
        agent.train(sess, args, actor, critic, actor_noise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--time-steps', help='max time steps of 1 episode', default=5)
    
    # checkpoint_dir
    parser.add_argument('--checkpoint-dir', help='checkpoint saving directory', default='./model')

    args = parser.parse_args()

    main(args)
















































