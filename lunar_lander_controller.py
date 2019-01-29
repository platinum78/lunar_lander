from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import tensorflow as tf
import numpy as np
import time

class Lander:
    def __init__(self):
        self.position = np.array([0, 0])
        self.velocity = np.array([0, 0])
        self.distance = np.array([0, 0])
        self.altitude = 0
        self.rotation = 0
        self.fuel = 0
        self.destination = np.array([688.5, 638])

class Game:
    def __init__(self):
        self.time = 0.0
        self.state = 0

class lunar_lander_controller:
    def __init__(self, browser):
        # Open game and initialize environment
        if browser == "chrome":
            self.driver = webdriver.Chrome()
        elif browser == "firefox":
            self.driver = webdriver.Firefox()
        self.driver.set_window_size(750, 750)
        self.driver.set_window_position(100, 100)
        self.driver.get("http://moonlander.seb.ly/")
        self.game = self.driver.find_element_by_tag_name("body")
        self.lander = Lander()
        self.game = Game()
        self.get_lander_state()

        # Define key action chains
        self.arrow_up_press = ActionChains(self.driver).key_down(Keys.ARROW_UP)
        self.arrow_down_press = ActionChains(self.driver).key_down(Keys.ARROW_DOWN)
        self.arrow_left_press = ActionChains(self.driver).key_down(Keys.ARROW_LEFT)
        self.arrow_right_press = ActionChains(self.driver).key_down(Keys.ARROW_RIGHT)
        self.arrow_up_release = ActionChains(self.driver).key_up(Keys.ARROW_UP)
        self.arrow_down_release = ActionChains(self.driver).key_up(Keys.ARROW_DOWN)
        self.arrow_left_release = ActionChains(self.driver).key_up(Keys.ARROW_LEFT)
        self.arrow_right_release = ActionChains(self.driver).key_up(Keys.ARROW_RIGHT)

        # I/O parameters definition
        self.input_tensor_old = np.zeros([1, 6])
        self.input_tensor_new = np.zeros([1, 6])

        
    
    def get_lander_position(self, update=True):
        if update:
            self.lander.position = self.driver.execute_script("return [lander.pos.x, lander.pos.y]")
        else:
            return self.driver.execute_script("return [lander.pos.x, lander.pos.y]")
    
    def get_lander_velocity(self, update=True):
        if update:
            self.lander.velocity = self.driver.execute_script("return [lander.vel.x, lander.vel.y]")
        else:
            return self.driver.execute_script("return [lander.vel.x, lander.vel.y]")

    def get_lander_rotation(self, update=True):
        if update:
            self.lander.rotation = self.driver.execute_script("return lander.rotation")
        else:
            return self.driver.execute_script("return lander.rotation")
    
    def get_lander_altitude(self, update=True):
        if update:
            self.lander.altitude = self.driver.execute_script("return lander.altitude")
        else:
            return self.driver.execute_script("return lander.altitude")
    
    def get_lander_fuel(self, update=True):
        if update:
            self.lander.fuel = self.driver.execute_script("return lander.fuel")
        else:
            return self.driver.execute_script("return lander.fuel")
    
    def get_lander_state(self):
        self.get_lander_position()
        self.get_lander_velocity()
        self.get_lander_rotation()
        self.get_lander_altitude()
        self.get_lander_fuel()
        self.lander.distance = self.lander.destination - self.lander.position
    
    def get_game_time(self, update=True):
        if update:
            self.game.time = self.driver.execute_script("return counter * mpf") / 1000
        else:
            return self.driver.execute_script("return counter * mpf") / 1000
    
    def get_game_play_state(self, update=True):
        if update:
            self.game.state = self.driver.execute_script("return gameState")
        else:
            return self.driver.execute_script("return gameState")

    def get_game_state(self):
        self.get_game_time()
        self.get_game_play_state()
    
    def make_input_tensor(self):
        input_tensor = np.zeros([1, 6])
        input_tensor[0, 0] = self.lander.position[0]
        input_tensor[0, 1] = self.lander.position[1]
        input_tensor[0, 2] = self.lander.velocity[0]
        input_tensor[0, 3] = self.lander.velocity[1]
        input_tensor[0, 4] = self.lander.rotation
        return input_tensor
    
    def refresh_page(self):
        self.driver.refresh()
        self.game = self.driver.find_element_by_tag_name("body")
    
    def q_network_model(self):
        # Inputs will be as follows: [distance.x, distance.y, velocity.x, velocity.y, altitude, rotation]
        # Direction outputs will be as follows: [left, none, right]
        # Thrust outputs will be as follows: [go, no-go]

        # Input layer definition
        # Input tensor: R x C = 1 x 5
        self.inputs = tf.placeholder(shape=[1, 6], dtype=tf.float32)
        
        # Hidden layer #1 definition
        # W1: R x C = 5 x 20
        # Activation function: Sigmoid
        self.W1 = tf.Variable(tf.random_uniform([6, 20], 0, 0.01))
        self.b1 = tf.Variable(tf.random_normal([1]))
        self.z1 = tf.matmul(self.inputs, self.W1) + self.b1
        self.h1 = tf.nn.sigmoid(self.z1)
        
        # Hidden layer #2 definition
        # W2: R x C = 20 x 50
        # Activation function: Sigmoid
        self.W2 = tf.Variable(tf.random_uniform([20, 50], 0, 0.01))
        self.b2 = tf.Variable(tf.random_normal([1]))
        self.z2 = tf.matmul(self.h1, self.W2) + self.b2
        self.h2 = tf.nn.sigmoid(self.z2)

        # Hidden layer #3 definition
        self.W3_direction = tf.Variable(tf.random_uniform([50, 3], 0, 0.01))
        self.W3_thrust = tf.Variable(tf.random_uniform([50, 2], 0, 0.01))
        
        # Outputs and loss definition
        self.q_output_direction = tf.matmul(self.h2, self.W3_direction)
        self.q_output_thrust = tf.matmul(self.h2, self.W3_thrust)

        self.action_direction = tf.argmax(self.q_output_direction, 1)
        self.action_thrust = tf.argmax(self.q_output_thrust, 1)

        self.q_next_direction = tf.placeholder(shape=[1, 3], dtype=tf.float32)
        self.q_next_thrust = tf.placeholder(shape=[1, 2], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.q_next_direction - self.q_output_direction)) + \
                    tf.reduce_sum(tf.square(self.q_next_thrust - self.q_output_thrust))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.update_model = self.train_op.minimize(self.loss)
    
    def train(self, discount_rate, episode_num, loop_interval=0.1):
        print("Train %d times..." % episode_num)
        Y = discount_rate   # Y is used thanks to its morphological analogy to gamma, not as the alphabet

        with tf.Session() as sess:
            model_init = tf.global_variables_initializer()
            sess.run(model_init)

            # Run episodes in given number.
            for episode in range(episode_num):
                print("Episode %4d / %4d" % (episode + 1, episode_num))
                # Refresh the page, so that all the parameters are reset.
                self.refresh_page()
                self.game.click()
                self.get_lander_state()
                self.get_game_state()
                self.input_tensor_old = self.make_input_tensor()
                control_num = 0
                time.sleep(0.5)
                
                # Iterate the loop while the game is active.
                while True:
                    control_num += 1
                    # print("Control Sequence #%3d" % control_num)
                    # Record current time
                    loop_init_time = time.time()

                    # Update lander and game state
                    action_direction, action_thrust, q_direction_all, q_thrust_all = \
                        sess.run([self.action_direction, self.action_thrust, self.q_output_direction, self.q_output_thrust], \
                                 feed_dict={self.inputs: self.input_tensor_old})
                    self.keyboard_control(code=10*action_direction[0] + action_thrust[0])
                    
                    self.get_lander_state()
                    self.get_game_state()
                    self.input_tensor_new = self.make_input_tensor()
                    reward = self.compute_reward()
                    q1_direction, q1_thrust = sess.run([self.q_output_direction, self.q_output_thrust], \
                                                       feed_dict={self.inputs: self.input_tensor_new})
                    q1_direction_max = np.max(q1_direction)
                    q1_thrust_max = np.max(q1_thrust)
                    q_direction_target = q_direction_all
                    q_thrust_target = q_thrust_all
                    q_direction_target[0, action_direction] = reward + Y * q1_direction_max
                    q_thrust_target[0, action_thrust] = reward + Y * q1_thrust_max

                    sess.run([self.update_model], feed_dict={self.inputs: self.input_tensor_old,
                                                             self.q_next_direction: q_direction_target,
                                                             self.q_next_thrust: q_thrust_target})
                    self.input_tensor_old[:] = self.input_tensor_new[:]
                    # Sleep until loop time becomes designated interval
                    loop_end_time = time.time()
                    sleep_time = loop_end_time - loop_init_time
                    if sleep_time <= 0:
                        pass
                    else:
                        time.sleep(sleep_time)
                    if self.game.state != 1:
                        if self.game.state == 2:
                            print("Successfully landed!")
                        elif self.game.state == 3:
                            print("Crashed!")
                        break

    def compute_reward(self):
        # reward_time = 140 - self.game.time
        # if reward_time < 0:
        #     reward_time = 0
        # reward_fuel = self.lander.fuel - 700
        reward_survival = 50

        # reward = reward_time + reward_fuel + reward_survival
        reward = reward_survival

        if self.lander.altitude > 750:
            reward -= 500

        if self.game.state == 2:
            reward += 8000
        elif self.game.state == 3:
            reward -= 7750
        
        reward_distance = (500 - np.sqrt(np.sum((self.lander.destination - self.lander.position)**2))) / 50
        reward += reward_distance

        reward *= 0.01
        # print(reward_distance, reward)
        return reward
    
    def keyboard_control(self, code):
        # Direction outputs will be as follows: [left, none, right]
        # Thrust outputs will be as follows: [go, no-go]
        # Code will be given as follows: 10 x direction_argmax + thrust_argmax

        # First, release all keys
        self.arrow_left_release.perform()
        self.arrow_right_release.perform()
        self.arrow_down_release.perform()
        self.arrow_up_release.perform()

        if code == -1:
            # print("Control off!")
            pass
        elif code == 0:
            self.arrow_left_press.perform()
            self.arrow_up_press.perform()
            # print("Left / Go")
        elif code == 10:
            self.arrow_up_press.perform()
            # print("Center / Go")
            pass
        elif code == 20:
            self.arrow_right_press.perform()
            self.arrow_up_press.perform()
            # print("Right / Go")
        elif code == 1:
            self.arrow_left_press.perform()
            # print("Left / No-Go")
            pass
        elif code == 11:
            # print("Center / No-Go")
            pass
        elif code == 21:
            self.arrow_right_press.perform()
            # print("Right / No-Go")
            pass
            