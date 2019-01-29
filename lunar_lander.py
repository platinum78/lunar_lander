from lunar_lander_controller import *
import sys

if __name__ == "__main__":
    lunar_lander = lunar_lander_controller("firefox")
    lunar_lander.q_network_model()
    lunar_lander.train(discount_rate=0.9, episode_num=int(sys.argv[1]))