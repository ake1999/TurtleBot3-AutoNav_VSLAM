import rclpy
import env
import numpy as np
import time 
rclpy.init()
e = env.RLEnvTurtlebot()
# rclpy.spin_once(e, timeout_sec=5)
# e.velocity


import threading
spin_thread = threading.Thread(target=rclpy.spin, args=(e,), daemon=True)
spin_thread.start()

time.sleep(2)

for i in range(200):
    #print('velocity: ', e.velocity)
    #print('position: ', e.position)
    # print(e.polar_minimap)
    #print(e.step([0.0,0.2]))
    e.visualize_minimap(e.polar_minimap)
    #if i % 10 == 9:
    #    e.reset()


rclpy.shutdown()
