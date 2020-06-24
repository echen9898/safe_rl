## Writing Assignment
See [here](solution.ipynb)

# Particle Filter Localization


This code implements the MCL algorithm for the RACECAR. 

[![YouTube Demo](./media/thumb.jpg)](https://www.youtube.com/watch?v=-c_0hSjgLYw)

For high efficiency in Python, it uses Numpy arrays and RangeLibc for fast 2D ray casting.

# Configuration
```
cd ~/racecar_ws/src/localization-solution/src
python setup.py build_ext --inplace
```

# Usage

The majority of parameters you might want to tweak are in the param.yaml file. You may have to modify the "odom_topic", "scan_topic", "num_beams_per_particle" and "angle_step" parameters in the launch file to match your environment.

In sumulation environment:
```
roslaunch racecar_simulator simulate.launch
roslaunch localization-solution localize_simulation.launch
```

In RACECAR environment:
```
roslaunch localization-solution localize_real_env.launch
```


Once the particle filter is running, you can visualize the map and other particle filter visualization message in RViz. Use the "2D Pose Estimate" tool from the RViz toolbar to initialize the particle locations.
