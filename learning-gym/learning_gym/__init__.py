from gym.envs.registration import register

register(
    id='Learning-v0',
    entry_point='learning_gym.envs:LearningEnv',
    max_episode_steps=200,
    reward_threshold=30.0,
)