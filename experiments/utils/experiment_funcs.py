from utils.functions import *

def gatherConditionalActions(env, observation, obs_horizon, start_iter):
    img_hist, vel_hist, haction_hist = [], [], [] # Initialize history
    for i in range(obs_horizon):
        # We move using the PD controller
        a = calculateAction(observation, target_velocity = 30)
        # We calculate the "human" policy ie sinonoidal trajectory
        h_action = action_sinusoidalTrajectory(i+start_iter, 1/100, observation, 12 , target_velocity = 30) #Unsafe driver
        obs, r, done, info = env.step(a) # Move while "AI" policy (PD controller) is still in control

        observation = {
            "image": obs,
            "velocity": env.return_absolute_velocity(),     # This function was added manually to car racing environment
            "track": env.return_track_flag()                # This function was added manually to car racing environment
        }

        # We store the history of observations, velocities and actions by "human" policy
        img_hist.append(observation["image"])
        vel_hist.append(observation["velocity"])
        haction_hist.append(h_action)

    img_hist = np.array(img_hist, dtype=np.float32)
    vel_hist = np.array(vel_hist, dtype=np.float32)
    vel_hist = vel_hist[:,None]
    haction_hist = np.array(haction_hist, dtype=np.float32)
    return img_hist, vel_hist, haction_hist

