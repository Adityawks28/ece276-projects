from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.core.world_object import Wall

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)

def read_map(env, info) :
    """
    read the wall, key, door, goal positions
    """
    m = { # for the map it contains walls, height, width, key, door, goal positions
            "walls":  set(),
            "height": env.unwrapped.height,
            "width": env.unwrapped.width,
        }

    for i in range(env.unwrapped.width) :
        for j in range(env.unwrapped.height) :
            if isinstance(env.unwrapped.grid.get(j, i), Wall) :
                m["walls"].add((j, i))
            elif isinstance(env.unwrapped.grid.get(j, i), Key):
                m["key_pos"] = (j, i)
            elif isinstance(env.unwrapped.grid.get(j, i), Door):
                m["door_pos"] = (j, i)
            elif isinstance(env.unwrapped.grid.get(j, i), Goal):
                m["goal_pos"] = (j, i)

    return m

def all_states(m) :
    states = []
    for x in range(m["width"]) :
        for y in range(m["height"]) : 
            for dir in range(4) : # direction
                for key in range(2) : # have key or not
                    for open in range(2) : #door open or not
                        states.append((x, y, dir, key, open))
    return states

def initial_state(env, info) :
    """
    reads the starting (x, y, d, k, o from simulator)
    """
    x, y = info["init_agent_pos"]
    d = env.unwrapped.agent_dir
    k = int(env.unwrapped.carrying is not None)
    o = int(env.unwrapped.grid.get(info["door_pos"][0], info["door_pos"][1]).is_open)

    return (x, y, d, k, o)


# Direction number -> (dx, dy) step, in MiniGrid's order: 0 right, 1 down, 2 left, 3 up.
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def motion_model(state, action, m) :
    """
    f(x, u): the state after taking `action` in `state` on map `m`
    this returns a new (x, y, d, k, o) tuple. An action that does nothing returns `state` unchanged
    """
    x, y, d, k, o = state
    dx, dy = DIR_VEC[d]
    front = (x + dx, y + dy)  # the cell the robot is facing

    if action == TL:
        return (x, y, (d - 1) % 4, k, o)
    if action == TR:
        return (x, y, (d + 1) % 4, k, o)
    if action == MF:
        blocked = (front in m["walls"]
                   or (front == m["key_pos"] and not k)    # key still lying there, can't move
                   or (front == m["door_pos"] and not o))  # door closed
        if blocked:
            return state
        return (front[0], front[1], d, k, o)
    if action == PK:
        if front == m["key_pos"] and not k:
            return (x, y, d, 1, o)
        return state
    if action == UD:
        if front == m["door_pos"]:
            if o:  # open door: UD closes it
                return (x, y, d, k, 0)
            if k:  # closed door and holding the key: UD unlocks and opens it
                return (x, y, d, k, 1)
        return state
    raise ValueError(f"unknown action {action}")


def doorkey_problem(env):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """
    optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq


def partA():
    env_path = "./envs/known_envs/doorkey-5x5-normal.env"
    env, info = load_env(env_path)  # load an environment
    seq = doorkey_problem(env)  # find the optimal action sequence
    # print(f"This is the info : {info}")

    draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save


def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)


if __name__ == "__main__":
    #example_use_of_gym_env()
    partA()
    # partB()

