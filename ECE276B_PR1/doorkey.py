from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.core.world_object import Wall
from pathlib import Path
from itertools import product

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

def read_map(env) :
    """
    read the wall, key, door, goal positions
    """
    u = env.unwrapped

    m = { # for the map it contains walls, height, width, key, door, goal positions
            "walls":  set(),
            "height": u.height,
            "width": u.width,
            "key_positions": [],
            "goal_positions": [],
            "doors": []
        }

    for y in range(u.height) :
        for x in range(u.width) :
            cell = u.grid.get(x, y)
            if isinstance(cell, Wall) :
                m["walls"].add((x, y))
            elif isinstance(cell, Key):
                m["key_positions"].append((x, y))
            elif isinstance(cell, Door):
                m["doors"].append((x, y))
            elif isinstance(cell, Goal):
                m["goal_positions"].append((x, y))
    return m

def all_states(m) :
    states = []
    for x in range(m["width"]) :
        for y in range(m["height"]) : 
            if (x, y) in m["walls"] : continue
            for d, k, opens, key_pos_idx, goal_pos_idx in product(range(4), range(2), product(range(2), repeat=len(m["doors"])), range(len(m["key_positions"])), range(len(m["goal_positions"]))) :
                states.append((x, y, d, k, opens, key_pos_idx, goal_pos_idx)) #d for direction facing, k for key picked or not, opens for door open or not
    return states

def initial_state(env, m) :
    """
    reads the starting (x, y, direction facing, carrying key or not, door open or not, key position index, goal position index)
    """
    u = env.unwrapped
    x, y = u.agent_pos
    d = u.agent_dir
    opens = tuple(int(u.grid.get(*p).is_open) for p in m["doors"])
    key_pos_idx = next(i for i, p in enumerate(m["key_positions"]) if isinstance(u.grid.get(*p), Key))
    goal_pos_idx = next(i for i, p in enumerate(m["goal_positions"]) if isinstance(u.grid.get(*p), Goal))

    return (int(x), int(y), d, int(u.carrying is not None), opens, key_pos_idx, goal_pos_idx)


# Direction number -> (dx, dy) step, in MiniGrid's order: 0 right, 1 down, 2 left, 3 up.
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def motion_model(state, action, m) :
    """
    f(x, u): the state after taking `action` in `state` on map `m`
    this returns a new (x, y, d, k, o) tuple. An action that does nothing returns `state` unchanged
    """
    x, y, d, k, opens, key_pos_idx, goal_pos_idx = state
    dx, dy = DIR_VEC[d]
    front = (x + dx, y + dy)  # the cell the robot is facing

    if action == TL:
        return (x, y, (d - 1) % 4, k, opens, key_pos_idx, goal_pos_idx)
    if action == TR:
        return (x, y, (d + 1) % 4, k, opens, key_pos_idx, goal_pos_idx)
    key_ahead = front == m["key_positions"][key_pos_idx] and not k # if the front of the robot is indeed key positions and is actually holding the key
    door_idx = m["doors"].index(front) if front in m["doors"] else None # which door is front, if there is any
    if action == MF:
        off_grid = not (0 <= front[0] < m["width"] and 0 <= front[1] < m["height"])
        if off_grid or front in m["walls"] or key_ahead or (door_idx is not None and not opens[door_idx]) :
            return state
        return (front[0], front[1], d, k, opens, key_pos_idx, goal_pos_idx)

    if action == PK:
        return (x, y, d, 1, opens, key_pos_idx, goal_pos_idx) if key_ahead else state

    if action == UD :
        # a door that started open and gets closed is really unlocked, so 0/1 cant really determine so
        # i assume, all doors started as locked, well also guessing that optimal plan never closes door since it has no reason to waste energy to that!
        if door_idx is None or (not opens[door_idx] and not k) :
            return state
        new = list(opens)
        new[door_idx] = 1 - opens[door_idx] # a trick to flip open to closed, and closed (with key) to open ( 1 - 1(open) = 0 and 1 - 0(closed) = 1)
        return (x, y, d, k, tuple(new), key_pos_idx, goal_pos_idx)
    raise ValueError(f"unknown action {action}")

def terminal_cost(state, m) :
    x, y, goal_pos_idx = state[0], state[1], state[6]
    if (x, y) == m["goal_positions"][goal_pos_idx] : return 0
    else : return np.inf

def solve(m, max_iter = 1000) : #the dp part
    X = list(all_states(m))
    actions = [MF, TL, TR, PK, UD]
    V = {x: terminal_cost(x, m) for x in X}

    for _ in range(max_iter):
        V_new = {}
        policy = {}
        for x in X:
            if terminal_cost(x, m) == 0:
                V_new[x] = 0
                policy[x] = None  # No action needed, at the goal cost 0
                continue

            best_cost = np.inf
            best_action = None
            for u in actions:
                next_x = motion_model(x, u, m) # x_next = f(x, u)
                cost = step_cost(u) + V[next_x]
                if best_action is None or cost < best_cost:
                    best_cost = cost
                    best_action = u

            V_new[x] = best_cost
            policy[x] = best_action

        if V_new == V:
            return V_new, policy

        V = V_new
    raise RuntimeError("Value iteration did not converge within max_iter")

def extract_sequence(init_state, policy, m, max_iter = 10000) :
    x = init_state
    actions = []
    for _ in range(max_iter):
        u = policy[x]
        actions.append(u)
        x  = motion_model(x, u, m)
        if terminal_cost(x, m) == 0 :
            return actions, len(actions) 
    raise RuntimeError("Iteration did not converge within max_iter")

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
    m = read_map(env)
    init_state = initial_state(env, m) # initial state is basically x0
    V, policy = solve(m)
    action_seq, n = extract_sequence(init_state, policy, m) # actions sequence and the length of the sequence
    # optim_act_seq = extract_sequence[init_state, policy, m]
    return action_seq


def partA():
    env_folder = Path("./envs/known_envs")
    Path("./gif/partA").mkdir(parents=True, exist_ok=True)
    for env_path in sorted(env_folder.glob("*.env")):
    #env_path = "./envs/known_envs/doorkey-5x5-normal.env"
        env, info = load_env(env_path)  # load an environment
        action_seq = doorkey_problem(env)  # find the optimal action sequence
        name = env_path.stem
        print(f"{name}: {len(action_seq)} actions \n")
        draw_gif_from_seq(action_seq, load_env(env_path)[0], f"./gif/partA/{name}.gif")  # draw a GIF & save


def partB():
    env_folder = Path("./envs/random_envs")
    Path("./gif/partB").mkdir(parents=True, exist_ok=True)
    env, _ = load_env(next(env_folder.glob("*.env"))) # any map has walls and doors shared
    m = read_map(env)
    m["key_positions"] = [(2, 2), (2, 3), (1, 6)]       # all 3 candidates (from the spec)
    m["goal_positions"] = [(6, 1), (7, 3), (6, 6)] 
    V, policy = solve(m) # for B, one policy for all 36 maps
    for env_path in sorted(env_folder.glob("*.env")):
    #env_path = "./envs/known_envs/doorkey-5x5-normal.env"
        env, _ = load_env(env_path)  # load an environment
        init_state = initial_state(env, m) # each map only supplies initial or start stae
        action_seq, n = extract_sequence(init_state, policy, m)  # find the optimal action sequence
        # print(f"This is the info : {info}")
        name = env_path.stem
        print(f"{name}: {n} actions, V[x0] = {V[init_state]}\n")
        draw_gif_from_seq(action_seq, load_env(env_path)[0], f"./gif/partB/{name}.gif")  # draw a GIF & save
        


if __name__ == "__main__":
    #example_use_of_gym_env()
    partA()
    partB()

