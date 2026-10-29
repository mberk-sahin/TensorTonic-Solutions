def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    # Write code here
    new_val = []
    num_state = len(rewards)
    num_actions = len(rewards[0])
    for s_i in range(num_state):
        max_val = None
        for a_i in range(num_actions):
            exp_rewards = 0 
            for sp_i in range(num_state):
                exp_rewards += transitions[s_i][a_i][sp_i] * values[sp_i]
            exp_rewards = rewards[s_i][a_i] + exp_rewards * gamma

            if max_val is None or max_val <= exp_rewards:
                max_val = exp_rewards
        new_val.append(max_val)
    return new_val