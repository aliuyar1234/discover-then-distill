from llm_mhc_sdft_tttd.training.ttt_discover import ToyStringMatchEnv


def test_toy_env_reward_overlap():
    env = ToyStringMatchEnv(problem_description="toy", target="HELLO")
    assert env.initial_state() == ""
    assert env.reward("") == 0.0
    assert env.reward("xxxx") == 0.0
    assert env.reward("hello") == 1.0
    assert env.reward("ole") > 0.0
