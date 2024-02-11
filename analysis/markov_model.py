import numpy as np
import pandas as pd
from hmmlearn import hmm


def train_and_predict_hidden_markov_model(df):
    window_size = 10

    # train_df = df[['point_won', 'point_loss', 'ace', 'winner', 'double_fault', 'unf_err', 'net_point', 'net_point_won', 'break_pt', 'break_pt_won', 'break_pt_miss']]

    train_df = df
    #         "p1_winner",
    #         "p2_winner",
    #         "winner_shot_type",
    #         "p1_double_fault",
    #         "p2_double_fault",
    #         "p1_unf_err",
    #         "p2_unf_err",
    #         "p1_net_pt_won",
    #         "p2_net_pt_won",
    #         "p1_break_pt_won",
    #         "p2_break_pt_won",
    #         "rally_count",
    #         "serve_width",
    #         "serve_depth",
    #         "return_depth"
    df["observation"] = 0

    # mapping = {}
    # counter = 0
    # for i in range(len(train_df)):
    #     cur_combination = train_df.iloc[i].to_list()
    #
    #     if str(cur_combination) not in mapping.keys():
    #         mapping[str(cur_combination)] = counter
    #         df.loc[i, "observation"] = counter
    #         counter += 1
    #     else:
    #         df.loc[i, "observation"] = mapping[str(cur_combination)]

    observation_list = df["observation"].to_list()

    # value_separated_observation_list = [observation_list[i - window_size: i] for i in range(window_size, len(observation_list))]
    # value_separated_observation_list = [[0] * window_size] * window_size + value_separated_observation_list

    observations = np.array([np.sum(np.array([train_df.iloc[j].to_list() for j in range(i-window_size, i)]).astype(int), axis=0) for i in range(window_size, len(train_df))])

    observations = abs(np.min(observations)) + observations

    observations = observations.astype(int)

    m_observations = np.concatenate(
        (np.array([observations[0].tolist()] * window_size), observations),
        axis=0
    )

    df = pd.concat([df, pd.DataFrame({"window_observation": m_observations.tolist()})], axis=1)

    hidden_markov_model = hmm.MultinomialHMM(n_components=5, n_iter=50, tol=0.01)

    hidden_markov_model.fit(observations)

    start_prob = hidden_markov_model.startprob_
    transition_prob = hidden_markov_model.transmat_
    emission_prob = hidden_markov_model.emissionprob_

    neg_log_likelihood, pred = calculate_momentum(df, hidden_markov_model, m_observations)

    _, hidden2observation = hidden_markov_model.score_samples(observations)

    state_impacts = np.sum(hidden2observation, axis=0)

    return state_impacts, neg_log_likelihood, pred, start_prob, transition_prob, emission_prob

    state_impacts = np.zeros((num_states, num_obs))

    for t in range(num_obs):
        for i in range(num_states):
            state_impacts[i, t] = (forward_prob[t, i] * backward_prob[t, i]) / np.sum(
                forward_prob[t, :] * backward_prob[t, :])

    return neg_log_likelihood, pred, start_prob, transition_prob, emission_prob


def calculate_momentum(df, hidden_markov_model, m_observations):
    # pred_list = []
    # neg_log_likelihood_list = []
    # for i in range(len(df)):
    #     neg_log_likelihood, pred = hidden_markov_model.decode(np.array([df.loc[i, "window_observation"]]))
    #     pred_list.append(pred[0])
    #     neg_log_likelihood_list.append(neg_log_likelihood)
    #
    # return pred_list, neg_log_likelihood_list

    neg_log_likelihood, pred = hidden_markov_model.decode(m_observations)

    return neg_log_likelihood, pred

