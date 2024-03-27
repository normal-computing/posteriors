import torch
import matplotlib.pyplot as plt
import pandas as pd
import posteriors


def download_data(start=21, end=23):
    urls = [
        f"https://www.football-data.co.uk/mmz4281/{y}{y+1}/E0.csv"
        for y in range(start, end)
    ]

    origin_date = pd.to_datetime(f"20{start}-08-01")
    data = pd.concat(pd.read_csv(url) for url in urls)
    data = data.dropna()
    data["Timestamp"] = pd.to_datetime(data["Date"], dayfirst=True)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], unit="D")
    data["TimestampDays"] = (
        (data["Timestamp"] - origin_date).astype("timedelta64[D]").astype(int)
    )

    players_arr = pd.unique(pd.concat([data["HomeTeam"], data["AwayTeam"]]))
    players_arr.sort()
    players_name_to_id_dict = {a: i for i, a in enumerate(players_arr)}
    players_id_to_name_dict = {i: a for i, a in enumerate(players_arr)}

    data["HomeTeamID"] = data["HomeTeam"].apply(lambda s: players_name_to_id_dict[s])
    data["AwayTeamID"] = data["AwayTeam"].apply(lambda s: players_name_to_id_dict[s])

    match_times = torch.tensor(data["TimestampDays"].to_numpy(), dtype=torch.float64)
    match_player_indices = torch.tensor(data[["HomeTeamID", "AwayTeamID"]].to_numpy())

    home_goals = torch.tensor(data["FTHG"].to_numpy())
    away_goals = torch.tensor(data["FTAG"].to_numpy())

    match_results = torch.where(
        home_goals > away_goals, 1, torch.where(home_goals < away_goals, 2, 0)
    )

    dataset = torch.utils.data.StackDataset(
        match_times=match_times,
        match_player_indices=match_player_indices,
        match_results=match_results,
    )

    return (
        dataset,
        players_id_to_name_dict,
        players_name_to_id_dict,
    )


(
    dataset,
    players_id_to_name_dict,
    players_name_to_id_dict,
) = download_data()


dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

epsilon = 1.0
transition_sd_scale = 0.1


def log_likelihood(params, batch):
    player_indices = batch["match_player_indices"]
    match_results = batch["match_results"]

    player_skills = params[player_indices]

    home_win_prob = torch.sigmoid(player_skills[:, 0] - player_skills[:, 1] - epsilon)
    away_win_prob = 1 - torch.sigmoid(
        player_skills[:, 0] - player_skills[:, 1] + epsilon
    )
    draw_prob = 1 - home_win_prob - away_win_prob
    result_probs = torch.vstack([draw_prob, home_win_prob, away_win_prob]).T
    log_liks = torch.log(result_probs[torch.arange(len(match_results)), match_results])
    return log_liks, result_probs


num_teams = len(players_id_to_name_dict)

init_means = torch.zeros((num_teams,))
init_sds = torch.ones((num_teams,))

state = posteriors.ekf.diag_fisher.init(init_means, init_sds)
all_means = init_means.unsqueeze(0)
all_sds = init_sds.unsqueeze(0)
previous_time = 0.0
for match in dataloader:
    match_time = match["match_times"]

    state = posteriors.ekf.diag_fisher.update(
        state,
        match,
        log_likelihood,
        lr=1.0,
        per_sample=True,
        transition_sd=torch.sqrt(transition_sd_scale**2 * (match_time - previous_time)),
    )
    all_means = torch.vstack([all_means, state.params.unsqueeze(0)])
    all_sds = torch.vstack([all_sds, state.sd_diag.unsqueeze(0)])
    previous_time = match_time


# Plot the last season
last_season_start = (len(all_means) - 1) // 2
times = dataset.datasets["match_times"][last_season_start:]
means = all_means[last_season_start + 1 :]
sds = all_sds[last_season_start + 1 :]

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(num_teams):
    team_name = players_id_to_name_dict[i]
    if team_name in ("Arsenal", "Man City"):
        c = "skyblue" if team_name == "Man City" else "red"
        ax.plot(times, means[:, i], c=c, zorder=2, label=team_name)
        ax.fill_between(
            times,
            means[:, i] - sds[:, i],
            means[:, i] + sds[:, i],
            color=c,
            alpha=0.3,
            zorder=1,
        )
    else:
        ax.plot(times, means[:, i], c="grey", alpha=0.2, zorder=0, linewidth=0.75)

ax.set_xticklabels([])
ax.set_xlabel("2022/23 season")
ax.set_ylabel("Skill")

ax.legend()
fig.tight_layout()
fig.savefig("ekf_premier_league.png", dpi=300)
