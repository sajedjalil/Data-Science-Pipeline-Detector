import plotly.express as px
import plotly.graph_objects as go
from pdb import set_trace as bp
import pandas as pd
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import numpy as np
pd.options.mode.chained_assignment = None
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

def plot_playmaking_skills(scores_df):
    fig = px.scatter(scores_df,
                        x="eir",
                        y="epa",
                        color="overall_score",
                        color_continuous_scale="portland",
                        hover_data=["name"],
                        title="Playmaking Skills",
                        labels={
                        "eir": "EIR",
                        "epa": "EPA",
                        "overall_score": "Overall score",
                        "name": "Name"
                    })
    fig.update_traces(marker=dict(size=12,
                                line=dict(width=2,
                                            color='DarkSlateGrey')),
                    selector=dict(mode='markers'))
    fig.add_shape(type="rect",
        x0=20, y0=0, x1=60, y1=0.4,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_shape(type="rect",
        x0=48, y0=-0.8, x1=60, y1=0.4,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_trace(go.Scatter(
        x=[30, 55],
        y=[0.28, -0.5],
        text=["Top<br>Playmaking<br>Ability", "Top<br>Tracking<br>Ability"],
        mode="text",
        textfont_size=16
    ))
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis_title="Expected Incompletion Rate (EIR) (%)",
        yaxis_title="Expected Points Added (EPA)",
        font=dict(
            size=14,
        )
    )
    fig.show()

def plot_ball_skills(scores_df):
    fig = px.scatter(scores_df,
                        x="irae",
                        y="int_rate",
                        color="overall_score",
                        color_continuous_scale="portland",
                        hover_data=["name"],
                        title="Ball Skills",
                        labels={
                         "irae": "IRAE",
                         "int_rate": "INT rate",
                         "overall_score": "Overall score",
                         "name": "Name"
                     })
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.add_shape(type="rect",
        x0=-15, y0=4, x1=17, y1=12,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_shape(type="rect",
        x0=5, y0=-1, x1=17, y1=12,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_trace(go.Scatter(
        x=[-8, 13],
        y=[10, 0.5],
        text=["Top<br>Takeaway<br>Ability", "Top Pass<br>Breakup<br>Ability"],
        mode="text",
        textfont_size=16
    ))
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis_title="Incompletion Rate Above Expectation (IRAE) (%)",
        yaxis_title="Interception Rate (%)",
        font=dict(
            size=14,
        )
    )
    fig.show()

def plot_coverage_skills(scores_df):
    fig = px.scatter(scores_df,
                        x="ipa",
                        y="inc_rate",
                        color="overall_score",
                        color_continuous_scale="portland",
                        hover_data=["name"],
                        title="Coverage Skills",
                        labels={
                         "ipa": "IPA",
                         "inc_rate": "INC rate",
                         "overall_score": "Overall score",
                         "name": "Name"
                     })
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.add_shape(type="rect",
        x0=-2, y0=50, x1=18, y1=60,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_shape(type="rect",
        x0=12, y0=15, x1=18, y1=60,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_trace(go.Scatter(
        x=[2, 15],
        y=[55, 22],
        text=["Top<br>Shutdown<br>Ability", "Top True<br>Coverage<br>Ability"],
        mode="text",
        textfont_size=16
    ))
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis_title="Incompletion Probability Added (IPA) (%)",
        yaxis_title="Incompletion Rate (%)",
        font=dict(
            size=14,
        )
    )
    fig.show()

# get final rankings table for safeties and DBs
def get_final_saf_rankings_table(n_players):
    scores_df = pd.read_csv('../input/nfl-bdb-data/saf_scores.csv')
    rankings_df = pd.read_csv('../input/nfl-bdb-data/saf_rankings.csv')
    final_table = rankings_df
    final_table = final_table.drop(columns=["position", "n_throws"])
    
    final_table.loc[final_table.name == 'Derwin James', "name"] = 'Derwin James *'
    final_table.loc[final_table.name == 'Eric Weddle', "name"] = 'Eric Weddle *'
    final_table.loc[final_table.name == 'Jamal Adams', "name"] = 'Jamal Adams *'
    final_table.loc[final_table.name == 'Adrian Phillips', "name"] = 'Adrian Phillips *'
    final_table.loc[final_table.name == 'Eddie Jackson', "name"] = 'Eddie Jackson *'
    final_table.loc[final_table.name == 'Harrison Smith', "name"] = 'Harrison Smith *'
    final_table.loc[final_table.name == 'Landon Collins', "name"] = 'Landon Collins *'
    final_table.loc[final_table.name == 'Malcolm Jenkins', "name"] = 'Malcolm Jenkins *'
    final_table.loc[final_table.name == 'Michael Thomas', "name"] = 'Michael Thomas *'

    indexNames = final_table[ final_table.index >= n_players ].index
    final_table.drop(indexNames , inplace=True)

    final_table = final_table.round(1)

    final_table.inc_rate = scores_df.inc_rate.astype(str) + " (" + final_table.inc_rate.astype(str) + ")"
    final_table.int_rate = scores_df.int_rate.astype(str) + " (" + final_table.int_rate.astype(str) + ")"
    final_table.epa = scores_df.epa.astype(str) + " (" + final_table.epa.astype(str) + ")"
    final_table.eir = scores_df.eir.astype(str) + " (" + final_table.eir.astype(str) + ")"
    final_table.irae = scores_df.irae.astype(str) + " (" + final_table.irae.astype(str) + ")"
    final_table.ipa = scores_df.ipa.astype(str) + " (" + final_table.ipa.astype(str) + ")"
    final_table.overall_score = final_table.overall_score.astype(str)

    final_table = final_table.rename(columns={"name": "Name", "inc_rate": "INC Rate", "int_rate": "INT Rate", "epa": "EPA"})
    final_table = final_table.rename(columns={"eir": "EIR", "irae": "IRAE", "ipa": "IPA", "raw_score": "Raw Score", "overall_score": "Overall Score"})

    final_table.index += 1

    pd.set_option('display.max_rows', final_table.shape[0]+1)
    final_table = final_table.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    final_table = final_table.set_properties(**{'text-align': 'center'})
    return final_table

# get final rankings table for linebackers
def get_final_lb_rankings_table(n_players):
    scores_df = pd.read_csv('../input/nfl-bdb-data/lb_scores.csv')
    rankings_df = pd.read_csv('../input/nfl-bdb-data/lb_rankings.csv')
    final_table = rankings_df
    final_table = final_table.drop(columns=["position", "n_throws"])
    
    final_table.loc[final_table.name == 'Khalil Mack', "name"] = 'Khalil Mack *'
    final_table.loc[final_table.name == 'Ryan Kerrigan', "name"] = 'Ryan Kerrigan *'
    final_table.loc[final_table.name == 'Anthony Barr', "name"] = 'Anthony Barr *'
    final_table.loc[final_table.name == 'Olivier Vernon', "name"] = 'Olivier Vernon *'
    final_table.loc[final_table.name == 'Luke Kuechly', "name"] = 'Luke Kuechly *'
    final_table.loc[final_table.name == 'Bobby Wagner', "name"] = 'Bobby Wagner *'
    final_table.loc[final_table.name == 'Leighton Vander Esch', "name"] = 'Leighton Vander Esch *'
    final_table.loc[final_table.name == 'Cory Littleton', "name"] = 'Cory Littleton *'
    final_table.loc[final_table.name == 'Von Miller', "name"] = 'Von Miller *'
    final_table.loc[final_table.name == 'Jadeveon Clowney', "name"] = 'Jadeveon Clowney *'
    final_table.loc[final_table.name == 'Dee Ford', "name"] = 'Dee Ford *'
    final_table.loc[final_table.name == 'T.J. Watt', "name"] = 'T.J. Watt *'
    final_table.loc[final_table.name == 'C.J. Mosley', "name"] = 'C.J. Mosley *'
    final_table.loc[final_table.name == 'Benardrick McKinney', "name"] = 'Benardrick McKinney *'

    indexNames = final_table[ final_table.index >= n_players ].index
    final_table.drop(indexNames , inplace=True)

    final_table = final_table.round(1)

    final_table.inc_rate = scores_df.inc_rate.astype(str) + " (" + final_table.inc_rate.astype(str) + ")"
    final_table.int_rate = scores_df.int_rate.astype(str) + " (" + final_table.int_rate.astype(str) + ")"
    final_table.epa = scores_df.epa.astype(str) + " (" + final_table.epa.astype(str) + ")"
    final_table.eir = scores_df.eir.astype(str) + " (" + final_table.eir.astype(str) + ")"
    final_table.irae = scores_df.irae.astype(str) + " (" + final_table.irae.astype(str) + ")"
    final_table.ipa = scores_df.ipa.astype(str) + " (" + final_table.ipa.astype(str) + ")"
    final_table.overall_score = final_table.overall_score.astype(str)

    final_table = final_table.rename(columns={"name": "Name", "inc_rate": "INC Rate", "int_rate": "INT Rate", "epa": "EPA"})
    final_table = final_table.rename(columns={"eir": "EIR", "irae": "IRAE", "ipa": "IPA", "raw_score": "Raw Score", "overall_score": "Overall Score"})

    final_table.index += 1

    pd.set_option('display.max_rows', final_table.shape[0]+1)
    final_table = final_table.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    final_table = final_table.set_properties(**{'text-align': 'center'})
    return final_table
    

# get final rankings table for cornerbacks
def get_final_cb_rankings_table(n_players):
    scores_df = pd.read_csv('../input/nfl-bdb-data/cb_scores.csv')
    rankings_df = pd.read_csv('../input/nfl-bdb-data/cb_rankings.csv')
    final_table = rankings_df
    final_table = final_table.drop(columns=["position", "n_throws"])

    final_table.loc[final_table.name == 'Xavien Howard', "name"] = 'Xavien Howard *'
    final_table.loc[final_table.name == 'Jalen Ramsey', "name"] = 'Jalen Ramsey *'
    final_table.loc[final_table.name == 'Stephon Gilmore', "name"] = 'Stephon Gilmore *'
    final_table.loc[final_table.name == 'Denzel Ward', "name"] = 'Denzel Ward *'
    final_table.loc[final_table.name == 'Chris Harris', "name"] = 'Chris Harris *'
    final_table.loc[final_table.name == 'Kyle Fuller', "name"] = 'Kyle Fuller *'
    final_table.loc[final_table.name == 'Patrick Peterson', "name"] = 'Patrick Peterson *'
    final_table.loc[final_table.name == 'Darius Slay', "name"] = 'Darius Slay *'
    final_table.loc[final_table.name == 'Byron Jones', "name"] = 'Byron Jones *'

    indexNames = final_table[ final_table.index >= n_players ].index
    final_table.drop(indexNames , inplace=True)

    final_table = final_table.round(1)

    final_table.inc_rate = scores_df.inc_rate.astype(str) + " (" + final_table.inc_rate.astype(str) + ")"
    final_table.int_rate = scores_df.int_rate.astype(str) + " (" + final_table.int_rate.astype(str) + ")"
    final_table.epa = scores_df.epa.astype(str) + " (" + final_table.epa.astype(str) + ")"
    final_table.eir = scores_df.eir.astype(str) + " (" + final_table.eir.astype(str) + ")"
    final_table.irae = scores_df.irae.astype(str) + " (" + final_table.irae.astype(str) + ")"
    final_table.ipa = scores_df.ipa.astype(str) + " (" + final_table.ipa.astype(str) + ")"
    final_table.overall_score = final_table.overall_score.astype(str)

    final_table = final_table.rename(columns={"name": "Name", "inc_rate": "INC Rate", "int_rate": "INT Rate", "epa": "EPA"})
    final_table = final_table.rename(columns={"eir": "EIR", "irae": "IRAE", "ipa": "IPA", "raw_score": "Raw Score", "overall_score": "Overall Score"})

    final_table.index += 1

    pd.set_option('display.max_rows', final_table.shape[0]+1)
    final_table = final_table.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    final_table = final_table.set_properties(**{'text-align': 'center'})
    return final_table

# gets data for a specific play
def get_play_data(week, gameId, playId):
    train_df = pd.read_csv('../input/nfl-bdb-data/training_data.csv')
    play = train_df.query('week == {:.0f} and game == {:.0f} and play == {:.0f}'.format(week, gameId, playId))
    play.result = 'INCOMPLETE'
    play = play.rename(columns={"week": "Week", "game": "Game ID", "play": "Play ID", \
                                          "lat_dist": "Lateral pass distance", "lon_dist": "Longitudinal pass distance", \
                                          "ball_prox": "WR-ball proximity", "db_prox": "WR-DB proximity", \
                                          "sl_prox": "WR-sideline proximity", "bl_prox": "QB-blitzer proximity", \
                                          "qb_speed": "QB speed", "t_throw": "Time to throw", "result": "Result", "route": "Route"})
    return play.iloc[0]

# Completion Probability Network (CPNet)
class CPNet(nn.Module):
    def __init__(self, B, M, H1, H2, H3, C, p_dropout):
        super(CPNet, self).__init__()
        self.linear1 = nn.Linear(M, H1)
        self.bn1 = nn.BatchNorm1d(H1)
        self.dropout1 = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(H1, H2)
        self.bn2 = nn.BatchNorm1d(H2)
        self.dropout2 = nn.Dropout(p_dropout/2.0)
        self.linear3 = nn.Linear(H2, H3)
        self.bn3 = nn.BatchNorm1d(H3)
        self.dropout3 = nn.Dropout(p_dropout/4.0)
        self.head = nn.Linear(H3, C)
        # self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.activation(self.dropout1(self.bn1(self.linear1(x))))
        x = self.activation(self.dropout2(self.bn2(self.linear2(x))))
        x = self.activation(self.dropout3(self.bn3(self.linear3(x))))
        x = self.head(x)
        return x

# Completion Probability dataset
class CPDataset(Dataset):
    def __init__(self, cp_df):
        self.cp_df = cp_df

    def __len__(self):
        return len(self.cp_df)

    def __getitem__(self, idx):
        data = self.cp_df.iloc[idx,3:-1].values
        labels = self.cp_df.iloc[idx,-1]
        sample = {'data': data, 'labels': labels}

        return sample


class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for idx, sample in enumerate(valid_loader):
                input = sample["data"]
                label = sample["labels"]
                input = input.cuda().float()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels.long()).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels.long())
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels.long()).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels.long()).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    def __init__(self, n_bins=10):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    
# load and clean data
def load_and_clean_data(keep_routes=False):
    # load data
    train_df = pd.read_csv('../input/nfl-bdb-data/training_data.csv')

    # clean up data
    train_df = train_df.query('not (ball_prox > 10 and result == 1)') # sometimes receiver is not tracked
    train_df.bl_prox[train_df.bl_prox.isnull()] = np.max(train_df.bl_prox) # set nan values to max value

    if not keep_routes:
        train_df = train_df.drop(columns=["route"])
    
    return train_df

# set up neural network
def setup_network(train_df):
    B = 128
    M = train_df.shape[1]-4
    H1 = 100
    H2 = 50
    H3 = 10
    C = 2
    p_dropout = 0.4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpnet = CPNet(B, M, H1, H2, H3, C, p_dropout).to(device)
    return cpnet

# run AI model
def run_model(play_df):
    model_path = "../input/nfl-bdb-data/best_0194_0.483_calib.pt"
    train_df = load_and_clean_data()
    cpnet = setup_network(train_df)
    cpnet_calib = ModelWithTemperature(cpnet)
    cpnet_calib.load_state_dict(torch.load(model_path, map_location='cpu')["model_state_dict"])
    cpnet_calib.eval()
    x = torch.from_numpy(play_df.iloc[3:11].values.astype(float)).float().unsqueeze(0)
    logits = cpnet_calib(x)
    softmax = F.softmax(logits, dim=1)
    comp_prob = 100*softmax[0,1].item()
    incomp_prob = 100-comp_prob
    return comp_prob, incomp_prob

# print probabilities
def print_probabilities(comp_prob, incomp_prob):
    print("Pass completion probability: {:.1f}%".format(comp_prob))
    print("Pass incompletion probability: {:.1f}%".format(incomp_prob))