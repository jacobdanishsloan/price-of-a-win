import csv
import re
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statistics as statistics
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

#read in data, add winning percentage, convert salary to millions
teams = pd.read_csv("teams.txt")
teams["win_pct"]= teams.wins/(teams.losses + teams.wins)
teams.loc[teams["playoffs"]==1,"playoffs"] = "yes"
teams.loc[teams["playoffs"]==0,"playoffs"] = "no"
teams.salary  = teams.salary/1000000 


#save figures winning percentages vs. payroll for each league
sns.set_theme(style="darkgrid")
p = sns.lmplot(x="salary", y="win_pct", hue="playoffs", palette="Dark2_r", line_kws={'color': 'tomato'}, 
               scatter_kws={"alpha":0.9, "s":45},
                     data=teams[teams["league"]=="MLB"], fit_reg=False)
p.fig.suptitle('Winning Pct. vs. Payroll \n (MLB 2019)', fontsize=18)
p.fig.subplots_adjust(top=0.85);
q =sns.regplot(x="salary", y="win_pct", line_kws={"color": "tomato"},
               data=teams[teams["league"]=="MLB"], scatter=False, ax=p.axes[0, 0])
plt.xlabel("Payroll ($ Millions)", fontsize=16)
plt.ylabel("Winning Pct.", fontsize=16)
q.set(xlim=(70, 235))
plt.savefig('mlb_win_payroll.png', dpi = 600, bbox_inches = "tight")


sns.set_theme(style="darkgrid")
p = sns.lmplot(x="salary", y="win_pct", hue="playoffs", palette="Dark2_r", line_kws={'color': 'tomato'}, 
               scatter_kws={"alpha":0.9, "s":45},
                     data=teams[teams["league"]=="NBA"], fit_reg=False)
p.fig.suptitle('Winning Pct. vs. Payroll \n (NBA 2018-2019)', fontsize=22)
p.fig.subplots_adjust(top=0.85);
q = sns.regplot(x="salary", y="win_pct", line_kws={"color": "tomato"}, 
                data=teams[teams["league"]=="NBA"], scatter=False, ax=p.axes[0, 0])
q.set(xlim=(78, 155))
plt.xlabel("Payroll ($ Millions)", fontsize=21)
plt.ylabel("Winning Pct.", fontsize=16)
plt.savefig('nba_win_payroll.png', dpi = 600,  bbox_inches = "tight")


sns.set_theme(style="darkgrid")
p = sns.lmplot(x="salary", y="win_pct", hue="playoffs", palette="Dark2_r", line_kws={'color': 'tomato'}, 
               scatter_kws={"alpha":0.9, "s":45},
                     data=teams[teams["league"]=="NFL"], fit_reg=False)
p.fig.suptitle('Winning Pct. vs. Payroll \n (NFL 2019)', fontsize=18)
p.fig.subplots_adjust(top=0.85);
q = sns.regplot(x="salary", y="win_pct", data=teams[teams["league"]=="NFL"], line_kws={"color": "tomato"},
                scatter=False, ax=p.axes[0, 0])
q.set(xlim=(170, 221))
plt.xlabel("Payroll ($ Millions)", fontsize=16)
plt.ylabel("Winning Pct.", fontsize=16)
plt.savefig('nfl_win_payroll.png', dpi = 600,  bbox_inches = "tight")


#calculate mean and standard dev. for wins and salary for each league 
#for purpose of converting wins and salaries to standard normal distributions
nba_wins_mean = statistics.mean(teams.loc[teams.league=="NBA","wins"])
nfl_wins_mean = statistics.mean(teams.loc[teams.league=="NFL","wins"])
mlb_wins_mean = statistics.mean(teams.loc[teams.league=="MLB","wins"])
nba_wins_sd = statistics.stdev(teams.loc[teams.league=="NBA","wins"])
nfl_wins_sd = statistics.stdev(teams.loc[teams.league=="NFL","wins"])
mlb_wins_sd = statistics.stdev(teams.loc[teams.league=="MLB","wins"])

nba_salary_mean = statistics.mean(teams.loc[teams.league=="NBA","salary"])
nfl_salary_mean = statistics.mean(teams.loc[teams.league=="NFL","salary"])
mlb_salary_mean = statistics.mean(teams.loc[teams.league=="MLB","salary"])
nba_salary_sd = statistics.stdev(teams.loc[teams.league=="NBA","salary"])
nfl_salary_sd = statistics.stdev(teams.loc[teams.league=="NFL","salary"])
mlb_salary_sd = statistics.stdev(teams.loc[teams.league=="MLB","salary"])

#convert these values to standard normal distribution
teams.loc[teams.league=="MLB","wins_adj"] = (teams.loc[teams.league=="MLB","wins"] - mlb_wins_mean)/mlb_wins_sd
teams.loc[teams.league=="NBA","wins_adj"] = (teams.loc[teams.league=="NBA","wins"] - nba_wins_mean)/nba_wins_sd
teams.loc[teams.league=="NFL","wins_adj"] = (teams.loc[teams.league=="NFL","wins"] - nfl_wins_mean)/nfl_wins_sd

teams.loc[teams.league=="MLB","salary_adj"] = (teams.loc[teams.league=="MLB","salary"] - mlb_salary_mean)/mlb_salary_sd
teams.loc[teams.league=="NBA","salary_adj"] = (teams.loc[teams.league=="NBA","salary"] - nba_salary_mean)/nba_salary_sd
teams.loc[teams.league=="NFL","salary_adj"] = (teams.loc[teams.league=="NFL","salary"] - nfl_salary_mean)/nfl_salary_sd


#create QQ plots for residuals for winning predicted by salary via simple linear regression
Y = teams.loc[teams['league']=="MLB","win_pct"]
X = teams.loc[teams['league']=="MLB","salary"]
mlb_model = sm.OLS(Y, X, missing='drop')
mlb_model_result = mlb_model.fit()
sm.qqplot(mlb_model_result.resid, line='s');
plt.ylabel("Sample Quantiles", fontsize=16)
plt.xlabel("Theoretical Quantiles", fontsize=16)
plt.savefig('qq_mlb.png', dpi = 600,  bbox_inches = "tight")

Y = teams.loc[teams['league']=="NBA","win_pct"]
X = teams.loc[teams['league']=="NBA","salary"]
nba_model = sm.OLS(Y, X, missing='drop')
nba_model_result = nba_model.fit()
sm.qqplot(nba_model_result.resid, line='s');
plt.ylabel("Sample Quantiles", fontsize=16)
plt.xlabel("Theoretical Quantiles", fontsize=16)
plt.savefig('qq_nba.png', dpi = 600,  bbox_inches = "tight")


#plot residuals vs. fitted values
sns.set_theme(style="darkgrid")
p = sns.relplot(x=mlb_model_result.fittedvalues, y=mlb_model_result.resid)
p.fig.subplots_adjust(top=0.85);
plt.title('Residuals vs. Fitted Values (MLB)', fontsize=18)
plt.xlabel("Fitted Value", fontsize=16)
plt.ylabel("Residual", fontsize=16)
plt.axhline(0)
p.set(ylim=(-0.4, 0.4))
plt.savefig('mlb_resid.png', dpi = 600,  bbox_inches = "tight")

sns.set_theme(style="darkgrid")
p = sns.relplot(x=nba_model_result.fittedvalues, y=nba_model_result.resid)
p.fig.subplots_adjust(top=0.85);
plt.title('Residuals vs. Fitted Values (NBA)', fontsize=18)
plt.xlabel("Fitted Value", fontsize=16)
plt.ylabel("Residual", fontsize=16)
plt.axhline(0)
p.set(ylim=(-0.4, 0.4))
plt.savefig('nba_resid.png', dpi = 600,  bbox_inches = "tight")


#logistic regression to predict playoffs from salary
logr = LogisticRegression()
mlb_logr = logr.fit(np.array(teams.loc[teams.league=="MLB","salary"]).reshape((-1, 1)),
       np.array(teams.loc[teams.league=="MLB","playoffs"]))

nba_logr = logr.fit(np.array(teams.loc[teams.league=="NBA","salary"]).reshape((-1, 1)),
       np.array(teams.loc[teams.league=="NBA","playoffs"]))

teams.loc[teams["playoffs"]=="yes","playoffs"] = 1
teams.loc[teams["playoffs"]=="no","playoffs"] = 0

#predict probability of playoffs within MLB
xx = teams.loc[teams["league"]=="MLB","salary"]
yy = teams.loc[teams["league"]=="MLB","playoffs"]
sns.regplot(x=xx, y=yy.astype(int), logistic=True, ci=None, line_kws={'color': 'tomato'}, 
               scatter_kws={"alpha":0.7, "color":"seagreen", "s":45})
plt.ylabel("Est. Playoffs Probability", fontsize=16)
plt.xlabel("Payroll ($ Millions)", fontsize=16)
plt.suptitle('Probability of Playoffs (MLB)', fontsize=18)
plt.savefig('mlb_log.png', dpi = 600,  bbox_inches="tight")

#predict probability of playoffs within NBA
xx = teams.loc[teams["league"]=="NBA","salary"]
yy = teams.loc[teams["league"]=="NBA","playoffs"]
sns.set_theme(style="darkgrid")
sns.regplot(x=xx, y=yy.astype(int), logistic=True, ci=None, line_kws={'color': 'tomato'}, 
               scatter_kws={"alpha":0.7, "color":"seagreen", "s":45} )
plt.ylabel("Est. Playoffs Probability", fontsize=16)
plt.xlabel("Payroll ($ Millions)", fontsize=16)
plt.suptitle('Probability of Playoffs (NBA)', fontsize=18)
plt.savefig('nba_log.png', dpi = 600, bbox_inches="tight")


#standardized wins vs. standardized salary, across all leagues 
teams.loc[teams["playoffs"]==1,"playoffs"] = "yes"
teams.loc[teams["playoffs"]==0,"playoffs"] = "no"
sns.set_theme(style="darkgrid")
p = sns.lmplot(x="salary_adj", y="wins_adj", hue="playoffs", palette="Dark2_r", line_kws={'color': 'tomato'}, 
               scatter_kws={"alpha":0.65, "s":45},
                     data=teams, fit_reg=False)
p.fig.suptitle('Standardized Wins vs. Standardized Payroll \n (MLB/NBA/NFL)', fontsize=18)
p.fig.subplots_adjust(top=0.85);
q =sns.regplot(x="salary_adj", y="wins_adj", line_kws={"color": "tomato"},
               data=teams, scatter=False, ax=p.axes[0, 0])
plt.xlabel("Payroll (Std. Devs. above Mean)", fontsize=16)
plt.ylabel("Wins (Std. Devs. above Mean)", fontsize=16)
q.set(xlim=(-3, 3))
plt.savefig('std_win_payroll.png', dpi = 600, bbox_inches="tight")


#histogram of wins (standardized)
sns.histplot(data = teams, x = "wins_adj", bins=15, kde=True, color="seagreen")
plt.xlabel("Distribution of Win Totals (Standardized) \n (MLB/NBA/NFL)", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.savefig('win_hist.png', dpi = 600, bbox_inches="tight")

#histogram of salary (standardized)
sns.histplot(data = teams, x = "salary_adj", bins=15, kde=True, color="seagreen")
plt.xlabel("Distribution of Payroll Totals (Standardized) \n (MLB/NBA/NFL)", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.savefig('salary_hist.png', dpi = 600, bbox_inches="tight")


#for MLB, winning versus natural log of payroll 
#this relationship is more linear than winning versus payroll
teams.loc[teams["playoffs"]=="y","playoffs"] = "yes"
teams.loc[teams["playoffs"]=="n","playoffs"] = "no"
import math
for i in range(0,len(teams)):
    teams.loc[i,"log_salary"] = math.log(teams.loc[i,"salary"], 10)
teams["win_pct_sq"] = teams["win_pct"]**0.5
    
sns.set_theme(style="darkgrid")
p = sns.lmplot(x="log_salary", y="win_pct", hue="playoffs", palette="Dark2_r", line_kws={'color': 'tomato'}, 
               scatter_kws={"alpha":0.9, "s":45},
                     data=teams[teams["league"]=="MLB"], fit_reg=False)
p.fig.suptitle('Winning Pct. vs. Log of Payroll \n (MLB 2019)', fontsize=18)
p.fig.subplots_adjust(top=0.85);
q =sns.regplot(x="log_salary", y="win_pct", line_kws={"color": "tomato"},
               data=teams[teams["league"]=="MLB"], scatter=False, ax=p.axes[0, 0])
plt.xlabel("Base-10 Log of Payroll ($ Millions)", fontsize=16)
plt.ylabel("Winning Pct.", fontsize=16)

plt.savefig('mlb_log_lm.png', dpi = 600, bbox_inches = "tight")





