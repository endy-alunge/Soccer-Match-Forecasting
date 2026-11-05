"""Serializable Dixon–Coles and XGBoost with temperature scaling."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax
from scipy.stats import poisson
from xgboost import XGBClassifier, DMatrix
from .data import MARKETS

def log_loss(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())

def brier(p, y):
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(axis=1).mean())

def scale(p, temperature):
    return softmax(np.log(np.clip(p, 1e-15, 1)) / temperature, axis=1)

def calibrate(p, y):
    r = minimize_scalar(lambda t: log_loss(scale(p, np.exp(t)), y),
                        bounds=(np.log(.25), np.log(4)), method="bounded")
    if not r.success:
        raise ValueError("Temperature optimization failed.")
    return float(np.exp(r.x))

def fit_xgb(x, y):
    if not np.array_equal(np.unique(y), np.arange(len(np.unique(y)))):
        raise ValueError("Every market class must occur in training.")
    model = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=.05,
                          random_state=42, n_jobs=1, eval_metric="mlogloss" if len(set(y)) == 3 else "logloss")
    return model.fit(x, y)

def contributions(model, x):
    """Native exact TreeSHAP on raw margins, including a bias term."""
    dm = DMatrix(x, feature_names=list(x.columns))
    values = model.get_booster().predict(dm, pred_contribs=True, strict_shape=True)
    margins = model.get_booster().predict(dm, output_margin=True, strict_shape=True)
    if not np.allclose(values.sum(axis=-1), margins, atol=2e-5, rtol=1e-5):
        raise ValueError("TreeSHAP additivity check failed.")
    return values, margins

@dataclass
class DixonColes:
    teams: list
    parameters: list

    def unpack(self):
        n = len(self.teams)
        p = np.array(self.parameters)
        att = np.r_[p[:n] - p[:n].mean(), 0.]
        defence = np.r_[p[n:2*n] - p[n:2*n].mean(), 0.]
        intercept, advantage, dependence = p[-3:]
        base = np.exp(intercept + att.max() + defence.max())
        home = base * np.exp(advantage)
        lower, upper = -.99 / max(home, base), .99 * min(1., 1 / (home * base))
        rho = dependence * (upper if dependence >= 0 else -lower)
        return att, defence, intercept, advantage, rho

    def rates(self, frame):
        att, defence, intercept, advantage, rho = self.unpack()
        lookup = {t: i for i, t in enumerate(self.teams)}
        h = frame.home_team.map(lambda t: lookup.get(t, len(self.teams))).to_numpy()
        a = frame.away_team.map(lambda t: lookup.get(t, len(self.teams))).to_numpy()
        return np.exp(intercept + advantage + att[h] + defence[a]), np.exp(intercept + att[a] + defence[h]), rho

    @staticmethod
    def correction(h, a, hr, ar, rho):
        return np.select([(h==0)&(a==0), (h==0)&(a==1), (h==1)&(a==0), (h==1)&(a==1)],
                         [1-hr*ar*rho, 1+hr*rho, 1+ar*rho, 1-rho], default=1.)

    @classmethod
    def fit(cls, frame):
        teams = sorted(set(frame.home_team) | set(frame.away_team))
        n = len(teams)
        h, a = frame.home_goals.to_numpy(), frame.away_goals.to_numpy()
        def objective(p):
            m = cls(teams, p.tolist())
            hr, ar, rho = m.rates(frame)
            return -(poisson.logpmf(h, hr) + poisson.logpmf(a, ar) +
                     np.log(m.correction(h, a, hr, ar, rho))).sum()
        bounds = ([(-2.,2.)]*(n-1)+[(0.,0.)])*2 + [(-2.,2.),(-2.,2.),(-1.,1.)]
        r = minimize(objective, np.zeros(2*n+3), bounds=bounds, method="L-BFGS-B",
                     options={"maxiter": 1500})
        if not r.success:
            raise ValueError(f"Dixon–Coles fit failed: {r.message}")
        return cls(teams, r.x.tolist())

    def predict(self, frame):
        hr, ar, rho = self.rates(frame)
        g = max(10, int(poisson.ppf(1-1e-10, max(hr.max(), ar.max()))))
        h, a = np.ogrid[:g+1, :g+1]
        grid = poisson.pmf(h, hr[:,None,None]) * poisson.pmf(a, ar[:,None,None])
        grid *= self.correction(h,a,hr[:,None,None],ar[:,None,None],rho)
        mass = grid.sum(axis=(1,2), keepdims=True)
        if (grid < 0).any() or not np.allclose(mass, 1, atol=1e-8, rtol=0):
            raise ValueError("Invalid scoreline probability mass.")
        grid /= mass
        masks = {"1X2":[h>a,h==a,h<a], "O/U 1.5":[h+a<2,h+a>=2],
                 "BTTS":[(h==0)|(a==0),(h>0)&(a>0)]}
        p = {m:np.column_stack([grid[:,mask].sum(axis=1) for mask in masks[m]]) for m in MARKETS}
        return p, hr, ar, grid
