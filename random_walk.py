import numpy.random as rd
import operator as op
import plotly.offline as ply
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time


def ne(arg):
    return op.ne(arg, 0)


def gt(arg):
    return op.gt(arg, 0)


def ge(arg):
    return op.ge(arg, 0)


def ok(arg):
    return True


class BaseError(Exception):
    pass


class WrongArgType(BaseError):
    pass


class WrongValue(BaseError):
    pass


class WrongArgsNumber(BaseError):
    pass


class Random_walk:
    DISTRIBUTIONS = {'normal': (rd.normal, (ok, ge)),
                     'gauss': (rd.normal, (ok, ge)),
                     'exp': (rd.exponential, ne),
                     'exponential': (rd.exponential, ne),
                     'gamma': (rd.gamma, (gt, gt))
                     }

    def __init__(self, *args, steps=10000, dim=1, distribution='normal'):
        if distribution in Random_walk.DISTRIBUTIONS:
            self.distribution = distribution
            self.distribution_func, self.conditions = Random_walk.DISTRIBUTIONS[distribution]
        if Random_walk.validate_type(steps, dim):
            self.steps = steps
            self.dim = dim
        if Random_walk.validate_values(*args, conditions=self.conditions) and \
                Random_walk.validate_type(*args, types=(int, float)):
            self.args = args
        self.random = np.array(self.gen_random_num())
        self.path = np.array(self.gen_path())
        self.mean = self.path.mean()
        self.std = self.path.std()
        self.fig = go.Figure()
        self.ax_distance = 2
        self.axes_range = [[item.min() * self.ax_distance, item.max() * self.ax_distance] for item in self.random]

    @staticmethod
    def validate_type(*args, types=(int,)):
        for arg in args:
            if not isinstance(arg, types):
                raise WrongArgType(arg)
            return True

    @staticmethod
    def validate_values(*args, conditions):
        if len(args) > len(conditions):
            raise WrongArgsNumber('Too many args')
        elif len(args) < len(conditions):
            raise WrongArgsNumber('Not enough args')
        for index, cond in enumerate(conditions):
            if not cond(args[index]):
                raise WrongValue(args[index])
        return True

    def gen_random_num(self):
        res = []
        for index in range(self.dim):
            res.append(self.distribution_func(*self.args, self.steps))
        return res

    def gen_random_num_rd(self):
        res = []
        for index in range(self.dim):
            res.append([self.distribution_func(*self.args) for i in range(self.steps)])
        return res

    def gen_path(self):
        res = [[0] for i in range(self.dim)]
        for i in range(self.dim):
            for j in range(self.steps):
                res[i].append(res[i][-1] + self.random[i][j])
        return res

    def plot(self):
        if self.dim == 1:
            self.plot1d()
        elif self.dim == 2:
            self.plot2d_rand()
            time.sleep(1)
            self.plot2d_path()
        elif self.dim == 3:
            self.plot3d_rand()
            self.plot2d_path()

    def plot1d(self):
        nbins = int(self.steps ** 0.5)
        subplots = make_subplots(rows=2, cols=1)
        subplots.add_trace(go.Histogram(x=self.random[0], nbinsx=nbins, xaxis='x1', yaxis='y1', name='Distribution'),
                           row=1, col=1
                           )
        subplots.add_trace(go.Scatter(y=self.path[0], mode='lines+markers',
                                      marker=dict(size=1, color='olive'), name='1D random walk', xaxis='x2', yaxis='y2'
                                      )
                           )
        subplots.update_layout(template="plotly_white",
                               autosize=False,
                               width=1800,
                               height=2000,
                               xaxis1=dict(zeroline=False, showgrid=True, range=self.axes_range[0]),
                               xaxis2=dict(zeroline=False, showgrid=True,),
                               yaxis=dict(zeroline=False, showgrid=True)
                               )
        ply.plot(go.Figure(subplots))

    def plot2d_rand(self):
        nbins = int(self.steps ** 0.5)
        fig = go.Figure()
        fig.add_trace(go.Histogram2dContour(x=self.random[0], y=self.random[1],
                                            xaxis='x', yaxis='y',
                                            line=dict(width=1),
                                            contours=dict(coloring='none',
                                                          showlabels=True,
                                                          labelfont=dict(size=18)
                                                          ),
                                            visible='legendonly',
                                            name='Contour lines'
                                            )
                      )
        fig.add_trace(go.Scatter(x=self.random[0], y=self.random[1],
                                 xaxis='x', yaxis='y',
                                 mode='markers', marker=dict(size=2),
                                 name='X-Y distribution'
                                 )
                      )
        fig.add_trace(go.Histogram(x=self.random[0], yaxis='y2', name='X distribution', nbinsx=nbins,
                                   hovertemplate='value:%{x:.2f}' +
                                                 '<br>probability:%{y:.2f}<br>',
                                   histnorm="probability"
                                   )
                      )
        fig.add_trace(go.Histogram(y=self.random[1], xaxis='x2', name='Y distribution', nbinsy=nbins,
                                   hovertemplate='value:%{y:.2f}' +
                                                 '<br>probability:%{x:.2f}<br>',
                                   histnorm="probability"
                                   )
                      )
        fig.update_layout(template="plotly_white",
                          autosize=True,
                          xaxis=dict(zeroline=False,
                                     domain=[0, 0.85],
                                     showgrid=True,
                                     range=self.axes_range[0]
                                     ),
                          yaxis=dict(zeroline=False,
                                     domain=[0, 0.85],
                                     showgrid=True,
                                     range=self.axes_range[1]
                                     ),
                          xaxis2=dict(zeroline=False,
                                      domain=[0.9, 1],
                                      showgrid=False,
                                      ),
                          yaxis2=dict(zeroline=False,
                                      domain=[0.9, 1],
                                      showgrid=False
                                      ),
                          legend=dict(font=dict(size=18),
                                      bgcolor="White",
                                      itemsizing='constant'
                                      )
                          )
        ply.plot(fig)

    def plot2d_path(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.path[0], y=self.path[1],
                                 mode='lines+markers',
                                 marker=dict(size=1, color='olive'),
                                 name='2D random walk',
                                 )
                      )
        fig.update_layout(template="plotly_white",
                          autosize=True,
                          xaxis=dict(zeroline=False,
                                     showgrid=True,
                                     ),
                          yaxis=dict(zeroline=False,
                                     showgrid=True,
                                     )
                          )
        ply.plot(fig)

    def plot3d_rand(self):
        x, y, z = self.random
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers',
                                   marker=dict(size=1, color='olive'),
                                   name='X-Y-Z distribution',
                                   )
                      )

        fig.add_trace(go.Scatter3d(x=x, y=y, z=np.ones(len(z)) * z.min() * 2,
                                   mode='markers',
                                   marker=dict(size=1, color='red'),
                                   name='X-Y distribution',
                                   hovertemplate='x:%{x:.2f}' +
                                                 '<br>y:%{y:.2f}<br>'
                                   )
                      )

        fig.add_trace(go.Scatter3d(x=np.ones(len(x)) * x.min() * 2, y=y, z=z,
                                   mode='markers',
                                   marker=dict(size=1, color='green'),
                                   name='Y-Z distribution',
                                   hovertemplate='y:%{y:.2f}' +
                                                 '<br>z:%{z:.2f}<br>'
                                   )
                      )

        fig.add_trace(go.Scatter3d(x=x, y=np.ones(len(y)) * min(y) * 2, z=z,
                                   mode='markers',
                                   marker=dict(size=1, color='blue'),
                                   name='X-Z distribution',
                                   hovertemplate='x:%{x:.2f}' +
                                                 '<br>z:%{z:.2f}<br>'
                                   )
                      )

        fig.update_layout(template="plotly_white",
                          legend=dict(font=dict(size=18),
                                      itemsizing='constant'
                                      ),
                          scene=dict(xaxis=dict(range=self.axes_range[0]),
                                     yaxis=dict(range=self.axes_range[1]),
                                     zaxis=dict(range=self.axes_range[2])
                                     )
                          )
        ply.plot(fig)

    def plot3d_path(self):
        x, y, z = self.random
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                        mode='lines+markers',
                                        marker=dict(size=1, color='olive'),
                                        name='3D random walk',
                                        )
                           )

        self.fig.update_layout(template="plotly_white",
                               scene=dict(xaxis=dict(range=self.axes_range[0]),
                                          yaxis=dict(range=self.axes_range[1]),
                                          zaxis=dict(range=self.axes_range[2])),
                               legend=dict(font=dict(size=18),
                                           paper_bgcolor="White",
                                           template="plotly_white",
                                           itemsizing='constant'
                                           )
                               )
        ply.plot(self.fig)

