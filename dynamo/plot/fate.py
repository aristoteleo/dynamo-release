import numpy as np
from scipy.integrate import odeint
from ..tools.utils import vector_field_function
from .fate_utilities import Animation


class StreamFuncAnim(Animation):  #
    def __init__(self, VecFld, dt=0.05, xlim=(-1, 1), ylim=None):
        import matplotlib.pyplot as plt

        self.dt = dt
        # Initialize velocity field and displace *functions*
        self.f = lambda x, t=None: vector_field_function(x, VecFld=VecFld)
        self.displace = lambda x, dt: odeint(self.f, x.flatten(), [0, dt])
        # Save bounds of plot
        self.xlim = xlim
        self.ylim = ylim if ylim is not None else xlim
        # Animation objects must create `fig` and `ax` attributes.
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")

    def init_background(self):
        """Draw background with streamlines of flow.

        Note: artists drawn here aren't removed or updated between frames.
        """
        u = np.linspace(self.xlim[0], self.xlim[1], 100)
        v = np.linspace(self.ylim[0], self.ylim[1], 100)
        uu, vv = np.meshgrid(u, v)

        # Compute derivatives
        u_vel = np.empty_like(uu)
        v_vel = np.empty_like(vv)
        for i in range(uu.shape[0]):
            for j in range(uu.shape[1]):
                u_vel[i, j], v_vel[i, j] = self.f(np.array([uu[i, j], vv[i, j]]), None)

        streamline = self.ax.streamplot(uu, vv, u_vel, v_vel, color="0.7")

        return streamline

    def update(self):
        """Update locations of "particles" in flow on each frame frame."""
        pts = []
        while True:
            pts = list(pts)
            pts.append((random_xy(self.xlim), random_xy(self.ylim)))
            pts = [
                self.displace(np.array(cur_pts).reshape(-1, 1), self.dt)
                for cur_pts in pts
            ]
            pts = np.asarray(pts)
            pts = remove_particles(pts, self.xlim, self.ylim)
            self.ax.lines = []

            x, y = np.asarray(pts).transpose()
            (lines,) = self.ax.plot(x, y, "ro")
            yield lines,  # return line so that blit works properly


def random_xy(lim):
    return list(np.random.uniform(lim[0], lim[1], 1))


def remove_particles(pts, xlim, ylim):
    if len(pts) == 0:
        return []
    outside_xlim = (pts[:, 0] < xlim[0]) | (pts[:, 0] > xlim[1])
    outside_ylim = (pts[:, 1] < ylim[0]) | (pts[:, 1] > ylim[1])
    keep = ~(outside_xlim | outside_ylim)
    return pts[keep]


class Flow(StreamFuncAnim):
    def init_background(self):
        StreamFuncAnim.init_background(self)


"""
# use example 
VF = lambda x, t=None: vector_field_function(x=x, t=t, VecFld=VecFld)
toggle_flow = Flow(VecFld, xlim=(0, 6), ylim=(0, 6), dt=1e-500)
toggle_flow.animate(blit=False)
plt.show()
"""
