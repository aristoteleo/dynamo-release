# source code modified from:  https://github.com/tonysyu/mpltools/blob/master/mpltools/animation.py#L27
# This software is licensed under the Modified BSD License.
#
# Copyright (c) 2012, Tony S. Yu
# All rights reserved.

"""
Animation class.
This implementation is a interface for Matplotlib's FuncAnimation class, but
with different interface for:
* Easy reuse of animation code.
* Logical separation of setup parameter (passed to `__init__`) and animation
  parameters (passed to `animate`).
* Unlike Matplotlib's animation class, this Animation class clearly must be
  assigned to a variable (in order to call the `animate` method). The
  FuncAnimation object needs to be assigned to a variable so that it isn't
  garbage-collected, but this requirement is confusing, and easily forgotten,
  because the user never uses the animation object directly.
"""
# from future.builtins import object
# import warnings
# import matplotlib.animation as _animation
#
#
# __all__ = ["Animation"]
#
#
# class Animation(object):
#     """Base class to create animation objects.
#     To create an animation, simply subclass `Animation` and override the
#     `__init__` method to create a plot (`self.fig` needs to be assigned to the
#     figure object here), and override `update` with a generator that updates
#     the plot:
#     .. code-block:: python
#        class RandomPoints(Animation):
#            def __init__(self, width=10):
#                self.fig, self.ax = plt.subplots()
#                self.width = width
#                self.ax.axis([0, width, 0, width])
#                self.num_frames = 20
#            def update(self):
#                artists = []
#                self.ax.lines = [] # Clean up plot when repeating animation.
#                for i in np.arange(self.num_frames):
#                    x, y = np.random.uniform(0, self.width, size=2)
#                    artists.append(self.ax.plot(x, y, 'ro'))
#                    yield artists
#        pts = RandomPoints()
#        pts.animate()
#     Note: if you want to use blitting (see docstring for `Animation.animate`),
#     You must yield a sequence of artists in `update`.
#     This Animation class does not subclass any of Matplotlib's animation
#     classes because the `__init__` method takes arguments for creating the
#     plot, while `animate` method is what accepts arguments that alter the
#     animation.
#     """
#
#     def __init__(self):
#         """Initialize plot for animation.
#         Replace this method to initialize the plot. The only requirement is
#         that you must create a figure object assigned to `self.fig`.
#         """
#         raise NotImplementedError
#
#     def init_background(self):
#         """Initialize background artists.
#         Note: This method is passed to `FuncAnimation` as `init_func`.
#         """
#         pass
#
#     def update(self):
#         """Update frame.
#         Replace this method to with a generator that updates artists and calls
#         an empty `yield` when updates are complete.
#         """
#         raise NotImplementedError
#
#     def animate(self, **kwargs):
#         """Run animation.
#         Parameters
#         ----------
#         interval : float, defaults to 200
#             Time delay, in milliseconds, between frames.
#         repeat : {True | False}
#             If True, repeat animation when the sequence of frames is completed.
#         repeat_delay : None
#             Delay in milliseconds before repeating the animation.
#         blit : {False | True}
#             If True, use blitting to optimize drawing. Unsupported by some
#             backends.
#         init_background : function
#             If None, the results of drawing
#             from the first item in the frames sequence will be used. This can
#             also be added as a class method instead of passing to `animate`.
#         save_count : int
#             If saving a movie, `save_count` determines number of frames saved.
#             If not defined, use `num_frames` attribute (if defined); otherwise,
#             set to 100 frames.
#         """
#         reusable_generator = lambda: iter(self.update())
#         kwargs["init_background"] = self.init_background
#
#         self._warn_num_frames = False
#         if hasattr(self, "num_frames") and "save_count" not in kwargs:
#             kwargs["save_count"] = self.num_frames
#         if "save_count" not in kwargs:
#             kwargs["save_count"] = 100
#             self._warn_num_frames = True
#
#         self._ani = _GenAnimation(self.fig, reusable_generator, **kwargs)
#
#     def save(self, filename, **kwargs):
#         """Saves a movie file by drawing every frame.
#         Parameters
#         ----------
#         filename : str
#             The output filename.
#         writer : :class:`matplotlib.animation.MovieWriter` or str
#             Class for writing movie from animation. If string, must be 'ffmpeg'
#             or 'mencoder', which identifies the MovieWriter class used.
#             If None, use 'animation.writer' rcparam.
#         fps : float
#             The frames per second in the movie. If None, use the animation's
#             specified `interval` to set the frames per second.
#         dpi : int
#             Dots per inch for the movie frames.
#         codec :
#             Video codec to be used. Not all codecs are supported by a given
#             writer. If None, use 'animation.codec' rcparam.
#         bitrate : int
#             Kilobits per seconds in the movie compressed movie. A higher
#             value gives a higher quality movie, but at the cost of increased
#             file size. If None, use `animation.bitrate` rcparam.
#         extra_args : list
#             List of extra string arguments passed to the underlying movie
#             utility. If None, use 'animation.extra_args' rcParam.
#         metadata : dict
#             Metadata to include in the output file. Some keys that may be of
#             use include: title, artist, genre, subject, copyright, comment.
#         """
#
#         if not hasattr(self, "_ani"):
#             raise RuntimeError("Run `animate` method before calling `save_fig`!")
#             return
#         if self._warn_num_frames:
#             msg = "%s `num_frames` attribute. Animation may be truncated."
#             warnings.warn(msg % self.__class__.__name__)
#         self._ani.save(filename, **kwargs)
#
#
# # matplotlib.animation.FuncAnimation(fig, func, frames=None, init_func=None, fargs=None, save_count=None, *, cache_frame_data=True, **kwargs)[source]
# class _GenAnimation(_animation.FuncAnimation):
#     def __init__(
#         self,
#         fig,
#         frames,
#         init_background=None,
#         save_count=None,
#         cache_frame_data=True,
#         **kwargs
#     ):
#         self._iter_gen = frames
#         self._cache_frame_data = cache_frame_data
#
#         self._init_func = init_background
#         self.save_count = save_count if save_count is not None else 100
#
#         # Dummy args and function for compatibility with FuncAnimation
#         self._args = ()
#         self._func = lambda args: args
#
#         self._save_seq = []
#         _animation.TimedAnimation.__init__(self, fig, **kwargs)
#         # Clear saved seq since TimedAnimation.__init__ adds a single frame.
#         self._save_seq = []
#

def remove_particles(pts, xlim, ylim):
    if len(pts) == 0:
        return []
    outside_xlim = (pts[:, 0] < xlim[0]) | (pts[:, 0] > xlim[1])
    outside_ylim = (pts[:, 1] < ylim[0]) | (pts[:, 1] > ylim[1])
    keep = ~(outside_xlim | outside_ylim)
    return pts[keep]
