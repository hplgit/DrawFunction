# Extensions:
# Allow curves returned from compute_function to be dicts;
# {'label1': (x1,y1), 'label2': (x2,y2)} instead of just (x,y)
# Allow the returned curve to be a object for function animation
# (of several curves).
# http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

import numpy as np
import sys

def smooth_function_from_coordinates(
    x, y,  # coordinates of some curve
    sample_fraction=1.0,  # fraction of x,y points used for resampling
    spline_smoothing=4,   # degree of spline (1 gives piecewise linear)
    ):

    """
    Given a set of coordinates in the `x` and `y` arrays, create a
    smooth function from these coordinates by 1) resampling n
    uniformly distributed points by linear interpolation of the `x`
    and `y` coordinates, where n is given by `sample_fraction` times
    the length of `x`; and 2) interpolating the resampled points by
    a smooth spline, where `spline_smoothing` is an integer holding
    the degree of the piecewise polynomial pieces of the spline
    (0 and 1 gives a piecewise linear function, 2 and higher gives
    splines of that order). Return the smooth function as a
    Python function of x, together with the (uniformly distributed)
    resampled points on which the smooth function is based.
    """
    # Construct linear interpolator of data points
    from Scientific.Functions.Interpolation \
         import InterpolatingFunction
    linear = InterpolatingFunction([x], y)
    # Resample
    xp = np.linspace(x[0], x[-1],
                     sample_fraction*len(x))
    yp = np.array([linear(xi) for xi in xp])
    # Spline smoothing or linear interpolation, based on (xp,yp)
    if spline_smoothing >= 2:
        from scipy.interpolate import UnivariateSpline as Spline
        function = Spline(xp, yp, s=0, k=spline_smoothing)
    else:
        function = InterpolatingFunction([xp], yp)
    return function, xp, yp


def extrapolate_endpoints(_x, _y, xmin, xmax):
    """
    Given a drawing as a set of coordinates `_x` and `_y`,
    supposed to be in the interval from `xmin` to `max`,
    the drawn coordinates will never match the end points.
    The idea here is to add end points to `_x` and `_y`
    such that the x coordinate starts at `xmin` and ends
    at `xmax`. The y coordinate of the two end points is
    computed by linear extrapolation."""
    # x and y are the arrays to be returned, including end points
    x = np.zeros(len(_x)+2)
    y = np.zeros(len(_y)+2)
    # Use the drawn _x and _y coordinates as internal points
    x[1:-1] = _x
    y[1:-1] = _y

    # xmin boundary
    x[0] = xmin
    dx = _x[1] - _x[0]
    if dx > 1E-10:
        # Use linear extrapolation formula if dx is not too small
        y[0] = _y[0] + (_y[1] - _y[0])/dx*(xmin - _x[0])
    else:
        y[0] = y[1]

    # xmax boundary
    x[-1] = xmax
    dx = _x[-1] - _x[-2]
    if dx > 1E-10:
        # Use linear extrapolation formula if dx is not too small
        y[-1] = _y[-2] + (_y[-1] - _y[-2])/dx*(xmax - _x[-2])
    else:
        y[-1] = y[-2]

    return x, y


class DrawFunction:
    def __init__(self,
                 xmin=0, xmax=1, ymin=0, ymax=1,
                 sample_fraction=1.0,  # fraction used for spline
                 resolution=100,       # no of samples from spline
                 xlabel=None,
                 ylabel=None,
                 title='help string',
                 sliders=True,
                 spline_smoothing=True,
                 response=None,
                 animated_response=False,
                 fig=None):
        self.xmin, self.xmax, self.ymin, self.ymax = \
                   xmin, xmax, ymin, ymax
        self.resolution = resolution
        self.sample_fraction = sample_fraction
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.spline_smoothing = spline_smoothing
        self.response = response
        self.animated_response = animated_response
        self.sliders = sliders

        self._x = self._y = []
        self.x = self.y = None
        self.spline = None
        self.key = None

        if response is not None:
            if not isinstance(response, dict) or \
               'curves' not in response or \
               not isinstance(response['curves'], (list,tuple)):
                raise ValueError("""response must be dict like
    response = dict(
        curves=[
            dict(ymin=U_0, ymax=U_L, xlabel='x', ylabel='u'),
            dict(xlabel='x', ylabel='flux'),
            ],
        compute_function=mysimulator)
""")
            self.num_curves = len(response['curves']) + 1
        else:
            self.num_curves = 1

        self.fig = fig

    # Maybe menu and the others should be interactive plotting classes?

    def menu(self, key):
        """Menu mapped to key: c is clear, r is compute response."""
        if key == 'r':
            # Plot responses
            if self.animated_response:
                self.update_response_animation()
            else:
                self.update_response()
        elif key == 'c':
            # Clear responses and be ready for new drawing
            self.clear_drawing()
            self.clear_response()

    # If compute_function is an iterator we can iterate here, otherwise
    # do one call, pass self.response_axes and plt to compute_function
    # and rely on the animation loop there

    def update_response(self):
        raise NotImplementedError

    def update_response_animation(self):
        raise NoteImplementedError

    def clear_drawing(self):
        raise NoteImplementedError

    def clear_responses(self):
        raise NoteImplementedError

    def begin_drawing(self):
        """Initialize drawing."""
        self._x = []
        self._y = []
        self.spline = None
        self.x = self.y = None

    def end_drawing(self):
        # Add end points so that xmin and xmax are included
        self.x, self.y = extrapolate_endpoints(
            self._x, self._y, self.xmin, self.xmax)

        self._x = self._y = []  # clear recorded points

        # Check that self.y = f(self.x) is a function
        # (is guaranteed in on_move)
        if (np.sort(self.x) - self.x).max() != 0:
            print 'Not a valid function - draw again'


    def add_pt(self, x_pt, y_pt):
        """Add a (x,y) point."""
        if x_pt is None or y_pt is None:
            return

        # 1st point?
        if len(self._x) == 0 and len(self._y) == 0:
            self._x.append(x_pt)
            self._y.append(y_pt)
        else:
            # Do not allow present x smaller than last recorded x, but
            # with a small eps larger in x coordinate
            if x_pt <= self._x[-1]:
                eps = 1E-4
                self._x.append(self._x[-1] + eps)
            else:
                self._x.append(x_pt)
            self._y.append(y_pt)

    def get_curve(self, resolution=None):
        if resolution is None:
            resolution = self.resolution
        if self.spline:
            x = np.linspace(self.x[0], self.x[-1], resolution+1)
            y = self.spline(x)
            return x, y
        else:
            raise TypeError('Wrong usage - smoothed drawing is not computed')


import matplotlib.pyplot as plt
import matplotlib.widgets
import matplotlib.axes
from StringIO import StringIO
import base64

class DrawFunctionMpl(DrawFunction):
    def __init__(self, *args, **kwargs):
        DrawFunction.__init__(self, *args, **kwargs)

        # Interactive plot on the screen?
        self.interactive = kwargs.get('interactive', True)
        self.png_plot = None  # stores a PNG version of the plot as a string
        self._num_clicks = 0

        if self.fig is None:
            self.fig = plt.figure()

        # Make subplots with self.num_curves (= one drawn curve
        # plus response functions) below each other.
        # self.ax: axes of the drawn curve
        # self.response_axes: axes of the responses
        self.ax = self.fig.add_subplot(self.num_curves, 1, 1)
        self.response_axes = []
        for i in range(2, self.num_curves+1):
            self.response_axes.append(
                self.fig.add_subplot(self.num_curves, 1, i))

        # Move subplot into figure: [0,1]x[0,1] is to total
        # figure area; here we let the subplot area have lower
        # left corner at (0.1,0.2), leaving the space below
        # for slider widgets.
        plt.subplots_adjust(left=0.1, bottom=0.2)

        # Init plot with axes extent, labels, etc.
        self.clear_drawing()
        if self.response:
            self.clear_response()

        # Bind events to the upper plot (the drawn curve: self.ax)
        self.widget = matplotlib.widgets.AxesWidget(self.ax)
        self.widget.connect_event('button_press_event', self.on_click)
        self.widget.connect_event('motion_notify_event', self.on_move)
        self.widget.connect_event('key_press_event', self.on_key)

        if not self.sliders:
            return
        # Make sliders to adjust the number of data points from
        # the drawing we use for defining the spline (sample_fraction)
        # and the number of intervals we sample from the spline
        # (resolution).
        # (Axes does not work:)
        #self.slider_ax = matplotlib.axes.Axes(
        #    self.fig, [0.1, 0.1, 0.7, 0.05])
        self.slider_ax = plt.axes([0.15, 0.05, 0.7, 0.035])
        self.slider_resolution = matplotlib.widgets.Slider(
            self.slider_ax, 'samples',
            valmin=0, valmax=2*self.resolution,
            valinit=self.resolution, valfmt='%d')
        self.slider_resolution.on_changed(self.update_drawing)

        self.slider_ax = plt.axes([0.15, 0.1, 0.7, 0.035])
        self.slider_sample_fraction = matplotlib.widgets.Slider(
            self.slider_ax, '% points',
            valmin=0, valmax=120,
            valinit=int(100*self.sample_fraction), valfmt='%d')
        self.slider_sample_fraction.on_changed(self.update_drawing)

    def on_key(self, event):
        """Called when a key is pressed."""
        DrawFunction.menu(self, event.key)


    def _clear_plot(self, ax,
                    xmin=None, xmax=None,
                    ymin=None, ymax=None,
                    xlabel=None, ylabel=None,
                    title=None):
        """
        Make empty plot for given axes `ax`.
        `xmin`, etc. can be specified to fix the extent of axes,
        put labels on axes, and a title.
        """
        ax.cla()
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        # Plot empty curve to realize the removal of previous curves
        ax.plot([], [])

    def clear_drawing(self):
        """Erase the drawing."""
        if self.title == 'help string':
            title = 'Click to start drawing, click to end; repeat until satisfied'
        elif self.title is not None:
            title = self.title
        else:
            title = None

        self._clear_plot(self.ax,
                         xmin=self.xmin, xmax=self.xmax,
                         ymin=self.ymin, ymax=self.ymax,
                         xlabel=self.xlabel, ylabel=self.ylabel,
                         title=title)

    def clear_response(self):
        """Make empty plots for the responses."""
        c = self.response['curves']
        for i, curve in enumerate(self.response['curves']):
            ymin = curve.get('ymin', None)
            ymax = curve.get('ymax', None)
            title = curve.get('title', None)
            xlabel = curve.get('xlabel', None)
            ylabel = curve.get('ylabel', None)
            self._clear_plot(self.response_axes[i],
                             xmin=self.xmin, xmax=self.xmax,
                             ymin=ymin, ymax=ymax,
                             xlabel=xlabel, ylabel=ylabel,
                             title=title)

    def _make_png_str(self):
        """Return current plot as PNG string."""
        figfile = StringIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        return base64.b64encode(figfile.buf)

    def update_drawing(self, dummy=-1):
        """Transform the drawing to a smooth spline curve."""
        # Argument val is not used - we need to grab val in each slider

        # 1. Interpolate linearly between the recorded coordinates.
        # 2. Resample uniformly.
        # 3. Fit spline curve to the resampled points.
        # 4. Draw to smoothed curve.
        # Pt 2 is necessary to make splines work

        # Get slider values
        self.resolution = self.slider_resolution.val
        self.sample_fraction = float(self.slider_sample_fraction.val)/100

        # Compute smooth function self.spline from recorded coordinates
        # self.x and self.y
        if self.x is None or self.y is None:
            return
        self.spline, xp, yp = smooth_function_from_coordinates(
            self.x, self.y,
            self.sample_fraction,
            4 if self.spline_smoothing else 1)

        # Plot the smooth version of the drawing and mark the
        # points used for smoothing
        x, y = self.get_curve(self.resolution)
        self.clear_drawing()
        self.ax.plot(xp, yp, marker='o', color='blue', markersize=4,
                     linestyle=' ')
        self.ax.plot(x, y, linestyle='-', color='blue')
        if self.interactive:
            plt.draw()
        self.png_plot = self._make_png_str()

    def update_response(self):
        if not 'compute_function' in self.response:
            raise KeyError('response dict had no "compute_function" key')
        compute = self.response['compute_function']

        x, y = self.get_curve()  # input
        response = compute(x, y)
        if isinstance(response[0], np.ndarray):
            # response is x, u rather than [(x, y)]
            response = [response]
        for i, curve in enumerate(response):
            self.response_axes[i].plot(curve[0], curve[1])
        if self.interactive:
            plt.draw()
        self.png_plot = self._make_png_str()

    def update_response_animation(self):
        if not 'compute_function' in self.response:
            raise KeyError('response dict had no "compute_function" key')
        compute = self.response['compute_function']

        def process(response):
            if isinstance(response[0], np.ndarray):
                # response is x, u rather than [(x, y)]
                response = [response]
            for i, curve in enumerate(response):
                self.response_axes[i].plot(curve[0], curve[1])
            if self.interactive:
                plt.draw()
            self.png_plot = self._make_png_str()

        x, y = self.get_curve()  # input
        import collection
        if isinstance(compute, collection.Iterable):
            # Do the plotting here
            for response in compute(x, y):
                process(response)
        else:
            # Leave plotting to compute
            compute(x, y, self.response_axes, plt)

    def on_click(self, event):
        """Called when a mouse button is clicked."""
        # Act only when in plot area axes (self.ax)
        if event.inaxes != self.ax:
            return

        self._num_clicks += 1
        if self._num_clicks % 2 == 1:
            # Odd click inside plot area, start new drawing
            self.clear_drawing()
            DrawFunction.begin_drawing(self)
            DrawFunction.add_pt(self, event.xdata, event.ydata)
        else:
            # Even click, end it all
            DrawFunction.end_drawing(self)
            self._num_clicks = 0
            self.update_drawing()

    def on_move(self, event):
        """Called when mouse is moved."""
        # Act only when in plot area axes (self.ax)
        print 'moving mouse:', event.xdata, event.ydata
        if event.inaxes != self.ax:
            print 'outside axes!'
            return

        if self._num_clicks == 0:
            return

        DrawFunction.add_pt(self, event.xdata, event.ydata)
        if len(self._x) > 1:
            self.ax.plot(self._x, self._y)
            if self.interactive:
                plt.draw()
            self.png_plot = self._make_png_str()


def drawing2file():
    """
    Command-line application: run this file as a program with
    command-line arguments,

      * xmin, ymin, xmax, ymax of the plot area
      * resolution of the final discrete function from the drawing
      * filename

    The coordinates of a smoothed version of the drawing are
    stored in two columns in the file with the given name.
    """
    try:
        xmin, xmax, ymin, ymax, resolution, filename = sys.argv[1:]
    except:
        print 'Usage: %s xmin xmax ymin ymax resolution filename' \
              % sys.argv[0]
        sys.exit(1)
    plotter = DrawFunctionMpl(
        xmin=float(xmin), xmax=float(xmax),
        ymin=float(ymin), ymax=float(ymax),
        sample_fraction=1.0,
        resolution=int(resolution))
    plt.show()
    data = plotter.get_curve()
    if data is not None:
        f = open(filename, 'w')
        for xi, yi in zip(*data):
            f.write('%10.3g %10.3g\n' % (xi, yi))
        f.close()

def demo():
    print """

Click far left in the K(x) window, draw a curve, click when it is ended.

Move slider: % points chooses the percentage of the recorded drawn
points that are selected for spline smoothing (if turned on).
Move slider: samples represents the no of uniform intervals in the
discrete representation of K(x).

Press r to compute a solution u(x) of a differential equation that depends
on K(x): (K*u')'=0.

Press c to clear the previous u graph the next time you press r to
compute more. You can draw, press r, press c, draw, press r, draw,
press r (without pressing c the previous u curve is visible).
"""
    L = 2
    U_0 = 0
    U_L = 4

    def flow(x, K):
        """
        Given K(x), return the solution of (K(x)*u'(x))'=0,
        u(0)=U_0, u(L)=U_L, and the flux -K*u'.
        """
        # Use midpoint integration rule
        K_midpoints = (K[1:] + K[:-1])/2.0
        K_m1 = 1/K_midpoints
        dx = x[1] - x[0]  # uniform partition of x
        K_m1_integral = dx*np.cumsum(K_m1)
        u = np.zeros_like(K)
        u[1:] = U_0 + (U_L - U_0)*K_m1_integral/K_m1_integral[-1]
        u[0] = U_0
        flux = - K_midpoints*(u[1:] - u[:-1])/2.0
        x_flux = (x[1:] - x[:-1])/2.0
        #return [(x, u), (x_flux, flux)]
        return [(x, u)]

    # Perform simple K=const test
    x_test = np.linspace(0, L, 5)
    K_test = np.ones(5)
    curves = flow(x_test, K_test)
    x, u = curves[0]
    diff = (u - 2*x).max()
    assert abs(diff) < 1E-14, diff
    if len(curves) > 1:
        x_flux, flux = curves[1]
        assert abs(flux.max() + 0.5) < 1E-14

    response = dict(
        curves=[
            dict(ymin=U_0, ymax=U_L, xlabel='x', ylabel='u'),
            #dict(xlabel='x', ylabel='flux'),
            ],
        compute_function=flow)
    K_min = 0; K_max = 10
    plotter = DrawFunctionMpl(
        xmin=0, xmax=L, ymin=K_min, ymax=K_max,
        xlabel='x', ylabel='K(x)',
        response=response)
    plt.show()

# Nose test
def test_DrawFunction1():
    """Test with K(x)=x+2."""
    plotter = DrawFunction(
        xmin=0, xmax=1, ymin=0, ymax=1)

    x = np.linspace(0, 1, 5)
    x = x**2  # deform x coordinates
    x_drawn = x[1:-1]
    y_drawn = x_drawn + 2
    # (Linear extrapolation should be exact)
    plotter.begin_drawing()
    for x_pt, y_pt in zip(x_drawn, y_drawn):
        plotter.add_pt(x_pt, y_pt)
    plotter.end_drawing()
    x_diff = np.abs(plotter.x - np.array([ 0., 0.0625, 0.25, 0.5625, 1.])).max()
    y_diff = np.abs(plotter.y - np.array([ 2., 2.0625, 2.25, 2.5625, 3.])).max()
    assert x_diff < 1E-14
    assert y_diff < 1E-14

if __name__ == '__main__':
    if len(sys.argv) == 1:
        demo()
    else:
        drawing2file()
