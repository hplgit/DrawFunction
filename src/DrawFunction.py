# Extensions:
# Allow curves returned from compute_function to be dicts;
# {'label1': (x1,y1), 'label2': (x2,y2)} instead of just (x,y)
# Allow the returned curve be a object for function animation
# (of several curves).
# http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

import matplotlib.pyplot as plt
import matplotlib.widgets
import matplotlib.axes
import numpy as np
from scipy.interpolate import UnivariateSpline as Spline
from Scientific.Functions.Interpolation \
     import InterpolatingFunction
import sys

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
        self._num_clicks = 0
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

        if fig is None:
            self.fig = plt.figure()
        else:
            self.fig = fig

        self.ax = self.fig.add_subplot(self.num_curves, 1, 1)
        self.response_axes = []
        for i in range(2, self.num_curves+1):
            self.response_axes.append(
                self.fig.add_subplot(self.num_curves, 1, i))

        plt.subplots_adjust(left=0.1, bottom=0.2) # move subplot into figure

        # Init plot with axes extent, labels, etc.
        self.clear_drawing()
        if self.response:
            self.clear_response()

        self.widget = matplotlib.widgets.AxesWidget(self.ax)
        self.widget.connect_event('button_press_event', self.on_click)
        self.widget.connect_event('motion_notify_event', self.on_move)
        self.widget.connect_event('key_press_event', self.on_key)

        if not sliders:
            return
        # Make sliders to adjust the number of data points from
        # the drawing we use for defining the spline (sample_fraction)
        # and the number of intervals we sample from the spline
        # (resolution).
        self.slider_ax = matplotlib.axes.Axes(
            self.fig, [0.1, 0.1, 0.7, 0.05])
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
        if event.key == 'r':
            if self.animated_response:
                self.update_response_animation()
            else:
                self.update_response()
        elif event.key == 'c':
            self.clear_response()

    def clear_plot(self, ax,
                   xmin=None, xmax=None,
                   ymin=None, ymax=None,
                   xlabel=None, ylabel=None,
                   title=None):
        """Make empty plot for given axes `ax`."""
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
        """Make empty plot for the drawing."""
        if self.title == 'help string':
            title = 'Click to start drawing, click to end; repeat until satisfied'
        elif self.title is not None:
            title = self.title
        else:
            title = None

        self.clear_plot(self.ax,
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
            self.clear_plot(self.response_axes[i],
                            xmin=self.xmin, xmax=self.xmax,
                            ymin=ymin, ymax=ymax,
                            xlabel=xlabel, ylabel=ylabel,
                            title=title)

    def update_drawing(self, val):
        self.resolution = self.slider_resolution.val
        self.sample_fraction = float(self.slider_sample_fraction.val)/100
        self.smooth_and_plot_drawing()

    # If compute_function is an iterator we can iterate here, otherwise
    # do one call, pass self.response_axes and plt to compute_function
    # and rely on the animation loop there

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
        plt.draw()

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
            plt.draw()

        x, y = self.get_curve()  # input
        import collection
        if isinstance(compute, collection.Iterable):
            # Do the plotting here
            for response in compute(x, y):
                process(response)
        else:
            # Leave plotting to compute
            compute(x, y, self.response_axes, plt)

    def smooth_and_plot_drawing(self):
        # Resample uniformly (helps on making splines work)
        if self.x is None or self.y is None:
            return
        xp = np.linspace(self.x[0], self.x[-1],
                        self.sample_fraction*len(self.x))
        yp = np.array([self.linear(xi) for xi in xp])
        if self.spline_smoothing:
            self.spline = Spline(xp, yp, s=0, k=4)
        else:
            self.spline = InterpolatingFunction([self.xp], self.yp)
        x, y = self.get_curve(self.resolution)
        self.clear_drawing()
        self.ax.plot(xp, yp, marker='o', color='blue', markersize=4,
                     linestyle=' ')
        self.ax.plot(x, y, linestyle='-', color='blue')
        plt.draw()

    def on_click(self, event):
        """Called when a mouse button is clicked."""
        # Act only when in plot area axes (self.ax)
        if event.inaxes != self.ax:
            return

        self._num_clicks += 1
        if self._num_clicks % 2 == 1:
            # Odd click inside plot area, start new drawing

            self._x = [event.xdata]
            self._y = [event.ydata]
            self.clear_drawing()
            self.spline = None
            self.x = self.y = None
        else:
            # Even click, end drawing

            # Add end points so that xmin and xmax are included
            self.x = np.zeros(len(self._x)+2)
            self.y = np.zeros(len(self._y)+2)
            self.x[1:-1] = self._x
            self.y[1:-1] = self._y
            # Extrapolate end points
            self.x[0] = self.xmin
            dx = self._x[1] - self._x[0]
            if dx > 1E-10:
                self.y[0] = self._y[0] + \
                (self.xmin - self._x[0])/dx*(self._y[1] - self._y[0])
            else:
                self.y[0] = self.y[1]
            self.x[-1] = self.xmax
            dx = self._x[-1] - self._x[-2]
            if dx > 1E-10:
                self.y[-1] = self._y[-2] + \
                (self.xmax - self._x[-1])/dx*(self._y[-1] - self._y[-2])
            else:
                self.y[-1] = self.y[-2]

            self._x = self._y = []  # clear recorded points

            # Check that self.y = f(self.x) is a function
            # (is guaranteed in on_move)
            if (np.sort(self.x) - self.x).max() != 0:
                print 'Not a valid function - draw again'

            # Construct linear interpolator of data points
            self.linear = InterpolatingFunction([self.x], self.y)
            self.smooth_and_plot_drawing()

    def on_move(self, event):
        """Called when mouse is moved."""
        # Act only when in plot area axes (self.ax)
        if event.inaxes != self.ax:
            return

        if self._x and self._y: # are we in a drawing?
            if event.xdata is not None and event.ydata is not None:
                # Do not allow x smaller than last one
                if event.xdata < self._x[-1]:
                    self._x.append(self._x[-1])
                else:
                    self._x.append(event.xdata)
                self._y.append(event.ydata)
                if len(self._x) > 1:
                    self.ax.plot(self._x, self._y)
                    plt.draw()

    def get_curve(self, resolution=None):
        if resolution is None:
            resolution = self.resolution
        if self.spline:
            x = np.linspace(self.x[0], self.x[-1], resolution+1)
            y = self.spline(x)
            return x, y


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
    plotter = DrawFunction(float(xmin), float(xmax),
                           float(ymin), float(ymax),
                           1.0, int(resolution))
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
    plotter = DrawFunction(0, L, K_min, K_max,
                           xlabel='x', ylabel='K(x)',
                           response=response)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        demo()
    else:
        drawing2file()
