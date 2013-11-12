from DrawFunction import DrawFunction, smooth_function_from_coordinates, \
     extrapolate_endpoints

class DrawFunctionCanvas(DrawFunction):
    def __init__(self, *args, **kwargs):
        DrawFunction.__init__(self, *args, **kwargs)
