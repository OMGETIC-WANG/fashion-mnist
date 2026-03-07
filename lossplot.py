import matplotlib.pyplot as plt


class _Line:
    def __init__(self, axes, name: str, color: str = ""):
        (self.line,) = axes.plot([], [], color, label=name)
        axes.legend(loc="best")
        self.xdata = []
        self.ydata = []

    def Add(self, x, y):
        self.xdata.append(x)
        self.ydata.append(y)
        self.line.set_data(self.xdata, self.ydata)


class LossPlot:
    def __init__(self, title: str = "Loss plot"):
        self.figure, self.axes = plt.subplots()
        self.figure.suptitle(title)
        self.lines: dict[str, _Line] = {}
        self.xvalue = 0
        plt.pause(0.001)

    def Update(self, vals: dict[str, float]):
        self.xvalue += 1
        for line, val in vals.items():
            if not line in self.lines:
                self.lines[line] = _Line(self.axes, line)
            self.lines[line].Add(self.xvalue, val)
        self.axes.relim()
        self.axes.autoscale_view()

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
