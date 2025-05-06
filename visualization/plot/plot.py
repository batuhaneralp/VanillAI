from abc import ABC, abstractmethod
from visualization.ppm.ppm import PPMImage
from .font import FontEngine


class Plot(ABC):
    def __init__(self, width=400, height=400, background=(255, 255, 255)):
        self.width = width
        self.height = height
        self.background = background
        self.image = PPMImage(width, height, background)
        self.font = FontEngine()

    @abstractmethod
    def render(self):
        pass

    def save(self, path: str):
        self.image.save(path)

    def show(self):
        self.image.show()

    def draw_x_ticks(self, min_val, max_val, tick_values, margin, plot_width):
        """
        Draw X-axis tick labels using FontEngine.

        Args:
            min_val (float): Minimum x value.
            max_val (float): Maximum x value.
            tick_values (list): List of tick values.
            margin (int): Margin space around plot.
            plot_width (int): Width of the plot area.
        """
        range_val = max_val - min_val or 1
        for tick in tick_values:
            px = int((tick - min_val) / range_val * plot_width) + margin
            self.font.draw_text(
                self.image, px - 5, self.height - margin + 5, str(tick), color=(0, 0, 0), scale=1
            )

    def draw_y_ticks(self, min_val, max_val, tick_values, margin, plot_height):
        """
        Draw Y-axis tick labels using FontEngine.

        Args:
            min_val (float): Minimum y value.
            max_val (float): Maximum y value.
            tick_values (list): List of tick values.
            margin (int): Margin space around plot.
            plot_height (int): Height of the plot area.
        """
        range_val = max_val - min_val or 1
        for tick in tick_values:
            py = int((tick - min_val) / range_val * plot_height)
            py = self.height - py - margin
            self.font.draw_text(self.image, 10, py - 3, str(tick), color=(0, 0, 0), scale=1)
