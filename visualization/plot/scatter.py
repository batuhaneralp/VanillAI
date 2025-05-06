"""
Scatter Plot Module

This module defines the Scatter class, a visual plot renderer that draws scatter plots
using a PPM-based raster image system. It supports plot features such as custom dot colors,
axis labels, tick marks, and titles — all rendered without any third-party libraries.

All drawing operations are pixel-based using PPMImage and text is rendered using a custom
5x7 pixel font engine.

"""


from .plot import Plot


class Scatter(Plot):
    """
    A pixel-rendered scatter plot using PPMImage and a 5x7 font.

    This class allows rendering of 2D data points on a manually scaled pixel canvas.
    It supports axis labeling, tick marks, titles, and custom colors per point.

    Attributes:
        x_data (list of float): X-coordinates of the data points.
        y_data (list of float): Y-coordinates of the data points.
        color (tuple[int, int, int]): Default RGB color to use for points.
        dot_colors (list[tuple[int, int, int]] or None): Optional RGB color per point.
        point_size (int): Radius of each point in pixels.
        x_label (str): X-axis label.
        y_label (str): Y-axis label (non-rotated).
        title (str): Plot title.
        x_ticks (list[float]): X-axis tick values.
        y_ticks (list[float]): Y-axis tick values.
        margin (int): Margin around the plot content.

    Example:
        >>> from visualization.plot.scatter import Scatter
        >>> x = [1, 2, 3, 4]
        >>> y = [1, 4, 9, 16]
        >>> plot = Scatter(x, y, title="Example", x_label="X", y_label="Y")
        >>> plot.render()
        >>> plot.save("example.ppm")
    """

    def __init__(
        self,
        x_data,
        y_data,
        color=(0, 0, 0),
        dot_colors=None,
        point_size=2,
        x_label="",
        y_label="",
        title="",
        x_ticks=None,
        y_ticks=None,
        **kwargs
    ):
        """
        A pixel-rendered scatter plot using PPMImage and a 5x7 font.

        This class allows rendering of 2D data points on a manually scaled pixel canvas.
        It supports axis labeling, tick marks, titles, and custom colors per point.

        Attributes:
            x_data (list of float): X-coordinates of the data points.
            y_data (list of float): Y-coordinates of the data points.
            color (tuple[int, int, int]): Default RGB color to use for points.
            dot_colors (list[tuple[int, int, int]] or None): Optional RGB color per point.
            point_size (int): Radius of each point in pixels.
            x_label (str): X-axis label.
            y_label (str): Y-axis label (non-rotated).
            title (str): Plot title.
            x_ticks (list[float] or None): X-axis tick values.
            y_ticks (list[float] or None): Y-axis tick values.
            margin (int): Margin around the plot content.

        Example:
            >>> from visualization.plot.scatter import Scatter
            >>> x = [1, 2, 3, 4]
            >>> y = [1, 4, 9, 16]
            >>> plot = Scatter(x, y, title="Example", x_label="X", y_label="Y")
            >>> plot.render()
            >>> plot.save("example.ppm")
        """
        super().__init__(**kwargs)
        self.x_data = x_data
        self.y_data = y_data
        self.color = color
        self.dot_colors = dot_colors
        self.point_size = point_size
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.x_ticks = x_ticks
        self.y_ticks = y_ticks

        # Computed at render time
        self.margin = 50
        self.plot_w = None
        self.plot_h = None
        self.range_x = None
        self.range_y = None
        self.min_x = None
        self.min_y = None

    def render(self):
        """
        Render the scatter plot.

        Computes scaling and dimensions, then draws:
        - Axes
        - Dots
        - Labels
        - Tick marks
        """
        if len(self.x_data) != len(self.y_data):
            raise ValueError("x_data and y_data must be the same length")

        self.plot_w = self.width - 2 * self.margin
        self.plot_h = self.height - 2 * self.margin

        self.min_x, max_x = min(self.x_data), max(self.x_data)
        self.min_y, max_y = min(self.y_data), max(self.y_data)

        self.range_x = max_x - self.min_x or 1
        self.range_y = max_y - self.min_y or 1

        self._draw_points()

        # Draw X and Y axes
        for i in range(self.plot_w + 1):
            self.image.set_pixel(self.margin + i, self.height - self.margin, (0, 0, 0))
        for i in range(self.plot_h + 1):
            self.image.set_pixel(self.margin, self.height - self.margin - i, (0, 0, 0))

        # Title
        if self.title:
            self.font.draw_text(
                self.image,
                self.width // 2 - len(self.title) * 6 // 2,
                10,
                self.title,
                color=(0, 0, 0),
                scale=1,
            )

        # X-axis label
        if self.x_label:
            self.font.draw_text(
                self.image,
                self.width // 2 - len(self.x_label) * 6 // 2,
                self.height - 25,
                self.x_label,
                color=(0, 0, 0),
                scale=1,
            )

        # Y-axis label (not rotated)
        if self.y_label:
            self.font.draw_text(
                self.image,
                5,
                self.margin,
                self.y_label[:10],
                color=(0, 0, 0),
                scale=1,
            )

        # Tick marks
        if self.x_ticks:
            self.draw_x_ticks(
                self.min_x, self.min_x + self.range_x, self.x_ticks, self.margin, self.plot_w
            )
        if self.y_ticks:
            self.draw_y_ticks(
                self.min_y, self.min_y + self.range_y, self.y_ticks, self.margin, self.plot_h
            )

    def _draw_points(self):
        """
        Plot all data points as filled circles on the canvas.
        Uses scaled coordinates and handles custom per-point colors.
        """
        for i, (x, y) in enumerate(zip(self.x_data, self.y_data)):
            px = int((x - self.min_x) / self.range_x * self.plot_w) + self.margin
            py = int((y - self.min_y) / self.range_y * self.plot_h)
            py = self.height - py - self.margin
            dot_color = (
                self.dot_colors[i] if self.dot_colors and i < len(self.dot_colors) else self.color
            )
            self.image.draw_dot(px, py, radius=self.point_size, color=dot_color)
