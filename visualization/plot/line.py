"""
Line Plot Module

This module defines the Line class, a visual plot renderer that draws connected
line plots using a PPM-based raster image system. It supports plot features
such as line thickness, axis labels, tick marks, and titles — all rendered
without any third-party libraries.

All drawing operations are pixel-based using PPMImage and text is rendered using a custom
5x7 pixel font engine.
"""

from .plot import Plot


class Line(Plot):
    """
    A pixel-rendered line plot using PPMImage and a 5x7 font.

    This class renders a 2D line plot by connecting sequential data points using
    pixel-based line drawing. It supports axis labeling, tick marks, and titles.

    Attributes:
        x_data (list of float): X-coordinates of the data points.
        y_data (list of float): Y-coordinates of the data points.
        color (tuple[int, int, int]): RGB color of the line.
        line_thickness (int): Thickness of the line in pixels.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis (non-rotated).
        title (str): Plot title.
        x_ticks (list of float or None): X-axis tick values.
        y_ticks (list of float or None): Y-axis tick values.
        margin (int): Margin around the plot content.

    Example:
        >>> from visualization.plot.line import Line
        >>> x = [0, 1, 2, 3]
        >>> y = [0, 1, 4, 9]
        >>> plot = Line(x, y, title="Line Example", x_label="X", y_label="Y")
        >>> plot.render()
        >>> plot.save("line_output.ppm")
    """

    def __init__(
        self,
        x_data,
        y_data,
        color=(0, 0, 0),
        line_thickness=1,
        x_label="",
        y_label="",
        title="",
        x_ticks=None,
        y_ticks=None,
        **kwargs
    ):
        """
        Initialize the Line plot.

        Args:
            x_data (list of float): X-coordinates of data points.
            y_data (list of float): Y-coordinates of data points.
            color (tuple): RGB color of the line.
            line_thickness (int): Thickness of the connecting lines in pixels.
            x_label (str): Label for the X-axis.
            y_label (str): Label for the Y-axis (non-rotated).
            title (str): Plot title.
            x_ticks (list of float or None): Optional tick values for X-axis.
            y_ticks (list of float or None): Optional tick values for Y-axis.
            **kwargs: Additional arguments forwarded to the Plot base class.
        """
        super().__init__(**kwargs)
        self.x_data = x_data
        self.y_data = y_data
        self.color = color
        self.line_thickness = line_thickness
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
        Render the line plot.

        Computes scaling and dimensions, then draws:
        - Axes
        - Connecting lines
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

        self._draw_lines()

        # Draw axes
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

        # Y-axis label
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

    def _draw_lines(self):
        """
        Draw lines connecting the data points.

        Uses scaled coordinates and Bresenham’s line algorithm to rasterize lines.
        """
        pixel_coords = []
        for x, y in zip(self.x_data, self.y_data):
            px = int((x - self.min_x) / self.range_x * self.plot_w) + self.margin
            py = int((y - self.min_y) / self.range_y * self.plot_h)
            py = self.height - py - self.margin
            pixel_coords.append((px, py))

        for (x0, y0), (x1, y1) in zip(pixel_coords[:-1], pixel_coords[1:]):
            self._draw_line(x0, y0, x1, y1)

    def _draw_line(self, x0, y0, x1, y1):
        """
        Draw a line between two pixels using Bresenham's algorithm.

        Args:
            x0, y0: Starting pixel coordinates.
            x1, y1: Ending pixel coordinates.
        """
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            self.image.draw_dot(x0, y0, radius=self.line_thickness, color=self.color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
