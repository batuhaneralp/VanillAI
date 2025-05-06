import os
import tempfile
import platform
import subprocess


class PPMImage:
    """
    Simple PPM image generator using vanilla Python.
    Supports RGB pixel writing and outputs .ppm (P3) format.
    """

    def __init__(self, width: int, height: int, background=(255, 255, 255)):
        """
        Create a blank PPM image.

        Args:
            width (int): Width of the image in pixels.
            height (int): Height of the image in pixels.
            background (tuple): RGB color tuple for the background.
        """
        self.width = width
        self.height = height
        self.bg_color = background
        self.pixels = [[self.bg_color for _ in range(width)] for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: tuple):
        """
        Set the color of a single pixel.

        Args:
            x (int): X coordinate.
            y (int): Y coordinate (0 at top).
            color (tuple): RGB color tuple.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = color

    def draw_dot(self, x: int, y: int, radius: int = 1, color=(0, 0, 0)):
        """
        Draw a filled square dot around a center pixel.

        Args:
            x (int): X coordinate of center.
            y (int): Y coordinate of center.
            radius (int): Half-size of the dot (square).
            color (tuple): RGB color tuple.
        """
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                self.set_pixel(x + dx, y + dy, color)

    def save(self, path: str):
        """
        Save the image to a .ppm file.

        Args:
            path (str): Output file path.
        """
        with open(path, "w") as f:
            f.write("P3\n")
            f.write(f"{self.width} {self.height}\n255\n")
            for row in self.pixels:
                for r, g, b in row:
                    f.write(f"{r} {g} {b} ")
                f.write("\n")

    def show(self):
        """
        Save the image to a temporary .ppm file and open it with the default image viewer.
        """
        # Create temporary file path
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ppm")
        self.save(tmp_file.name)

        # Determine platform and open
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", tmp_file.name])
        elif platform.system() == "Windows":
            os.startfile(tmp_file.name)
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", tmp_file.name])
        else:
            print(f"Unsupported platform: {platform.system()}")
