import unittest
import os
from visualization.ppm.ppm import PPMImage
from visualization.plot.scatter import Scatter
from visualization.plot.line import Line
from visualization.plot.font import FontEngine


class TestPPMImage(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_output.ppm"
        self.img = PPMImage(10, 10, background=(255, 255, 255))

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_set_pixel(self):
        self.img.set_pixel(5, 5, (0, 0, 0))
        self.assertEqual(self.img.pixels[5][5], (0, 0, 0))

    def test_save_file_exists(self):
        self.img.set_pixel(1, 1, (123, 123, 123))
        self.img.save(self.test_file)
        self.assertTrue(os.path.exists(self.test_file))


class TestScatterPlot(unittest.TestCase):
    def setUp(self):
        self.output_file = "test_scatter.ppm"
        self.x_data = [0, 1, 2, 3, 4, 5]
        self.y_data = [0, 1, 4, 9, 16, 25]

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_render_and_save(self):
        plot = Scatter(
            self.x_data,
            self.y_data,
            title="Scatter Test",
            x_label="X",
            y_label="Y",
            x_ticks=[0, 2, 4],
            y_ticks=[0, 10, 20],
            dot_colors=[(255, 0, 0)] * 6,
            width=300,
            height=300,
        )
        plot.render()
        plot.save(self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

    @unittest.skipIf(os.environ.get("CI") == "true", "Skipping show in CI.")
    def test_scatter_show(self):
        plot = Scatter(self.x_data, self.y_data, title="Scatter Show", width=300, height=300)
        plot.render()
        plot.show()


class TestLinePlot(unittest.TestCase):
    def setUp(self):
        self.output_file = "test_line.ppm"
        self.x_data = [0, 1, 2, 3, 4, 5]
        self.y_data = [0, 1, 4, 9, 16, 25]

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_render_and_save(self):
        plot = Line(
            self.x_data,
            self.y_data,
            title="Line Test",
            x_label="X",
            y_label="Y",
            x_ticks=[0, 2, 4],
            y_ticks=[0, 10, 20],
            color=(0, 0, 255),
            line_thickness=1,
            width=300,
            height=300,
        )
        plot.render()
        plot.save(self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

    @unittest.skipIf(os.environ.get("CI") == "true", "Skipping show in CI.")
    def test_line_show(self):
        plot = Line(self.x_data, self.y_data)
        plot = Line(
            self.x_data,
            self.y_data,
            title="Line Test",
            x_label="X",
            y_label="Y",
            x_ticks=[0, 2, 4],
            y_ticks=[0, 10, 20],
            color=(0, 0, 255),
            line_thickness=1,
            width=300,
            height=300,
        )
        plot.render()
        plot.show()


class TestFontRendering(unittest.TestCase):
    def setUp(self):
        self.output_file = "test_font_output.ppm"
        self.font = FontEngine()
        self.image = PPMImage(320, 100, background=(255, 255, 255))

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_draw_all_characters(self):
        """Draw the full A-Z and 0-9 set using the 5x7 font."""
        all_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        all_digits = "0123456789-"
        self.font.draw_text(self.image, 5, 10, all_letters, color=(0, 0, 0), scale=1)
        self.font.draw_text(self.image, 5, 30, all_digits, color=(0, 0, 255), scale=1)
        self.image.save(self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

    @unittest.skipIf(os.environ.get("CI") == "true", "Skip font show in CI.")
    def test_show_font_image(self):
        """Show image with font preview (manual)."""
        all_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789-"
        self.font.draw_text(self.image, 5, 20, all_text, color=(0, 0, 0), scale=1)
        self.image.show()


if __name__ == "__main__":
    unittest.main()
