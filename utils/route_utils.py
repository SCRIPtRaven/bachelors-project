import math
from colorsys import hsv_to_rgb


class RouteColorManager:
    """
    Advanced color management system that ensures maximum visual distinction between routes,
    especially for neighboring routes, using perceptual color spacing and contrast optimization.
    """

    def __init__(self):
        self.golden_ratio = (1 + math.sqrt(5)) / 2

        self.base_hues = [
            0.0,  # Red
            0.33,  # Green
            0.66,  # Blue
            0.15,  # Yellow-Orange
            0.45,  # Turquoise
            0.75,  # Purple
            0.05,  # Orange
            0.25,  # Yellow-Green
            0.55,  # Sky Blue
            0.85  # Magenta
        ]

        self.saturation_levels = [1.0, 0.85, 0.7]
        self.brightness_levels = [0.95, 0.75, 0.55]

        self.patterns = [
            {'type': 'solid', 'weight': 4, 'dash_array': None},
            {'type': 'dashed', 'weight': 4, 'dash_array': '10, 10'},
            {'type': 'dotted', 'weight': 4, 'dash_array': '3, 7'}
        ]

        self.style_cache = {}
        self.used_combinations = set()

    def get_route_style(self, index, total_routes):
        """
        Generates a visually distinct style for each route, ensuring neighboring routes
        are easily distinguishable.
        """
        cache_key = (index, total_routes)
        if cache_key in self.style_cache:
            return self.style_cache[cache_key]

        hue_index = int((index * self.golden_ratio) * len(self.base_hues)) % len(self.base_hues)
        base_hue = self.base_hues[hue_index]

        sat_index = (index // len(self.base_hues)) % len(self.saturation_levels)
        bright_index = (index // (len(self.base_hues) * len(self.saturation_levels))) % len(self.brightness_levels)

        pattern_index = index % len(self.patterns)

        hsv_color = (
            base_hue,
            self.saturation_levels[sat_index],
            self.brightness_levels[bright_index]
        )

        rgb_color = hsv_to_rgb(*hsv_color)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255)
        )

        pattern = self.patterns[pattern_index]

        style = {
            'color': hex_color,
            'weight': pattern['weight'],
            'dash_array': pattern['dash_array'],
            'opacity': 0.8 if pattern['type'] != 'solid' else 0.9
        }

        self.style_cache[cache_key] = style
        return style
