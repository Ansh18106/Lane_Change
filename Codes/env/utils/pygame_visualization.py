import colorsys
import pygame


class LaneRenderer:

    def __init__(self, x_end, y_min, y_max, lane_centers, lane_width, grid_rows=10, grid_cols=10):

        pygame.init()
        self.clock = pygame.time.Clock()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.x_end = x_end
        self.y_min = y_min
        self.y_max = y_max
        self.lane_centers = lane_centers
        self.lane_width = lane_width
        num_lanes = len(lane_centers)
        self.screen_width = num_lanes * 300
        self.screen_height = 300
        self.lane_colors = self.generate_k_colors(num_lanes)

        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height)
        )

        pygame.display.set_caption("Lane Change Simulation")


    def darken(self, color, factor=0.6):
        return tuple(int(c * factor) for c in color)

    def generate_k_colors(self, k):
        colors = []
        for i in range(k):
            h = i / k
            s = 0.4   # LOW saturation (key change)
            v = 0.9   # HIGH brightness
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgb = tuple(int(255 * x) for x in rgb)
            colors.append(rgb)
        return colors

    def world_to_screen(self, x, y):

        sx = int((x / self.x_end) * self.screen_width)

        sy = int(
            self.screen_height
            - (y - self.y_min)
            / (self.y_max - self.y_min)
            * self.screen_height
        )

        return sx, sy
    
    def draw_grid(self):
        if self.grid_rows is None or self.grid_cols is None:
            return  

        # light grey grid
        grid_color = (200, 200, 200)

        # vertical lines
        for c in range(1, self.grid_cols):
            x = (c / self.grid_cols) * self.screen_width
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.screen_height), 1)

        # horizontal lines
        for r in range(1, self.grid_rows):
            y = (r / self.grid_rows) * self.screen_height
            pygame.draw.line(self.screen, grid_color, (0, y), (self.screen_width, y), 1)


    def draw_lanes(self):
        for i, center in enumerate(self.lane_centers):

            color = self.lane_colors[i]

            y1 = center - self.lane_width / 2
            y2 = center + self.lane_width / 2

            # convert both corners
            sx_left, sy_top = self.world_to_screen(0, y2)
            sx_right, sy_bottom = self.world_to_screen(self.x_end, y1)

            # create rectangle (width, height must be positive)
            rect = pygame.Rect(
                sx_left,
                sy_top,
                sx_right - sx_left,
                sy_bottom - sy_top
            )

            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.line(self.screen, (0, 0, 0), (sx_left, sy_top), (sx_right, sy_top), 2)
            pygame.draw.line(self.screen, (0, 0, 0), (sx_left, sy_bottom), (sx_right, sy_bottom), 2)

    def draw_cars(self, agents):
        for i, agent in agents.items():

            sx, sy = self.world_to_screen(agent["x"], agent["y"])

            car_width = 40
            car_height = 20

            rect = pygame.Rect(
                sx - car_width // 2,
                sy - car_height // 2,
                car_width,
                car_height
            )

            lane_idx = agent["target_lane"]
            base_color = self.lane_colors[lane_idx]
            car_color = self.darken(base_color)

            pygame.draw.rect(self.screen, car_color, rect)

    def render(self, agents):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((255, 255, 255))

        # draw lanes
        self.draw_lanes()        

        # draw grid
        self.draw_grid()

        # draw cars
        self.draw_cars(agents)

        pygame.display.flip()

        self.clock.tick(60)