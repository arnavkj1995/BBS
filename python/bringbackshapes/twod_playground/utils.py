import pymunk
import config


def flipy(p):
    """Convert chipmunk physics to pygame coordinates."""
    return pymunk.Vec2d(p[0], -p[1] + config.env_config.canvas_dim_y)
