import typing
import numpy as np
import random
import math
import secrets

import pygame
import pymunk
import pymunk.pygame_util

from bringbackshapes.twod_playground import config
import pygame.font

pygame.font.init()

try:
    import pygame.display
    pygame.display.init()
except Exception:
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame.display
    pygame.display.init()


class ObjectMetaData(typing.NamedTuple):
    """
    ObjectMetaData is a named tuple that contains information about an object.
    """
    object_body: pymunk.Body
    name: str
    mass: float
    moment: float
    shape: pymunk.Poly
    color: typing.Tuple[int, int, int, int]
    elasticity: float
    friction: float


class AgentMetaData(typing.NamedTuple):
    """
    AgentMetaData is a named tuple that contains information about an agent.
    """
    agent_body: pymunk.Body
    name: str
    mass: float
    moment: float
    shape: pymunk.Circle
    color: typing.Tuple[int, int, int, int]
    elasticity: float
    friction: float


class ObjectState(object):
    position: typing.Tuple[float, float]
    angle: float
    velocity: typing.Tuple[float, float]
    angular_velocity = float


class AgentState(object):
    position: typing.Tuple[float, float]
    angle: float
    velocity: typing.Tuple[float, float]
    angular_velocity = float


class Arena:
    def __init__(
        self,
        debug=True,
        user_overide=True,
        render_game=True,
        random_start=True,
        max_objects=5,
        max_distractors=1,
        variable_num_objects=False,
        variable_num_distractors=False,
        variable_goal_position=False,
        agent_view_size=125,
        arena_scale=1.0
    ):
        self.width = int(config.ARENA_WIDTH * arena_scale)
        self.length = int(config.ARENA_LENGTH * arena_scale)
        self.wall_thickness_x = config.WALL_THICKNESS_X
        self.wall_thickness_y = config.WALL_THICKNESS_Y

        self.agent_view_size = agent_view_size
        self.agent_radius = config.AGENT_RADIUS
        assert self.agent_view_size <= self.width and \
            self.agent_view_size <= self.length
        self.angle_momentum = config.ANGLE_MOMENTUM
        self.goal_bounce_factor = config.GOAL_BOUNCE_FACTOR
        self.max_acc = config.MAX_ACC
        self.max_vel = config.MAX_VEL
        self.user_overide = user_overide
        self.overide = False
        self.overide_key = None
        self.collision_types = config.COLLISION_TYPES
        self.render_game = render_game

        self.fps = config.FPS
        self.frameskip = config.FRAMESKIP
        
        self.debug = debug
        self.max_objects = max_objects
        self.random_start = random_start
        self.max_distractors = max_distractors
        self.add_brownian_distractor = self.max_distractors > 0

        self.variable_num_objects = variable_num_objects
        self.variable_num_distractors = variable_num_distractors
        self.variable_goal_position = variable_goal_position
        self.float_generator = secrets.SystemRandom()
        self.__init_graphic_backbone()
        self.__init_physical_world()
        self.define_goal_area()
        self.create_goal_zone()
        self.init_walls()

        self.add_collision_handlers()

        self.agent_state = AgentState()
        self.objects_states = []
        self.reset_num_objects_and_distractors()
        self.reset_state()
        self.reset_entities()

        self.check_running_status()
        self.update()

    def init_walls(self):
        walls: typing.List[pymunk.Shape] = [
            pymunk.Poly(
                self.space.static_body,
                [
                    (0, 0),
                    (0, self.length),
                    (self.wall_thickness_x, self.length),
                    (self.wall_thickness_x, 0),
                ],
            ),
            pymunk.Poly(
                self.space.static_body,
                [
                    (0, 0),
                    (0, self.wall_thickness_y),
                    (self.width, self.wall_thickness_y),
                    (self.width, 0),
                ],
            ),
            pymunk.Poly(
                self.space.static_body,
                [
                    (self.width - self.wall_thickness_x, 0),
                    (self.width - self.wall_thickness_x, self.length),
                    (self.width, self.length),
                    (self.width, 0),
                ],
            ),
            pymunk.Poly(
                self.space.static_body,
                [
                    (0, self.length - self.wall_thickness_y),
                    (0, self.length),
                    (self.width, self.length),
                    (self.width, self.length - self.wall_thickness_y),
                ],
            ),
            pymunk.Poly(
                self.space.static_body,
                [
                    (
                        self.width - self.wall_thickness_x - self.goal_width,
                        self.wall_thickness_y,
                    ),
                    (
                        self.width - self.wall_thickness_x - self.goal_width,
                        self.goal_block_height + self.wall_thickness_y,
                    ),
                    (
                        self.width - self.wall_thickness_x,
                        self.goal_block_height + self.wall_thickness_y,
                    ),
                    (
                        self.width - self.wall_thickness_x,
                        self.wall_thickness_y,
                    ),
                ],
            ),
            pymunk.Poly(
                self.space.static_body,
                [
                    (
                        self.width - self.wall_thickness_x - self.goal_width,
                        self.length
                        - self.wall_thickness_y
                        - self.goal_block_height,
                    ),
                    (
                        self.width - self.wall_thickness_x - self.goal_width,
                        self.length - self.wall_thickness_y,
                    ),
                    (
                        self.width - self.wall_thickness_x,
                        self.length - self.wall_thickness_y,
                    ),
                    (
                        self.width - self.wall_thickness_x,
                        self.length
                        - self.goal_block_height
                        - self.wall_thickness_y,
                    ),
                ],
            ),
        ]
        for wall in walls:
            wall.friction = 1.0
            wall.elasticity = 0.7
            wall.group = 1
        self.space.add(*walls)

    def define_goal_area(self):
        self.goal_width = config.GOAL_WIDTH
        self.goal_height_factor = config.GOAL_HEIGHT_FACTOR
        self.goal_height = self.goal_height_factor * (
            self.length - 2 * self.wall_thickness_y
        )
        self.goal_block_height = (
            self.length - self.goal_height - 2 * self.wall_thickness_y
        ) / 2
        self.goal_x = self.width - self.wall_thickness_x - self.goal_width
        self.goal_y = self.wall_thickness_y + self.goal_block_height

    def create_agent(self, initial_position: typing.Tuple[int, int] = None):
        self.agent_body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.agent_body.mass = 10
        self.agent_body.moment = 1e6
        self.agent_shape = pymunk.Circle(self.agent_body, self.agent_radius)
        self.agent_shape.color = pygame.Color("steelblue")
        self.agent_shape.friction = 0.5
        self.agent_shape.collision_type = self.collision_types["agent"]
        self.agent_shape.density = 0.1 
        if initial_position is None:
            game_width = self.width - 2 * (
                self.wall_thickness_x + self.agent_radius
            )
            game_height = self.length - 2 * (
                self.wall_thickness_y + self.agent_radius
            )
            self.agent_body.position = (
                np.random.uniform(-game_width / 2, game_width / 2)
                + (self.width // 2),
                np.random.uniform(-game_height / 2, game_height / 2)
                + (self.length // 2),
            )
        else:
            self.agent_body.position = initial_position
        agent_name = "deterministic"

        self.agent_state.position = self.agent_body.position
        self.agent_state.angle = 0.0
        self.agent_state.velocity = self.agent_body.velocity
        self.agent_state.angular_velocity = self.agent_body.angular_velocity
        self.agent_metadata = AgentMetaData(
            self.agent_body,
            agent_name,
            self.agent_body.mass,
            self.agent_body.moment,
            self.agent_shape,
            self.agent_shape.color,
            self.agent_shape.elasticity,
            self.agent_shape.friction,
        )
        self.space.add(self.agent_body, self.agent_shape)
        self.handler.data["agent"] = self.agent_body

    def define_arena_object(self, position, shape=None, color=None):
        object_body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        object_body.position = position
        object_body.mass = 20
        object_body.moment = 1e6
        object_body.angle = np.random.random_sample() * 2 * math.pi
        if shape is None:
            rand_num = np.random.random_sample()
            if rand_num <= 0.2:
                # create a triangle
                object_shape = pymunk.Poly(
                    object_body, [(0, 0), (50, 0), (25, 25)]
                )
                name = "triangle"
            elif rand_num <= 0.4:
                # create a square
                object_shape = pymunk.Poly(
                    object_body, [(0, 0), (50, 0), (50, 50), (0, 50)]
                )
                name = "square"
            elif rand_num <= 0.6:
                # create a rectangle
                object_shape = pymunk.Poly(
                    object_body, [(0, 0), (50, 0), (50, 25), (0, 25)]
                )
                name = "rectangle"
            elif rand_num <= 0.8:
                # create a pentagon
                object_shape = pymunk.Poly(
                    object_body,
                    [(5, -17), (-19, 0), (-10, 28), (19, 28), (28, 0)],
                )
                name = "pentagon"
            else:
                # create a hexagon
                object_shape = pymunk.Poly(
                    object_body,
                    [
                        (17, -14),
                        (-8, -14),
                        (-20, 7),
                        (-8, 29),
                        (17, 29),
                        (30, 8),
                    ],
                )
                name = "hexagon"
        else:
            object_shape = shape
            name = shape.name
        if color is None:
            # pick a colour from the tableau palette
            color = random.choice(["orange", "blue", "green", "red", "purple"])
            object_shape.color = pygame.Color(color)
        object_shape.elasticity = 1.0
        object_shape.friction = 0.5
        object_shape.collision_type = self.collision_types["objects"]

        object_state = ObjectState()
        object_state.position = object_body.position
        object_state.angle = 0.0
        object_state.velocity = [0.0, 0.0]
        object_state.angular_velocity = [0.0, 0.0]

        # [bookeeping] populate the metadata for each object
        object_metadata = ObjectMetaData(
            object_body,
            name,
            object_body.mass,
            object_body.moment,
            object_shape,
            color,
            object_shape.elasticity,
            object_shape.friction,
        )
        return object_body, object_shape, object_metadata, object_state

    def create_arena_objects(self, num_objects):
        self.object_shapes: typing.List[pymunk.Shape] = []
        self.object_bodies: typing.List[pymunk.Body] = []
        self.objects_metadata = []
        self.handler.post_solve = self.post_object_goal_collision_callback
        self.handler.data["objects"] = self.object_bodies
        for _ in range(num_objects):
            (
                object_body,
                object_shape,
                object_metadata,
                object_state,
            ) = self.define_arena_object(
                (
                    self.wall_thickness_x + np.random.uniform(0, 0.7 * (self.width - 2 * self.wall_thickness_x)),
                    np.random.uniform(
                        self.wall_thickness_y,
                        self.length - self.wall_thickness_y),
                )
            )
            self.object_shapes.append(object_shape)
            self.object_bodies.append(object_body)
            self.objects_metadata.append(object_metadata)
            self.objects_states.append(object_state)
            self.space.add(object_body, object_shape)
            self.handler = self.space.add_collision_handler(
                self.collision_types["objects"], self.collision_types["agent"]
            )

    def define_brownian_distractor(
            self, num_distractors,
            initial_position: typing.Tuple[int, int] = None):
        
        self.distractor_shapes: typing.List[pymunk.Shape] = []
        self.distractor_bodies: typing.List[pymunk.Body] = []
        self.handler.data["distractors"] = self.distractor_bodies
        for _ in range(num_distractors):
            distractor_body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
            distractor_body.mass = 10
            distractor_body.moment = 1e6
            self.distractor_radius = config.DISTRACTOR_RADIUS
            vs = [(-30, 0), (0, 15), (10, 0), (0, -15)]
            distractor_shape = pymunk.Poly(
                    distractor_body, vs)
            distractor_shape.color = pygame.Color("rosybrown")
            distractor_shape.friction = 0.5
            distractor_shape.collision_type = self.collision_types["agent"]
            distractor_shape.density = 0.1
            if initial_position is None:
                game_width = self.width - 2 * (
                    self.wall_thickness_x + self.distractor_radius
                )
                game_height = self.length - 2 * (
                    self.wall_thickness_y + self.distractor_radius
                )
                distractor_body.position = (
                    np.random.uniform(-game_width / 2, game_width / 2)
                    + (self.width // 2),
                    np.random.uniform(-game_height / 2, game_height / 2)
                    + (self.length // 2),
                )
            else:
                distractor_body.position = initial_position
            self.space.add(distractor_body, distractor_shape)
            self.handler.data["distractor"] = distractor_body
            self.distractor_shapes.append(distractor_shape)
            self.distractor_bodies.append(distractor_body)

    def create_goal_zone(self):
        self.goal = pygame.draw.rect(
            self.screen,
            (0, 130, 65),
            pygame.Rect(
                [self.goal_x, self.goal_y, self.goal_width, self.goal_height]
            ),
        )

    def add_collision_handlers(self):
        self.handler = self.space.add_collision_handler(
            self.collision_types["walls"], self.collision_types["agent"]
        )
        self.handler = self.space.add_collision_handler(
            self.collision_types["walls"], self.collision_types["objects"]
        )
        self.handler = self.space.add_collision_handler(
            self.collision_types["objects"], self.collision_types["agent"]
        )
        self.handler = self.space.add_collision_handler(
            self.collision_types["distractors"], self.collision_types["agent"]
        )
        self.handler = self.space.add_collision_handler(
            self.collision_types["distractors"],
            self.collision_types["objects"]
        )

    def define_env_camera(self):
        camera_type = config.CAMERA_TYPE
        box_width = 2 * self.agent_view_size
        if camera_type == "agent_centric":
            camera_x = np.clip(
                self.agent_body.position.x - self.agent_view_size,
                0,
                self.width - box_width,
            )
            camera_y = np.clip(
                self.agent_body.position.y - self.agent_view_size,
                0,
                self.length - box_width,
            )
            self.camera_rect = [
                camera_x,
                camera_y,
                box_width,
                box_width,
            ]
        elif camera_type == "global":
            self.camera_rect = [
                0,
                0,
                config.ARENA_WIDTH,
                config.ARENA_LENGTH,
            ]
        camera_surface = pygame.Surface(
            pygame.Rect(self.camera_rect).size, pygame.SRCALPHA
        )
        pygame.Surface.set_alpha(camera_surface, 100)
        pygame.draw.rect(
            camera_surface,
            (128, 128, 128),
            camera_surface.get_rect(),
        )
        self.screen.blit(camera_surface, self.camera_rect)

    def check_done(self):
        for object_body in self.object_bodies:
            object_body.velocity_func = self.limit_velocity
        for object_body, object_shape in zip(
            self.object_bodies, self.object_shapes
        ):
            vanishing_x = self.goal_x + 5
            if object_body.position.x >= vanishing_x:
                self.space.remove(object_body, object_shape)
                self.object_bodies.remove(object_body)
                self.object_shapes.remove(object_shape)
                self.dones = self.num_objects - len(self.object_bodies)
                if self.debug:
                    print(len(self.object_bodies))

    def update(self):
        self.check_done()
        self.screen.fill(pygame.Color("black"))
        self.create_goal_zone()
        self.space.debug_draw(self.draw_options)

        if self.debug:
            self.screen.blit(
                self.font.render(
                    f"fps: {self.clock.get_fps()}, steps: {self.steps}, overide: ({self.overide}, {self.overide_key})",
                    True,
                    pygame.Color("white"),
                ),
                (0, 0),
            )
            self.screen.blit(
                self.font.render(
                    "Agent kicks objects into the green drop-off zone",
                    True,
                    pygame.Color("darkgrey"),
                ),
                (5, self.length - 35),
            )
            self.screen.blit(
                self.font.render(
                    "Press ESC or Q to quit", True, pygame.Color("darkgrey")
                ),
                (5, self.length - 20),
            )
        self.define_env_camera()
        pygame.display.flip()

        self.overide = False
        self.overide_key = None

        dt = 1.0 / self.fps
        self.space.step(dt)

    def apply_action(self, action=None):
        if action is None:
            action = self.sample_action()
        if self.user_overide and self.overide and self.overide_key is not None:
            print("Action Overide")
            action = (math.pi / 2.0 * self.overide_key, self.max_acc)
            self.debug and print("Angle: ", action[0] * 180 / math.pi)
        if self._last_angle is not None:
            action = (
                action[0] * (1 - self.angle_momentum)
                + self._last_angle * self.angle_momentum,
                action[1],
            )
        self._last_angle = action[0]
        for _ in range(self.frameskip):
            self.agent_body.apply_force_at_world_point(
                (
                    math.cos(action[0]) * action[1],
                    math.sin(action[0]) * action[1],
                ),
                (self.agent_body.position.x, self.agent_body.position.x),
            )
            if self.agent_body.position.x + self.agent_radius > self.goal_x:
                self.agent_body.position = (
                    self.goal_x - self.agent_radius,
                    self.agent_body.position.y,
                )
                self.agent_body.velocity = (
                    -self.goal_bounce_factor * self.agent_body.velocity
                )
        self.steps += 1

    def move_distractor(self, num_distractors):
        for distractor in range(num_distractors):
            if self.distractor_steps[distractor] == 0:
                distractor_action = self.sample_action()
                self.distractor_actions[distractor] = distractor_action
                self.distractor_steps[distractor] = np.random.randint(2, 10)

        actions = self.distractor_actions
        self.distractor_steps[:] = [
            distractor_steps - 1 for distractor_steps in self.distractor_steps]

        for _ in range(self.frameskip):
            for distractor_body, action in zip(
                self.distractor_bodies, actions
            ):
                distractor_body.apply_force_at_world_point(
                    (
                        math.cos(action[0]) * (
                            action[1] * config.DISTRACTOR_ACTION_SCALE),
                        math.sin(action[0]) * (
                            action[1] * config.DISTRACTOR_ACTION_SCALE),
                    ),
                    (
                        distractor_body.position.x,
                        distractor_body.position.y,
                    ),
                )
                if distractor_body.position.x + self.distractor_radius > \
                        self.goal_x:
                    distractor_body.position = (
                        self.goal_x - self.distractor_radius,
                        distractor_body.position.y,
                    )
                    distractor_body.velocity = (
                        -self.goal_bounce_factor * distractor_body.velocity
                    )

    def get_observation(self):
        subscreen = self.screen.subsurface(self.camera_rect)
        env_image = pygame.surfarray.array3d(subscreen)
        env_image = env_image.swapaxes(0, 1).astype(np.uint8)
        return env_image

    def get_states(self, state_type="absolute"):
        object_positions: typing.List = []
        object_angles: typing.List = []
        object_velocities: typing.List = []
        object_angular_velocities: typing.List = []

        # update the agent states in the agent metadata
        self.agent_state.position = np.asarray(self.agent_body.position)
        self.agent_state.angle = np.asarray(self.agent_body.angle)
        self.agent_state.velocity = np.asarray(self.agent_body.velocity)
        self.agent_state.angular_velocity = np.asarray(
            self.agent_body.angular_velocity
        )

        # update the objects' states in the metadata of all the objects
        for object_metadata, object_state in zip(
            self.objects_metadata, self.objects_states
        ):
            arena_object = object_metadata.object_body
            object_state.position = np.asarray(arena_object.position)
            object_state.angle = np.asarray(arena_object.angle)
            object_state.velocity = np.asarray(arena_object.velocity)
            object_state.angular_velocity = np.asarray(
                arena_object.angular_velocity
            )
            if state_type == "absolute":
                object_position = object_state.position
                object_angle = object_state.angle
                object_velocity = object_state.velocity
                object_angular_velocity = object_state.angular_velocity

                agent_position = self.agent_state.position
                agent_angle = self.agent_state.angle
                agent_velocity = self.agent_state.velocity
                agent_angular_velocity = self.agent_state.angular_velocity

            elif state_type == "relative":
                object_position = (
                    object_state.position - self.agent_state.position
                )
                object_angle = object_state.angle - self.agent_state.angle
                object_velocity = (
                    object_state.velocity - self.agent_state.velocity
                )
                object_angular_velocity = (
                    object_state.angular_velocity
                    - self.agent_state.angular_velocity
                )

                agent_position = (
                    self.agent_state.position - self.agent_state.position
                )
                agent_angle = self.agent_state.angle - self.agent_state.angle
                agent_velocity = (
                    self.agent_state.velocity - self.agent_state.velocity
                )
                agent_angular_velocity = (
                    self.agent_state.angular_velocity
                    - self.agent_state.angular_velocity
                )

            object_positions.append(object_position)
            object_angles.append(object_angle)
            object_velocities.append(object_velocity)
            object_angular_velocities.append(object_angular_velocity)

            return {
                "agent_position": agent_position,
                "agent_angle": agent_angle,
                "agent_velocity": agent_velocity,
                "agent_angular_velocity": agent_angular_velocity,
                "object_positions": object_positions,
                "object_angles": object_angles,
                "object_velocities": object_velocities,
                "object_angular_velocities": object_angular_velocities,
            }

    def check_running_status(self):
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and (event.key in [pygame.K_ESCAPE, pygame.K_q])
            ):
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pygame.image.save(self.screen, "balls.png")
                elif self.user_overide:
                    if event.key == pygame.K_d:
                        self.overide = True
                        self.overide_key = 0
                        print("D Pressed")
                    elif event.key == pygame.K_w:
                        self.overide = True
                        self.overide_key = 3
                        print("W Pressed")
                    elif event.key == pygame.K_a:
                        self.overide = True
                        self.overide_key = 2
                        print("A Pressed")
                    elif event.key == pygame.K_s:
                        self.overide = True
                        self.overide_key = 1
                        print("S Pressed")

    def reset_num_objects_and_distractors(self):
        if self.variable_num_objects:
            self.num_objects = secrets.choice(range(1, self.max_objects + 1))
        else:
            self.num_objects = self.max_objects
        if self.variable_num_distractors:
            self.num_distractors = secrets.choice(
                range(1, self.max_distractors + 1))
        else:
            self.num_distractors = self.max_distractors

    def reset_entities(self):
        if self.random_start:
            self.create_agent()
            if self.add_brownian_distractor:
                self.define_brownian_distractor(self.num_distractors)
        else:
            self.create_agent((self.width / 2, self.height / 2))
        
        self.create_arena_objects(self.num_objects)
        if self.variable_goal_position:
            self.goal_y = self.float_generator.uniform(
                self.wall_thickness_y,
                self.length - self.wall_thickness_y - self.goal_height)
            self.goal.y = self.goal_y

    def reset_state(self):
        self._last_angle = None
        self.distractor_steps = [0] * self.num_distractors
        self.distractor_actions = [(0.0, 0.0)] * self.num_distractors
        self.dones = 0
        self.steps = 0
        self.running = True

    def clear_arena(self):
        self.reset_state()
        for object_body, object_shape in zip(
            self.object_bodies, self.object_shapes
        ):
            self.space.remove(object_body, object_shape)
        self.object_shapes.clear()
        self.object_bodies.clear()
        self.space.remove(self.agent_body, self.agent_shape)
        if self.add_brownian_distractor:
            for distractor_body, distractor_shape in zip(
                self.distractor_bodies, self.distractor_shapes
            ):
                self.space.remove(distractor_body, distractor_shape)
            self.distractor_shapes.clear()
            self.distractor_bodies.clear()

    def close(self):
        self.running = False
        self.reset_state()
        for event in pygame.event.get():
            exit()

    @staticmethod
    def limit_velocity(body, gravity, damping, dt):
        # to prevent tunneling http://www.pymunk.org/en/latest/overview.html
        max_velocity = config.MAX_VEL
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        if body.velocity.length > max_velocity:
            body.velocity = body.velocity * 0.99

    @staticmethod
    def post_object_goal_collision_callback(arbiter, space, data):
        pass

    def sample_action(self):
        force = self.max_acc * np.random.random_sample()
        # sample angles between 0 and 360
        angle = random.randrange(0, 360) * math.pi / 180
        return (angle, force)

    def __init_graphic_backbone(self):
        # pygame.init() # pygame.mixer keeps throwing errors on ComputeCanada
        if self.render_game:
            self.screen = pygame.display.set_mode((self.width, self.length))
        else:
            self.screen = pygame.display.set_mode(
                (self.width, self.length), pygame.HIDDEN
            )
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont("Arial", 16)

    def __init_physical_world(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0.7
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
