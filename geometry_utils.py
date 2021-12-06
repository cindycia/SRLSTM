import math
import numpy as np
from typing import List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class AgentGeometry:
    def __init__(self, _len_ref_to_front, _len_ref_to_side, _len_ref_to_back):
        self.len_ref_to_front = _len_ref_to_front
        self.len_ref_to_side = _len_ref_to_side
        self.len_ref_to_back = _len_ref_to_back


DEFAULT_AGENT_PARAMS = dict()
ped_tag = "People"
ped_len_ref_to_front = 0.1
ped_len_ref_to_side = 0.22
ped_len_ref_to_back = 0.1
DEFAULT_AGENT_PARAMS[ped_tag] = AgentGeometry(
    ped_len_ref_to_front, ped_len_ref_to_side, ped_len_ref_to_back)
scooter_tag = "Scooter"
scooter_len_ref_to_front = 1.25
scooter_len_ref_to_side = 0.6
scooter_len_ref_to_back = 0.4
DEFAULT_AGENT_PARAMS[scooter_tag] = AgentGeometry(
    scooter_len_ref_to_front, scooter_len_ref_to_side, scooter_len_ref_to_back)
car_tag = "Car"
car_len_ref_to_front = 3.8
car_len_ref_to_side = 1.0
car_len_ref_to_back = 1.0
DEFAULT_AGENT_PARAMS[scooter_tag] = AgentGeometry(
    car_len_ref_to_front, car_len_ref_to_side, car_len_ref_to_back)
van_tag = "Van"
van_len_ref_to_front = 4.2
van_len_ref_to_side = 1.1
van_len_ref_to_back = 1.1
DEFAULT_AGENT_PARAMS[van_tag] = AgentGeometry(
    van_len_ref_to_front, van_len_ref_to_side, van_len_ref_to_back)
bus_tag = "Bus"
bus_len_ref_to_front = 4.5
bus_len_ref_to_side = 1.5
bus_len_ref_to_back = 1.5
DEFAULT_AGENT_PARAMS[bus_tag] = AgentGeometry(
    bus_len_ref_to_front, bus_len_ref_to_side, bus_len_ref_to_back)
jeep_tag = "Jeep"
jeep_len_ref_to_front = 4.2
jeep_len_ref_to_side = 1.1
jeep_len_ref_to_back = 1.1
DEFAULT_AGENT_PARAMS[jeep_tag] = AgentGeometry(
    jeep_len_ref_to_front, jeep_len_ref_to_side, jeep_len_ref_to_back)
bicycle_tag = "Bicycle"
bicycle_len_ref_to_front = 1.25
bicycle_len_ref_to_side = 0.5
bicycle_len_ref_to_back = 0.4
DEFAULT_AGENT_PARAMS[bicycle_tag] = AgentGeometry(
    bicycle_len_ref_to_front, bicycle_len_ref_to_side, bicycle_len_ref_to_back)
electric_tricycle_tag = "Electric_Tricycle"
electric_tricycle_len_ref_to_front = 1.25
electric_tricycle_len_ref_to_side = 0.8
electric_tricycle_len_ref_to_back = 0.4
DEFAULT_AGENT_PARAMS[electric_tricycle_tag] = AgentGeometry(
    electric_tricycle_len_ref_to_front, electric_tricycle_len_ref_to_side, electric_tricycle_len_ref_to_back)
gyro_scooter_tag = "Gyro_Scooter"
gyro_scooter_len_ref_to_front = 0.25
gyro_scooter_len_ref_to_side = 0.3
gyro_scooter_len_ref_to_back = 0.25
DEFAULT_AGENT_PARAMS[gyro_scooter_tag] = AgentGeometry(
    gyro_scooter_len_ref_to_front, gyro_scooter_len_ref_to_side, gyro_scooter_len_ref_to_back)


def rotate_cw(xy, angle):
    # rotate vector(x, y) counterclockwise by the given angle (angle in radians)
    x = xy[0] * math.cos(angle) - xy[1] * math.sin(angle)
    y = xy[0] * math.sin(angle) + xy[1] * math.cos(angle)
    return np.asarray([x, y])


def get_geometry(tag: str):
    return DEFAULT_AGENT_PARAMS[tag]


def get_bounding_box_corners(xy:np.ndarray, heading:np.ndarray, ped_type:str):
    heading_rotate_90_clockwise = rotate_cw(xy, -math.pi / 2)
    g = get_geometry(ped_type)
    ref_to_front = g.len_ref_to_front
    ref_to_side = g.len_ref_to_side
    ref_to_back = g.len_ref_to_back
    front_right = xy + ref_to_front * heading + ref_to_side * heading_rotate_90_clockwise
    front_left = xy + ref_to_front * heading - ref_to_side * heading_rotate_90_clockwise
    back_right = xy - ref_to_back * heading + ref_to_side * heading_rotate_90_clockwise
    back_left = xy - ref_to_back * heading - ref_to_side * heading_rotate_90_clockwise

    corners = [back_left, back_right, front_right, front_left]
    return corners


def point_on_left_side(edge_start: np.ndarray, edge_end: np.ndarray, p: np.ndarray):
    edge = edge_end - edge_start
    dir = p - edge_start
    print(f'np.outer(edge, dir)={np.outer(edge, dir)}')
    return np.outer(edge, dir) > 0


def in_rectangle(p: np.ndarray, rect: List[np.ndarray]):
    rect.append(rect[0])  # to form a closed polygon
    for i in range(len(rect) - 1):
        if not point_on_left_side(rect[i], rect[i + 1], p):
            return False
        return True


def in_collision(rect_1: List[np.ndarray], rect_2: List[np.ndarray]):
    # check whether there exists one vertex of rect_1 inside rect_2
    for p1 in rect_1:
        point = Point(p1[0], p1[1])
        polygon = Polygon(rect_2)
        if polygon.contains(point):  # if in_rectangle(p1, rect_2):
            return True
    # check whether there exists one vertex of rect_2 inside rect_1
    for p2 in rect_2:
        point = Point(p2[0], p2[1])
        polygon = Polygon(rect_1)
        if polygon.contains(point):  # if in_rectangle(p1, rect_2):
            return True
    return False


def collision_check(xy1, heading1, ped_type1, xy2, heading2, ped_type2):
    # xy1, xy2, heading1, heading2 = xy1.cpu().detach().numpy(), xy2.cpu().detach().numpy(),\
                                    # heading1.cpu().detach().numpy(), heading2.cpu().detach().numpy()

    heading1 = heading1 / np.linalg.norm(heading1)
    heading2 = heading2 / np.linalg.norm(heading2)

    rect1 = get_bounding_box_corners(xy1, heading1, ped_type1)
    rect2 = get_bounding_box_corners(xy2, heading2, ped_type2)
    return in_collision(rect1, rect2)
