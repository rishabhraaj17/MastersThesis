from enum import Enum


class SDDVideoClasses(Enum):
    BOOKSTORE = "bookstore"
    COUPA = "coupa"
    DEATH_CIRCLE = "deathCircle"
    GATES = "gates"
    HYANG = "hyang"
    LITTLE = "little"
    NEXUS = "nexus"
    QUAD = "quad"


class ObjectClasses(Enum):
    PEDESTRIAN = "Pedestrian"
    BUS = "Bus"
    BIKER = "Biker"
    CAR = "Car"
    SKATER = "Skater"
    CART = "Cart"


OBJECT_CLASS_COLOR_MAPPING = {
    ObjectClasses.PEDESTRIAN: 'b',
    ObjectClasses.BUS: 'g',
    ObjectClasses.BIKER: 'black',
    ObjectClasses.CAR: 'white',
    ObjectClasses.SKATER: 'pink',
    ObjectClasses.CART: 'yellow'
}

ANNOTATION_COLUMNS = ["track_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "label"]

VIDEO_FILE_NAME = "video.mov"
ANNOTATION_FILE_NAME = "annotations.txt"
