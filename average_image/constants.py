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


class SDDVideoDatasets(Enum):
    BOOKSTORE = "Bookstore"
    COUPA = "Coupa"
    DEATH_CIRCLE = "deathCircle"
    GATES = "Gates"
    HYANG = "Hyang"
    LITTLE = "Little"
    NEXUS = "Nexus"
    QUAD = "Quad"


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
    ObjectClasses.CAR: 'purple',
    ObjectClasses.SKATER: 'pink',
    ObjectClasses.CART: 'yellow'
}

ANNOTATION_COLUMNS = ["track_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "label"]

VIDEO_FILE_NAME = "video.mov"
ANNOTATION_FILE_NAME = "annotations.txt"

COLORS = [[252, 69, 3], [3, 3, 252], [3, 252, 57], [29, 31, 29], [156, 20, 219], [176, 170, 9], [62, 64, 138],
          [201, 127, 167], [7, 41, 97], [138, 182, 255], [133, 201, 171], [21, 59, 42], [255, 182, 13], [219, 160, 121],
          [109, 122, 140], [237, 197, 201], [120, 120, 122], [247, 82, 98], [113, 64, 148], [255, 0, 208],
          [0, 255, 204]]

COLORS2 = [[230, 41, 28], [240, 184, 180], [110, 61, 57], [240, 196, 38], [110, 86, 3], [200, 232, 21], [69, 79, 16],
           [5, 237, 210], [73, 230, 16], [26, 138, 161], [125, 0, 25], [126, 24, 163], [8, 8, 13], [171, 230, 255],
           [255, 255, 255], [119, 123, 125], [138, 184, 145], [143, 160, 219], [161, 100, 121], [250, 242, 167]]

COLORS3 = []
for color in COLORS2:
    COLORS3.append([color[0]/255, color[1]/255, color[2]/255])
