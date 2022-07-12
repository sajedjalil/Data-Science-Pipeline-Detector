import heapq
import itertools
import math
import os
import json
import time
import logging
import csv
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union, Iterable

import numpy as np
from matplotlib import colors, pyplot as plt

MAX_NLL = 9.5
SELECTOR_TARGET_REQUIRES_ENTITY_PENALTY = 1.
EMPTY_SELECT_PENALTY = 1.
MISMATCHED_OUTPUT_PENALTY = 10.
SELECTING_SAME_PROPERTY_BONUS = 1.
SELECT_NOT_0_NLL = np.log(2.)
NEW_COLOR_BONUS = 1.
SELECTOR_MAX_NLL_CORRECTION = np.log(2.)
DIM_CHANGE_PENALTY = np.log(10.)
SELECTOR_PAIR_PENALTY = np.log(2)
NONCONSTANT_BOOL_PENALTY = 2.
PROPERTY_CONSTRUCTION_COST = np.log(2)
SELECTOR_CONSTRUCTION_COST = np.log(2)
POINT_PROP_COST = 2.
ALLOW_COMPOSITE_SELECTORS = False
USE_TRANSFORMER_NAMES = True
MAKE_PROPERTY_LIST = False
MAX_SMALL_TIME = 250.
MAX_LARGE_TIME = 250.
MAX_PREDICTORS = 300_000
MAX_PARTIAL_PREDICTORS = 3
CONSTANTS = (f"MAX_NLL = {MAX_NLL}",
             SELECTOR_TARGET_REQUIRES_ENTITY_PENALTY,
             EMPTY_SELECT_PENALTY,
             MISMATCHED_OUTPUT_PENALTY,
             ALLOW_COMPOSITE_SELECTORS,
             SELECTING_SAME_PROPERTY_BONUS,
             SELECT_NOT_0_NLL,
             NEW_COLOR_BONUS,
             SELECTOR_MAX_NLL_CORRECTION,
             DIM_CHANGE_PENALTY,
             SELECTOR_PAIR_PENALTY,
             PROPERTY_CONSTRUCTION_COST,
             POINT_PROP_COST
             )
TYPES = frozenset(
    {'x_length', 'y_length', 'x_coordinate', 'y_coordinate', 'quantity', 'shape', 'uncolored_shape', 'vector', 'color',
     'bool', 'point', 'line', 'group'})
STRAIGHT_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
ALL_DIRECTIONS = tuple(itertools.product([1, 0, -1], [1, 0, -1]))
LINE_MAPPING = {0: 'horizontal', 1: 'vertical', 0.5: 'forward diagonal', -0.5: 'backward diagonal'}
FORWARD_DIAGONAL = 0.5
BACKWARD_DIAGONAL = -0.5

logging.basicConfig(filename='property_select_transform.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')

expense_logger = logging.getLogger('expense')
file_handler = logging.FileHandler('expense.log')
expense_logger.addHandler(file_handler)
expense_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
file_handler.setFormatter(formatter)

CONSTANT_STRINGS = tuple((str(constant) for constant in CONSTANTS))
expense_logger.info("New test: " + ", ".join(CONSTANT_STRINGS))


class Entity:
    """
    Properties:
        colors = subset of range(10)
        group = D4, Z4, Z2, Z2xZ2
    """
    all_entities = {}
    count = 0
    counts_to_entities = []

    def __init__(self, positions: dict, grid: tuple):
        self.positions = positions
        self.grid = grid
        self.shape_ = None
        self.colors_ = None
        self.uncolored_shape_ = None
        self.freeze_ = None
        self.count = self.__class__.count
        self.__class__.count += 1
        self.__class__.counts_to_entities.append(self.count)

    def __repr__(self):
        return f'Entity({self.positions})'

    def __eq__(self, other):
        return self.positions == other.positions and self.grid == other.grid

    def shape(self):
        if self.shape_ is None:
            shape_dict = {}
            center_0, center_1 = self.center(0), self.center(1)
            for position, color in self.positions.items():
                shape_dict[(position[0] - center_0, position[1] - center_1)] = color
            self.shape_ = frozenset(shape_dict.items())
        return self.shape_

    def uncolored_shape(self):
        if self.uncolored_shape_ is None:
            shape_set = set()
            center_0, center_1 = self.center(0), self.center(1)
            for position, color in self.positions.items():
                shape_set.add((position[0] - center_0, position[1] - center_1))
            self.uncolored_shape_ = frozenset(shape_set)
        return self.uncolored_shape_

    def colorblind_equals(self, other):
        return set(self.positions.keys()) == set(other.positions.keys())

    # def shape_equal(self, other):
    #     return set(self.positions.keys()) == set(other.positions.keys())

    def copy(self):
        return self.__class__(deepcopy(self.positions), self.grid)

    def change_grid(self, new_grid):
        new_positions = self.positions.copy()
        if new_grid and self.grid and (len(new_grid) != len(self.grid) or len(new_grid[0]) != len(self.grid[0])):
            for position, _ in self.positions.items():
                if position[0] >= len(new_grid) or position[1] >= len(new_grid[0]):
                    del new_positions[position]
        if not new_positions:
            return None
        return self.__class__.make(new_positions, new_grid)

    def colors(self):
        if self.colors_ is None:
            self.colors_ = frozenset(self.positions.values())
        return self.colors_

    def min_coord(self, axis=0):
        return int(min([key[axis] for key in self.positions.keys()]))

    def max_coord(self, axis=0):
        return int(max([key[axis] for key in self.positions.keys()]))

    def center(self, axis=0):
        output = (self.min_coord(axis) + self.max_coord(axis)) / 2.
        # if output.is_integer():
        #     output = int((self.min_coord(axis) + self.max_coord(axis))/2.)
        # else:
        #     output = float(output)
        return float(output)

    def num_points(self):
        return len(self.positions)

    def symmetry_group(self):
        pass

    def display(self):
        new_grid = np.array(self.grid)
        for i, j in itertools.product(range(new_grid.shape[0]), range(new_grid.shape[1])):
            if (i, j) not in set(self.positions):
                new_grid[i, j] = 10
        display_case(new_grid)

    def freeze(self):
        if self.freeze_ is None:
            self.freeze_ = frozenset(self.positions.items())
        return self.freeze_

    def is_a_rectangle(self):
        if len(self.positions) < 6:
            return False
        min_0, max_0 = self.min_coord(0), self.max_coord(0)
        min_1, max_1 = self.min_coord(1), self.max_coord(1)
        rectangle = {(sides_0, x) for x in range(min_1, max_1 + 1) for sides_0 in (min_0, max_0)}
        rectangle |= {(y, sides_1) for y in range(min_0 + 1, max_0) for sides_1 in (min_1, max_1)}
        return rectangle.issubset(self.positions.keys())

    def is_a_square(self):
        min_0, max_0 = self.min_coord(0), self.max_coord(0)
        min_1, max_1 = self.min_coord(1), self.max_coord(1)
        return max_0 - min_0 == max_1 - max_1 and self.is_a_rectangle()

    def is_a_line(self):
        return len(self.positions) > 2 and \
               (len({x for (x, y) in self.positions.keys()}) == 1 or len({y for (x, y) in self.positions.keys()}) == 1)

    @classmethod
    def make(cls, positions: dict, grid: tuple):
        key = frozenset(positions.items()), grid
        if key not in cls.all_entities:
            cls.all_entities[key] = cls(positions, grid)
        return cls.all_entities[key]

    @classmethod
    def reset(cls):
        cls.count = 0
        cls.all_entities = {}
        cls.counts_to_entities = []

    @staticmethod
    def counts(entities):
        return frozenset((entity.count for entity in entities))

    @staticmethod
    def freeze_entities(entities):
        return frozenset((entity.freeze() for entity in entities))

    @staticmethod
    def shapes(entities):
        return Counter((entity.shape() for entity in entities))

    @staticmethod
    def uncolored_shapes(entities):
        return Counter((entity.uncolored_shape() for entity in entities))


class EntityFinder:
    def __init__(self, base_finder: callable, nll: float = 0., name: str = 'find all entities'):
        self.base_finder = base_finder
        self.nll = nll
        self.name = name
        self.cache = {}
        self.grid_distance_cache = {}

    def __call__(self, grid):
        if grid in self.cache:
            return self.cache[grid]
        out = self.base_finder(grid)
        self.cache[grid] = out
        return out

    def __str__(self):
        return self.name

    def grid_distance(self, grid1: tuple, grid2: tuple, shape_only=False) -> float:
        if not grid1 or not grid2:
            return float('inf')
        if (grid1, grid2) in self.grid_distance_cache:
            return self.grid_distance_cache[(grid1, grid2)]
        # print(f'len(cache) = {len(self.cache)}')
        entities1, entities2 = self(grid1), self(grid2)
        if shape_only:
            dist = 0
            for (i, entity1), (j, entity2) in itertools.product(enumerate(entities1), enumerate(entities2)):
                if i <= j and entity1.shape() == entity2.shape():
                    dist -= entity1.num_points()
        else:
            arr1, arr2 = np.array(grid1), np.array(grid2)
            if arr1.shape != arr2.shape:
                dist = max(arr1.shape[0] * arr1.shape[1], arr2.shape[0] * arr2.shape[1])
                # dist -= sum(np.nditer(arr1 == 0))
                # dist -= sum(np.nditer(arr2 == 0))
                # dist = np.abs(arr1.shape[0] - arr2.shape[0]) * np.abs(arr1.shape[1] - arr2.shape[1])
                # min_y = min(arr1.shape[0], arr2.shape[0])
                # min_x = min(arr1.shape[1], arr2.shape[1])
                # dist = sum(np.nditer(arr1[:min_y, :min_x] != arr2[:min_y, :min_x]))
                # dist += sum([arr.shape[0]*arr.shape[1] - min_y*min_x for arr in [arr1, arr2]])
                # dist = 1_000_000
                set1, set2 = {entity1.shape() for entity1 in entities1}, {entity2.shape() for entity2 in entities2}

                for shape in set1 & set2:
                    dist -= 0.5 * len(shape)
            else:
                dist = sum(np.nditer(arr1 != arr2))
                completely_the_sames = {frozenset(entity1.positions.items()) for entity1 in entities1} & {
                    frozenset(entity2.positions.items()) for entity2 in entities2}
                completely_the_sames = {frozenset((key for key, value in positions)) for positions in
                                        completely_the_sames}
                set1, set2 = {frozenset(entity1.positions.keys()) for entity1 in entities1}, {
                    frozenset(entity2.positions.keys()) for entity2 in entities2}
                for positions in (set1 & set2) - completely_the_sames:
                    dist -= 0.5 * len(positions)
        self.grid_distance_cache[(grid1, grid2)] = dist
        return dist

    def compose(self, selector: callable, composite: bool = False):
        return self.__class__(lambda grid: selector.select(self(grid), composite), self.nll + selector.nll,
                              name=f'{self.name} where {selector.name} {"as composite" if composite else ""}')

    def reset(self):
        self.cache = {}

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.name < other.name


@dataclass
class OrdinalProperty:
    base_property: callable
    nll: float
    name: str
    input_types: frozenset = frozenset({})

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.base_property(*args, **kwargs)

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.name < other.name


def nth_ordered(lst, n=0, use_max=True):
    if None in lst:
        return None
    ordered_list = list(sorted(lst, reverse=True))
    # Filter out repeats
    ordered_list = [element for i, element in enumerate(ordered_list) if i == 0 or element != ordered_list[i - 1]]
    if len(ordered_list) > n:
        return ordered_list[n] if use_max else ordered_list[-1 - n]
    else:
        return None


def pick_the_unique_value(lst):
    if None in lst:
        return None
    value_count = Counter(lst)
    output = None
    for value, count in value_count.items():
        if count == 1:
            if output is None:
                output = value
            else:
                return None
    return output


def my_sum(lst: list, n: int):
    if None in lst or lst == []:
        return None
    try:
        return sum(lst)
    except TypeError:
        return None


SINGLE_VALUE = OrdinalProperty(lambda x: next(iter(x)) if len(x) == 1 else None,
                               nll=0,
                               name=f'the single element',
                               input_types=TYPES)
ORDINAL_PROPERTIES = [SINGLE_VALUE]


# 45

# 46 - 51
def ordinal(n):
    """Obtained from Stack overflow via user Gareth: https://codegolf.stackexchange.com/users/737/gareth"""
    return "%d%s" % (n, "tsnrhtdd"[(math.floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10::4])


ORDINAL_PROPERTIES.extend([OrdinalProperty(lambda x, n=n: nth_ordered(x, n, use_max=True),
                                           nll=n * np.log(2),
                                           name=f'take the {ordinal(n + 1)} largest',
                                           input_types=frozenset(
                                               {'x_length', 'y_length', 'x_coordinate', 'y_coordinate', 'quantity'}))
                           for n in range(6)])
# 52 - 57
ORDINAL_PROPERTIES.extend([OrdinalProperty(lambda x, n=n: nth_ordered(x, n, use_max=False),
                                           nll=n * np.log(2),
                                           name=f'take the {ordinal(n + 1)} smallest',
                                           input_types=frozenset(
                                               {'x_length', 'y_length', 'x_coordinate', 'y_coordinate', 'quantity'}))
                           for n in range(6)])

# 57.5
ORDINAL_PROPERTIES.append(OrdinalProperty(lambda x: pick_the_unique_value(x),
                                          nll=np.log(2),
                                          name=f'take the value that is unique',
                                          input_types=TYPES))

# 57.75
ORDINAL_PROPERTIES.append(OrdinalProperty(lambda x: any(x),
                                          nll=np.log(2),
                                          name=f'at least one is true of',
                                          input_types=frozenset({'bool'})))

ORDINAL_PROPERTIES.sort()


@dataclass
class PropertyInput:
    """
    A dataclass to help with standardization and readability
    """
    entity: Optional[Entity]
    entities: Optional[list]
    grid: Optional[tuple]

    def __init__(self, entity: Optional[Entity] = None,
                 entities: Optional[list] = None,
                 grid: Optional[tuple] = None):
        assert entity is None or isinstance(entity, Entity)
        self.entity = entity
        self.entities = entities
        self.grid = grid


class Property:
    count = 0
    if MAKE_PROPERTY_LIST:
        property_list = []
    of_type = {typ: [] for typ in TYPES}
    signatures_to_output_types = defaultdict(set)
    signatures_to_best_estimator = {}

    def __init__(self, prop: callable, nll: float, entity_finder: callable,
                 count=None, name=None, output_types: Optional[frozenset] = None, select=lambda x: x,
                 requires_entity=False,
                 is_constant=False, register=False, associated_objects=None):
        """

        :param prop: (entity, entities, grid) -> value
        :param nll:
        :param entity_finder:
        :param count:
        :param name:
        :param associated_objects: List of other properties that this property is derived from
        :param output_types: used to make combining different types of data more unlikely. Allowable inputs: 'color',
         'x_coordinate', 'y_coordinate', 'x_length', 'y_length', 'quantity'
        """
        self.prop = prop
        self.nll = nll
        self.entity_finder = entity_finder
        self.select = select
        self.requires_entity = requires_entity
        if name is None:
            name = f'Property({self.prop}, {self.nll}, {self.count}}})'
        self.name = name
        if output_types is None:
            output_types = TYPES
        self.output_types = output_types
        self.is_constant = is_constant
        self.associated_objects = associated_objects if associated_objects is not None else []
        self.cache = {}
        if register:
            self.count = count if count is not None else self.__class__.count
            self.__class__.count += 1
            if MAKE_PROPERTY_LIST:
                self.__class__.property_list.append(self)
            for output_type in output_types:
                if output_type in self.__class__.of_type:
                    self.__class__.of_type[output_type].append(self)
                else:
                    self.__class__.of_type[output_type] = [self]
        else:
            self.count = -1

    def __call__(self, entity: Optional[Entity], grid: Optional[tuple]):
        key = (entity.count if entity is not None else None, grid)
        if key in self.cache:
            return self.cache[key]
        output = self.prop(PropertyInput(entity, self.select(self.entity_finder(grid)), grid))
        self.cache[key] = output
        return output

    def register(self):
        self.count = self.__class__.count
        self.__class__.count += 1
        if MAKE_PROPERTY_LIST:
            self.__class__.property_list.append(self)
        for output_type in self.output_types:
            if output_type in self.__class__.of_type:
                self.__class__.of_type[output_type].append(self)
            else:
                self.__class__.of_type[output_type] = [self]

    def generate_output_signature(self, task):
        inputs = [case['input'] for case in task['train']] + [case['input'] for case in task['test']]
        if not self.requires_entity:
            signature = tuple((self(None, inp) for inp in inputs))
        else:
            signature = tuple(
                (frozenset({(entity.count, self(entity, inp)) for entity in self.entity_finder(inp)}) for inp in
                 inputs))
        return signature

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.count < other.count

    def __repr__(self):
        return f'Property({self.prop}, {self.nll}, {self.count}}})'

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return self.name

    def add_selector(self, selector, ordinal_prop=SINGLE_VALUE, register=False):
        """
        Combines with a selector to generate a unique grid property

        :param selector:
        :param ordinal_prop: A function that takes a set, and returns a single value
        :param register: Determines if the property is automatically added to the list of properties "of_type"
        :return:
        """
        assert self.requires_entity
        new_nll = combine_property_selector_nll(self, selector, ordinal_prop)

        # if not (self.output_types & ordinal_prop.input_types):
        #     new_nll += MISMATCHED_OUTPUT_PENALTY

        def new_base_property(prop_input: PropertyInput):
            set_of_props = {self(entity, prop_input.grid)
                            for entity in selector.select(prop_input.entities)}
            return ordinal_prop(set_of_props)

        return self.__class__(prop=new_base_property,
                              nll=new_nll,
                              entity_finder=self.entity_finder,
                              output_types=self.output_types,
                              name=f'({self.entity_finder.name}) where ({selector.name}) then find the ({ordinal_prop})'
                                   f' of {self.name}',
                              register=register,
                              requires_entity=False,
                              associated_objects=[self])

    def create_distance_property(self, other, register=False, nll_red=0.):
        output_types = set()
        if 'x_coordinate' in self.output_types and 'x_coordinate' in other.output_types:
            output_types.add('x_length')
        if 'y_coordinate' in self.output_types and 'y_coordinate' in other.output_types:
            output_types.add('y_length')
        return self.__class__(prop=lambda x: float(self(x.entity, x.grid) - other(x.entity, x.grid))
        if self(x.entity, x.grid) is not None and other(x.entity, x.grid) is not None else None,
                              nll=self.nll + other.nll - nll_red,
                              entity_finder=self.entity_finder,
                              output_types=frozenset(output_types),
                              name=f'({self}) - ({other})',
                              register=register,
                              requires_entity=self.requires_entity or other.requires_entity)

    def coord_to_line_property(self, typ, register=False, nll_red=0.):
        """
        Creates a constant-type line property out of a coordinate property

        :param typ: 0., 1., 0.5, -0.5 corresponding to horizontal, vertical, forward diagonal, and backward diagonal line
        :param register: Determines if the property is automatically added to the list of properties "of_type"
        :param nll_red: Additional nll adjustment if necessary
        :return: the new property
        """
        return self.__class__(prop=lambda x: (float(self(x.entity, x.grid)), float(typ))
        if self(x.entity, x.grid) is not None else None,
                              nll=self.nll - nll_red + np.log(2),
                              entity_finder=self.entity_finder,
                              output_types=frozenset({'line'}),
                              name=f'the {LINE_MAPPING[typ]} line at position ({self})',
                              register=register,
                              requires_entity=self.requires_entity)

    def validate_and_register(self, task: dict, extra_validation: callable = lambda x: True) -> bool:
        output_signature = self.generate_output_signature(task)
        if self.requires_entity or (None not in output_signature and extra_validation(output_signature)):

            if output_signature in self.__class__.signatures_to_best_estimator:
                combined_nll = union_nlls(self.nll, self.__class__.signatures_to_best_estimator[output_signature].nll)
            else:
                combined_nll = self.nll
            if output_signature not in self.__class__.signatures_to_best_estimator or \
                    self.__class__.signatures_to_best_estimator[output_signature].nll > self.nll:
                self.__class__.signatures_to_best_estimator[output_signature] = self
            self.__class__.signatures_to_best_estimator[output_signature].nll = combined_nll

            if not self.output_types.issubset(
                    self.__class__.signatures_to_output_types[output_signature]):
                self.__class__.signatures_to_output_types[output_signature] |= self.output_types

                self.register()
                return True
        return False

    @classmethod
    def create_point_property(cls, first, second, register=False, nll_red=0.):
        return cls(prop=lambda x: (float(first(x.entity, x.grid)), float(second(x.entity, x.grid)))
        if first(x.entity, x.grid) is not None and second(x.entity, x.grid) is not None else None,
                   nll=first.nll + second.nll - nll_red + POINT_PROP_COST,
                   entity_finder=first.entity_finder,
                   output_types=frozenset({'point'}),
                   name=f'point (({first}), ({second}))',
                   register=register,
                   requires_entity=first.requires_entity or second.requires_entity)

    @classmethod
    def reset(cls):
        cls.count = 0
        cls.property_list = []
        for key, value in cls.of_type.items():
            cls.of_type[key] = []
        cls.signatures_to_output_types = defaultdict(set)
        cls.signatures_to_best_estimator = {}

    @classmethod
    def from_relation_selector(cls, relation, selector, entity_finder,
                               ordinal_property: OrdinalProperty = SINGLE_VALUE,
                               register=True):
        new_nll = combine_relation_selector_nll(relation, selector, ordinal_property)

        def base_property(prop_input: PropertyInput):
            relations = [relation(prop_input.entity, selected_entity) for selected_entity in
                         selector.select(prop_input.entities)]
            return ordinal_property(relations) if ordinal_property(relations) is not None else None

        return cls(prop=base_property,
                   nll=new_nll,
                   entity_finder=entity_finder,
                   output_types=relation.output_types,
                   name=f'({ordinal_property.name}) of ({relation.name}) all entities where ({selector.name})',
                   register=register,
                   requires_entity=True)

    @classmethod
    def from_entity_prop_and_ordinal(cls, entity_prop, ordinal_property: OrdinalProperty = SINGLE_VALUE):
        nll = entity_prop.nll + ordinal_property.nll
        if not (entity_prop.output_types & ordinal_property.input_types):
            nll += MISMATCHED_OUTPUT_PENALTY

        def base_property(prop_input):
            prop_values = {entity_prop(entity, prop_input.grid) for entity in prop_input.entities}
            return ordinal_property(prop_values)

        return cls(base_property,
                   nll=nll,
                   entity_finder=entity_prop.entity_finder,
                   name=f'the ({ordinal_property}) of ({entity_prop})',
                   output_types=entity_prop.output_types,
                   requires_entity=False)

    @classmethod
    def xy_length_to_vector(cls, vert_prop, horiz_prop, register=False):
        return cls(lambda x: (
            vert_prop(x.entity, x.grid), horiz_prop(x.entity, x.grid)),
                   nll=combine_move_nll(vert_prop, horiz_prop),
                   entity_finder=vert_prop.entity_finder,
                   output_types=frozenset({'vector'}),
                   name=f'vertically ({vert_prop}) and horizontally ({horiz_prop})',
                   register=register,
                   requires_entity=vert_prop.requires_entity or horiz_prop.requires_entity)

    @classmethod
    def points_to_vector(cls, source_pt, target_pt, register=False):
        assert source_pt.entity_finder == target_pt.entity_finder
        return cls(lambda x: tuple((
            target_pt(x.entity, x.grid)[i] - source_pt(x.entity, x.grid)[i]
            for i in range(2)
        )) if target_pt(x.entity, x.grid) is not None and source_pt(x.entity, x.grid) is not None else None,
                   nll=source_pt.nll + target_pt.nll + np.log(2),
                   entity_finder=source_pt.entity_finder,
                   output_types=frozenset({'vector'}),
                   name=f'the vector from {source_pt} to {target_pt}',
                   register=register,
                   requires_entity=source_pt.requires_entity or target_pt.requires_entity)

    @classmethod
    def length_to_vector(cls, length_prop, axis, register=False):
        return cls(lambda x: tuple((
            (float(length_prop(x.entity, x.grid))) if i == axis else 0.
            for i in range(2)
        )) if length_prop(x.entity, x.grid) is not None else None,
                   nll=length_prop.nll + np.log(2),
                   entity_finder=length_prop.entity_finder,
                   output_types=frozenset({'vector'}),
                   name=f'the vector in the direction of axis={axis} with length={length_prop}',
                   register=register,
                   requires_entity=length_prop.requires_entity)


class Relation:
    """
    This is a raw relation_or_property which doesn't depend on the grid or entity finder in order to allow for relations
    between entities from different examples

    :param base_relation: (entity1, entity2) -> bool
    """
    count = 0
    relations = []

    def __init__(self, base_relation: callable, nll: float, name: str, output_types: frozenset):
        self.base_relation = base_relation
        self.nll = nll
        self.name = name
        self.output_types = output_types
        self.cache = {}
        self.count = self.__class__.count
        self.__class__.count += 1
        self.__class__.relations.append(self)

    def __call__(self, entity1: Entity, entity2: Entity) -> bool:
        # key = (entity1.freeze(), entity2.freeze())
        # if key in self.cache:
        #     return self.cache[key]
        out = self.base_relation(entity1, entity2)
        # self.cache[key] = out
        return out

    def __str__(self):
        return self.name

    @classmethod
    def reset(cls):
        cls.count = 0
        cls.relations = []

    @classmethod
    def from_coordinate_properties(cls, property1: Property, property2: Property, reverse=False):
        sign = -1 if reverse else 1
        return cls(
            lambda entity1, entity2: sign * (property1(entity1, entity1.grid) - property2(entity2, entity2.grid)),
            nll=property1.nll + property2.nll,
            name=f'({property1}) minus ({property2})' if reverse else
            f'({property2}) minus ({property1})',
            output_types=frozenset({'vector'}))


class Selector:
    count = 0
    previous_selectors = set()
    output_signatures = {}

    def __init__(self, restriction: callable, nll=np.log(2), count=None, name=None, fixed_properties=None):
        if fixed_properties is None:
            fixed_properties = []
        if name is None:
            name = f'(Selector, restriction = {restriction}, NLL={nll})'
        self.restriction = restriction
        self.nll = nll
        self.name = name
        self.cache = {}
        self.fixed_properties = fixed_properties
        self.count = count if count is not None else self.__class__.count
        self.__class__.count += 1

    def __call__(self, entity, entities):
        return self.restriction(entity, entities)

    def select(self, entities: list, composite: bool = False):
        key = (Entity.counts(entities), composite)
        if key in self.cache:
            return self.cache[key]
        output = [entity for entity in entities if self(entity, entities)]
        if composite and output:
            new_dict = {position: color for entity in output for position, color in entity.positions.items()}
            composite_entity = Entity.make(new_dict, output[0].grid)
            output = [composite_entity]
        self.cache[key] = output
        return output

    def __str__(self):
        return self.name

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.count < other.count

    def generate_output_signature(self, base_entity_finder, task, output_raw_entities=False):
        inputs = [case['input'] for case in task['train']] + [case['input'] for case in task['test']]
        entities_list = [base_entity_finder(inp) for inp in inputs]
        if not output_raw_entities:
            return tuple((Entity.counts(self.select(entities)) for entities in entities_list))
        else:
            return tuple((Entity.counts(self.select(entities)) for entities in entities_list)), tuple(
                (Entity.counts(entities) for entities in entities_list))

    def intersect(self, other):
        new_selector = self.__class__(lambda entity, entities: other(entity, entities) & self(entity, entities),
                                      nll=combine_pair_selector_nll(self, other),
                                      name=f'({self.name}) and ({other.name})')
        return new_selector

    def validate_and_register(self, task: dict, base_entity_finder, max_nll,
                              extra_validation: callable = lambda x: True) -> bool:
        output_signature, raw_signature = self.generate_output_signature(base_entity_finder, task,
                                                                         output_raw_entities=True)
        empty_selections = len([0 for x in output_signature if x == frozenset()])
        self.nll += empty_selections * EMPTY_SELECT_PENALTY

        for training_case, raw_case in zip(output_signature, raw_signature):
            if not training_case:
                self.nll += 10

            if self.name != 'true' and training_case == raw_case:
                # Add penalties for each training task that isn't affected by this selection
                self.nll += 2

        if self.nll > max_nll or str(self) in Selector.previous_selectors or not extra_validation(output_signature):
            return False
        if output_signature in Selector.output_signatures:
            Selector.output_signatures[output_signature].nll = \
                union_nlls(self.nll, Selector.output_signatures[output_signature].nll)
            Selector.previous_selectors.add(str(self))
            return False
        else:
            Selector.output_signatures[output_signature] = self
            Selector.previous_selectors.add(str(self))
            return True

    @classmethod
    def reset(cls):
        cls.count = 0
        cls.previous_selectors = set()
        cls.output_signatures = {}

    @classmethod
    def make_property_selector(cls, entity_property, target_property, the_same=True):
        def base_selector(entity, _):
            return (entity_property(entity, entity.grid) == target_property(entity, entity.grid)) == the_same

        new_nll = combine_selector_nll(entity_property, target_property)
        fixed_properties = [entity_property.count]
        prop_selector = cls(base_selector,
                            nll=new_nll,
                            name=f"({entity_property}) is "
                                 f"{'equal' if the_same else 'not equal'} to ({target_property})",
                            fixed_properties=fixed_properties)
        prop_selector.entity_prop = entity_property
        prop_selector.target_prop = target_property
        return prop_selector


class Transformer:
    """
    :param base_transform: A function that takes a set of entities and a grid, and applies a transformation to them all
    """

    def __init__(self, base_transform: callable, nll=np.log(2), name='<no name>', **kwargs):
        self.base_transform = base_transform
        self.nll = nll
        self.name = name if USE_TRANSFORMER_NAMES else ''
        self.kwargs = kwargs
        self.cache = {}

    def transform(self, entities: list, grid: Optional[tuple] = None):
        if len(entities) > 0:
            key = (Entity.counts(entities), entities[0].grid)
        else:
            return {}, ()
        if key in self.cache:
            return self.cache[key]

        new_entities, new_grid = self.base_transform(entities, grid, **self.kwargs)
        self.cache[key] = (new_entities, new_grid)
        return new_entities, new_grid

    def __str__(self):
        return self.name

    def compose(self, other):
        return self.__class__(lambda entities, grid: other.transform(*self.transform(entities, grid)),
                              nll=self.nll + other.nll,
                              name=f'({self.name}) then also ({other.name})')

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.name < other.name

    # def create_entity_transformer(self, leaving_color=0,
    #                               entering_color_map=lambda orig_color, entity_color: entity_color, copy=False,
    #              extend_grid=True, nll=True, **property_kwargs):
    #     def base_transformer(entities: list, grid: Optional[tuple] = None):
    #         if len(entities) > 0:
    #             grid = entities[0].grid
    #             new_grid = np.array(grid)
    #         else:
    #             return {}, ()
    #         # new_grid = np.array(grid)
    #         new_entities = []
    #         new_positions_list = []
    #         for entity in entities:
    #             property_values = {key: prop(entity, grid) for key, prop in property_kwargs.items()}
    #             if None in property_values:
    #                 return {}, to_tuple(np.full_like(new_grid, 10))
    #             if not copy:
    #                 for position, color in entity.positions.items():
    #                     new_grid[position[0], position[1]] = leaving_color
    #             new_positions = move_entity(entity, grid, entering_color_map, extend_grid=extend_grid,
    #                                           **property_kwargs)
    #             if extend_grid and new_positions:
    #                 # First we compute the new shape of the grid
    #                 max_coordinates = [max((position[i] for position in new_positions.keys())) for i in range(2)]
    #                 positives = [max(max_coordinate, original_max - 1) + 1 for max_coordinate, original_max in
    #                              zip(max_coordinates, new_grid.shape)]
    #                 if tuple(positives) != new_grid.shape:
    #                     extended_grid = np.zeros(positives)
    #                     extended_grid[:new_grid.shape[0], :new_grid.shape[1]] = new_grid
    #                     new_grid = extended_grid
    #             for position, color in new_positions.items():
    #                 new_grid[position[0], position[1]] = new_positions[position[0], position[1]]
    #             new_positions_list.append(new_positions)
    #         new_grid_tuple = to_tuple(new_grid)
    #         new_entities = [Entity.make(new_positions, new_grid_tuple) for new_positions in new_positions_list]
    #         return new_entities, new_grid_tuple
    #     return Transformer(base_transformer, nll=nll)


@dataclass
class TransformerList:
    transformers: list
    nll: float

    @property
    def name(self):
        return ', '.join([str(transformer) for transformer in self.transformers])

    def __iter__(self):
        return iter(self.transformers)

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.name < other.name


class Predictor:
    count = 0

    def __init__(self, entity_finder: Union[EntityFinder, Iterable], transformer: Union[Transformer, Iterable],
                 nll=None, parallel=True):
        """

        :param entity_finder: A EntityFinder or iterable of EntityFinders that selects the desired entities
        :param transformer: A Transformer of iterable of Transformers that transforms the selected entities
        :param nll: The NLL of the predictor. If None, this will be automatically calculated
        :param parallel: Determines if the entity-transformer pairs are done in parallel or in sequence
        """
        self.parallel = parallel
        if isinstance(transformer, Transformer):
            self.transformers = (transformer,)
        else:
            self.transformers = tuple(transformer)
        if isinstance(entity_finder, EntityFinder):
            self.entity_finders = tuple((entity_finder for _ in self.transformers))
        else:
            self.entity_finders = tuple(entity_finder)
        if len(self.transformers) != len(self.entity_finders):
            raise Exception(
                f'number of entity finders ({len(self.entity_finders)}) '
                f'and transformers ({len(self.transformers)}) unequal')
        unique_entity_finders = {(entity_finder.name, entity_finder.nll) for entity_finder in self.entity_finders}
        unique_transformers = {(transformer.name, transformer.nll) for transformer in self.transformers}
        self.nll = sum([entity_finder_nll for entity_finder_name, entity_finder_nll in unique_entity_finders]) + \
                   sum([transformer_nll for transformer_name, transformer_nll in
                        unique_transformers]) if nll is None else nll
        self.net_distance_ = None
        # self.cache = {}
        self.count = self.__class__.count
        self.__class__.count += 1

    def predict(self, grid):
        out = grid
        # if grid in self.cache:
        #     return self.cache[grid]
        if self.parallel:
            selected_entities_list = [entity_finder(out) for entity_finder in self.entity_finders]
            for selected_entities, transformer in zip(selected_entities_list, self.transformers):
                edited_entities = []
                for entity in selected_entities:
                    new_entity = entity.change_grid(new_grid=out)
                    if new_entity is not None:
                        edited_entities.append(new_entity)
                _, out = transformer.transform(edited_entities)
        else:
            for entity_finder, transformer in zip(self.entity_finders, self.transformers):
                selected_entities = entity_finder(out)
                _, out = transformer.transform(selected_entities)
        # self.cache[grid] = out
        return out

    def __str__(self):
        return f"first ({[str(entity_finder) for entity_finder in self.entity_finders]}) then " \
               f"({[str(transformer) for transformer in self.transformers]}) " \
               f"{'in parallel' if self.parallel else 'sequentially'}"

    def __len__(self):
        assert len(self.entity_finders) == len(self.transformers)
        return len(self.entity_finders)

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.count < other.count

    def copy(self, parallel=None):
        if parallel is None:
            parallel = self.parallel
        return self.__class__(self.entity_finders, self.transformers,
                              self.nll, parallel=parallel)

    # def add_transformer(self, new_transformer: Transformer):
    #     return self.__class__(entity_finder=self.entity_finder,
    #                           transformer=self.transformer.compose(new_transformer))

    def compose(self, other_predictor, parallel=None):
        if parallel is None:
            parallel = self.parallel
        return self.__class__(self.entity_finders + other_predictor.entity_finders,
                              self.transformers + other_predictor.transformers,
                              parallel=parallel)

    @classmethod
    def reset(cls):
        cls.count = 0


def reset_all():
    Relation.reset()
    Property.reset()
    Selector.reset()
    Predictor.reset()
    Entity.reset()
    # global move_entity_cache
    # move_entity_cache = {}
    adjacent_direction_cache = {}
    find_color_entities_cache = {}
    collision_directions_cache = {}


def selector_iterator(task: dict, base_entity_finder: EntityFinder, max_nll: float = 20.):
    start_time = time.perf_counter()
    inputs = [case['input'] for case in task['train']]
    inputs.extend([case['input'] for case in task['test']])
    earlier_selectors = []
    rounds = 0
    grid_properties, entity_properties = generate_base_properties(task, base_entity_finder)
    grid_properties, entity_properties = filter_unlikelies(grid_properties, max_nll), filter_unlikelies(
        entity_properties, max_nll)

    entity_properties.sort()
    entity_properties = [entity_property for entity_property in entity_properties
                         if entity_property.validate_and_register(task)]

    # # Make all grid properties that are the same for all training examples less likely
    # for grid_property in grid_properties:
    #     if len({grid_property(None, case['input']) for case in task['train']}) == 1 and not grid_property.is_constant:
    #         grid_property.nll += 1

    grid_properties = filter_unlikelies(grid_properties, max_nll)
    grid_properties.sort()
    grid_properties = [grid_property for grid_property in grid_properties if grid_property.validate_and_register(task)]

    entity_properties.sort()
    grid_properties.sort()

    all_properties = entity_properties + grid_properties
    all_properties.sort()

    trivial_selector = Selector(lambda entity, grid: True, name='true', nll=0)
    queue = [trivial_selector]
    for entity_property in entity_properties:
        for target_property in all_properties:
            if (entity_property.count != target_property.count) and (
                    combine_selector_nll(entity_property, target_property) <= max_nll):
                for the_same in [True, False]:
                    new_selector = Selector.make_property_selector(entity_property, target_property, the_same)
                    if str(entity_property) == "the colors" and str(target_property) == "color 0" and not the_same:
                        # Specially make selecting all the non-0 color entities more likely
                        new_selector.nll = SELECT_NOT_0_NLL
                    if new_selector.validate_and_register(task, base_entity_finder, max_nll):
                        heapq.heappush(queue, new_selector)

    earlier_selectors = []
    while queue:
        my_selector = heapq.heappop(queue)

        Selector.previous_selectors.add(str(my_selector))
        yield my_selector
        rounds += 1
        # print(my_selector, my_selector.nll)
        if time.perf_counter() - start_time > MAX_SMALL_TIME:
            return

        # Makes properties where the selection produces unique values in all the training cases
        new_grid_properties = []
        common_property_indices = []
        for i, training_case in enumerate(task['train']):
            training_grid = training_case['input']
            training_entities = base_entity_finder(training_grid)
            training_selected_entities = my_selector.select(training_entities)
            if not training_selected_entities:
                common_property_indices = [set()]
                break
            common_properties = {(i, prop(training_selected_entities[0], training_grid))
                                 for i, prop in enumerate(entity_properties) if
                                 prop.count not in my_selector.fixed_properties}
            for entity in training_selected_entities:
                common_properties &= {(i, prop(entity, training_grid)) for i, prop in enumerate(entity_properties)
                                      if prop.count not in my_selector.fixed_properties}
            # Extract which entity properties give the same result for all entities selected
            common_property_indices.append({prop[0] for prop in common_properties})
        valid_common_properties = set.intersection(*common_property_indices)

        common_grid_properties = [entity_properties[index].add_selector(my_selector) for index in
                                  valid_common_properties]
        common_grid_properties.sort()

        for prop in (prop for prop in entity_properties if prop.count not in my_selector.fixed_properties):
            for ordinal_property in ORDINAL_PROPERTIES:
                if combine_property_selector_nll(prop, my_selector, ordinal_property) <= max_nll:
                    grid_property = prop.add_selector(my_selector, ordinal_property)
                    if grid_property in common_grid_properties:
                        # not a problem
                        grid_property.nll -= 1

                    if grid_property.validate_and_register(task):
                        new_grid_properties.append(grid_property)

        # Makes sure properties are potentially valid on training examples
        # all_cases = task['train'] + task['test']
        # for i, case in enumerate(all_cases):
        #     training_grid = case['input']
        #     training_entities = base_entity_finder(training_grid)
        #     valid_indices = []
        #     for j, entity_prop in enumerate(new_entity_properties):
        #         for entity in training_entities:
        #             if entity_prop(entity, training_grid) is None:
        #                 break
        #         else:
        #             valid_indices.append(j)

        #     new_entity_properties = [new_entity_properties[valid_index] for valid_index in valid_indices]

        # # Makes relational entity properties from a chosen entity to some selection
        # new_entity_properties = [
        #                          for relation, ordinal_property in
        #                          itertools.product(Relation.relations, ORDINAL_PROPERTIES)
        #                          if combine_relation_selector_nll(relation, my_selector,
        #                                                           ordinal_property) <= max_nll
        #                          ]
        # # Now register the properties
        # for prop in new_entity_properties:
        #     prop.validate_and_register(task)

        new_entity_properties = []
        for relation, ordinal_property in combine_sorted_queues((Relation.relations, ORDINAL_PROPERTIES),
                                                                max_nll - my_selector.nll - PROPERTY_CONSTRUCTION_COST):
            if combine_relation_selector_nll(relation, my_selector, ordinal_property) < max_nll:
                prop = Property.from_relation_selector(relation, my_selector,
                                                       entity_finder=base_entity_finder,
                                                       ordinal_property=ordinal_property,
                                                       register=False)
                if prop.validate_and_register(task):
                    new_entity_properties.append(prop)

        # Make new ordinal grid properties
        for entity_prop, selector, ordinal_prop in combine_sorted_queues((new_entity_properties, earlier_selectors,
                                                                          ORDINAL_PROPERTIES),
                                                                         max_nll - PROPERTY_CONSTRUCTION_COST):
            if combine_property_selector_nll(entity_prop, selector, ordinal_prop) <= max_nll:
                grid_property = entity_prop.add_selector(selector, ordinal_prop)
                if grid_property.validate_and_register(task):
                    new_grid_properties.append(grid_property)

        new_grid_properties.sort()
        # Now add in the new selectors to the queue
        for entity_prop, new_prop in combine_sorted_queues((entity_properties, new_grid_properties),
                                                           max_nll - SELECTOR_CONSTRUCTION_COST):
            # Makes a new selector from the base property and the new property
            if combine_selector_nll(entity_prop, new_prop) <= max_nll:
                for the_same in [True, False]:
                    new_selector = Selector.make_property_selector(entity_prop, new_prop, the_same=the_same)
                    if new_selector.validate_and_register(task, base_entity_finder, max_nll):
                        heapq.heappush(queue, new_selector)

        grid_properties.extend(new_grid_properties)
        grid_properties.sort()

        all_properties = grid_properties + entity_properties
        all_properties.sort()

        for new_prop, grid_prop in combine_sorted_queues((new_entity_properties, all_properties),
                                                         max_nll - SELECTOR_CONSTRUCTION_COST):
            # Makes a new selector from the base property and the new property
            if combine_selector_nll(new_prop, grid_prop) <= max_nll:
                for the_same in [True, False]:
                    new_selector = Selector.make_property_selector(new_prop, grid_prop, the_same=the_same)
                    if new_selector.validate_and_register(task, base_entity_finder, max_nll):
                        heapq.heappush(queue, new_selector)

        entity_properties.extend(new_entity_properties)
        entity_properties.sort()

        earlier_selectors.append(my_selector)
        earlier_selectors.sort()

        # if not (rounds % 10):
        #     print(f'my_selector.nll = {my_selector.nll}')
        #     print(f'len(entity_properties) = {len(entity_properties)}')
        #     print(f'len(properties) = {len(grid_properties)}')


def create_predictor_queue(task, max_nll, base_entity_finder, allow_selector_pairs=False):
    for i, example in enumerate(task['train']):
        if len(base_entity_finder(example['input'])) == 0:
            return []
    start_time = time.perf_counter()
    selector_list = list(selector_iterator(task, base_entity_finder, max_nll=max_nll - SELECTOR_MAX_NLL_CORRECTION))
    selector_list.sort()
    print(f"selecting time = {time.perf_counter() - start_time}")

    if MAKE_PROPERTY_LIST:
        Property.property_list.sort()
        print(f"len(Property.property_list) = {len(Property.property_list)}")

    print(f'built selector list (1), length={len(selector_list)}')

    if allow_selector_pairs:
        for selector1, selector2 in itertools.combinations(selector_list, 2):
            if combine_pair_selector_nll(selector1, selector2) < max_nll - SELECTOR_MAX_NLL_CORRECTION:
                new_selector = selector1.intersect(selector2)
                if new_selector.validate_and_register(task, base_entity_finder, max_nll - SELECTOR_MAX_NLL_CORRECTION):
                    selector_list.append(new_selector)
        if time.perf_counter() - start_time > MAX_SMALL_TIME:
            print('Out of time')
            return []

    selector_list.sort()
    print(f'built selector list (2), length={len(selector_list)}')
    # print('Time after selectors created = ', time.perf_counter() - start_time)
    # Create distance properties out of coordinate properties
    Property.of_type['x_coordinate'].sort()
    Property.of_type['y_coordinate'].sort()
    # LENGTH PROPERTIES
    x_length_props = (prop1.create_distance_property(prop2, register=False)
                      for prop1, prop2 in
                      combine_sorted_queues((Property.of_type['x_coordinate'], Property.of_type['x_coordinate']),
                                            max_nll - np.log(2))
                      if prop1.count != prop2.count
                      and (not prop1.is_constant or not prop2.is_constant))
    y_length_props = (prop1.create_distance_property(prop2, register=False)
                      for prop1, prop2 in
                      combine_sorted_queues((Property.of_type['y_coordinate'], Property.of_type['y_coordinate']),
                                            max_nll - np.log(2))
                      if prop1.count != prop2.count
                      and (not prop1.is_constant or not prop2.is_constant))

    length_props = sorted(list(itertools.chain(x_length_props, y_length_props)))
    for length_prop in length_props:
        length_prop.validate_and_register(task,
                                          extra_validation=lambda output_signature:
                                          all((value.is_integer() for value in output_signature)))

    if time.perf_counter() - start_time > MAX_SMALL_TIME:
        print('Out of time')
        return []

    Property.of_type['x_length'].sort()
    Property.of_type['y_length'].sort()

    # Constructing point properties
    point_props = [Property.create_point_property(prop1, prop2, register=False)
                   for prop1, prop2 in
                   combine_sorted_queues((Property.of_type['y_coordinate'], Property.of_type['x_coordinate']),
                                         max_nll - 2 - POINT_PROP_COST)]
    for point_prop in point_props:
        point_prop.validate_and_register(task)
    Property.of_type['point'].sort()

    if time.perf_counter() - start_time > MAX_SMALL_TIME:
        print('Out of time')
        return []

    # Constructing vector properties

    # Create vectors from single lengths
    for axis, name in enumerate(['y_length', 'x_length']):
        for length in Property.of_type[name]:
            vect_prop = Property.length_to_vector(length, axis, register=False)
            vect_prop.validate_and_register(task)

    # Create vectors from pairs of points
    for source_pt, target_pt in combine_sorted_queues((Property.of_type['point'],
                                                       Property.of_type['point']),
                                                      max_nll - np.log(2)):
        vect_prop = Property.points_to_vector(source_pt, target_pt, register=False)
        vect_prop.validate_and_register(task,
                                        extra_validation=lambda output_signature: all(
                                            (value[i].is_integer() for value in output_signature for i in range(2))))
        if time.perf_counter() - start_time > MAX_SMALL_TIME:
            print('Out of time')
            return []

    penalize_dim_change = True if all(
        (len(case['input']) == len(case['output']) and len(case['input'][0]) == len(case['output'][0]) for case in
         task['train'])) else False

    transformers = (
        # 34
        Transformer(
            lambda entities, grid, vector_prop=vector_prop, copy=copy: move(entities,
                                                                            vector_property=vector_prop,
                                                                            copy=copy,
                                                                            extend_grid=not penalize_dim_change),
            nll=vector_prop.nll + np.log(2),
            name=f"{'copy' if copy else 'move'} them by ({vector_prop})")
        for vector_prop in Property.of_type['vector']
        for copy in [True, False] if vector_prop.nll + np.log(2) <= max_nll
    )
    if time.perf_counter() - start_time > MAX_SMALL_TIME:
        print('Out of time')
        return []
    Property.of_type['color'].sort()
    # 35
    composite_transformers = (Transformer(lambda entities, grid, offsets=offsets:
                                          crop_entities(entities, grid, offsets=offsets),
                                          nll=np.log(2) + sum(
                                              (abs(offset) for offset in offsets)) * np.log(2) + \
                                              penalize_dim_change * DIM_CHANGE_PENALTY,
                                          name=f'crop them with offset {offsets}')
                              for offsets in itertools.product([-1, 0, 1], repeat=4)
                              if np.log(2) + sum((abs(offset) for offset in offsets)) * np.log(2) + \
                              penalize_dim_change * DIM_CHANGE_PENALTY < max_nll)
    if any(({entry for row in case['input'] for entry in row} == {entry for row in case['output'] for entry in row}
            for case in task['train'])):
        new_colors = False
    else:
        new_colors = True

    # 36
    composite_transformers = itertools.chain(composite_transformers,
                                             (Transformer(lambda entities, grid, offsets=offsets,
                                                                 source_color_prop=source_color_prop,
                                                                 target_color_prop=target_color_prop:
                                                          replace_colors_in_entities_frame(entities, grid,
                                                                                           offsets=offsets,
                                                                                           source_color_prop=source_color_prop,
                                                                                           target_color_prop=target_color_prop),
                                                          nll=np.log(
                                                              2) + source_color_prop.nll + target_color_prop.nll + sum(
                                                              (abs(offset) for offset in offsets)) * np.log(2)
                                                              - new_colors * NEW_COLOR_BONUS,
                                                          name=f'replace ({source_color_prop}) '
                                                               f'with ({target_color_prop}) '
                                                               f'in a box around them with offsets {offsets}')
                                              for source_color_prop, target_color_prop in
                                              combine_sorted_queues(
                                                  (Property.of_type['color'], Property.of_type['color']),
                                                  max_nll=max_nll - np.log(2) + new_colors * NEW_COLOR_BONUS)
                                              for offsets in [(0, 0, 0, 0), (1, -1, 1, -1)]))
    # 37
    composite_transformers = itertools.chain(composite_transformers,
                                             (Transformer(lambda entities, grid,
                                                                 source_color_prop=source_color_prop,
                                                                 target_color_prop=target_color_prop:
                                                          replace_color(entities,
                                                                        source_color_prop=source_color_prop,
                                                                        target_color_prop=target_color_prop),
                                                          nll=source_color_prop.nll + target_color_prop.nll + np.log(
                                                              2) - new_colors * NEW_COLOR_BONUS,
                                                          name=f'recolor ({source_color_prop}) with ({target_color_prop})')
                                              for source_color_prop, target_color_prop in
                                              combine_sorted_queues(
                                                  (Property.of_type['color'], Property.of_type['color']),
                                                  max_nll - np.log(2) + new_colors * NEW_COLOR_BONUS)))
    Property.of_type['shape'].sort()
    # 38
    transformers = itertools.chain(transformers,
                                   (Transformer(lambda entities, grid,
                                                       point_prop=point_prop,
                                                       shape_prop=shape_prop,
                                                       color_strategy=color_strategy:
                                                place_shape(entities,
                                                            point_prop=point_prop,
                                                            shape_prop=shape_prop,
                                                            color_strategy=color_strategy),
                                                nll=point_prop.nll + shape_prop.nll + np.log(2),
                                                name=f'place ({shape_prop}) at position ({point_prop})')
                                    for point_prop, shape_prop in
                                    combine_sorted_queues((Property.of_type['point'],
                                                           Property.of_type['shape']), max_nll - np.log(2))
                                    for color_strategy in ('original', 'extend_non_0', 'replace_0'))
                                   )

    reflections = [
        Transformer(lambda entities, grid, line_prop=line_prop: reflect_about_line(entities,
                                                                                   line_prop=line_prop,
                                                                                   extend_grid=not penalize_dim_change),
                    nll=line_prop.nll + np.log(2), name=f'reflect about {line_prop}') for line_prop in
        Property.of_type['line']]

    # rotations = [Transformer(lambda entities, grid, line_prop1=line_prop1, line_prop2=line_prop2:
    #                          rotate_via_reflects(entities, grid, line_prop1, line_prop2,
    #                                              extend_grid=not penalize_dim_change),
    #                          nll=line_prop1.nll + line_prop2.nll + np.log(2),
    #                          name=f'reflect about ({line_prop1}) then ({line_prop2})')
    #              for line_prop1, line_prop2 in itertools.permutations(Property.of_type['line'], 2)
    #              if line_prop1.nll + line_prop2.nll + np.log(2) < max_nll]

    rotations = [Transformer(lambda entities, grid, point_prop=point_prop:
                             rotate_about_point(entities, grid, point_prop,
                                                extend_grid=not penalize_dim_change),
                             nll=point_prop.nll + np.log(2),
                             name=f'rotate {steps} steps clockwise about {point_prop}')
                 for point_prop in Property.of_type['point']
                 for steps in range(1, 4)
                 if point_prop.nll + np.log(2) < max_nll]

    # rotation_groups = [Transformer(lambda entities, grid, line_prop1=line_prop1, line_prop2=line_prop2:
    #                                apply_rotation_group_old(entities, grid, line_prop1, line_prop2,
    #                                                         extend_grid=not penalize_dim_change),
    #                                nll=line_prop1.nll + line_prop2.nll + 2 * np.log(2),
    #                                name=f'rotate via ({line_prop1}, {line_prop2}) 4 times')
    #                    for line_prop1, line_prop2 in itertools.permutations(Property.of_type['line'], 2)
    #                    if line_prop1.nll + line_prop2.nll + np.log(2) + 2 * np.log(2) < max_nll]

    rotation_groups = [Transformer(lambda entities, grid, point_prop=point_prop:
                                   apply_rotation_group(entities, grid, point_prop,
                                                        extend_grid=not penalize_dim_change),
                                   nll=point_prop.nll + np.log(2),
                                   name=f'apply full rotation group about {point_prop}')
                       for point_prop in Property.of_type['point']
                       if point_prop.nll + 2 * np.log(2) < max_nll]

    klein_viers = [Transformer(lambda entities, grid, line_prop1=line_prop1, line_prop2=line_prop2:
                               apply_klein_vier_group(entities, grid, line_prop1, line_prop2,
                                                      extend_grid=not penalize_dim_change),
                               nll=line_prop1.nll + line_prop2.nll + 2 * np.log(2),
                               name=f'apply the group generated by {line_prop1} and {line_prop2}')
                   for line_prop1, line_prop2 in itertools.combinations(Property.of_type['line'], 2)
                   if line_prop1.nll + line_prop2.nll + 2 * np.log(2) < max_nll]

    transformers = itertools.chain(transformers,
                                   reflections,
                                   rotations,
                                   rotation_groups,
                                   klein_viers
                                   )
    # print('transformer lengths:')
    # print(len(reflections))
    # print(len(Property.of_type['point']))
    # print(len(rotations))
    # print(len(rotation_groups))
    # print(len(klein_viers))

    # rotations.sort()
    # for rotation in rotations:
    #     print(rotation, rotation.nll)

    # for shape_prop in Property.of_type['shape']:
    #     print(shape_prop)
    # transformers = itertools.chain(transformers,
    #                                (Transformer(lambda entities, grid, shape_prop=shape_prop:
    #                                                           output_shape_as_grid(entities, grid, shape_prop),
    #                                                           nll=shape_prop.nll + np.log(2))
    #                                 for shape_prop in Property.of_type['shape']
    #                                 if shape_prop.nll + np.log(2) <= max_nll-2 and not shape_prop.requires_entity))
    # print('Time after transformers list =', time.perf_counter() - start_time)
    # print(f"sys.getsizeof(transformers) = {sys.getsizeof(transformers)}", f"len(transformers) = {len(transformers)}")

    transformers = itertools.chain(transformers, composite_transformers)
    transformers = list(transformers)
    transformers.sort()

    if not ALLOW_COMPOSITE_SELECTORS:
        transformers = itertools.chain(transformers, composite_transformers)
        transformers = list(transformers)
        transformers.sort()
        entity_finders = [base_entity_finder.compose(selector) for selector in selector_list if
                          selector.nll + base_entity_finder.nll <= max_nll]
        predictor_queue = [Predictor(entity_finder, transformer)
                           for entity_finder, transformer in
                           combine_sorted_queues((entity_finders, transformers), max_nll)]
    else:
        composite_transformers = list(composite_transformers)
        transformers = list(transformers)
        transformers.sort()
        composite_transformers.sort()
        entity_finders_noncomposite = [base_entity_finder.compose(selector, False) for selector in
                                       selector_list if selector.nll + base_entity_finder.nll <= max_nll]

        entity_finders_composite = entity_finders_noncomposite + \
                                   [base_entity_finder.compose(selector, True) for selector in selector_list
                                    if selector.nll + base_entity_finder.nll <= max_nll]

        entity_finders_composite.sort()

        predictor_queue = [Predictor(entity_finder, transformer)
                           for entity_finder, transformer in
                           combine_sorted_queues((entity_finders_composite, transformers), max_nll)]
        predictor_queue += [Predictor(entity_finder, transformer)
                            for entity_finder, transformer in
                            combine_sorted_queues((entity_finders_noncomposite, composite_transformers), max_nll)]
    # print('Time after predictor queue =', time.perf_counter() - start_time)
    # for key, properties in Property.of_type.items():
    #     print(key, len(properties))
    if time.perf_counter() - start_time > MAX_SMALL_TIME:
        print('Out of time')
        return []
    print(f'built predictor queue, length = {len(predictor_queue)}')
    predictor_queue.sort()
    print('sorted predictor queue')

    return predictor_queue


def test_case(task,
              max_nll=10.,
              base_entity_finder=EntityFinder(lambda grid: find_components(grid, directions=ALL_DIRECTIONS)),
              allow_multiple_predictors=False,
              allow_selector_pairs=False):
    start_time = time.perf_counter()
    reset_all()
    base_entity_finder.reset()
    generate_base_relations()
    inputs = [case['input'] for case in task['train']]
    inputs.extend([case['input'] for case in task['test']])
    predictor_queue = create_predictor_queue(task=task,
                                             max_nll=max_nll,
                                             base_entity_finder=base_entity_finder,
                                             allow_selector_pairs=allow_selector_pairs)
    my_count = 0

    if len(predictor_queue) > MAX_PREDICTORS:
        predictor_queue = predictor_queue[:MAX_PREDICTORS]

    predictions = set()
    predictor_indices = []
    good_predictors = []

    for i, predictor in enumerate(predictor_queue):
        if time.perf_counter() - start_time > MAX_SMALL_TIME:
            print('Out of time')
            return []

        if all((predictor.predict(case['input']) == case['output'] for case in task['train'])):
            good_predictors.append(predictor)
            if len(good_predictors) == 3:
                break
        if not good_predictors and allow_multiple_predictors:
            prediction = tuple((predictor.predict(case['input']) for case in task['train'] + task['test']))
            if () not in prediction and prediction not in predictions:
                predictions.add(prediction)
                predictor_indices.append(i)

    print(f"before filtering: {len(predictor_queue)}")
    predictor_queue = [predictor_queue[i] for i in predictor_indices]
    print(f"after filtering: {len(predictor_queue)}")
    # If there is no single predictor solution, expand our search to multiple-predictor solutions
    partial_predictors = []
    old_partial_predictors = partial_predictors
    if allow_multiple_predictors:
        previous_predictor_outputs = set()
        original_grids = [case['input'] for case in task['train'] + task['test']]
        depth = 0
        while not good_predictors and depth < 10:
            # print(f'len(adjacent_direction_cache)= {len(adjacent_direction_cache)}')
            # print(f'len(find_color_entities_cache)= {len(find_color_entities_cache)}')
            # print(f'len(collision_directions_cache)= {len(collision_directions_cache)}')
            if time.perf_counter() - start_time > MAX_LARGE_TIME:
                print('Out of time')
                return []
            print(f'round {my_count + 1}')
            old_partial_predictors = partial_predictors
            partial_predictors = []
            for predictor in predictor_queue:
                if partial_predictors and predictor.nll > partial_predictors[0][0].nll + 3:
                    break
                # desired_string = 'replace (color 0) with (color 1) in a box around them'
                # if desired_string in str(predictor):
                #     print(predictor, predictor.nll)
                if not old_partial_predictors:
                    # On the first pass, we create a trivial list of just the predictor
                    predictors = [(predictor, original_grids)]
                else:
                    # On future passes, we combine the "predictor" index with all of the useful partial predictors
                    predictors = ((partial_predictor.compose(predictor, parallel), old_predictions) for
                                  partial_predictor, old_predictions in
                                  old_partial_predictors for parallel in ([True, False]
                                                                          if depth == 1 else [None]))
                for new_predictor, old_predictions in predictors:
                    # test_output = [new_predictor.predict(train_case['input']) for train_case in task['test']]
                    # if full_output in previous_predictor_outputs:
                    #     continue
                    # else:
                    #     previous_predictor_outputs.add(full_output)
                    # if 'place' in str(predictor):
                    #     print(predictor)
                    no_superfluous_changes = all(
                        (base_entity_finder.grid_distance(case['output'], new_predictor.predict(case['input'])) +
                         base_entity_finder.grid_distance(new_predictor.predict(case['input']), old_prediction) ==
                         base_entity_finder.grid_distance(case['output'], old_prediction)
                         for case, old_prediction in
                         zip(task['train'], old_predictions)))
                    # Note: old predictions also contains the test predictions, but zip cuts this part off

                    if no_superfluous_changes:
                        # We test if the prediction is on a "straight line" between the input and the output to prevent
                        # the algorithm making lots of horizontal changes
                        new_predicted = [new_predictor.predict(train_case['input']) for train_case in task['train']]
                        test_output = [new_predictor.predict(train_case['input']) for train_case in task['test']]
                        if () in test_output:
                            continue
                        full_output = tuple(new_predicted + test_output)
                        if full_output in previous_predictor_outputs:
                            continue
                        else:
                            previous_predictor_outputs.add(full_output)

                        if new_predictor.entity_finders[-1].nll + new_predictor.transformers[-1].nll > 5. or depth < 3:
                            min_changes = 2
                        else:
                            min_changes = 1
                        num_changes = 0
                        for prediction, old_prediction in zip(full_output, old_predictions):
                            if prediction != old_prediction:
                                num_changes += 1
                                if num_changes >= min_changes:
                                    non_trivial_change = True
                                    break
                        else:
                            non_trivial_change = False

                        if non_trivial_change:
                            # If we've made a change, by the previous equality we must have made an improvement

                            if all((prediction == case['output'] for prediction, case in
                                    zip(new_predicted, task['train']))):
                                # If there's a perfect match, we add it to the good_predictors list
                                good_predictors.append(new_predictor)
                                if len(good_predictors) == 3:
                                    break
                                else:
                                    continue

                            if not good_predictors:
                                # If there's no perfect match, we look for a partial match
                                new_predictor.net_distance_ = sum(
                                    (base_entity_finder.grid_distance(case['output'], prediction)
                                     for prediction, case in zip(new_predicted, task['train'])))
                                if len(partial_predictors) < MAX_PARTIAL_PREDICTORS:
                                    # If the list of predictors is small we just add any new predictor
                                    partial_predictors.append((new_predictor,
                                                               full_output))
                                    partial_predictors.sort(key=lambda x: (x[0].net_distance_, x[0]))
                                elif (new_predictor.net_distance_, new_predictor) < \
                                        (partial_predictors[-1][0].net_distance_, partial_predictors[-1][0]):
                                    # If the list of predictors is greater than MAX_PARTIAL_PREDICTORS we check if it's
                                    # better then the worst one in the list
                                    partial_predictors.pop()
                                    partial_predictors.append((new_predictor,
                                                               full_output))
                                    partial_predictors.sort(key=lambda x: (x[0].net_distance_, x[0]))
                if len(good_predictors) >= 3:
                    break
            for predictor, predictions in partial_predictors:
                print(predictor)
            my_count += 1
            print(f'len(partial_predictors) = {len(partial_predictors)}')
            print(f'len(good_predictors) = {len(good_predictors)}')
            if len(partial_predictors) == 0:
                break
            depth += 1
    if not good_predictors:
        print("No good predictors")
        good_predictors = [predictor for predictor, _ in old_partial_predictors]
    return good_predictors


def flattener(pred):
    """Function provided by the challenge"""
    if isinstance(pred[0], tuple):
        str_pred = str([list(row) for row in pred])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
    else:
        str_pred = "|0|"
    return str_pred


def multi_array_flattener(arrays):
    string_list = [flattener(array) for array in arrays]
    return " ".join(string_list)


def combine_sorted_queues(queues, max_nll):
    """First queue can be a generator"""
    if len(queues) == 2:
        first_queue, other_queue = queues
        for y in other_queue:
            for x in first_queue:
                if x.nll + y.nll > max_nll:
                    break
                else:
                    yield x, y
    elif len(queues) > 2:
        first_queue, *other_queues = queues
        for ys in combine_sorted_queues(other_queues, max_nll):
            for x in first_queue:
                if x.nll + sum((y.nll for y in ys)) > max_nll:
                    break
                else:
                    yield (x,) + ys


def plot_task(task, num=0):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'])
    norm = colors.Normalize(vmin=0, vmax=10)
    train_len = len(task['train'])
    fig, axs = plt.subplots(1, 2 * train_len, figsize=(15, 15))
    # print(len(task['train']))
    for i in range(train_len):
        axs[0 + 2 * i].imshow(task['train'][i]['input'], cmap=cmap, norm=norm)
        axs[0 + 2 * i].axis('off')
        axs[0 + 2 * i].set_title(f'Train Input {num}')
        axs[1 + 2 * i].imshow(task['train'][i]['output'], cmap=cmap, norm=norm)
        axs[1 + 2 * i].axis('off')
        axs[1 + 2 * i].set_title(f'Train Output {num}')
    plt.tight_layout()
    plt.show()


def to_tuple(lst):
    out = tuple((tuple((int(entry) for entry in row)) for row in lst))
    return out


def display_case(grid, title=''):
    if len(grid) == 0:
        return
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'])
    norm = colors.Normalize(vmin=0, vmax=10)
    fig, axs = plt.subplots(1, figsize=(12, 4))
    axs.imshow(grid, cmap=cmap, norm=norm)
    if title:
        plt.title(title)
    # plt.tight_layout()
    plt.show()


def add_one(properties):
    for prop in properties:
        prop.nll += 1


def list2d_to_set(lst):
    return {x for sublist in lst for x in sublist}


def filter_unlikelies(properties, max_nll):
    return [prop for prop in properties if prop.nll <= max_nll]


def extend_grid_via_extreme_coordinates(min_coords, max_coords, grid_arr):
    if min_coords == (0, 0) and all(
            max_coord + 1 == grid_length for max_coord, grid_length in zip(max_coords, grid_arr.shape)):
        return grid_arr
    shape = tuple((max_coord + 1 - min_coord for min_coord, max_coord in zip(min_coords, max_coords)))
    new_grid_arr = np.zeros(shape, dtype=grid_arr.dtype)
    for i, j in itertools.product(range(grid_arr.shape[0]), range(grid_arr.shape[1])):
        new_grid_arr[i + min_coords[0], j + min_coords[1]] = grid_arr[i, j]
    return new_grid_arr


def extend_entities_positions(min_coords, entities_positions):
    adjustments = [min(min_coord, 0) for min_coord in min_coords]
    new_entities_positions = []
    for entity_positions in entities_positions:
        new_entity_positions = {}
        for position, color in entity_positions.items():
            new_position = tuple((coord - adjustment for coord, adjustment in zip(position, adjustments)))
            new_entity_positions[new_position] = color
        new_entities_positions.append(new_entity_positions)
    return new_entities_positions


def stamp_entities_positions(entities_positions, grid_arr):
    for entity_positions in entities_positions:
        for position, color in entity_positions.items():
            grid_arr[position[0], position[1]] = color
    return grid_arr


def tuplefy_task(task, test=False):
    if test:
        tuplefied_task = {'train': [{'input': to_tuple(case['input']), 'output': to_tuple(case['output'])} for case in
                                    task['train']],
                          'test': [{'input': to_tuple(case['input'])} for case in
                                   task['test']]}
    else:
        tuplefied_task = {'train': [{'input': to_tuple(case['input']), 'output': to_tuple(case['output'])} for case in
                                    task['train']],
                          'test': [{'input': to_tuple(case['input']), 'output': to_tuple(case['output'])} for case in
                                   task['test']]}
    return tuplefied_task


class WindowsInhibitor:
    """Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx"""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def __enter__(self):
        self.inhibit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninhibit()

    @staticmethod
    def inhibit():
        import ctypes
        print("Preventing Windows from going to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS |
            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    @staticmethod
    def uninhibit():
        import ctypes
        print("Allowing Windows to go to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)


def union_nlls(nll1, nll2):
    return min(nll1, nll2)


def combine_relation_selector_nll(relation_or_property, selector, ordinal_property):
    new_nll = relation_or_property.nll + selector.nll + ordinal_property.nll + PROPERTY_CONSTRUCTION_COST
    if not (relation_or_property.output_types & ordinal_property.input_types):
        new_nll += MISMATCHED_OUTPUT_PENALTY
    # if_neg(new_nll)
    return new_nll


combine_property_selector_nll = combine_relation_selector_nll


def combine_selector_nll(entity_property, target_property):
    new_nll = entity_property.nll + target_property.nll + SELECTOR_CONSTRUCTION_COST
    if not (entity_property.output_types & target_property.output_types):
        new_nll += MISMATCHED_OUTPUT_PENALTY
    if entity_property.output_types == frozenset({'bool'}) and not target_property.is_constant:
        new_nll += NONCONSTANT_BOOL_PENALTY
    if target_property.requires_entity:
        new_nll += SELECTOR_TARGET_REQUIRES_ENTITY_PENALTY
    if entity_property in target_property.associated_objects:
        new_nll -= SELECTING_SAME_PROPERTY_BONUS
        new_nll = max(new_nll, 0)
    return new_nll


def combine_pair_selector_nll(selector1, selector2):
    return selector1.nll + selector2.nll + SELECTOR_PAIR_PENALTY


def combine_move_nll(property0, property1):
    combined_nll = property0.nll + property1.nll
    if 'y_length' not in property0.output_types:
        combined_nll += MISMATCHED_OUTPUT_PENALTY
    if 'x_length' not in property1.output_types:
        combined_nll += MISMATCHED_OUTPUT_PENALTY
    return combined_nll


def combine_color_nll(property_pairs: Iterable):
    nll = 0
    for property1, property2 in property_pairs:
        nll += property1.nll + property2.nll
    return nll


def same_color(point1, point2):
    return point1[0] == point2[0]


def not_zero(_, point2):
    return bool(point2[0])


def find_components(grid: tuple, relation='same_color', directions=STRAIGHT_DIRECTIONS, ignore_zero=False):
    # Uses breadth first search to find all connected components and puts them in a list
    if len(grid) == 0:
        return {}
    arr_grid = np.array(grid)
    if relation == 'same_color':
        relation_function = same_color
    elif relation == 'not_zero':
        relation_function = not_zero
        ignore_zero = True
    else:
        raise Exception(f'Invalid input for relation_or_property: {relation}')
    array_directions = [np.array(direction) for direction in directions]
    entities = []
    reached = np.full_like(arr_grid, 0)
    for i, j in itertools.product(range(arr_grid.shape[0]), range(arr_grid.shape[1])):
        positions = {}
        if reached[i, j] or (ignore_zero and not arr_grid[i, j]):
            continue
        queue = deque([(i, j)])
        reached[i, j] = 1
        while queue:
            current = queue.pop()
            positions[current] = arr_grid[current[0], current[1]]
            for k, ell in array_directions:
                if (-1 < current[0] + k < arr_grid.shape[0]) and (-1 < current[1] + ell < arr_grid.shape[1]):
                    old_color = positions[current]
                    new_color = arr_grid[current[0] + k, current[1] + ell]
                    if not reached[current[0] + k, current[1] + ell] and \
                            relation_function((old_color, current[0], current[1]),
                                              (new_color, current[0] + k, current[1] + ell)):
                        reached[current[0] + k, current[1] + ell] = 1
                        queue.appendleft((current[0] + k, current[1] + ell))
        entities.append(Entity.make(positions, grid))
    return entities


find_color_entities_cache = {}


def find_color_entities(grid: tuple):
    # Uses breadth first search to find all connected components and puts them in a list
    if grid in find_color_entities_cache:
        return find_color_entities_cache[grid]
    if not grid:
        return {}

    appearing_colors = list2d_to_set(grid)
    arr_grid = np.array(grid)

    entity_dict = {color: [] for color in appearing_colors}
    for i, j in itertools.product(range(arr_grid.shape[0]), range(arr_grid.shape[1])):
        entity_dict[arr_grid[i, j]].append((i, j))

    entities = []
    for color, positions in entity_dict.items():
        new_entity = {}
        for position in positions:
            new_entity[position] = color
        entities.append(Entity.make(new_entity, grid))
    find_color_entities_cache[grid] = entities
    return entities


def move_entity(entity: Entity, new_grid: np.array, old_grid: tuple, vector: tuple,
                entering_color_map: callable, extend_grid: bool = False):
    # global move_entity_cache
    # if entity.freeze() in move_entity_cache:
    #     return move_entity_cache[entity.freeze()]
    old_grid = entity.grid
    new_positions = {}
    for position, color in entity.positions.items():
        if (extend_grid and 0 <= position[0] + vector[0] and 0 <= position[1] + vector[1]) or \
                (0 <= position[0] + vector[0] < new_grid.shape[0] and 0 <= position[1] + vector[1] < new_grid.shape[1]):
            if position[0] + vector[0] >= len(old_grid) or position[1] + vector[1] >= len(old_grid[0]):
                new_positions[(position[0] + vector[0], position[1] + vector[1])] = entering_color_map(0, old_grid[
                    position[0]][position[1]])
                new_color = 0
            else:
                new_color = old_grid[position[0] + vector[0]][position[1] + vector[1]]
            new_positions[(position[0] + vector[0], position[1] + vector[1])] = \
                entering_color_map(new_color, old_grid[position[0]][position[1]])
    # move_entity_cache[entity.freeze()] = new_positions
    return new_positions


def move(entities: list, grid: Optional[tuple] = None, vector_property: callable = None,
         leaving_color=0, entering_color_map=lambda orig_color, entity_color: entity_color, copy=False,
         extend_grid=True) -> tuple:
    # start_time = time.perf_counter()
    if len(entities) > 0:
        grid = entities[0].grid
        new_grid = np.array(grid)
    else:
        return {}, ()
    # accumulated_time[0] += time.perf_counter() - start_time
    new_positions_list = []
    for entity in entities:
        vector = vector_property(entity, grid)
        if vector is None \
                or not all((isinstance(length, int) or isinstance(length, np.int32) or (
                isinstance(length, float) and length.is_integer())) for length in vector):
            return {}, ()
        else:
            vector = tuple((int(length) for length in vector))
        if not copy:
            for position, color in entity.positions.items():
                new_grid[position[0], position[1]] = leaving_color
        # start_time = time.perf_counter()
        new_positions = move_entity(entity, new_grid, grid, vector, entering_color_map, extend_grid=extend_grid)
        # accumulated_time[1] += time.perf_counter() - start_time
        # start_time = time.perf_counter()
        if extend_grid and new_positions:
            # First we compute the new shape of the grid
            max_coordinates = [max((position[i] for position in new_positions.keys())) for i in range(2)]
            positives = [max(max_coordinate, original_max - 1) + 1 for max_coordinate, original_max in
                         zip(max_coordinates, new_grid.shape)]
            if tuple(positives) != new_grid.shape:
                extended_grid = np.zeros(positives)
                extended_grid[:new_grid.shape[0], :new_grid.shape[1]] = new_grid
                new_grid = extended_grid
        # accumulated_time[2] += time.perf_counter() - start_time
        # start_time = time.perf_counter()
        for position, color in new_positions.items():
            new_grid[position[0], position[1]] = new_positions[position[0], position[1]]
        new_positions_list.append(new_positions)
        # accumulated_time[3] += time.perf_counter() - start_time
    # start_time = time.perf_counter()
    new_grid_tuple = to_tuple(new_grid)
    new_entities = [Entity.make(new_positions, new_grid_tuple) for new_positions in new_positions_list]
    # accumulated_time[4] += time.perf_counter() - start_time
    return new_entities, new_grid_tuple


def reflect_about_line(entities: list, grid: Optional[tuple] = None, line_prop: callable = None,
                       color_prop: callable = lambda x, y: next(iter(x.color())), leaving_color=0,
                       color_strategy='replace', copy=True, copy_entities=False, extend_grid=True,
                       change_grid=True) -> tuple:
    """
    Reflects each entities about a line coming from a line property

    :param change_grid: Determines whether to stamp the new entities onto the grid. Used primarily as part of a more
    complicated function
    :param copy_entities: Determines whether to retain the old entities. Primarily used for rotation group functions
    :param extend_grid: Whether to extend the grid when reflecting
    :param copy: Whether to leave the original entity
    :param leaving_color: The color to leave behind if we're not copying
    :param entities: entities to use to get the property values
    :param grid: not used, necessary for compatibility
    :param line_prop: property that tells us the line to reflect about
    :param color_prop: the default color for the object after reflection
    :param color_strategy: How to handle the existing color of the grid when placing the object
    :return: the modified entities and grid in a tuple
    """
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    if not old_grid:
        return {}, ()

    grid_arr = np.array(old_grid)
    new_entities = []
    positions_of_new_entities = []
    min_coordinates = [0., 0.]
    max_coordinates = [float(grid_arr.shape[0] - 1), float(grid_arr.shape[1] - 1)]
    for entity in entities:
        new_positions = {}
        line = line_prop(entity, entity.grid)
        if line is None:
            return {}, ()
        if not copy:
            for position in entity.positions:
                grid_arr[position[0], position[1]] = 0
        # color = color_prop(entity, entity.grid)
        for position, color in entity.positions.items():
            if line[1] in {0., 1.}:
                line1_int = int(line[1])
                new_coordinate = position[line1_int] + 2 * (line[0] - position[line1_int])
                if new_coordinate < min_coordinates[line1_int]:
                    min_coordinates[line1_int] = new_coordinate
                elif new_coordinate > max_coordinates[line1_int]:
                    max_coordinates[line1_int] = new_coordinate

                if not new_coordinate.is_integer():
                    return {}, ()
                new_position = tuple((position[i] if i != line1_int else int(new_coordinate) for i in range(2)))
            elif line[1] in {-0.5, 0.5}:
                sign = np.sign(line[1])
                steps = (position[0] + sign * position[1]) / 2 - sign * line[0]
                new_position = (position[0] - 2 * steps, position[1] - sign * 2 * steps)
                for i in range(2):
                    if new_position[i] < min_coordinates[i]:
                        min_coordinates[i] = new_position[i]
                    elif new_position[i] > max_coordinates[i]:
                        max_coordinates[i] = new_position[i]
                if any((not coordinate.is_integer() for coordinate in new_position)):
                    return {}, ()
                else:
                    new_position = tuple((int(coordinate) for coordinate in new_position))
            else:
                raise ValueError(f"line[1] is of invalid value {line[1]}")
            if extend_grid or (0 <= new_position[0] < grid_arr.shape[0] and 0 <= new_position[1] < grid_arr.shape[1]):
                new_positions[new_position] = color
            # grid_arr[new_position[0], new_position[1]] = color
        if new_positions:
            positions_of_new_entities.append(new_positions)
    if extend_grid:
        min_coordinates = tuple((int(min_coordinate) for min_coordinate in min_coordinates))
        max_coordinates = tuple((int(max_coordinate) for max_coordinate in max_coordinates))
        new_grid_arr = extend_grid_via_extreme_coordinates(min_coordinates, max_coordinates, grid_arr)
        positions_of_new_entities = extend_entities_positions(min_coordinates, positions_of_new_entities)
    else:
        new_grid_arr = grid_arr

    if change_grid:
        stamp_entities_positions(positions_of_new_entities, new_grid_arr)
    new_grid = to_tuple(new_grid_arr)
    new_entities = [Entity.make(positions, new_grid) for positions in positions_of_new_entities]

    if copy_entities:
        old_entities_positions = extend_entities_positions(min_coordinates, [entity.positions for entity in entities])
        old_entities = [Entity.make(positions, new_grid) for positions in old_entities_positions]
        new_entities.extend(old_entities)
    return new_entities, new_grid


def rotate_via_reflects(entities: list, grid: Optional[tuple] = None, line_prop1: callable = None,
                        line_prop2: callable = None,
                        color_prop: callable = lambda x, y: next(iter(x.color())), leaving_color=0,
                        color_strategy='replace', copy=True, copy_entities=False, extend_grid=True) -> tuple:
    """
    Applies two reflections sequentially giving a rotation

    :param entities:
    :param grid:
    :param line_prop1:
    :param line_prop2:
    :param color_prop:
    :param leaving_color:
    :param color_strategy:
    :param copy:
    :param extend_grid:
    :return:
    """
    line_prop2_values = [line_prop2(entity, entity.grid) for entity in entities]
    new_entities, new_grid = reflect_about_line(entities, grid, line_prop1, copy=True, copy_entities=copy_entities,
                                                change_grid=False, extend_grid=extend_grid)

    def const_line_prop2(entity, grid):
        return line_prop2_values[new_entities.index(entity)]

    new_entities, new_grid = reflect_about_line(new_entities, new_grid, const_line_prop2, copy=extend_grid,
                                                copy_entities=copy_entities, extend_grid=extend_grid)
    return new_entities, new_grid


def rotate_position(position: tuple, pivot: tuple, quarter_steps: int) -> tuple:
    if quarter_steps % 4 == 1:
        return position[1] + pivot[0] - pivot[1], pivot[0] + pivot[1] - position[0]
    elif quarter_steps % 4 == 2:
        return 2 * pivot[0] - position[0], 2 * pivot[1] - position[1]
    elif quarter_steps % 4 == 3:
        return pivot[0] + pivot[1] - position[1], position[0] + pivot[1] - pivot[0]
    else:
        return position[0], position[1]


def rotate_about_point(entities: list, grid: Optional[tuple] = None, point_prop: callable = None,
                       quarter_steps=1, color_prop: callable = lambda x, y: next(iter(x.color())), leaving_color=0,
                       color_strategy='replace', copy=True, extend_grid=True,
                       change_grid=True) -> tuple:
    """
    Applies two reflections sequentially giving a rotation

    :param entities:
    :param grid:
    :param line_prop1:
    :param line_prop2:
    :param color_prop:
    :param leaving_color:
    :param color_strategy:
    :param copy:
    :param extend_grid:
    :return:
    """
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    if not old_grid:
        return {}, ()

    grid_arr = np.array(old_grid)
    new_entities = []
    positions_of_new_entities = []
    min_coordinates = [0., 0.]
    max_coordinates = [float(grid_arr.shape[0] - 1), float(grid_arr.shape[1] - 1)]
    for entity in entities:
        new_positions = {}
        point = point_prop(entity, entity.grid)
        if point is None:
            return {}, ()
        if not copy:
            for position in entity.positions:
                grid_arr[position[0], position[1]] = 0
        for position, color in entity.positions.items():
            new_position = rotate_position(position, point, quarter_steps)

            if not new_position[0].is_integer() or not new_position[0].is_integer():
                return {}, ()
            else:
                new_position = tuple((int(coord) for coord in new_position))

            for axis in range(2):
                min_coordinates[axis] = min(min_coordinates[axis], new_position[axis])
                max_coordinates[axis] = max(max_coordinates[axis], new_position[axis])
            if extend_grid or (0 <= new_position[0] < grid_arr.shape[0] and 0 <= new_position[1] < grid_arr.shape[1]):
                new_positions[new_position] = color
        if new_positions:
            positions_of_new_entities.append(new_positions)
    if extend_grid:
        min_coordinates = tuple((int(min_coordinate) for min_coordinate in min_coordinates))
        max_coordinates = tuple((int(max_coordinate) for max_coordinate in max_coordinates))
        new_grid_arr = extend_grid_via_extreme_coordinates(min_coordinates, max_coordinates, grid_arr)
        positions_of_new_entities = extend_entities_positions(min_coordinates, positions_of_new_entities)
    else:
        new_grid_arr = grid_arr

    if change_grid:
        stamp_entities_positions(positions_of_new_entities, new_grid_arr)
    new_grid = to_tuple(new_grid_arr)
    new_entities = [Entity.make(positions, new_grid) for positions in positions_of_new_entities]
    return new_entities, new_grid


def apply_rotation_group_old(entities: list, grid: Optional[tuple] = None,
                             line_prop1=None, line_prop2=None, extend_grid=True):
    """
    Applies a rotation four times. If the element has order four, this gives the action of Z_4

    :param entities:
    :param grid:
    :param line_prop1:
    :param line_prop2:
    :return:
    """
    line_prop1_values = [line_prop1(entity, entity.grid) for entity in entities]
    line_prop2_values = [line_prop2(entity, entity.grid) for entity in entities]
    new_entities, new_grid = rotate_via_reflects(entities, grid, line_prop1, line_prop2, extend_grid=extend_grid)
    for _ in range(2):
        def const_line_prop1(entity, grid):
            return line_prop1_values[new_entities.index(entity)]

        def const_line_prop2(entity, grid):
            return line_prop2_values[new_entities.index(entity)]

        new_entities, new_grid = rotate_via_reflects(new_entities, new_grid, const_line_prop1, const_line_prop2,
                                                     extend_grid=True)
    return new_entities, new_grid


def apply_rotation_group(entities: list, grid: Optional[tuple] = None,
                         point_prop=None, extend_grid=True):
    """
    Applies a rotation four times. If the element has order four, this gives the action of Z_4

    :param entities:
    :param grid:
    :param point_prop:
    :param extend_grid:
    :return:
    """
    point_values = [point_prop(entity, entity.grid) for entity in entities]

    # We apply rotations to the same entities to allow for better caching
    for i in range(1, 4):
        def const_point_prop(entity, grid):
            return point_values[entities.index(entity)]

        entities, grid = rotate_about_point(entities, grid, const_point_prop,
                                            extend_grid=extend_grid,
                                            quarter_steps=i)

    return entities, grid


def apply_klein_vier_group(entities: list, grid: Optional[tuple] = None, line_prop1=None, line_prop2=None,
                           extend_grid=True):
    """
    Applies the combinations of actions of two perpendicular reflections, giving a four element group

    :param entities:
    :param grid:
    :param line_prop1:
    :param line_prop2:
    :return:
    """
    line_prop1_values = [line_prop1(entity, entity.grid) for entity in entities]
    line_prop2_values = [line_prop2(entity, entity.grid) for entity in entities]

    new_entities, new_grid = reflect_about_line(entities, grid, line_prop1, extend_grid=extend_grid)

    def const_line_prop2(entity, grid):
        return line_prop2_values[new_entities.index(entity)]

    new_entities, new_grid = reflect_about_line(new_entities, new_grid, const_line_prop2, extend_grid=extend_grid)

    def const_line_prop1(entity, grid):
        return line_prop1_values[new_entities.index(entity)]

    new_entities, new_grid = reflect_about_line(new_entities, new_grid, const_line_prop1, extend_grid=extend_grid)

    return new_entities, new_grid


def place_line(entities: list, grid: Optional[tuple] = None, line_prop: Property = None,
               color_prop: Property = None,
               color_strategy: str = 'original') -> tuple:
    """
    INCOMPLETE

    :param entities: entities to use to get the property values
    :param grid: not used, necessary for compatibility
    :param line_prop: property that tells us the line to draw
    :param color_prop: the default color for the line
    :param color_strategy: How to handle the existing color of the grid when placing the line
    :return: the modified entities and grid in a tuple
    """
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    grid_arr = np.array(old_grid)
    new_entities = []
    for entity in entities:
        line = line_prop(entity, entity.grid)
        color = color_prop(entity, entity.grid)
        ...

    return new_entities, to_tuple(grid_arr)


def place_line_segment(entities: list, grid: Optional[tuple] = None, point_prop1: Property = None,
                       point_prop2: Property = None, color_prop: Property = None,
                       color_strategy: str = 'original') -> tuple:
    """
    INCOMPLETE

    :param entities: entities to use to get the property values
    :param grid: not used, necessary for compatibility
    :param point_prop1: the start of the line
    :param point_prop2: the end of the line
    :param color_prop: the default color for the line
    :param color_strategy: How to handle the existing color of the grid when placing the line
    :return: the modified entities and grid
    """
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    grid_arr = np.array(old_grid)
    new_entities = []
    for entity in entities:
        point1 = point_prop1(entity, entity.grid)
        point2 = point_prop2(entity, entity.grid)
        color = color_prop(entity, entity.grid)
        ...

    return new_entities, to_tuple(grid_arr)


def place_shape(entities: list, grid: Optional[tuple] = None, point_prop: Property = None,
                shape_prop: Property = None, color_strategy: str = 'original') -> tuple:
    """

    :param entities: entities to use to get the property values
    :param grid: not used, necessary for compatibility
    :param point_prop: location for the center of the shape to be
    :param shape_prop: the shape to be placed
    :param color_strategy: Determines how to color the placed shape. Options are 'original', 'extend_non_0', and 'replace_0'
    :return:
    """
    # Adds a shape to the original board
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    if color_strategy not in {'original', 'extend_non_0', 'replace_0'}:
        raise Exception(f'Invalid color strategy: {color_strategy}')
    grid_arr = np.array(old_grid)
    new_entities = []
    for entity in entities:
        new_entity_positions = entity.positions.copy()
        point = point_prop(entity, entity.grid)
        fixed_shape = shape_prop(entity, entity.grid)
        if fixed_shape is None or point is None:
            return {}, ()

        y_coord, x_coord = point
        color_map = {}

        if color_strategy == 'original' or color_strategy == 'replace_0':
            color_map = {i: i for i in range(10)}
        elif color_strategy == 'extend_non_0':
            for position, color in fixed_shape:
                new_position_0, new_position_1 = position[0] + y_coord, position[1] + x_coord
                if new_position_0 < 0 or new_position_1 < 0:
                    return {}, ()
                if not isinstance(new_position_0, int) and new_position_0.is_integer():
                    new_position_0 = int(new_position_0)
                if not isinstance(new_position_1, int) and new_position_1.is_integer():
                    new_position_1 = int(new_position_1)
                try:
                    if grid_arr[new_position_0, new_position_1] != 0:
                        if color in color_map and grid_arr[new_position_0, new_position_1] != color_map[color]:
                            return {}, ()
                        else:
                            color_map[color] = grid_arr[new_position_0, new_position_1]
                except IndexError or KeyError:
                    return {}, ()
            for i in range(10):
                if i not in color_map:
                    color_map[i] = i

        for position, color in fixed_shape:
            new_position_0, new_position_1 = position[0] + y_coord, position[1] + x_coord
            if new_position_0 < 0 or new_position_1 < 0:
                return {}, ()
            if not isinstance(new_position_0, int) and new_position_0.is_integer():
                new_position_0 = int(new_position_0)
            if not isinstance(new_position_1, int) and new_position_1.is_integer():
                new_position_1 = int(new_position_1)
            try:
                new_color = color_map[color] \
                    if (color_strategy != 'replace_0' or grid_arr[new_position_0, new_position_1] == 0) \
                    else grid_arr[new_position_0, new_position_1]
                grid_arr[new_position_0, new_position_1] = new_color
                new_entity_positions[(new_position_0, new_position_1)] = new_color
            except IndexError or KeyError:
                return {}, ()
        new_entities.append(Entity.make(new_entity_positions, entity.grid))
    return new_entities, to_tuple(grid_arr)


def output_shape_as_grid(entities: list, grid: Optional[tuple] = None, shape_prop: Property = None):
    if len(entities) > 0:
        if grid is None:
            grid = entities[0].grid
    # else:
    #     return {}, ()

    shape = shape_prop(None, grid)
    if shape is None:
        return {}, ()
    min_y, max_y = min((y for (y, x), color in shape)), max((y for (y, x), color in shape))
    min_x, max_x = min((x for (y, x), color in shape)), max((x for (y, x), color in shape))
    new_grid = np.zeros((int(max_y + 1 - min_y), int(max_x + 1 - min_x)))
    for (y, x), color in shape:
        # print((y, x), color)
        new_grid[int(y + min_y), int(x + min_x)] = color
    return dict(shape), to_tuple(new_grid)


def replace_color(entities: list, grid: Optional[tuple] = None,
                  source_color_prop: Optional[Property] = None,
                  target_color_prop: Optional[Property] = None):
    if len(entities) > 0:
        if grid is None:
            grid = entities[0].grid
        new_grid = np.array(grid)
    else:
        return {}, ()

    if len(grid) == 0 or len(grid) != len(entities[0].grid) or len(grid[0]) != len(entities[0].grid[0]):
        return {}, ()

    new_positions_list = []
    for entity in entities:
        source_colors = source_color_prop(entity, entity.grid)
        target_colors = target_color_prop(entity, entity.grid)
        if source_colors is None or target_colors is None or len(target_colors) != 1:
            # If target color has more than just one element the map becomes ambiguous, so just return a trivial result
            return {}, ()
        target_color = next(iter(target_colors))

        new_positions = {}
        for position, color in entity.positions.items():
            new_color = target_color if color in source_colors else color
            new_positions[position] = new_color
            if new_color != color:
                new_grid[position[0], position[1]] = new_color
        new_positions_list.append(new_positions)
    new_entities = [Entity.make(new_positions, to_tuple(new_grid)) for new_positions in new_positions_list]
    return new_entities, to_tuple(new_grid)


def replace_color_in_frame(box_y_min, box_y_max, box_x_min, box_x_max, grid: tuple,
                           source_colors: frozenset,
                           target_color: int):
    min_y = len(grid) - 1
    max_y = 0
    assert len(grid) > 0
    min_x = len(grid[0]) - 1
    max_x = 0
    grid_arr = np.array(grid)
    min_y = min(min_y, box_y_min)
    max_y = max(max_y, box_y_max)
    min_x = min(min_x, box_x_min)
    max_x = max(max_x, box_x_max)
    source_colors = list(source_colors)
    places_grid_in_source = np.isin(grid_arr[min_y: max_y + 1, min_x: max_x + 1], source_colors)
    grid_arr[min_y: max_y + 1, min_x: max_x + 1] = np.where(places_grid_in_source, target_color,
                                                            grid_arr[min_y: max_y + 1, min_x: max_x + 1])
    return to_tuple(grid_arr)


def replace_colors_in_entities_frame(entities, grid: Optional[tuple] = None, offsets=(0, 0, 0, 0),
                                     source_color_prop: Optional[Property] = None,
                                     target_color_prop: Optional[Property] = None):
    if len(entities) > 0:
        grid = entities[0].grid
    else:
        # print('No entities')
        return {}, ()
    grid_min_y = len(grid) - 1
    grid_max_y = 0
    assert len(grid) > 0
    grid_min_x = len(grid[0]) - 1
    grid_max_x = 0
    # print(source_color_prop, target_color_prop)
    for entity in entities:
        source_colors = source_color_prop(entity, entity.grid)
        target_colors = target_color_prop(entity, entity.grid)
        # print(source_colors, target_colors)
        if source_colors is None or target_colors is None or len(target_colors) != 1:
            # If target color has more than just one element the map becomes ambiguous, so just return a trivial result
            # print(f'source_colors={source_colors}, target_colors={target_colors}')
            return {}, ()
        else:
            target_color = next(iter(target_colors))
        min_y = min(grid_min_y, entity.min_coord(0)) + offsets[0]
        max_y = max(grid_max_y, entity.max_coord(0)) + offsets[1]
        min_x = min(grid_min_x, entity.min_coord(1)) + offsets[2]
        max_x = max(grid_max_x, entity.max_coord(1)) + offsets[3]
        grid = replace_color_in_frame(min_y, max_y, min_x, max_x, grid, source_colors, target_color)
    return {}, grid


def crop_grid(grid: tuple, y_range: tuple, x_range: tuple) -> tuple:
    new_grid = np.array(grid)
    y_begin, y_end = y_range
    x_begin, x_end = x_range
    return to_tuple(new_grid[y_begin:y_end + 1, x_begin:x_end + 1])


def crop_entities(entities, grid: Optional[tuple] = None, offsets=(0, 0, 0, 0)):
    if len(entities) > 0:
        grid = entities[0].grid
    else:
        return {}, ()
    min_y = len(grid) - 1
    max_y = 0
    assert len(grid) > 0
    min_x = len(grid[0]) - 1
    max_x = 0
    for entity in entities:
        min_y = min(min_y, entity.min_coord(0)) + offsets[0]
        max_y = max(max_y, entity.max_coord(0)) + offsets[1]
        min_x = min(min_x, entity.min_coord(1)) + offsets[2]
        max_x = max(max_x, entity.max_coord(1)) + offsets[3]
    new_grid = crop_grid(grid, (min_y, max_y), (min_x, max_x))
    new_entities = []
    for entity in entities:
        temp_position = {}
        for position, color in entity.positions.items():
            temp_position[(position[0] - min_y, position[1] - min_x)] = color
        new_entities.append(Entity.make(temp_position, new_grid))

    return new_entities, new_grid


adjacent_direction_cache = {}


def adjacent_direction(entity1: Entity, entity2: Entity):
    if entity1 is None or entity2 is None:
        return []
    key = (frozenset(entity1.positions.keys()), frozenset(entity2.positions.keys()))
    if key in adjacent_direction_cache:
        return adjacent_direction_cache[key]
    directions = []
    test_directions = list(ALL_DIRECTIONS)
    test_directions.remove((0, 0))
    for position1, position2 in itertools.product(entity1.positions.keys(), entity2.positions.keys()):
        for direction in test_directions:
            if position1[0] + direction[0] == position2[0] and position1[1] + direction[1] == position2[1]:
                directions.append(direction)
                test_directions.remove(direction)
                break
    adjacent_direction_cache[key] = directions
    return directions


def direction_sign_to_vector(direction, sign, value):
    vector = [0, 0]
    vector[direction] = value * sign
    return tuple(vector)


collision_directions_cache = {}


def collision_directions(entity1: Entity, entity2: Entity, adjustment=0):
    if entity1 is None or entity2 is None:
        return []
    cache_key = (frozenset(entity1.positions.keys()), frozenset(entity2.positions.keys()), adjustment)
    if cache_key in collision_directions_cache:
        return collision_directions_cache[cache_key]
    min_distances = {(i, sign): float('inf') for i, sign in itertools.product(range(2), (1, -1))}
    for position1, position2 in itertools.product(entity1.positions.keys(), entity2.positions.keys()):
        for i, sign in itertools.product(range(2), (1, -1)):
            if position1[int(not i)] == position2[int(not i)] and position1[i] * sign < position2[i] * sign:
                min_distances[(i, sign)] = min(min_distances[(i, sign)],
                                               position2[i] * sign - position1[i] * sign - 1)
    out = frozenset(
        direction_sign_to_vector(*key, value + (adjustment if value != 0 else 0)) for key, value in
        min_distances.items() if
        value != float('inf'))
    collision_directions_cache[cache_key] = out
    return out


def surrounded(entity1, entity2):
    return all((entity2.min_coord(axis) < entity1.min_coord(axis) for axis in range(2))) and all(
        (entity1.max_coord(axis) < entity2.max_coord(axis) for axis in range(2)))


def generate_base_relations():
    # 39
    Relation(lambda entity1, entity2: adjacent_direction(entity1, entity2)[0] if len(
        adjacent_direction(entity1, entity2)) == 1 else None,
             nll=1 + np.log(2),
             name='find the unique touching direction to',
             output_types=frozenset({'vector'}))
    # 40
    Relation(lambda entity1, entity2: True if adjacent_direction(entity1, entity2) else False,
             nll=1 + np.log(2),
             name='are touching',
             output_types=frozenset({'bool'}))
    # 41
    Relation(lambda entity1, entity2: next(iter(collision_directions(entity1, entity2))) if len(
        collision_directions(entity1, entity2)) == 1 else None,
             nll=1 + np.log(2), name='the unique pre-collision vector to',
             output_types=frozenset({'vector'}))
    # 41.5
    Relation(lambda entity1, entity2: next(iter(collision_directions(entity1, entity2, adjustment=1)))
    if len(collision_directions(entity1, entity2)) == 1 else None,
             nll=1 + np.log(2), name='the unique collision vector to',
             output_types=frozenset({'vector'}))
    # 42
    Relation(lambda entity1, entity2: entity1.colors() == entity2.colors(),
             nll=1 + np.log(2), name='shares a color with',
             output_types=frozenset({'bool'}))
    # 43
    Relation(lambda entity1, entity2: entity1.shape() == entity2.shape(),
             nll=1 + np.log(2), name='has the same shape as',
             output_types=frozenset({'bool'}))
    # 44
    Relation(lambda entity1, entity2: entity1.uncolored_shape() == entity2.uncolored_shape(),
             nll=1 + np.log(2), name='has the same uncolored shape as',
             output_types=frozenset({'bool'}))
    # # 44.5
    Relation(lambda entity1, entity2: surrounded(entity1, entity2),
             nll=1 + np.log(2), name='is surrounded by',
             output_types=frozenset({'bool'}))


def generate_base_properties(case, entity_finder):
    entity_properties = [
        # 0
        Property(lambda x: x.entity.num_points(), nll=np.log(2),
                 name='the number of points',
                 output_types=frozenset({'quantity'}),
                 entity_finder=entity_finder),
        # 1
        Property(lambda x: x.entity.colors(), nll=0,
                 name='the colors',
                 output_types=frozenset({'color'}),
                 entity_finder=entity_finder),

        # Coordinate Properties
        # 2
        Property(lambda x: x.entity.center(axis=0), nll=np.log(2),
                 name='the center y coordinate',
                 output_types=frozenset({'y_coordinate'}),
                 entity_finder=entity_finder),
        # 3
        Property(lambda x: x.entity.center(axis=1), nll=np.log(2),
                 name='the center x coordinate',
                 output_types=frozenset({'x_coordinate'}),
                 entity_finder=entity_finder),
        # 4
        Property(lambda x: float(x.entity.min_coord(axis=0)), np.log(4),
                 name='the smallest y coordinate',
                 output_types=frozenset({'y_coordinate'}),
                 entity_finder=entity_finder),
        # 5
        Property(lambda x: float(x.entity.min_coord(axis=1)), np.log(4),
                 name='the smallest x coordinate',
                 output_types=frozenset({'x_coordinate'}),
                 entity_finder=entity_finder),
        # 6
        Property(lambda x: float(x.entity.max_coord(axis=0)), np.log(4),
                 name='the largest y coordinate',
                 output_types=frozenset({'y_coordinate'}),
                 entity_finder=entity_finder),
        # 7
        Property(lambda x: float(x.entity.max_coord(axis=1)), np.log(4),
                 name='the largest x coordinate',
                 output_types=frozenset({'x_coordinate'}),
                 entity_finder=entity_finder),

        # Line properties
        Property(lambda x: (float(x.entity.center(axis=1)) / 2., 1.),
                 np.log(4),
                 name="the entity\'s vertical center line",
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.center(axis=0) / 2., 0.),
                 np.log(4),
                 name='the entity\'s horizontal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=1), -0.5),
                 np.log(4),
                 name='the entity\'s back diagonal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.max_coord(axis=1) / 2., 0.5),
                 np.log(4),
                 name='the entity\'s forward diagonal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),

        Property(lambda x: (x.entity.max_coord(axis=1) - 0.5, 1.),
                 np.log(4),
                 name='the entity\'s vertical right-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=1) - 0.5, 1.),
                 np.log(4),
                 name='the entity\'s vertical left-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(x.entity.max_coord(axis=0) - 0.5), 0.),
                 np.log(4),
                 name='the entity\'s horizontal bottom-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=0) - 0.5, 0.),
                 np.log(4),
                 name='the entity\'s horizontal top-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),

        # Point properties
        # 8
        Property(lambda x: (x.entity.center(axis=0), x.entity.center(axis=1)), nll=0,
                 name='the center point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=0) - 0.5, x.entity.min_coord(axis=1) - 0.5), nll=np.log(4),
                 name='the top-left corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=0) - 0.5, x.entity.max_coord(axis=1) + 0.5), nll=np.log(4),
                 name='the top-right corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.max_coord(axis=0) + 0.5, x.entity.min_coord(axis=1) - 0.5), nll=np.log(4),
                 name='the bottom-left corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.max_coord(axis=0) + 0.5, x.entity.max_coord(axis=1) + 0.5), nll=np.log(4),
                 name='the bottom-right corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),

        # Length Properties
        # 9
        Property(lambda x: float(x.entity.max_coord(axis=0) - x.entity.min_coord(axis=0) + 1),
                 np.log(4),
                 name='the y length',
                 output_types=frozenset({'y_length'}),
                 entity_finder=entity_finder),
        # 9.5
        Property(lambda x: -(float(x.entity.max_coord(axis=0) - x.entity.min_coord(axis=0) + 1)),
                 np.log(4),
                 name='the negative y length',
                 output_types=frozenset({'y_length'}),
                 entity_finder=entity_finder),
        # 10
        Property(lambda x: float(x.entity.max_coord(axis=1) - x.entity.min_coord(axis=1) + 1),
                 np.log(4),
                 name='the x length',
                 output_types=frozenset({'x_length'}),
                 entity_finder=entity_finder),
        # 10.5
        Property(lambda x: -(float(x.entity.max_coord(axis=1) - x.entity.min_coord(axis=1) + 1)),
                 np.log(4),
                 name='the negative x length',
                 output_types=frozenset({'x_length'}),
                 entity_finder=entity_finder),

        # Shape-based properties
        # 11
        Property(lambda x: x.entity.shape(), 0,
                 name='the shape',
                 output_types=frozenset({'shape'}),
                 entity_finder=entity_finder),
        # 12
        Property(lambda x: x.entity.is_a_rectangle(), np.log(4),
                 name='is a rectangle',
                 output_types=frozenset({'bool'}),
                 entity_finder=entity_finder),
        # 13
        Property(lambda x: x.entity.is_a_square(), np.log(2),
                 name='is a square',
                 output_types=frozenset({'bool'}),
                 entity_finder=entity_finder),
        # 14
        Property(lambda x: x.entity.is_a_line(), np.log(2),
                 name='is a line',
                 output_types=frozenset({'bool'}),
                 entity_finder=entity_finder)
    ]

    for prop in entity_properties:
        prop.requires_entity = True

    grid_properties = [
        # 15
        Property(lambda x: 2 + len(x.entities), 0, name='the number of entities',
                 output_types=frozenset({'quantity'}),
                 entity_finder=entity_finder),
        # 16
        Property(lambda x: float(np.array(x.grid).shape[0]), np.log(2), name='the height of the grid',
                 output_types=frozenset({'y_coordinate', 'y_length'}),
                 entity_finder=entity_finder),
        # 17
        Property(lambda x: float(np.array(x.grid).shape[1]), np.log(2), name='the width of the grid',
                 output_types=frozenset({'x_coordinate', 'x_length'}),
                 entity_finder=entity_finder),

        # 18
        Property(lambda x: (float(np.array(x.grid).shape[0] - 1) / 2., float(np.array(x.grid).shape[1] - 1) / 2.),
                 0,
                 name='the center point of the grid',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        # 101
        Property(lambda x: float(np.array(x.grid).shape[0] - 1) / 2.,
                 np.log(4),
                 name='the center y coordinate of the grid',
                 output_types=frozenset({'y_coordinate'}),
                 entity_finder=entity_finder),
        # 102
        Property(lambda x: float(np.array(x.grid).shape[1] - 1) / 2.,
                 np.log(4),
                 name='the center x coordinate of the grid',
                 output_types=frozenset({'x_coordinate'}),
                 entity_finder=entity_finder),
        # 19
        Property(lambda x: True,
                 0,
                 name='True',
                 output_types=frozenset({'bool'}),
                 is_constant=True,
                 entity_finder=entity_finder),

        Property(lambda x: (float(np.array(x.grid).shape[1] - 1) / 2., 1.),
                 np.log(4),
                 name='the vertical center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[0] - 1) / 2., 0.),
                 np.log(4),
                 name='the horizontal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (0., -0.5),
                 np.log(4),
                 name='the back diagonal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[1] - 1.) / 2., 0.5),
                 np.log(4),
                 name='the forward diagonal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),

        Property(lambda x: (float(np.array(x.grid).shape[1] - 0.5), 1.),
                 np.log(4),
                 name='the vertical right-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (- 0.5, 1.),
                 np.log(4),
                 name='the vertical left-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[0] - 0.5), 0.),
                 np.log(4),
                 name='the horizontal bottom-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (- 0.5, 0.),
                 np.log(4),
                 name='the horizontal top-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),

        # Grid Corners
        Property(lambda x: (0 - 0.5, 0 - 0.5), nll=np.log(4),
                 name='the grid top-left corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (0 - 0.5, float(np.array(x.grid).shape[1]) - 0.5), nll=np.log(4),
                 name='the grid top-right corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[0]) - 0.5, 0 - 0.5), nll=np.log(4),
                 name='the grid bottom-left corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[0]) - 0.5, float(np.array(x.grid).shape[1]) - 0.5),
                 nll=np.log(4),
                 name='the grid bottom-right corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder)
    ]
    # 20 - 24
    grid_properties.extend([Property(lambda x, i=i: i, nll=2 * i, name=f'{i}',
                                     output_types=frozenset({'x_coordinate', 'y_coordinate',
                                                             'x_length', 'y_length',
                                                             'quantity'}),
                                     is_constant=True,
                                     entity_finder=entity_finder) for i in range(0, 5)])
    # 25 - 29
    grid_properties.extend([Property(lambda x, i=i: -i, nll=2 * i, name=f'-{i}',
                                     output_types=frozenset({'x_length', 'y_length',
                                                             'quantity'}),
                                     is_constant=True,
                                     entity_finder=entity_finder) for i in range(1, 5)])
    appearing_colors = Counter()
    shape_counter = Counter()
    non_black_shape_counter = Counter()
    uncolored_shape_counter = Counter()
    y_counter = Counter()
    x_counter = Counter()
    point_counter = Counter()
    non_black_uncolored_shape_counter = Counter()
    non_black_entity_finder = EntityFinder(
        lambda grid: find_components(grid, relation='not_zero', directions=ALL_DIRECTIONS))

    for grid in case['train']:
        # We use the data to determine the likely constant properties, and set their likelihood
        for color in (list2d_to_set(grid['output']) - list2d_to_set(grid['input'])) | \
                     (list2d_to_set(grid['input']) - list2d_to_set(grid['output'])):
            appearing_colors[color] = 10
        appearing_colors.update(list2d_to_set(grid['input']))
        appearing_colors.update(list2d_to_set(grid['output']))

        output_entities = entity_finder(grid['output'])
        output_entities_non_black = non_black_entity_finder(grid['output'])

        shape_counter.update(Entity.shapes(output_entities))
        non_black_shape_counter.update(Entity.shapes(output_entities_non_black))

        non_black_uncolored_shape_counter.update(Entity.uncolored_shapes(output_entities_non_black))
        uncolored_shape_counter.update(Entity.uncolored_shapes(output_entities))

        # Point and coordinate constants
        y_counter.update({(entity.center(0)) for entity in output_entities})
        x_counter.update({(entity.center(1)) for entity in output_entities})
        point_counter.update({(entity.center(0), entity.center(1)) for entity in output_entities})
    for color, count in appearing_colors.items():
        # If the color appears in every example, it is also very likely to be used
        if count == len(case['train']) * 2:
            appearing_colors[color] = 10

    # We determine which shapes most likely to be used as constants by counting the number of times the shape appears
    # in an uncolored sense to allow for otherwise identical shapes with different colorings
    common_uncolored_shapes = {uncolored_shape for uncolored_shape, count in
                               itertools.chain(uncolored_shape_counter.items(),
                                               non_black_uncolored_shape_counter.items()) if
                               count > 1}

    combined_shapes = {shape: max(shape_counter[shape], non_black_shape_counter[shape]) for shape in
                       itertools.chain(shape_counter.keys(), non_black_shape_counter.keys())
                       if frozenset({x[0] for x in shape}) in common_uncolored_shapes}

    # 30
    grid_properties.extend(
        [Property(lambda x, color=color, count=count: frozenset({color}),
                  np.log(max(len(appearing_colors) - count, 1)),
                  name=f'color {color}',
                  output_types=frozenset({'color'}),
                  is_constant=True,
                  entity_finder=entity_finder) for color, count in appearing_colors.items()])
    # 31
    shapes = [Property(lambda x, shape=shape: shape, nll=max(np.log(10) - count * np.log(2), 0), name=f'shape {shape}',
                       output_types=frozenset({'shape'}), is_constant=True, entity_finder=entity_finder) for
              shape, count in combined_shapes.items()]
    grid_properties.extend(shapes)
    # for shape in shapes:
    #     print(shape, shape.nll)
    # 32
    grid_properties.extend(
        [Property(lambda x: y_coordinate,
                  nll=max(np.log(len(case['train']) + 1 - count), 0),
                  name=f'y = {y_coordinate}',
                  output_types=frozenset({'y_coordinate'}),
                  is_constant=True,
                  entity_finder=entity_finder)
         for y_coordinate, count in y_counter.items()
         if count > 1 and (y_coordinate > 4 or not float(y_coordinate).is_integer())])
    # 33
    grid_properties.extend(
        [Property(lambda x: x_coordinate,
                  nll=max(len(case['train']) - count, 0),
                  name=f'x = {x_coordinate}',
                  output_types=frozenset({'x_coordinate'}),
                  is_constant=True,
                  entity_finder=entity_finder)
         for x_coordinate, count in x_counter.items()
         if count > 1 and (x_coordinate > 4 or not float(x_coordinate).is_integer())])

    # 100
    grid_properties.extend(
        [Property(lambda x: point,
                  nll=max(np.log(3) - count, 0),
                  name=f'point {point}',
                  output_types=frozenset({'point'}),
                  is_constant=True,
                  entity_finder=entity_finder)
         for point, count in point_counter.items()
         if count > 1])
    add_one(grid_properties), add_one(entity_properties)
    return grid_properties, entity_properties


def main():
    file_prefix = '../input/abstraction-and-reasoning-challenge/test/'
    fieldnames = ['output_id', 'output']
    output_file_name = 'submission.csv'
    with open(output_file_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for i, filename in enumerate(os.listdir(file_prefix)):
        start_time = time.perf_counter()
        print(f"Starting task {i} at {start_time}")
        with open(file_prefix + filename) as f:
            raw_task = json.load(f)
        task = tuplefy_task(raw_task, 'test' in file_prefix)
        component_entity_finder = EntityFinder(
            lambda grid: find_components(grid, directions=ALL_DIRECTIONS))
        component_entities = component_entity_finder(task['train'][0]['input'])
        if len(component_entities) <= 30:
            base_entity_finder = component_entity_finder
        else:
            base_entity_finder = EntityFinder(lambda grid: find_color_entities(grid))

        predictors = test_case(task,
                               max_nll=MAX_NLL - 2,
                               base_entity_finder=base_entity_finder,
                               allow_multiple_predictors=False,
                               allow_selector_pairs=True)
        
        time_elapsed = time.perf_counter() - start_time
        print(f"time_elapsed = {time_elapsed}")
        # Complexity is roughly 3^n or 4^n, so if the first try fails we up the nll without going over 5 min
        if (not predictors and time_elapsed < 250/4) or (len(predictors) < 3 and time_elapsed < 2.5):
            if len(predictors) > 0:
                print(f"Found {len(predictors)} predictors, now looking for more")
            print(f'Second attempt, NLL = {MAX_NLL - (1 if time_elapsed > 18.75 else 0)}')
            predictors = test_case(task,
                                   max_nll=MAX_NLL - (1 if time_elapsed > 250/16 else 0),
                                   base_entity_finder=base_entity_finder,
                                   allow_multiple_predictors=True,
                                   allow_selector_pairs=True)
        for j, case in enumerate(task['test']):
            test_input = case['input']
            predictions = [predictor.predict(test_input) for predictor in predictors]
            prediction_set = set()
            for prediction in predictions:
                if prediction not in prediction_set:
                    prediction_set.add(prediction)

            if not prediction_set:
                prediction_set = [((0,),)]
            root, _ = os.path.splitext(filename)
            output_id = f"{os.path.basename(root)}_{j}"
            with open(output_file_name, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({'output_id': output_id, 'output': multi_array_flattener(prediction_set)})
                
        print(f"Time elapsed = {time.perf_counter()}")



# NLLs = 4.7, 29.3,

if __name__ == '__main__':
    main()