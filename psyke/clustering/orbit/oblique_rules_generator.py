import pandas as pd
from psyke.extraction.hypercubic.hypercube import ClosedCube
from sklearn.mixture import GaussianMixture
import heapq
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib import pyplot as plt
import numpy as np
from utils.draw import draw_data, draw_clusters
from typing import Tuple, List, Dict, Union
from psyke.clustering.orbit.container import Container
import math
from sklearn.metrics import accuracy_score


remove_dimensions = True


def generate_container(whole_dataframe: pd.DataFrame,
                       df: pd.DataFrame,
                       cube_indices,
                       iper_cube: ClosedCube,
                       steps: int,
                       min_accuracy_increase: float,
                       initial_size: int = 0,
                       max_disequation_num: int = 10
                       ) -> Container:
    """
    generate a container that contains the points with the most common cluster label.
        the container will have mixed constraints in form of intervals and disequations
    :param whole_dataframe: all the dataframe that can interest the accuracy of the obtained model
    :param df: dataframe containing the points that must be represented by the container
    :param cube_indices: the most common cluster of the points indicated by cube_indices is the "main cluster"
        that must be represented by the container
    :param iper_cube: cube containing the points that must be modelled trough container
    :param steps: max number of couples of most promising dimension that are considered to generate disequations
    :param min_accuracy_increase: min total accuracy increase in order to consider a set of
        "disequation_num" disequations over a couple of dimensions
    :param initial_size: total size of the whole starting dataframe
    :param max_disequation_num: number of disequtions for each couple of dimensions
    :return:
    """

    reduced_df = df.iloc[cube_indices]
    mode_cluster = reduced_df.iloc[:, -1].mode()[0]
    masked_predictions: np.ndarray = whole_dataframe.iloc[:, -1].to_numpy()
    true_predictions: np.ndarray = (masked_predictions == mode_cluster)

    cube_predictions = iper_cube.filter_indices(whole_dataframe.iloc[:, :-1])
    cube_accuracy = get_total_accuracy(true_predictions,
                                       cube_predictions,
                                       initial_size)

    all_data_size = masked_predictions.shape[0]
    reduced_df = reduced_df.copy()
    reduced_df = reduced_df[reduced_df.iloc[:, -1] == mode_cluster]
    reduced_df = reduced_df.iloc[:, :-1]
    covariance_matrix = reduced_df.cov().to_numpy()
    n_features = len(reduced_df.columns)

    # best_covariance = heapq.nlargest(dim_num, [(abs(covariance_matrix[i, j]), covariance_matrix[i, j], i, j)
    #                                      for i in range(n_features) for j in range(i+1, n_features)], key=lambda x: x[0])
    best_covariance = [(abs(covariance_matrix[i, j]), covariance_matrix[i, j], i, j)
                       for i in range(n_features) for j in range(i + 1, n_features)]
    best_covariance.sort(reverse=True, key=lambda x: x[0])

    dimensions = iper_cube.dimensions
    original_dimensions = dimensions.copy()
    disequations = {}
    # kneel_accuracies = []
    previous_accuracy = cube_accuracy
    # k = 0
    precision = 2
    convex_hulls = {}

    for step, (_, cov, i, j) in enumerate(best_covariance):
        # if abs(cov) > min_covariance:
        if step >= steps:
            break
        col_i_name = reduced_df.columns[i]
        col_j_name = reduced_df.columns[j]
        generated_disequations = generate_constraint(reduced_df[[col_i_name, col_j_name]], max_disequation_num)
        conv_hull = []
        best_accuracy = previous_accuracy
        dis_num = 3
        new_diequations = None
        c_h = None
        for new_dis_i_j, c_h in generated_disequations:
            new_diequations = disequations.copy()
            new_diequations[(col_i_name, col_j_name)] = new_dis_i_j
            reduced_dim = dimensions.copy()
            reduced_dim.pop(col_i_name, None)
            reduced_dim.pop(col_j_name, None)

            new_container = Container(reduced_dim, new_diequations)
            new_accuracy = get_total_accuracy(true_predictions,
                                              new_container.filter_indices(whole_dataframe.iloc[:, :-1]),
                                              initial_size)
            # measure the increase in accuracy by increasing the number of disequations by 1.
            #   In order to continue, this increase mut be positive, and greater than the min_accuracy_increase
            new_acc_increse = new_accuracy - best_accuracy
            if new_acc_increse < min_accuracy_increase and min_accuracy_increase >= 0:
                break
            best_accuracy = new_accuracy
            dis_num = len(new_dis_i_j)

        if min_accuracy_increase < 0 or (best_accuracy - previous_accuracy > min_accuracy_increase * dis_num) and new_diequations is not None:
            # reduce_dim_accuracy, reduced_dim = try_reducing_dimension(dimensions,
            #                                                           col_i_name,
            #                                                           new_diequations,
            #                                                           true_predictions,
            #                                                           initial_size,
            #                                                           whole_dataframe)
            # if reduce_dim_accuracy >= best_accuracy:
            #     best_accuracy = reduce_dim_accuracy
            #     dimensions = reduced_dim
            # reduce_dim_accuracy, reduced_dim = try_reducing_dimension(dimensions,
            #                                                           col_j_name,
            #                                                           new_diequations,
            #                                                           true_predictions,
            #                                                           initial_size,
            #                                                           whole_dataframe)
            # if reduce_dim_accuracy >= best_accuracy:
            #     best_accuracy = reduce_dim_accuracy
            #     dimensions = reduced_dim
            dimensions = reduced_dim
            previous_accuracy = best_accuracy
            disequations = new_diequations
            convex_hulls[(col_i_name, col_j_name)] = c_h

        # if k % precision == 0:
        #     kneel_accuracies.append(previous_accuracy)
        # k += 1

    if remove_dimensions:
        new_container = Container(dimensions, disequations, convex_hulls=(convex_hulls, mode_cluster))
    else:
        new_container = Container(original_dimensions, disequations, convex_hulls=(convex_hulls, mode_cluster))
    return new_container


def try_reducing_dimension(dimensions: dict, dim_name: str, disequations, true_predictions, initial_size, dataframe):
    reduced_dim = dimensions.copy()
    reduced_dim.pop(dim_name, None)
    container_reduced_dims = Container(reduced_dim, disequations)
    reduce_dim_accuracy = get_total_accuracy(true_predictions,
                                      container_reduced_dims.filter_indices(dataframe.iloc[:, :-1]),
                                      initial_size)
    return reduce_dim_accuracy, reduced_dim


def refine_constraints(constraint: dict[tuple[str, str], list[tuple[float, float, float]]],
                       dimensions: dict[str, tuple],
                       df: pd.DataFrame,
                       true_predictions: np.ndarray,
                       accuracy: float)\
        -> tuple[dict[str, tuple], dict[tuple[str, str], list[tuple[float, float, float]]]]:
    """
    this function is not used anymore
    :param constraint:
    :param dimensions:
    :param df:
    :param true_predictions:
    :param accuracy:
    :return:
    """
    dimensions = dimensions.copy()
    final_constraints = {}
    for dim1, dim2 in constraint:
        reduced_dim = dimensions.copy()
        reduced_dim.pop(dim1, None)
        reduced_dim.pop(dim2, None)
        new_constr = {(dim1, dim2): constraint[(dim1, dim2)]}
        reduced_container = Container(dimension=reduced_dim, disequation=new_constr)
        new_container_predictions = reduced_container.filter_indices(df)
        new_container_accuracy = accuracy_score(true_predictions, new_container_predictions)
        if new_container_accuracy > accuracy:
        # if True:
            dimensions = reduced_dim
            final_constraints[(dim1, dim2)] = constraint[(dim1, dim2)]
    return dimensions, final_constraints


def generate_constraint(df: pd.DataFrame, max_number_of_diequations: int=10) -> List[Tuple[List[Tuple[float, float, float]], List[Tuple]]]:
    """

    :param df: dataframe with only 2 dimensions
    :param number_of_diequations: number of diequations repreenting the points in df
    :return:
    """
    data = df.to_numpy()
    try:
        hull = ConvexHull(data)

        points = hull.points

        contour_line = np.array([points[simplex, :] for simplex in hull.simplices])
        disequations_list = []
        simple_hull_net_dict = simplify_convex_hull(contour_line, max_number_of_diequations)
        for n_disequations in range(3, max_number_of_diequations + 1):
            if n_disequations not in simple_hull_net_dict:
                continue
            simple_hull_net = simple_hull_net_dict[n_disequations]
            # simple_hull = np.array(list(simple_hull_net.keys()))
            disequations = generate_disequations(simple_hull_net)

            convex_hull_start = list(simple_hull_net.keys())[0]
            # final_convex_hull = [convex_hull_start]
            next = convex_hull_start
            prev = None
            actual = None
            final_hull_ordered = []
            while next != convex_hull_start or actual is None:
                prev = actual
                actual = next
                next = simple_hull_net[next][0] if prev != simple_hull_net[next][0] else simple_hull_net[next][1]
                final_hull_ordered.append(actual)
            disequations_list.append((disequations, final_hull_ordered))
        return disequations_list
    except:
        # the only reason i can think of for not being able to create the convex hull
        #   is that all points are positioned on a line
        if len(data) < 2:
            return []
        first_point = data[0, :]
        second_point = data[1, :]

        a, b, c = get_rect(first_point, second_point)

        # assume the points are an oblique line (if is vertical or horizontal there should be 0 covariance)
        lower_point = min(data[:, 0])
        higher_point = max(data[:, 0])

        constraints = [(a, b, c + Container.EPSILON),
                       (-a, -b, -c + Container.EPSILON),
                       (1, 0, higher_point + Container.EPSILON),   # constrain the first dimension to contain all points
                       (-1, 0, - lower_point + Container.EPSILON)]

        all_satisfied = np.full(data.shape[0], True, dtype=bool)
        for a, b, c in constraints:
            constra_sat = data[:, 0] * a + data[:, 1] * b <= c
            all_satisfied = np.logical_and(all_satisfied, constra_sat)
        if np.all(all_satisfied):
            return [(constraints, [lower_point, higher_point])]
        else:
            print("Unable to build additional constraints.")
            return []
    # plt.plot(simple_hull[:, 0], simple_hull[:, 1], 'ko')
    # plt.xlabel(col_i_name)
    # plt.ylabel(col_j_name)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # plt.scatter(df[col_i_name], df[col_j_name])
    # plt.xlabel(col_i_name)
    # plt.ylabel(col_j_name)
    # convex_hull_plot_2d(hull, ax=ax)
    # plt.show()


def simplify_convex_hull(hull_lines, max_final_point_num: int=10):
    """
    reduce the number of points (or lines) of the initial convex hull until it has number_of_points points
    :param hull_lines: set of lines representing the initial convex hull containing all points
    :param max_final_point_num: max number of points of the final polygon containing the points
    :return: a dictionary containing for each number of lines, a polygon with such number of line.
        The polygon contains the data and is in the form of a network where each point of the polygon is linked
        to it neighbours
    """
    assert max_final_point_num >= 3

    # contour_net: given a point returns 2 connected, adjacent points
    contour_net = generate_contour_net(hull_lines)

    elimination_cost = {}
    for point in contour_net:
        elimination_cost[point] = evaluate_elimination_cost(point, contour_net)

    assert len(hull_lines) == len(contour_net)
    assert len(hull_lines) == len(elimination_cost)
    contour_net_dict = {}
    while len(contour_net) > 3:
        point_to_eliminate = min(elimination_cost, key=elimination_cost.get)
        if elimination_cost[point_to_eliminate] == np.inf:
            break
        eliminate_point(point_to_eliminate, contour_net, elimination_cost)
        if len(contour_net) <= max_final_point_num:
            contour_net_dict[len(contour_net)] = contour_net

    return contour_net_dict


def generate_contour_net(hull_lines: List[List]) -> Dict[Tuple[float, float], Tuple[Tuple, Tuple]]:
    contour_net = {}
    for line in hull_lines:
        p0 = tuple(line[0])
        p1 = tuple(line[1])
        if p0 not in contour_net:
            contour_net[p0] = []
        if p1 not in contour_net:
            contour_net[p1] = []
        contour_net[p0].append(p1)
        contour_net[p1].append(p0)
    for point in contour_net:
        assert len(contour_net[point]) == 2
        contour_net[point] = tuple(contour_net[point])
    return contour_net


def generate_disequations(contour_net: Dict[Tuple[float, float], Tuple[Tuple, Tuple]]) -> List[Tuple[float, float, float]]:
    """

    :param contour_net: dictionary which for each point in the contour gives 2 adjacent points
    :return: constraints in the form aX + bY <= c
    """
    def get_disequation(p1, p2, p3):
        """
        returns the disequation of the rect passing from p1, p2, considering that p3 satisfies the constraint
        """
        a, b, c = get_rect(p1, p2)
        x3, y3 = p3
        if not (a * x3 + b * y3 <= c):
            # in case the constraint aX + bY <= c, invert the constraint transforming it to aX + bY >= c,
            #   or to keep it simpler, -aX - bY <= -c
            a = -1 * a
            b = -1 * b
            # the additional Container.EPSILON value is a small margin to avoid excluding points due to approximation
            c = -1 * c + max([abs(a), abs(b), abs(b)]) * Container.EPSILON
        return a, b, c

    equations = {}
    for point in contour_net.keys():
        p1, p2 = contour_net[point]
        if (p1, point) not in equations and (point, p1) not in equations:
            a, b, c = get_disequation(p1, point, p2)
            equations[(p1, point)] = a, b, c
        if (p2, point) not in equations and (point, p2) not in equations:
            a, b, c = get_disequation(p2, point, p1)
            equations[(p2, point)] = a, b, c
    return list(equations.values())


def evaluate_elimination_cost(point, contour_net):

    _, _, extra_area_0, extra_area_1, _, _, _, _ = get_new_points(point, contour_net)

    return extra_area_0 if extra_area_0 < extra_area_1 else extra_area_1


def eliminate_point(point, contour_net, elimination_cost):
    new_p0, new_p1, extra_area_0, extra_area_1, extern_point_0, extern_point_1, p0, p1 = get_new_points(point, contour_net)


    if extra_area_0 < extra_area_1:
        extern_point = extern_point_0
        new_p = new_p0
        inner_point_to_eliminate = p0
        inner_point_ok = p1
    else:
        extern_point = extern_point_1
        new_p = new_p1
        inner_point_to_eliminate = p1
        inner_point_ok = p0

    extern_point_p0, extern_point_p1 = contour_net[extern_point]
    if extern_point_p0 == inner_point_to_eliminate:
        extern_point_p0 = new_p
    else:
        extern_point_p1 = new_p
    contour_net[extern_point] = (extern_point_p0, extern_point_p1)

    inner_point_ok_p0, inner_point_ok_p1 = contour_net[inner_point_ok]
    if inner_point_ok_p0 == point:
        inner_point_ok_p0 = new_p
    else:
        inner_point_ok_p1 = new_p
    contour_net[inner_point_ok] = (inner_point_ok_p0, inner_point_ok_p1)
    contour_net.pop(inner_point_to_eliminate, None)
    contour_net.pop(point, None)
    contour_net[new_p] = [extern_point, inner_point_ok]

    elimination_cost.pop(point, None)
    elimination_cost.pop(inner_point_to_eliminate, None)
    elimination_cost[new_p] = evaluate_elimination_cost(new_p, contour_net)
    elimination_cost[inner_point_ok] = evaluate_elimination_cost(inner_point_ok, contour_net)
    elimination_cost[extern_point] = evaluate_elimination_cost(extern_point, contour_net)


def get_new_points(point: Tuple[float, float], contour_net: Dict):
    """

    :param point: point that could be removed
    :param contour_net:
    :return: given 2 directions (0 and 1) along the contour, starting from "point", the following values are returned:
    new_p0 and new_p1: 2 new possible points that could be used instead of "point" and "p0" or "p1"
    extra_area_0 and extra_area_1: the extra area added to the polygon inside the contour,
        which would be added by using the point "new_p0" or "new_p1". The area is np.inf if it is not possible to get such points
    p0 and 01: the 2 closest points to "point"
    extern_point_0, extern_point_1: the other 2 points close to "p0" and "p1"
    """
    p0 = contour_net[point][0]
    p1 = contour_net[point][1]

    extern_point_0 = contour_net[p0][0] if contour_net[p0][0] != point else contour_net[p0][1]
    extern_point_1 = contour_net[p1][0] if contour_net[p1][0] != point else contour_net[p1][1]

    extern_rect_0 = get_rect(extern_point_0, p0)
    extern_rect_1 = get_rect(extern_point_1, p1)
    intern_rect_0 = get_rect(p0, point)
    intern_rect_1 = get_rect(p1, point)
    new_p0 = get_intersection(extern_rect_0, intern_rect_1)
    new_p1 = get_intersection(extern_rect_1, intern_rect_0)
    if new_p0 is not None:
        if not is_middle_point(p0, extern_point_0, new_p0):
            new_p0 = None
    if new_p1 is not None:
        if not is_middle_point(p1, extern_point_1, new_p1):
            new_p1 = None
    extra_area_0 = get_area(p0, new_p0, point) if new_p0 is not None else np.inf
    extra_area_1 = get_area(p1, new_p1, point) if new_p1 is not None else np.inf
    return new_p0, new_p1, extra_area_0, extra_area_1, extern_point_0, extern_point_1, p0, p1


def get_rect(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float, float]:
    """
     define the rect passing bu p1 and p2 as:  a*x1 + b*x2 = c
    :return:
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    return a, b, c


def get_intersection(r1: Tuple[float, float, float], r2: Tuple[float, float, float]) -> Union[Tuple[float, float], None]:
    """

    :param r1: (a, b, c) representing rect ax + bx = c
    :param r2: (a, b, c) representing rect ax + bx = c
    :return: point of intersection, None if the two lines are parallel
    """
    a1, b1, c1 = r1
    a2, b2, c2 = r2

    # bring c on the other side of equal: ax + bx = c -> ax + bx - c = 0
    c1 = -1 * c1
    c2 = -1 * c2
    try:
        xp = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
        yp = (c1 * a2 - c2 * a1) / (a1 * b2 - a2 * b1)
        return xp, yp
    except ZeroDivisionError:
        return None


def get_area(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    return 0.5 * abs(x1 * (y2-y3) + x2 * (y3-y1) + x3 * (y1-y2))


def is_middle_point(p_middle, p1, p2):
    """
    given 3 points on the same rect(p_middle, p1, p2), return true if p_middle is between p1 and p2
    :return: True if p_middle is between p1 and p2
    """
    x_middle, y_middle = p_middle
    x1, y1 = p1
    x2, y2 = p2

    if x1 > x2:
        x_is_between = x1 >= x_middle and x_middle >= x2
    else:
        x_is_between = x2 >= x_middle and x_middle >= x1

    if y1 > y2:
        y_is_between = y1 >= y_middle and y_middle >= y2
    else:
        y_is_between = y2 >= y_middle and y_middle >= y1

    return x_is_between and y_is_between

def get_total_accuracy(true_values: np.ndarray, pred_values: np.ndarray, initial_size: int):
    pred_size = len(true_values)
    accuracy = accuracy_score(true_values, pred_values)
    correct_num = accuracy * pred_size          # TP + TN
    correct_num = (initial_size - pred_size) + correct_num      # consider non predicted data as TN
    total_accuracy = correct_num / initial_size
    return total_accuracy







