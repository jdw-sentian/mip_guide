import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from sentian_miami import get_solver

"""Minimize
c^T * x

Given

A*x >= b
x >= 0
"""

#nutrient_item2content[nutrient, item] = content
# A matrix
nutrient_item2content = np.array([[10, 2], 
                                  [0, 2],
                                  [2, 5]])

# nutrient2requirement[nutrient] = requirement
# b matrix
nutrient2requirement = np.array([20, 4, 15])

# item2price[item] = price
# c matrix
item2price = np.array([1, 3])

def sample_line(vec, d, scale=100, N=100):
    a, b = vec
    normal = np.array([b, -a])

    nearest = vec.T * d / np.linalg.norm(vec)**2
    return [nearest + normal * coeff for coeff in np.linspace(-scale, scale, N)]

def visualize_stigler():
    nutrients = ["carbohydrates", "fats", "proteins"]
    items = ["bread", "meat"]

    fig, ax = plt.subplots()
    for row, req in zip(nutrient_item2content, nutrient2requirement):
        xcoords, ycoords = zip(*sample_line(row, req))
        ax.plot(xcoords, ycoords)

    ax.axvline(0, color='k')
    ax.axhline(0, color='k')

    # plot a bunch of level curves for the price
    for price in np.arange(0, 30, 1):
        xcoords, ycoords = zip(*sample_line(item2price, price))
        ax.plot(xcoords, ycoords, linewidth=0.5, alpha=0.5, color='k')

    xcoords, ycoords = zip(*product(range(10), repeat=2))
    ax.scatter(xcoords, ycoords, c='k', alpha=0.5)

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 5])

    ax.set_xlabel(items[0])
    ax.set_ylabel(items[1])
    ax.set_title("Visualization of diet problem")
    ax.legend(nutrients, loc=1)

    plt.show()

def stigler_lp():
    """
    Using continuous x
    """

    solver = get_solver("CBC")

    # note that the natural constraint x >= 0 can be set at variable creation
    num_items = 2
    X = [solver.NumVar(lb=0, ub=None) for _ in range(num_items)]

    # setting the constraints
    for nutrient, row in enumerate(nutrient_item2content):
        total_nutrient = solver.Sum([x*content for x, content in zip(X, row)])
        # there is also a shorthand:
        # total_nutrient = solver.Dot(X, row)

        # this is how constraints are added.
        # can be made very much like natural language!
        requirement = nutrient2requirement[nutrient]
        solver.Add(total_nutrient >= requirement)

    # setting the objective
    # note that 'maximize' is a required parameter.
    # I recommend to always explicitly use the keyword,
    # in order to avoid sign errors. 
    price = solver.Dot(X, item2price)
    solver.SetObjective(price, maximize=False)

    # solving and recovering the solution
    solver.Solve(time_limit=10) # time limit in seconds
    # The time limit is useful for larger models, where
    # it is hard to know beforehand whether it will take
    # 30 seconds or 30 minutes to find an optimum.
    #
    # If the time limit is reached before a proven optimal
    # solution has been found, the best known solution is returned.
    # This often actually turns out to be the optimal solution,
    # and the solver has been struggling with proving optimality.

    quantities = [solver.solution_value(x) for x in X]
    return quantities

def stigler_ip():
    """
    Same as stigler_lp, with discrete X
    """

    solver = get_solver("CBC")

    num_items = 2
    X = [solver.IntVar(lb=0, ub=None) for _ in range(num_items)]

    # setting the constraints
    for nutrient, row in enumerate(nutrient_item2content):
        total_nutrient = solver.Sum([x*content for x, content in zip(X, row)])
        requirement = nutrient2requirement[nutrient]
        solver.Add(total_nutrient >= requirement)

    # setting the objective
    price = solver.Dot(X, item2price)
    solver.SetObjective(price, maximize=False)

    # solving and recovering the solution
    solver.Solve(time_limit=10) # time limit in seconds
    quantities = [solver.solution_value(x) for x in X]
    return quantities

if __name__ == '__main__':
    #stigler_lp()
    #stigler_ip()
    visualize_stigler()
