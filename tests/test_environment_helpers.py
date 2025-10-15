"""Tests for environment helper utilities."""

from miniverse.environment import (
    EnvironmentGraph,
    EnvironmentGrid,
    GridTile,
    GraphOccupancy,
    LocationNode,
    grid_shortest_path,
    shortest_path,
)


def test_graph_shortest_path_and_occupancy():
    graph = EnvironmentGraph(
        nodes={
            "ops": LocationNode(name="Ops", capacity=2),
            "stock": LocationNode(name="Stock", capacity=1),
            "loading": LocationNode(name="Loading"),
        },
        adjacency={
            "ops": ["stock", "loading"],
            "stock": ["ops"],
            "loading": ["ops"],
        },
    )

    path = shortest_path(graph, "ops", "loading")
    assert path == ["ops", "loading"]

    occupancy = GraphOccupancy(graph)
    assert occupancy.enter("stock", "alpha") is True
    assert occupancy.enter("stock", "beta") is False  # capacity reached
    occupancy.leave("stock", "alpha")
    assert occupancy.enter("stock", "beta") is True


def test_grid_shortest_path():
    grid = EnvironmentGrid(
        width=4,
        height=4,
        tiles={
            (1, 1): GridTile(collision=True),
            (1, 2): GridTile(collision=True),
        },
    )

    path = grid_shortest_path(grid, (0, 0), (2, 3))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)
    assert (1, 1) not in path and (1, 2) not in path
