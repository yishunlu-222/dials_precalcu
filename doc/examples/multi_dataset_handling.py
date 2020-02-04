from dials.array_family import flex
from dxtbx.model import Experiment, ExperimentList


def split_reflection_tables():
    """Example code for handling multi-dataset reflection tables."""
    ###First make an ExperimentList and a multi-dataset reflection table
    experiments = ExperimentList()
    for i in range(3):
        experiments.append(Experiment(identifier=str(i)))
    reflection_table = flex.reflection_table()
    reflection_table["id"] = flex.int([-1, 0, 1, 2])
    reflection_table.experiment_identifiers()[0] = "0"
    reflection_table.experiment_identifiers()[1] = "1"
    reflection_table.experiment_identifiers()[2] = "2"

    # Historically, an id of -1 was used to indicate unindexed reflections
    unindexed_reflections = reflection_table.select(reflection_table["id"] == -1)
    assert list(unindexed_reflections["id"]) == [-1]

    # Selecting one table at a time using .select().
    # Only works if identifiers are set.
    split_tables = [reflection_table.select(exp) for exp in experiments]
    for i, table in enumerate(split_tables):
        assert list(table["id"]) == [i]
        assert dict(table.experiment_identifiers()) == {i: str(i)}

    # Selecting manually with a boolean selection array.
    # Works whether identifiers are set or not.
    for i in set(reflection_table["id"]).difference(set([-1])):
        table = reflection_table.select(reflection_table["id"] == i)
        assert list(table["id"]) == [i]
        assert dict(table.experiment_identifiers()) == {i: str(i)}

    # to use split_by_experiment_id, one must first select ids >= 0
    # Works whether identifiers are set or not.
    indexed_reflection_table = reflection_table.select(reflection_table["id"] >= 0)
    split_tables_2 = indexed_reflection_table.split_by_experiment_id()
    for i, table in enumerate(split_tables_2):
        assert list(table["id"]) == [i]
        assert dict(table.experiment_identifiers()) == {i: str(i)}
