"""Definition of systematic absences check algorithm."""
import logging
from dials.util import tabulate

from cctbx import sgtbx

from dials.algorithms.symmetry.absences.laue_groups_info import (
    construct_laue_groups,
    score_screw_axes,
    score_space_groups,
)

logger = logging.getLogger("dials")


def run_systematic_absences_checks(
    experiments, merged_reflections, chiral=True, significance_level=0.95
):
    """Check for systematic absences in the data for the laue group.

    Using a reflection table containing merged data, test screw axes and score
    possible space groups. The crystals are updated with the most likely space
    group.
    """
    # Get the laue class from the space group.
    space_group = experiments[0].crystal.get_space_group()
    laue_group = str(space_group.build_derived_patterson_group().info())
    logger.info("Laue group: %s", laue_group)
    lauegroups = construct_laue_groups(chiral)
    if laue_group not in lauegroups:
        logger.info("No absences to check for this laue group")
        return

    # Score the screw axes.
    screw_axes, screw_axis_scores = score_screw_axes(
        lauegroups[laue_group], merged_reflections, significance_level
    )

    logger.info(
        "%s",
        tabulate(
            [
                [
                    a.name,
                    "%.3f" % score,
                    str(a.n_refl_used[0]),
                    str(a.n_refl_used[1]),
                    "%.3f" % a.mean_I,
                    "%.3f" % a.mean_I_abs,
                    "%.3f" % a.mean_I_sigma,
                    "%.3f" % a.mean_I_sigma_abs,
                ]
                for a, score in zip(screw_axes, screw_axis_scores)
            ],
            [
                "Screw axis",
                "Score",
                "No. present",
                "No. absent",
                "<I> present",
                "<I> absent",
                "<I/sig> present",
                "<I/sig> absent",
            ],
        ),
    )

    # Score the space groups from the screw axis scores.
    space_groups, scores = score_space_groups(screw_axis_scores, lauegroups[laue_group])

    logger.info(
        "%s",
        tabulate(
            [[sg, "%.4f" % score] for sg, score in zip(space_groups, scores)],
            ["Space group", "score"],
        ),
    )

    # Find the best space group and update the experiments.
    best_sg = space_groups[scores.index(max(scores))]
    logger.info("Recommended space group: %s", best_sg)
    if "enantiomorphic pairs" in lauegroups[laue_group]:
        if best_sg in lauegroups[laue_group]["enantiomorphic pairs"]:
            logger.info(
                "Space group with equivalent score (enantiomorphic pair): %s",
                lauegroups[laue_group]["enantiomorphic pairs"][best_sg],
            )
    if "equivalent_nonchiral_groups" in lauegroups[laue_group]:
        if best_sg in lauegroups[laue_group]["equivalent_nonchiral_groups"]:
            logger.info(
                "Space groups with equivalent scores (indistinguishable by absences): %s",
                ", ".join(
                    lauegroups[laue_group]["equivalent_nonchiral_groups"][best_sg]
                ),
            )

    new_sg = sgtbx.space_group_info(symbol=best_sg).group()
    for experiment in experiments:
        experiment.crystal.set_space_group(new_sg)
