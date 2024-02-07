import os
import time

import numpy as np
import pandas as pd

from profiling_utils import run_single_cp


def multiple_cp_runs():
    """Perform the CP on multiple data input for performance prifling."""

    # delete the file if it exists
    if os.path.exists("cp_results.csv"):
        os.remove("cp_results.csv")

    with open("cp_results.csv", "a", encoding="utf-8") as f:
        f.write(
            "method,n_opt,n_mach,running_time_seconds,num_constraints,num_variables,makespan,feasible_MILP,feasible_CP\n"
        )

        for n_opt_selected in np.arange(10, 94, 5):
        # for n_opt_selected in np.arange(10, 17, 5):
        # for n_opt_selected in [60]:
            new_row = run_single_cp(
                input_fname="gfjsp_10_5_1.txt",
                infinity=1.0e7,
                n_opt_selected=n_opt_selected,
                num_workers=32,
                verbose=False,
            )
            # contact all the values into a string with comma separated
            new_row = ",".join(map(str, new_row.values())) + "\n"
            f.write(new_row)


# the main script
if __name__ == "__main__":
    multiple_cp_runs()

