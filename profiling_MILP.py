import pandas as pd
import numpy as np
import time
import os

from profiling_utils import run_single_milp


def multiple_milp_runs():
    # write the new_row to the csv file

    # delete the file if it exists
    if os.path.exists("milp_results.csv"):
        os.remove("milp_results.csv")

    with open("milp_results.csv", "a", encoding="utf-8") as f:
        f.write(
            "method,n_opt,n_mach,running_time_seconds,num_constraints,num_variables,makespan,feasible_MILP,feasible_CP\n"
        )

        for n_opt_selected in np.arange(10, 94, 5):
        # for n_opt_selected in np.arange(10, 21, 5):
            new_row = run_single_milp(
                input_fname="gfjsp_10_5_1.txt",
                infinity=1.0e7,
                n_opt_selected=n_opt_selected,
                num_workers=16,
                verbose=False,
            )
            # contact all the values into a string with comma separated
            new_row = ",".join(map(str, new_row.values())) + "\n"
            f.write(new_row)


# the main script
if __name__ == "__main__":
    multiple_milp_runs()

