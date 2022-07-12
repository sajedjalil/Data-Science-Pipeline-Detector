# Split training data into separate sequences.
#
# Assumes a new sequence begins as soon as the time to failure
# increases between two observations

import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def export_sequence(ctr, acoustic_datas, time_to_failures):
    df = pd.DataFrame({'acoustic_data': acoustic_datas,
                       'time_to_failure': time_to_failures})
    df.to_csv(f"seq_{ctr}.csv.zip", index=False, compression='zip')


with open("../input/train.csv") as f:
    f.readline()
    last_ttf = None
    ctr = 0

    acoustic_datas = []
    time_to_failures = []
    for line in f:
        acoustic_data, time_to_failure = line.split(',')
        acoustic_data, time_to_failure = int(acoustic_data), float(time_to_failure)

        if last_ttf:
            delta = last_ttf - time_to_failure
            if delta < 0.0:
                logging.info(
                    f"Exporting sequence {ctr} of length {len(acoustic_datas)}")
                last_ttf = None
                export_sequence(ctr, acoustic_datas, time_to_failures)
                acoustic_datas = []
                time_to_failures = []
                ctr += 1

        last_ttf = time_to_failure

        acoustic_datas.append(acoustic_data)
        time_to_failures.append(time_to_failure)

# Export final sequence
export_sequence(ctr, acoustic_data, time_to_failures)