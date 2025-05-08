"""
Copyright 2025 Cristian Martella

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os
import multiprocessing as mp
import generator as gen
from pysmartdatamodels import pysmartdatamodels as sdm


subject = os.getenv("SDM-SUBJECT", "")
genIterations = int(os.getenv("GEN-ITERATIONS", "10"))
genDepth = int(os.getenv("GEN-DEPTH", "0"))
synBatchRatio = float(os.getenv("SYN-BATCH-RATIO", "0.0"))
enableSnakeCase = eval(os.getenv("ENABLE-SNAKE-CASE", "False"))
enableNormalizedOut = eval(os.getenv("OUT-NORMALIZED-ENABLED", "True"))
enableKeyValuesOut = eval(os.getenv("OUT-KEYVALUES-ENABLED", "False"))


if __name__ == "__main__":
    dataModelsForSubject = sdm.datamodels_subject(subject)
    print(f"List of data models for subject {subject}:\n{dataModelsForSubject}")

    print(f"Number of CPU: {mp.cpu_count()}")
    jobs = []

    # instantiate the generator processes
    for dataModel in dataModelsForSubject:
        logging.info(f"Generating {genDepth*genIterations} samples for data model {dataModel}...")
        job = mp.Process(target=gen.run_generator, args=(subject, dataModel, genDepth, genIterations, synBatchRatio, enableSnakeCase, enableNormalizedOut, enableKeyValuesOut))
        jobs.append(job)
        job.start()
    
    # wait for all processes to finish
    for job in jobs:
        job.join()

    logging.info("Done!")