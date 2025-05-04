# Smart Data Models entity samples generator

![Version: 0.4.1](https://img.shields.io/badge/Version-0.4.1-informational)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)

This tool aims to streamline the generation of random NGSI-LD entity samples to be used in the training and distillation of Small Language Models (SLMs) models.

<details>
<summary><b>Table of Contents</b></summary>

- [Smart Data Models entity samples generator](#smart-data-models-entity-samples-generator)
  - [Maintainers](#maintainers)
  - [Source Code](#source-code)
  - [Workflow](#workflow)
  - [Configuration parameters: .conf.env](#configuration-parameters-confenv)
  - [Run](#run)
    - [Using Docker](#using-docker)
    - [Using default shell](#using-default-shell)
  - [Output](#output)
  - [License](#license)

</details>

## Maintainers

| Name | Email |
| ---- | ------ |
| cristianmartella | <cristian.martella@unisalento.it> |

## Source Code

- <https://github.com/cristianmartella/sdm-samples-generator>

## Workflow

For a given Smart Data Model (SDM) identified by its name and subject, the tool generates a custom series of data entities, in normalized and/or key-value format.

The `pysmartdatamodels` Python library includes tools to list the properties of a given SDM and even generate a full sample. This tool uses such a library to retrieve the initial version of sample entities, that will then be further refined for the purposes of acting as examples in the training of distilled SLMs.

In particular, for a given generated data entity, the tool removes all the properties that are **shared** across all the SDMs that belong to the same subject. It is possible to further refine the resulting entity by preserving a custom set of **retained properties**. This process is performed recursively, ensuring that even nested properties are taken into account.

The script iteratively ## Configuration parameters: .conf.env generates a number of entities that is given by the product of the number of *iterations* and the value of the *depth* parameter. For each iteration up to the value of **depth** parameter, a batch of entities (whose size is specified by the **iterations** parameter) is further refined by removing an additional set of randomly picked unique properties. The size of this set corresponds to the current iteration index.

Some of the fields included in the samples generated with `pysmartdatamodels` contain gibberish text that could be troublesome for AI training. Hence, it is important to recognize such values and update them. From version 0.2.0 such properties are automatically identified via regex and cleared.

The tools also allows to replace a subset of the original properties using pseudo-generated synonyms. Moreover it allows to adopt snake case format for generated samples properties.

## Configuration parameters: .conf.env

| **Parameter** | **Description** | **Type** | **Default** | **Example** |
| --- | --- | --- | --- | --- |
| GEN-ITERATIONS | Size of generated entities set for each depth step | integer | 10 | 5 |
| GEN-DEPTH | Number of depth iterations (the deeper, the more random unique properties will be removed from the corresponding batches of generated entities) | integer | 0 | 3 |
| GEN-DEPTH-MAX-THR | Maximum depth threshold: depth is capped to this value if above this value | integer | 5 | 7 |
| SYN-BATCH-RATIO | Share of properties to replace with randomized synonyms | decimal between [0.0-1.0] | 0.0 | 0.3 |
| ENABLE-SNAKE-CASE | Enables the generation of entities with properties formatted in snake case (the default is camel case) | boolean | False | True |
| SDM-DOMAIN | The domain of the target SDM | string |  | SmartEnergy |
| SDM-SUBJECT | The subject of the target SDM | string |  | dataModel.GreenEnergy |
| SDM-NAME | The name of the target SDM | string |  | PhotovoltaicMeasurement |
| RETAINED-SHARED-PROPERTIES | The list of properties shared in a subject to retain | string | id,type,@context | id,type,@context,name,description |
| ANY-NEGATIVE-SUBJECT | Enable the generation of negative samples in other subjects (*see warning below). If disabled, only data models in the same subject will be randomly picked for the generation of negative samples | boolean | False | True |
| OUT-NORMALIZED-ENABLED | Enable the generation of normalized entity samples | boolean | False | True |
| OUT-KEYVALUES-ENABLED | Enable the generation of key-values entity samples | boolean | False | True |

> [!Warning]
> Enabling `ANY-NEGATIVE-SUBJECT` triggers resource intensive logic that will dramatically impact the generation rate of samples.

## Run

### Using Docker

The script can be containerized and executed using Docker. To this end, the first step is to build the related Docker image and, subsequently, a container corresponding to that image can be run.

Using `docker compose` allows to execute both such operations in one single command:

```shell
docker compose up
```

> [!Note]
> The first execution will require additional time to build the image. Once the image is built, docker compose will skip this step and run the previously built image right away.
> In case the image needs to be rebuilt (e.g., following a change in the code), run the above command with the option `--build` to force the rebuilding process.

### Using default shell

To execute the script, the required Python dependencies included in the `requirements.txt` file should be installed using the following command:

```shell
cd sdm-samples-generator
pip install --no-cache-dir -r requirements.txt
```

Then, environment variables included in `.conf.env` file must be made available to the script:

```shell
set -a
source .conf.env
set +a
```

> [!NOTE]
> Wrapping the sourcing of the *.conf.env* file with `set -a` and `set +a` ensures that the script can read the environment variables correctly and that the exported variables only exist as part of the execution of the script itself.

Finally, execute the script:

```shell
python3 src/main/app.py
```

## Output

The files generated by this tool are located in the `src/resources/output` folder and follow the following naming convention:

```text
{SDM-NAME}_{entities-format}.json
```

`SDM-NAME` is the name of the SDM, as described in the table [above](#configuration-parameters-confenv). Instead, `entities-format` refers to the format of the entities contained in the file, which can be either *normalized* or *key-values*.

The entities included in these collections are separated by new lines.
Each entity is structured as reported in the following table.

| **Property** | **Description** |
| --- | --- |
| target | The target positive sample, corresponding to the original format and property naming system of the target SDM |
| positive | A positively matching sample that includes changes in the property names and case format |
| negative | A negatively matching sample that includes changes in the property names and case format |

Furthermore, both `positive` and `negative` sample provide additional properties, as listed in the table below.

| **Property** | **Description** |
| --- | --- |
| unfittingProperties | The properties whose values have been removed from the original sample (reason: gibberish) |
| excludedProperties | The properties that have been excluded from the original sample (reason: ambiguity) |
| modifiedProperties | The original property names that have been modified from the original sample (reason: synonyms) |
| label | The match label (0...1): high score (e.g., 0.9) indicates a good match between the sample and the target entity, whereas a low score (e.g., 0.1) indicates a poor match |
| sdmMetadata | The metadata of the generated sample entity |

## License

Copyright 2025 Cristian Martella

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
