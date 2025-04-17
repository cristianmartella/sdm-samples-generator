
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
import json
import random
import utils
from nested_lookup import nested_delete
from pysmartdatamodels import pysmartdatamodels as sdm
#from ngsild_converter import normalized2keyvalues


genIterations = int(os.getenv("GEN-ITERATIONS", "10"))
genDepth = int(os.getenv("GEN-DEPTH", "0"))
depthMaxThreshold = int(os.getenv("GEN-DEPTH-MAX-THR", "5"))
domain = os.getenv("SDM-DOMAIN", "")
subject = os.getenv("SDM-SUBJECT", "")
name = os.getenv("SDM-NAME", "")
normalizedOutEnabled = os.getenv("OUT-NORMALIZED-ENABLED", "true")
keyvaluesOutEnabled = os.getenv("OUT-KEYVALUES-ENABLED", "false")

retainedProperties = set(['id', 'type'])
schemaUrl = f"https://raw.githubusercontent.com/smart-data-models/{subject}/master/{name}/schema.json"

dataModels = sdm.load_all_datamodels()

def get_domains() -> list:
    """
    Get the list of domains.
    :return: The list of domains.
    """

    domains = set()

    for dataModel in dataModels:
        # Get the list of domains
        dmDomains = set(dataModel["domains"])
        if dmDomains not in domains:
            domains = domains | dmDomains

    return list(domains)


def get_subjects_by_domain(domain:str) -> list:
    """
    Get the list of subjects by domain.
    :param domain: The domain.
    :return: The list of subjects.
    """

    subjects = set()

    for dataModel in dataModels:
        # Get the list of subjects
        dmSubject = dataModel["repoName"]
        if dmSubject not in subjects and domain in dataModel["domains"]:
            subjects.add(dmSubject)

    return list(subjects)



def get_shared_properties_by_subject(subject:str) -> set:
    """
    Get the list of shared properties by subject.
    :param subject: The subject.
    :return: The list of shared properties.
    """

    # Get the list of data models properties
    dataModels = sdm.datamodels_subject(subject)
    
    sharedProperties = set(sdm.attributes_datamodel(subject, dataModels[0]))

    for dataModel in dataModels[1:]:
        properties = set(sdm.attributes_datamodel(subject, dataModel))
        sharedProperties = sharedProperties.intersection(properties)

    return sharedProperties


def get_shared_properties_by_domain(domain:str, excludedSubjects:list) -> set:
    """
    Get the list of shared properties by domain.
    :param domain: The domain.
    :param excludedSubjects: The list of excluded subjects.
    :return: The list of shared properties.
    """

    # Get the list of subjects omitting the excluded subjects
    subjects = [subject for subject in get_subjects_by_domain(domain) if subject not in excludedSubjects]

    sharedProperties = set(get_shared_properties_by_subject(subjects[0]))

    for subject in subjects[1:]:
        properties = get_shared_properties_by_subject(subject)
        sharedProperties = sharedProperties.intersection(properties)

    return sharedProperties





# Get the list of shared properties for the given subject
sharedPropertiesBySubject = get_shared_properties_by_subject(subject)
logging.debug(f"get_shared_properties_by_subject({subject}): {sharedPropertiesBySubject}")


# Extract the full set of properties for the given data model
dmProperties = set(sdm.attributes_datamodel(subject, name))
logging.debug(f"properties: {dmProperties}")

# Remove the properties that are shared with the other data models in the same subject
uniqueProperties = dmProperties - sharedPropertiesBySubject
logging.debug(f"uniqueProperties: {uniqueProperties}")

# Cut off threshrold for the depth of the generated samples
if genDepth > depthMaxThreshold:
    genDepth = depthMaxThreshold


# Generate the normalized samples
if normalizedOutEnabled == 'True':
    # Generate a full sample using sdm module
    sampleNormalized = sdm.ngsi_ld_example_generator(schemaUrl)

    # search unfitting properties in the sample
    unfittingProperties = utils.match(utils.MATCHER_TYPE_SENTENCE, sampleNormalized)
    print(f"unfittingProperties: {unfittingProperties}")

    for ii in range(genDepth):
        logging.debug(f"Generating normalized samples excluding {ii} random properties...")

        for jj in range(genIterations):
            logging.debug(f"Generating {jj}-th sample...")

            # Generate the full sample using sdm module
            sampleNormalized = sdm.ngsi_ld_example_generator(schemaUrl)
            
            # compute the list of properties to remove (all the properties that are shared with other data models in the same subject, including a batch of random unique properties and excluding those in the list of properties to keep)
            excludedProperties = dmProperties - uniqueProperties - retainedProperties | set(random.sample(list(uniqueProperties), ii))

            # remove the excluded properties from the full sample
            for key in excludedProperties:
                sampleNormalized = nested_delete(sampleNormalized, key)

            # cleanup unfitting properties
            sampleNormalized = utils.clear_properties(sampleNormalized, unfittingProperties)

            # persist the resulting sample in a file
            with open(f"../output/{name}_normalized.json", "a") as f_norm:
                print(json.dumps(sampleNormalized), file=f_norm)


# Generate the key-values samples
if keyvaluesOutEnabled == 'True':
    # Generate a full sample using sdm module
    sampleKeyValues = sdm.ngsi_ld_example_generator(schemaUrl)

    # search unfitting properties in the sample
    unfittingProperties = utils.match(utils.MATCHER_TYPE_SENTENCE, sampleKeyValues)
    print(f"unfittingProperties: {unfittingProperties}")

    for ii in range(genDepth):
        logging.debug(f"Generating key-values samples excluding {ii} random properties...")

        for jj in range(genIterations):
            logging.debug(f"Generating {jj}-th sample...")
            
            # Generate the full sample using sdm module
            sampleKeyValues = sdm.ngsi_ld_keyvalue_example_generator(schemaUrl)

            # compute the list of properties to remove (all the properties that are shared with other data models in the same subject, including a batch of random unique properties and excluding those in the list of properties to keep)
            excludedProperties = dmProperties - uniqueProperties - retainedProperties | set(random.sample(list(uniqueProperties), ii))

            # remove the excluded properties from the full sample
            for key in excludedProperties:
                sampleKeyValues = nested_delete(sampleKeyValues, key)

            # cleanup unfitting properties
            sampleKeyValues = utils.clear_properties(sampleKeyValues, unfittingProperties)

            # persist the resulting sample in a file
            with open(f"../output/{name}_keyvalues.json", "a") as f_kv:
                print(json.dumps(sampleKeyValues), file=f_kv)


