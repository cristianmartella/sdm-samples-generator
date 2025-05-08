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

import os
import logging
import json
import random
import re
import multiprocessing as mp
import utils
from nested_lookup import nested_delete
from pysmartdatamodels import pysmartdatamodels as sdm


depthMaxThreshold = int(os.getenv("GEN-DEPTH-MAX-THR", "5"))
domain = os.getenv("SDM-DOMAIN", "")
retainedSharedProperties = set(os.getenv("RETAINED-SHARED-PROPERTIES", "id,type,@context").split(","))
anyNegativeSubject = eval(os.getenv("ANY-NEGATIVE-SUBJECT", "False"))

GEN_NORMALIZED = "normalized"
GEN_KEYVALUES = "keyvalues"
MATCH_LABEL_POSITIVE = 0.9
MATCH_LABEL_NEGATIVE = 0.1




dataModels = sdm.load_all_datamodels()


def get_schema_url(subject:str, name:str) -> str:
    """
    Get the schema URL for the subject and name of a given SDM.
    :param subject: The SDM subject.
    :param name: The SDM name.
    :return: The schema URL.
    """

    # Get the schema URL
    schemaUrl = f"https://raw.githubusercontent.com/smart-data-models/{subject}/master/{name}/schema.json"

    return schemaUrl

def get_subject_name_from_schema_url(schemaUrl:str) -> tuple[str, str]:
    """
    Extract the SDM subject and name from a given schemaUrl.
    :param schemaUrl: The schemaUrl to extract the SDM subject and name from.
    :return (subject, name): the extracted SDM subject and name.
    """
    path = re.sub("https://raw.githubusercontent.com/smart-data-models/|/schema.json", "", schemaUrl)
    info = re.split("/master/", path)

    return info[0], info[1]



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


def generate_sample(generator:str, schemaUrl:str, synBatchRatio:float, enableSnakeCase:bool, matchLabel:float, excludedProperties:set, retainedProperties:set) -> True:
    """
    Generate a sample for the given data model.
    :param generator: The generator type (normalized or keyvalues).
    :param schemaUrl: The schema URL.
    :param synBatchRatio: The ratio of synonyms to use.
    :param enableSnakeCase: Whether to enable snake case.
    :param matchLabel: The proposed match score of the generated sample (used for embedding training).
    :param excludedProperties: The list of excluded properties.
    :param retainedProperties: The list of retained properties.
    :return: The generated sample (includes the original generation, the modified version and the metadata of the generated sample).
    """
    # Generate a full sample using sdm module
    try:
        if generator == GEN_NORMALIZED:
            originalSample = sdm.ngsi_ld_example_generator(schemaUrl)
        elif generator == GEN_KEYVALUES:
            originalSample = sdm.ngsi_ld_keyvalue_example_generator(schemaUrl)
    except TypeError as te:
        raise ValueError(f"Invalid sample: {schemaUrl} failed to generate a sample. TypeError: {te}")

    if not isinstance(originalSample, dict):
        raise ValueError(f"Invalid sample: {schemaUrl} failed to generate a dict. Sample: {originalSample}")
    else:
        # discover unfitting properties in the sample
        unfittingProperties = utils.match(utils.MATCHER_TYPE_SENTENCE, originalSample)

        logging.debug(f"unfittingProperties: {unfittingProperties}")

        # remove the excluded properties from the full sample
        for key in excludedProperties:
            originalSample = nested_delete(originalSample, key)

        # cleanup unfitting properties
        originalSample = utils.clear_properties(originalSample, unfittingProperties)

        # replace batch of properties with random synonyms
        modifiedSample = originalSample.copy()
        if synBatchRatio > 0:
            # compute the number of properties to replace
            numPropertiesToReplace = int(len(modifiedSample.keys()) * synBatchRatio)
            # get a batch of random properties to replace (excluding the retained properties)
            propertiesToReplace = set(random.sample(list(set(modifiedSample.keys() - retainedProperties)), numPropertiesToReplace))
            # replace the properties with random synonyms
            modifiedProperties = {}
            for key in propertiesToReplace:
                modifiedProperties[key] = utils.randomize_camel_word(key)
                modifiedSample[modifiedProperties[key]] = modifiedSample.pop(key)

        # convert the properties to snake case if enabled
        if enableSnakeCase:
            modifiedSample = utils.dict_to_snake_keys(modifiedSample)
        
        # Extract SDM name and subject from schemaUrl
        name, subject = get_subject_name_from_schema_url(schemaUrl)

        # generated sample
        return {
            "modifiedSample": modifiedSample,    # the modified sample
            "originalSample": originalSample,    # the corresponding original sample
            "excludedProperties": list(excludedProperties),   # the properties that have been excluded from the original sample (reason: ambiguity)
            "unfittingProperties": list(unfittingProperties), # the properties whose values have been removed from the original sample (reason: gibberish)
            "modifiedProperties": modifiedProperties, # the original property names that have been modified from the original sample (reason: synonyms)
            "matchLabel": matchLabel,   # the match label of the generated sample
            "sdmMetadata": {
                "format": generator,
                "schemaUrl": schemaUrl,
                "domain": domain,
                "subject": subject,
                "name": name,
            } # the metadata of the generated sample
        }

    return True

def generate_samples(generator:str, schemaUrl:str, depth:int, iterations:int, synonymsBatchRatio:float, enableSnakeCase:bool) -> True:
    """
    Generate samples for the given data model.
    :param generator: The generator type (normalized or keyvalues).
    :param schemaUrl: The schema URL.
    :param depth: The depth of the generated samples.
    :param iterations: The number of iterations.
    :param synonymsBatchRatio: The ratio of synonyms to use.
    :param enableSnakeCase: Whether to enable snake case.
    :return: file(s) containing generated samples.
    """
    # Extract SDM subject and name from schemaUrl
    subject, name = get_subject_name_from_schema_url(schemaUrl)

    # Generate a full sample using sdm module (used to compute the unfitting properties)
    sample = sdm.ngsi_ld_keyvalue_example_generator(schemaUrl)

    # discover unfitting properties in the sample
    unfittingProperties = utils.match(utils.MATCHER_TYPE_SENTENCE, sample)
    logging.debug(f"unfittingProperties: {unfittingProperties}")

    # Get the list of shared properties for the given subject
    sharedPropertiesBySubject = get_shared_properties_by_subject(subject)
    logging.debug(f"get_shared_properties_by_subject({subject}): {sharedPropertiesBySubject}")

    # Extract the full set of properties for the given data model
    dmProperties = set(sdm.attributes_datamodel(subject, name))
    logging.debug(f"properties: {dmProperties}")

    # Remove the properties that are shared with the other data models in the same subject
    uniqueProperties = dmProperties - sharedPropertiesBySubject
    logging.debug(f"uniqueProperties: {uniqueProperties}")

    del sample

    for ii in range(depth):
        logging.debug(f"Generating normalized samples excluding {ii} random properties...")

        numExtraUniquePropertiesExcluded = ii if ii<len(uniqueProperties) else len(uniqueProperties)
        jj = 0

        while jj < iterations:
            logging.debug(f"Generating {ii}-{jj}-th sample...")
            
            # compute the list of properties to remove (all the properties that are shared with other data models in the same subject, including a batch of random unique properties and excluding those in the list of properties to keep)
            try:
                excludedProperties = sharedPropertiesBySubject - retainedSharedProperties | set(random.sample(list(uniqueProperties), numExtraUniquePropertiesExcluded))
            except ValueError as ve:
                logging.error(f"Error generating sample {ii}-{jj}: {ve}")
                continue

            # generate positive sample
            try:
                samplePositive = generate_sample(generator, schemaUrl, synonymsBatchRatio, enableSnakeCase, MATCH_LABEL_POSITIVE, excludedProperties, retainedSharedProperties)#, unfittingProperties)
                logging.debug(f"samplePositive: {samplePositive}")
            except ValueError as ve:
                logging.error(f"Error generating sample {ii}-{jj}: {ve}")
                continue
            
            # generate negative sample
            if anyNegativeSubject:
                negativeSubject = random.choice(list(get_subjects_by_domain(domain)))
                sharedPropertiesByNegativeSubject = get_shared_properties_by_subject(negativeSubject)
            else:
                negativeSubject = subject
                sharedPropertiesByNegativeSubject = sharedPropertiesBySubject

            negativeName = name
            while negativeName == name:
                negativeName = random.choice(list(sdm.datamodels_subject(negativeSubject)))
            
            negativeUrl = get_schema_url(negativeSubject, negativeName)

            ## compute the list of properties to exclude
            negativeDmProperties = set(sdm.attributes_datamodel(negativeSubject, negativeName))
            negativeUniqueProperties = negativeDmProperties - sharedPropertiesByNegativeSubject

            numExtraNegativeUniquePropertiesExcluded = ii if ii<len(negativeUniqueProperties) else len(negativeUniqueProperties)

            negativeExcludedProperties = sharedPropertiesByNegativeSubject - retainedSharedProperties | set(random.sample(list(negativeUniqueProperties), numExtraNegativeUniquePropertiesExcluded))
        
            ## generate the negative sample
            try:
                sampleNegative = generate_sample(generator, negativeUrl, synonymsBatchRatio, enableSnakeCase, MATCH_LABEL_NEGATIVE, negativeExcludedProperties, retainedSharedProperties)
                logging.debug(f"sampleNegative: {sampleNegative}")
            except ValueError as ve:
                logging.error(f"Error generating sample {ii}-{jj}: {ve}")
                continue

            # merge the two samples
            sample = {
                "target": samplePositive["originalSample"],     # the target sample corresponding to the correct data model representation
                "positive": {
                    "sample": samplePositive["modifiedSample"],   # the generated positive sample
                    "unfittingProperties": samplePositive["unfittingProperties"],   # the properties whose values have been removed from the original sample (reason: gibberish)
                    "excludedProperties": samplePositive["excludedProperties"],     # the properties that have been excluded from the original sample (reason: ambiguity)
                    "modifiedProperties": samplePositive["modifiedProperties"],     # the original property names that have been modified from the original sample (reason: synonyms)
                    "label": samplePositive["matchLabel"],   # the match label of the generated sample (high score indicates a good match)
                    "sdmMetadata": samplePositive["sdmMetadata"],   # the metadata of the generated sample
                },
                "negative": {
                    "sample": sampleNegative["originalSample"],   # the generated negative sample
                    "unfittingProperties": sampleNegative["unfittingProperties"],   
                    "excludedProperties": sampleNegative["excludedProperties"],
                    "modifiedProperties": sampleNegative["modifiedProperties"],
                    "label": sampleNegative["matchLabel"],      # the match label of the generated sample (low score indicates a bad match)
                    "sdmMetadata": sampleNegative["sdmMetadata"],
                },
            }

            # persist the resulting sample in a file
            try:
                outFile = f"../output/{subject}_{name}_{generator}.jsonl"
                with open(outFile, "a") as f_norm:
                    print(json.dumps(sample), file=f_norm)
            except OSError:
                logging.error(f"Could not open file {outFile} to store sample {ii}-{jj}")
            else:
                jj = jj+1
    
    logging.info(f"Done for {subject}.{name}-{generator}!")
    return True


def run_generator(subject:str, name:str, genDepth:int, genIterations:int, synBatchRatio:float, enableSnakeCase:bool, enableNormalisedOut:bool, enableKeyValuesOut:bool) -> True:    
    """
    Run the generator for the given data model.
    :param subject: The subject of the SDM.
    :param name: The name of the SDM.
    :param genDepth: The depth of the generated samples.
    :param genIterations: The number of iterations.
    :param synBatchRatio: The ratio of synonyms to use.
    :param enableSnakeCase: Whether to enable snake case.
    :param enableNormalizedOut: Whether to enable normalized output.
    :param enableKeyValuesOut: Whether to enable key-values output.
    :return: file(s) containing generated samples.
    """
    # Cut off threshrold for the depth of the generated samples
    if genDepth > depthMaxThreshold:
        genDepth = depthMaxThreshold

    # Generate the normalized samples
    positiveSchemaUrl = get_schema_url(subject, name)
    if enableNormalisedOut:
        generate_samples(GEN_NORMALIZED, positiveSchemaUrl, genDepth, genIterations, synBatchRatio, enableSnakeCase)
    # Generate the key-values samples
    if enableKeyValuesOut:
        job = mp.Process(target=generate_samples, args=(GEN_KEYVALUES, positiveSchemaUrl, genDepth, genIterations, synBatchRatio, enableSnakeCase))
        job.start()
        job.join()

    logging.info("Done!")
    return True