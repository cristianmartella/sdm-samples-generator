# To-Do list

### To do

- [ ] Cleanup generated samples:  
  - [ ] Update string properties with actual random samples (e.g. actual random addresses, names, etc., based on policies)  
- [ ] Replace subsets of property names with random synonyms  
  - [ ] Consider acronyms (synonyms may not be generated for acronyms)  

### In progress

- [ ] Allow CBOR encoding  
- [ ] Implement web API  

### Completed ✓

- [ ] Cleanup generated samples:  
  - [x] Clear strings of known faulty properties  
  - [x] Using regex: remove strings that have at least one space, exluding those that begin with a number followed by a dash (e.g. "02-Loss of communication")  
- [ ] Replace subsets of property names with random synonyms  
  - [x] Add batch ratio parameter to define the size of the batch of properties to be modified  
  - [x] Generate a list of synonyms for each given property and pick one randomly  
- [x] Allow the generation of entities with snake case properties  
- [x] Store original cleansed versions of the modified entities in a json document structured as follows:  

  ```json
  {
    "modidfiedSample": <modified sample json>,
    "originalSample": <original sample json>,
    "excludedProperties": [<omitted properties paths list>],
    ...
  }
  ```  