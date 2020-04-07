# wonderpy: Use Python3 and Selenium to automatically complete the CDC Wonder Online Form

## Disclaimer
This project is a work-in-progress. It works in some cases but may not work in many others (and may not be flexible enough for some users), and some extra code exists for functions that are not yet operational. This program has only been tested in Windows 10.


## Description
The CDC's WONDER API does not allow users to collect subnational data -- users have to use the online form. This package automates that form. Note that this is still a work in progress.

This package is intended to be used with the R package ["wondeR"](https://www.github.com/tlcaputi/wondeR).

## Installation

```
$ pip install wonderpy
```


## Usage

```{python}
from wonderpy.pulldata import wonder

wonder(
  ## wonderpy arguments
  RUN_NAME = "opioids", # Name of the run
  download_dir = "/path/to/downloads/folder", # Where to download the data
  existing_file = False, # True if the data file already exists on your local computer

  ## You want deaths matching the following ICD Codes
  MCD_CAUSE_CODE_1 = ["A10", "A20"], # Deaths that include ANY of these ICD Codes
  MCD_CAUSE_CODE_2 = ["A12", "A22"], # ...and ANY of these ICD Codes

  ## Group the deaths by these variables
  by_variables = ["sex", "age"], # Any of the following: "sex", "age", "race", "hispanic", "state", "year", "month"

  ## Only collect data of the following (None if all)
  AGEG = None, # age group
  GENDERG = None, # gender
  RACEG = None, # race
  HISPANICG = None, # hispanic status

  ## Other
  just_go = False # For use with more further and undeveloped cases. Keep as False.
)

```
