# cord19_spacy_pipeline
Uses scipacy to process cord19 papers

## Download CORD19
```bash
$ wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2021-02-15.tar.gz
$ tar -xzvf cord-19_2021-02-15.tar.gz
$ cd 2021-02-15
$ tar -xzvf document_parses.tar.gz
```

## Setup
Install using `pip install -r requirements.txt`. This installs spacy, numpy, and rpy2. Recommended to use `python -m venv` first for isolated environments.

## Running
Run `python parsing.py`. This pulls from the `parsing_test.rds` which is a rds file that contains a small subsection of Cord19 data. There is code to pull from raw data from Cord19 and output it as a CSV file in `cord19_to_csv.py`. Reading from the created csv requires code change in the main function of `parsing.py`.

This outputs files to `./output` that are csv files that contain found dependencies in text, abbreivations, and part of s
