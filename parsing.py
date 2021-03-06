import csv
import gzip
import time
from collections import defaultdict

import pandas
import rpy2.robjects as robjects
import spacy
from rpy2.robjects import pandas2ri

from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
from spacy.language import Language


def read_rds(input_filename: str) -> pandas.DataFrame:
    # Read RDS file and returns Dataframe
    print("Reading RDS file.")
    read_rds_function = robjects.r["readRDS"]
    rds_file = read_rds_function(input_filename)
    return pandas2ri.rpy2py_dataframe(rds_file)


def write_rds(input_df, output_filename: str) -> None:
    # Writes a RDS dataframe into
    r_data = pandas2ri.py2rpy(input_df)
    robjects.r.assign("my_df", r_data)
    robjects.r(f"save(my_df, file='{output_filename}')")


def iter_row(input_df):
    # Function to generate the tuples for nlp.pipe
    # This is needed because pipe takes in an iterator of data.
    # Usually people pass in the entire data structure
    # directly but you can get very creative by creating a generator and passing that instead.
    for _, row in input_df.iterrows():
        yield str(row["abstract"]), {"cord_uid": row["cord_uid"], "type": "abstract"}
        yield str(row["full_text"]), {"cord_uid": row["cord_uid"], "type": "full_text"}


@Language.component("serialize_abbreviation")
def replace_abbrev_with_json(spacy_doc):
    # https://github.com/allenai/scispacy/issues/205#issuecomment-597273144
    # Code is modified to work as a component
    new_abbrevs = []
    for short in spacy_doc._.abbreviations:
        short_text = short.text
        short_start = short.start
        short_end = short.end
        long = short._.long_form
        long_text = long.text
        long_start = long.start
        long_end = long.end
        serializable_abbr = {
            "short_text": short_text,
            "short_start": short_start,
            "short_end": short_end,
            "long_text": long_text,
            "long_start": long_start,
            "long_end": long_end,
        }
        short._.long_form = None
        new_abbrevs.append(serializable_abbr)
    spacy_doc._.abbreviations = new_abbrevs
    return spacy_doc


def abbrev_doc_iter(doc, context):
    # Takes in a Doc and outputs the CSV row of abbreviation output
    found_abbrevs = defaultdict(str)
    for abbrev in doc._.abbreviations:
        if abbrev["long_text"] not in found_abbrevs:
            found_abbrevs[abbrev["long_text"]] = abbrev["short_text"]

    for long_text, short_text in found_abbrevs.items():
        yield context["cord_uid"], context["type"], short_text, long_text


def pos_doc_iter(doc, context):
    # Takes in a Doc and context and outputs the CSV row of POS tagged words
    sentence = 0
    for sent in doc.sents:
        # Sentences are limited by brackets so the length could be found
        yield context["cord_uid"], context["type"], sentence, [
            f"{token.lemma_}//{token.tag_}" for token in sent if token.is_alpha and not token.is_stop
        ]
        sentence += 1


def dependencies_doc_iter(doc, context):
    # Takes Doc and context nd returns CSV rows of dependencies
    sentence = 0
    for sent in doc.sents:
        for token in sent:
            yield context["cord_uid"], context[
                "type"
            ], sentence, token.text, token.dep_, token.pos_, token.head.text, token.head.pos_, list(token.children)
        sentence += 1


def get_file(file_name, compress=False):
    # Takes in a file_name and returns either a gzip or normal file depending on compress flag
    if compress:
        file = gzip.open(file_name + ".gz", "wt", encoding="utf-8")
    else:
        file = open(file_name, "w")
    return file, csv.writer(file)


# TODO: Maybe change the code so it uses a list of files and writers as opposed to putting everything in parameters
def run(
    pipeline,
    iter_text,
    dependency_file_name,
    pos_file_name,
    abbreviation_file_name,
    compress=False,
    batch_size=5,
    n_process=-1,
) -> None:
    # Takes in a spacy pipeline, text file, and file names. Writes results to those file_names
    # compress is a flag to output files to csv
    # batch_size is the number of docs to cache for a process
    # n_process is the number of processes to run
    print("Finding dependencies")
    dep_file, dep_writer = get_file(dependency_file_name, compress)
    dep_writer.writerow(["cord_uid", "type", "sentence", "text", "dep", "pos", "head_text", "head_pos", "children"])

    print("Finding part of speech")
    pos_file, pos_writer = get_file(pos_file_name, compress)
    pos_writer.writerow(["cord_uid", "type", "sentence", "sentence_tagged"])

    print("Find abbreviation")
    abbrev_file, abbrev_writer = get_file(abbreviation_file_name, compress)
    abbrev_writer.writerow(["cord_uid", "type", "abbreviation", "full_definition"])

    start = time.time()
    documents_processed = 0
    for doc, context in pipeline.pipe(iter_text, as_tuples=True, batch_size=batch_size, n_process=n_process):
        dep_writer.writerows(dependencies_doc_iter(doc, context))
        pos_writer.writerows(pos_doc_iter(doc, context))
        abbrev_writer.writerows(abbrev_doc_iter(doc, context))
        documents_processed += 1
    end = time.time()
    print(
        f"{documents_processed} documents processed in {end - start: .2f} seconds. "
        f"{documents_processed / (end - start): .2f} documents per second"
    )

    dep_file.close()
    pos_file.close()
    abbrev_file.close()


if __name__ == "__main__":
    print("Loading nlp pipeline")
    # GPU does not work with multiprocessing
    # spacy.require_gpu()

    # Note to self: do not turn off tok2vec because its needed for sentences
    nlp = spacy.load("en_core_sci_sm", exclude=["ner"])
    nlp.add_pipe("abbreviation_detector")
    nlp.add_pipe("serialize_abbreviation", after="abbreviation_detector")

    df = read_rds("parsing_test.rds")

    run(
        nlp,
        iter_row(df),
        "output/dependencies.csv",
        "output/pos_tagged_text.csv",
        "output/found_abbreviations.csv",
    )
