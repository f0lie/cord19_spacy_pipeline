import gzip
from collections import defaultdict

import pandas
# os.environ['R_HOME'] = "C:/Program Files/R/R-4.0.4"
# os.environ['PATH'] += "C:/Program Files/R/R-4.0.4/bin/x64;"
import rpy2.robjects as robjects
import spacy
from spacy.language import Language
from rpy2.robjects import pandas2ri
# noinspection PyUnresolvedReferences
from scispacy.abbreviation import AbbreviationDetector

BATCH_SIZE = 5
N_PROCESS = -1


def read_rds(input_filename: str) -> pandas.DataFrame:
    # Read RDS file and returns Dataframe
    print("Reading RDS file.")
    read_rds_function = robjects.r['readRDS']
    rds_file = read_rds_function(input_filename)
    return pandas2ri.rpy2py_dataframe(rds_file)


def write_rds(input_df, output_filename: str) -> None:
    # Writes a RDS dataframe into
    r_data = pandas2ri.py2rpy(input_df)
    robjects.r.assign("my_df", r_data)
    robjects.r(f"save(my_df, file='{output_filename}')")


def iter_row(input_df):
    # Function to generate the tuples for nlp.pipe
    # This is needed because pipe takes in an iterator of data. Usually people pass in the entire data structure
    # directly but you can get very creative by creating a generator and passing that instead.
    for _, row in input_df.iterrows():
        yield str(row['abstract']), {"cord_uid": row["cord_uid"], "type": "abstract"}
        yield str(row['full_text']), {"cord_uid": row["cord_uid"], "type": "full_text"}


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
        serializable_abbr = {"short_text": short_text, "short_start": short_start, "short_end": short_end,
                             "long_text": long_text, "long_start": long_start, "long_end": long_end}
        short._.long_form = None
        new_abbrevs.append(serializable_abbr)
    spacy_doc._.abbreviations = new_abbrevs
    return spacy_doc


def abbrev_doc_iter(doc, context):
    found_abbrevs = defaultdict(str)
    for abbrev in doc._.abbreviations:
        if abbrev["long_text"] not in found_abbrevs:
            found_abbrevs[abbrev["long_text"]] = abbrev["short_text"]

    for long_text, short_text in found_abbrevs.items():
        yield f"{context['cord_uid']},{context['type']},{short_text},\"{long_text}\"\n"


def pos_doc_iter(doc, context):
    # Takes in doc and context and outputs the CSV row of POS tagged words
    sentence = 0
    for sent in doc.sents:
        # Sentences are limited by brackets so the length could be found
        result = ""
        for token in sent:
            if token.is_alpha and not token.is_stop:
                result += f"{token.lemma_}//{token.tag_} "
        yield f'{context["cord_uid"]},{context["type"]},{sentence},{result}\n'
        sentence += 1


def dependencies_doc_iter(doc, context):
    # Takes Doc and context nd returns CSV rows of dependencies
    def token_to_str(dep_tok):
        # Takes tokens and outputs the dependencies
        children = [f"{child}" for child in dep_tok.children]
        return f'"{dep_tok.text}",{dep_tok.dep_},"{dep_tok.pos_}",' \
               f'"{dep_tok.head.text}",{dep_tok.head.pos_},{children}\n'

    sentence = 0
    for sent in doc.sents:
        for token in sent:
            yield f"{context['cord_uid']},{context['type']},{sentence}," + token_to_str(token)
        sentence += 1


def get_file(file_name, compress=False):
    # Takes in a file_name and returns either a gzip or normal file depending on compress flag
    if compress:
        return gzip.open(file_name + ".gz", 'wt', encoding='utf-8')
    else:
        return open(file_name, 'wt', encoding='utf-8')


def run(pipeline, text_df, dependency_file_name, pos_file_name, abbreviation_file_name, compress=False) -> None:
    print("Finding dependencies")
    dep_file = get_file(dependency_file_name, compress)
    dep_file.write("cord_uid,type,sentence,text,dep,pos,head_text,head_pos,children\n")

    print("Finding part of speech")
    pos_file_ = get_file(pos_file_name, compress)
    pos_file_.write("cord_uid,type,sentence,sentence_tagged\n")

    print("Find abbreviation")
    abbrev_file = get_file(abbreviation_file_name, compress)
    abbrev_file.write("cord_uid,type,abbreviation,full_definition\n")

    for doc, context in pipeline.pipe(iter_row(text_df), as_tuples=True, batch_size=BATCH_SIZE, n_process=N_PROCESS):
        for row in dependencies_doc_iter(doc, context):
            dep_file.write(row)
        for row in pos_doc_iter(doc, context):
            pos_file_.write(row)
        for row in abbrev_doc_iter(doc, context):
            abbrev_file.write(row)

    dep_file.close()
    pos_file_.close()
    abbrev_file.close()


if __name__ == "__main__":
    print("Loading nlp pipeline")
    # spacy.require_gpu()

    # Note to self: do not turn off tok2vec because its needed for sentences
    nlp = spacy.load("en_core_sci_sm", exclude=['ner'])
    nlp.add_pipe("abbreviation_detector")  # load this pipeline before running get_abrv
    nlp.add_pipe("serialize_abbreviation", after="abbreviation_detector")

    # If you want it to be faster you can remove the parser
    # nlp = spacy.load("en_core_sci_sm", exclude=['parser', 'ner', 'tok2vec'])
    # nlp.add_pipe("sentencizer")

    df = read_rds('parsing_test.rds')

    run(nlp, df, "data/dependencies.csv", "data/pos_tagged_text.csv", "data/found_abbreviations.csv")
