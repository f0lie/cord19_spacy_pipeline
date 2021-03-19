import gzip
from collections import defaultdict

import pandas
# os.environ['R_HOME'] = "C:/Program Files/R/R-4.0.4"
# os.environ['PATH'] += "C:/Program Files/R/R-4.0.4/bin/x64;"
import rpy2.robjects as robjects
import spacy
from rpy2.robjects import pandas2ri
# noinspection PyUnresolvedReferences
from scispacy.abbreviation import AbbreviationDetector

BATCH_SIZE = 1000
N_PROCESS = 4


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


def get_abrv(pipeline, text_df, file_name, compress=False) -> None:
    # Given the spacy nlp and an pandas dataframe. Writes the results into file_name.
    print("Finding abbreviations")
    if compress:
        csv_file = gzip.open(file_name + ".gz", 'wt', encoding='utf-8')
    else:
        csv_file = open(file_name, 'wt', encoding='utf-8')

    csv_file.write("cord_uid,abbreviation,full_definition\n")
    for row in abbrev_iter(pipeline, iter_row(text_df)):
        csv_file.write(row)
    csv_file.close()


def abbrev_iter(pipeline, test_iter):
    # Generates the rows of the csv file from a iterator containing the data
    for doc, context in pipeline.pipe(test_iter, as_tuples=True, batch_size=BATCH_SIZE):
        # Finds all of the abbrevs on a coid_uid and type level. Abstracts and full text are treated differently.
        for row in abbrev_doc_iter(doc, context):
            yield row


def abbrev_doc_iter(doc, context):
    found_abbrevs = defaultdict(str)
    for abbrev in doc._.abbreviations:
        if abbrev._.long_form not in found_abbrevs:
            found_abbrevs[abbrev._.long_form] = abbrev.text

    for definition, text_abrv in found_abbrevs.items():
        yield f"{context['cord_uid']},{context['type']},{text_abrv},\"{definition}\"\n"


def pos_doc_iter(doc, context):
    # Takes in doc and context and outputs the CSV row of POS tagged words
    result = ""
    for sent in doc.sents:
        # Sentences are limited by brackets so the length could be found
        result += "["
        for token in sent:
            if token.is_alpha and not token.is_stop:
                result += f"{token.lemma_}//{token.tag_},"
        result += "]"
    yield f'{context["cord_uid"]},{context["type"]},{result}\n'


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


def run(pipeline, text_df, dependency_file_name, pos_file_name, compress=False) -> None:
    print("Finding dependencies")
    dep_file = get_file(dependency_file_name, compress)
    dep_file.write("cord_uid,type,sentence,text,dep,pos,head_text,head_pos,children\n")

    print("Finding part of speech")
    pos_file_ = get_file(pos_file_name, compress)
    pos_file_.write("cord_uid,type,text\n")

    for doc, context in pipeline.pipe(iter_row(text_df), as_tuples=True, batch_size=BATCH_SIZE, n_process=N_PROCESS):
        for row in dependencies_doc_iter(doc, context):
            dep_file.write(row)
        for row in pos_doc_iter(doc, context):
            pos_file_.write(row)

    dep_file.close()
    pos_file_.close()


if __name__ == "__main__":
    print("Loading nlp pipeline")
    # spacy.require_gpu()
    # Note to self: do not turn off tok2vec because its needed for sentences
    nlp = spacy.load("en_core_sci_sm", exclude=['ner'])

    # If you want it to be faster you can remove the parser
    # nlp = spacy.load("en_core_sci_sm", exclude=['parser', 'ner', 'tok2vec'])
    # nlp.add_pipe("sentencizer")

    df = read_rds('parsing_test.rds')

    # cProfile.run('get_dependencies(nlp, df, "data/dependencies.csv")')

    run(nlp, df, "data/dependencies.csv", "data/pos_tagged_text.csv")

    # WARNING: Do not run abbreviation_detector with the other functions, it does not play nice with mutliple processes
    # Manually removing it from the pipeline doesn't work either
    # Add the pipe after you run the other two
    # I think it's because it's scispacy's stuff not spacy
    # nlp.add_pipe("abbreviation_detector")  # load this pipeline before running get_abrv
    # get_abrv(nlp, df, "data/found_abbreviations.csv")
    # nlp.remove_pipe("abbreviation_detector")
