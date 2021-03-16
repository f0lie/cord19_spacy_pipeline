import gzip
import os
from collections import defaultdict
import spacy
# noinspection PyUnresolvedReferences
from scispacy.abbreviation import AbbreviationDetector
import pandas

#os.environ['R_HOME'] = "C:/Program Files/R/R-4.0.4"
#os.environ['PATH'] += "C:/Program Files/R/R-4.0.4/bin/x64;"
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

#import cProfile

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


def get_abrv(pipeline, text_df, file_name, compress=False) -> None:
    # Given the spacy nlp and an pandas dataframe. Writes the results into file_name.

    # Used a dict to store abbrevs because it's very efficient, it can scale to millions of rows
    # The first level is for cord_uid, it's more compact and easier to understand doing that
    # The second level is another dict to store the abbrevs of that document
    print("Finding abbreviations")
    found_abrv = defaultdict(dict)  # When we add a new cord_uid, it makes a new dict. Simplifies code.
    for _, row in text_df.iterrows():
        doc = pipeline(str(row['abstract']) + str(row['full_text']))
        for abbrev in doc._.abbreviations:
            # If the abbrev is not found, then add it to the dict.
            # Resolves dup issues when papers use the same abbrevs.
            if abbrev._.long_form not in found_abrv[row['cord_uid']]:
                found_abrv[row['cord_uid']][abbrev._.long_form] = abbrev.text

    if compress:
        csv_file = gzip.open(file_name + ".gz", 'wt', encoding='utf-8')
    else:
        csv_file = open(file_name, 'wt', encoding='utf-8')

    csv_file.write("cord_uid,abbreviation,full_definition\n")
    for cord_uid, abbrev in found_abrv.items():
        for definition, text_abrv in abbrev.items():
            csv_file.write(f"{cord_uid},{text_abrv},\"{definition}\"\n")

    csv_file.close()


def iter_row(input_df):
    # Function to generate the tuples for nlp.pipe
    # This is needed because pipe takes in an iterator of data. Usually people pass in the entire data structure
    # directly but you can get very creative by creating a generator and passing that instead.
    for _, row in input_df.iterrows():
        yield str(row['abstract']), {"cord_uid": row["cord_uid"], "type": "abstract"}
        yield str(row['full_text']), {"cord_uid": row["cord_uid"], "type": "full_text"}


def get_pos(pipeline, text_df, file_name, compress=False) -> None:
    print("Finding part of speech.")
    if compress:
        csv_file = gzip.open(file_name + ".gz", 'wt', encoding='utf-8')
    else:
        csv_file = open(file_name, 'wt', encoding='utf-8')

    csv_file.write("cord_uid,type,text\n")
    for doc, context in pipeline.pipe(iter_row(text_df), as_tuples=True, batch_size=BATCH_SIZE, n_process=N_PROCESS):
        result = ""
        for sent in doc.sents:
            result += "["
            for token in sent:
                if token.is_alpha and not token.is_stop:
                    result += f"{token.lemma_}//{token.tag_},"
            result += "]"
            csv_file.write(f'{context["cord_uid"]},{context["type"]},{result}\n')

    csv_file.close()


def get_dependencies(pipeline, text_df, file_name, compress=False) -> None:
    print("Finding the dependencies")
    if compress:
        csv_file = gzip.open(file_name + ".gz", 'wt', encoding='utf-8')
    else:
        csv_file = open(file_name, 'wt', encoding='utf-8')

    # Makes the csv write less ugly and easier to understand
    def token_to_str(dep_tok):
        children = [f"{child}" for child in dep_tok.children]
        return f'"{dep_tok.text}",{dep_tok.dep_},"{dep_tok.pos_}",' \
               f'"{dep_tok.head.text}",{dep_tok.head.pos_},{children}\n'

    # The every row of the CSV file is a single word.
    # type is whether it's the full_text or abstract
    # sentence is the serial id of the text that it belongs to
    csv_file.write("cord_uid,type,sentence,text,dep,pos,head_text,head_pos,children\n")
    for doc, context in pipeline.pipe(iter_row(text_df), as_tuples=True, batch_size=BATCH_SIZE, n_process=N_PROCESS):
        sentence = 0
        for sent in doc.sents:
            for token in sent:
                csv_file.write(f"{context['cord_uid']},{context['type']},{sentence}," + token_to_str(token))
            sentence += 1

    csv_file.close()


if __name__ == "__main__":
    print("Loading nlp pipeline")
    # spacy.require_gpu()
    # nlp = spacy.load("en_core_sci_lg")
    # Note to self: do not turn off tok2vec because its needed for sentences
    nlp = spacy.load("en_core_sci_sm", exclude=['ner'])

    # If you want it to be faster you can remove the parser
    # nlp = spacy.load("en_core_sci_sm", exclude=['parser', 'ner', 'tok2vec'])
    # nlp.add_pipe("sentencizer")

    df = read_rds('parsing_test.rds')

    # nlp.add_pipe("abbreviation_detector")  # load this pipeline before running get_abrv
    # get_abrv(nlp, df, "data/found_abbreviations.csv")
    get_pos(nlp, df, "data/pos_tagged_text.csv")
    # cProfile.run('get_dependencies(nlp, df, "data/dependencies.csv")')
    get_dependencies(nlp, df, "data/dependencies.csv")
