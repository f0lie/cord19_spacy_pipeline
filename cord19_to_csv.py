# Stolen and modified from here
# https://github.com/allenai/cord19/blob/master/README.md
import csv
import json

# Location of unpacked files
FILE_PATH = 'data/2021-02-15/'
OUTPUT_PATH = 'sample_cord.csv'
# LIMIT = float('inf')
LIMIT = 10000

# open the file
rows_read = 0
with open(OUTPUT_PATH, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["cord_uid", "title", "abstract", "body_text"])
    with open(FILE_PATH + 'metadata.csv') as f_in:
        reader = csv.DictReader(f_in)
        # Read metadata.csv row by row
        for row in reader:
            if rows_read > LIMIT:
                break
            rows_read += 1

            # access some metadata
            cord_uid = row['cord_uid']
            title = row['title']
            abstract = row['abstract']

            # Dump body_text into a list
            body_text = []
            if row['pmc_json_files']:
                for json_path in row['pmc_json_files'].split('; '):
                    with open(FILE_PATH + json_path) as f_json:
                        full_text_dict = json.load(f_json)

                        # grab introduction section from *some* version of the full text
                        for paragraph_dict in full_text_dict['body_text']:
                            paragraph_text = paragraph_dict['text']
                            body_text.append(paragraph_text)

            writer.writerow([cord_uid, title, abstract, body_text])
