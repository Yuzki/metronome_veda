import csv
import os

def extract_table(csv_file_path, column1_name, column2_name):
    result = []
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            item = [row[column1_name], row[column2_name]]
            result.append(item)
    
    return result

def transliterate(text:str, input_method:str, output_method:str):

    # load table
    table = extract_table(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transliteration.csv'), input_method, output_method)

    # transliterate
    orig_text = text
    for io_list in table:
        text = text.replace(io_list[0], io_list[1])
    
    return text

if __name__ == "__main__":
    s = transliterate(";agnim ii;le pur;ohitam", "tf", "iast")
    print(s)