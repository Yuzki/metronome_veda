import glob
import json
import os

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()
TEXT_PATH = os.environ.get("TEXT_PATH")
dir_name = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(dir_name, "src/text.json")
df_path = os.path.join(dir_name, "src/rigveda_text_poet_meter.csv")


def create_text_dataframe() -> DataFrame:
    if os.path.exists(json_path):
        print(f"load {os.path.basename(json_path)}")
        with open(json_path, mode="r", encoding="utf-8") as f:
            text_dict = json.load(f)
    else:
        print(f"creating {os.path.basename(json_path)}")
        text_dict = _create_text_dict()
        print(f"load {os.path.basename(json_path)}")

    if os.path.exists(df_path):
        print(f"load {os.path.basename(df_path)}")
        df = pd.read_csv(df_path)

    else:
        print(f"creating {os.path.basename(df_path)}")
        # read poet, meter, deity, verse number CSV
        df = pd.read_csv(os.path.join(dir_name, "src/rv_info.csv"))

        # add "work" columns which is identical to verse ID
        df["work"] = df.apply(
            lambda row: f"b{row['bookNum']:02d}_h{row['hymnNum']:03d}_{row['verseNum']:02d}",
            axis=1,
        )

        # add "text" columns
        df["text"] = df["work"].map(text_dict.get)

        # rename a column "poet" with "author"
        df.rename(columns={"poet": "author", "bookNum": "book"}, inplace=True)
        new_header = [
            "author",
            "work",
            "book",
            "meter",
            "text",
        ]

        # save dataframe as CSV
        csv_path = os.path.join(dir_name, "src/rigveda_text_poet_meter.csv")
        df[new_header].to_csv(csv_path, index=False)
        print(f"load {os.path.basename(df_path)}")

    return df


def _create_text_dict() -> dict:
    text_dir = os.path.join(dir_name, TEXT_PATH)
    text_path_list = glob.glob(os.path.join(text_dir, "*.tei"))

    text_dict = {}
    for text_path in text_path_list:
        with open(text_path, mode="r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "xml")
        lg_tags = soup.find_all("lg", {"source": "vnh"})

        for lg_tag in lg_tags:
            lg_id = lg_tag.get("xml:id").replace("_vnh", "")
            l_texts = [l_tag.text.strip() for l_tag in lg_tag.find_all("l")]
            l_text_all = "|".join(l_texts)
            l_text_all += "|"
            text_dict[lg_id] = l_text_all

    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(text_dict, f, indent=2)

    return text_dict


if __name__ == "__main__":
    df_rv = create_text_dataframe()
    print(df_rv)
