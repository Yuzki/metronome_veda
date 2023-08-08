import os
import argparse
import glob
import re
import csv
import json
import metronome as met
import pandas as pd
from bs4 import BeautifulSoup
from utils.transliteration import transliterate
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import seaborn as sns
import ast


class VedaMetronome():
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.src_dir = os.path.join(self.base_dir, 'src')
        self.json_path = os.path.join(self.src_dir, "text.json")

    def _create_text_dict(self):
        ''' Create a dictionary mapping verse identifiers to corresponding verses.
       
        This function reads TEI-encoded text files from a specified directory,
        extracts verse content, and constructs a dictionary where the keys
        are identifiers in the format bXX_hYYY_ZZ (representing book XX, hymn YY,
        and verse ZZ), and the values are the concatenated verses of the text.

        Returns:
            dict: A dictionary where keys are verse identifiers and values are
            concatenated verse texts.

        Note:
            The verse identifiers follow the pattern bXX_hYYY_ZZ, where:
            - XX represents the book number.
            - YY represents the hymn number.
            - ZZ represents the verse number within the hymn.
        '''

        text_path_list = glob.glob(os.path.join(self.src_dir, '*.tei'))

        text_dict = {}
        for text_path in text_path_list:
            with open(text_path, mode='r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'xml')
            lg_tags = soup.find_all('lg', {'source': 'vnh'})
            
            for lg_tag in lg_tags:
                lg_id = lg_tag.get('xml:id').replace("_vnh", "")
                l_texts = [l_tag.text.strip() for l_tag in lg_tag.find_all('l')]
                l_text_all = " |".join(l_texts)
                l_text_all += "|"
                text_dict[lg_id] = l_text_all

        with open(self.json_path, mode='w') as f:
            json.dump(text_dict, f)
        
        return text_dict

    def _convert_to_skeleton(self, text: str):
        '''Convert a given text into a phonemic skeleton representation.

        This function takes a text as input and processes each phoneme to generate a phonemic
        skeleton representation. The skeleton is constructed using the following symbols:
        - "V" for short vowels
        - "W" for long vowels
        - "C" for consonants
        - "H" for double consonants
        - "." for spaces
        - "|" for vertical lines
        Phonemes that do not match any of these categories are considered invalid.

        Args:
            text (str): The input text to be converted into a phonemic skeleton.

        Returns:
            str: The phonemic skeleton representation of the input text.
        
        Note:
            - The function handles both short and long vowels separately.
            - Accent marks such as ";" and ":" are ignored and not included in the skeleton.
            - If an invalid phoneme is encountered, an error message is printed and the
            function terminates, returning an incomplete skeleton.
        '''

        skeleton = ""

        for phoneme in text:
            if phoneme in self.short_vowel_list:
                # vowel
                skeleton += "V"
            elif phoneme in self.long_vowel_list:
                # double vowel
                skeleton += "W"
            elif phoneme in self.consonant_list:
                # consonant
                skeleton += "C"
            elif phoneme in self.double_consonant_list:
                # double consonant
                skeleton += "H"
            elif phoneme == " ":
                # space
                skeleton += "."
            elif phoneme == "|":
                # vertical line
                skeleton += "|"
            elif phoneme in [";", ":"]:
                continue
            else:
                print(f"{phoneme} is not in phoneme list.")
                break
            
        return skeleton

    def _clean_text(self, text: str):
        '''Clean the input text by removing specified characters.

        This function removes specific characters and patterns from the given input text
        using regular expressions. The characters and patterns removed include:
        - "3", "@", "\", "-", "+", "'", "&", "~", "*", "/", "`", and specified Unicode characters.
        
        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text with the specified characters and patterns removed.
        '''

        text = re.sub(r"[3@\\\-\+\'&~\*/\`\u0300]", "", text)

        return text

    def _transform_text_to_metronome(self, text: str):
        '''Transform the input text into a metronomic rhythm representation.

        This function takes a text in a specified transliteration scheme (IAST) and
        performs a series of transformations to generate a metronomic rhythm representation.
        The process involves converting the text to a phonemic skeleton, analyzing the
        syllable structure, and assigning metronomic symbols based on vowel weights and
        consonant counts.

        Args:
            text (str): The input text in the specified transliteration scheme (IAST).

        Returns:
            str: The metronomic rhythm representation of the input text.

        Note:
            - The function uses specific lists to identify short and long vowels, consonants,
            and double consonants.
            - Vowel weight is determined by a dictionary (vowel_weight) where "V" represents
            short vowels and "W" represents long vowels.
            - The syllable structure is determined using a phonemic skeleton generated by
            another function (_convert_to_skeleton).
            - The resulting metronomic rhythm representation uses "S" for stressed syllables
            and "w" for unstressed syllables.
        '''

        text_slp1 = transliterate(text, "iast", "slp1")
        text_slp1 = self._clean_text(text_slp1)

        self.short_vowel_list = ["a", "i", "u", "f", "x"]
        self.long_vowel_list = ["A", "I", "U", "F", "e", "o", "E", "O"]
        self.consonant_list = ["k", "K", "g", "G", "N", "c", "j", "J", "Y", "w", "W", "q", "R", "t", "T", "d", "D", "n", "p", "P", "b", "B", "m", "y", "r", "l", "v", "S", "z", "s", "h", "H", "M"]
        self.double_consonant_list = ["C", "Q"]

        # define the weight of vowels
        vowel_weight = {"V": 0, "W": 1}

        # get a skeleton (V, W, C, H, ., |)
        skeleton = self._convert_to_skeleton(text_slp1)
        # to calculate the final syllable
        skeleton += "C"

        # index of vowels in the skeleton
        vowel_idx_list = [idx for idx, phoneme in enumerate(skeleton) if phoneme in vowel_weight.keys()]

        # key: index of vowel, value: mora
        vowel_mora_dict = {}
        for i, vowel_idx in enumerate(vowel_idx_list):
            consonant_counter = 0

            # set the next index
            if i == len(vowel_idx_list) - 1:
                fin = len(skeleton)
            else:
                fin = vowel_idx_list[i+1]
            
            # sum up consonants
            for phoneme_idx in range(vowel_idx+1, fin):
                if skeleton[phoneme_idx] == "C":
                    consonant_counter += 1
                elif skeleton[phoneme_idx] == "H":
                    consonant_counter += 2
            
            # calculate mora from the weight of vowel and consonants in coda
            mora = vowel_weight[skeleton[vowel_idx]] + max(0, consonant_counter - 1)

            vowel_mora_dict[vowel_idx] = mora
        
        metronome = ""
        for idx, phoneme in enumerate(skeleton):
            if phoneme in [".", "|"]:
                metronome += phoneme
            else:
                try:
                    mora = vowel_mora_dict[idx]
                    if mora > 0:
                        metronome += "S"
                    else:
                        metronome += "w"
                except:
                    pass

        return metronome

    def _create_id(self, book_number: int, hymn_number: int, verse_number: int):
        '''Create a unique identifier for a verse based on book, hymn, and verse numbers.

        This function generates a unique identifier for a verse using the provided
        book number, hymn number, and verse number. The generated identifier follows
        the format bXX_hYYY_ZZ, where:
        - XX represents the book number with leading zeros.
        - YYY represents the hymn number with leading zeros.
        - ZZ represents the verse number with leading zeros.

        Args:
            book_number (int): The number of the book.
            hymn_number (int): The number of the hymn.
            verse_number (int): The number of the verse.

        Returns:
            str: A unique identifier for the verse in the format bXX_hYYY_ZZ.

        Example:
            Given book_number = 2, hymn_number = 15, verse_number = 4
            Created identifier: "b02_h015_04"
        '''

        return f"b{book_number:02d}_h{hymn_number:03d}_{verse_number:02d}"


    def _create_csv(self):
        '''Create a CSV file containing information about Rigvedic verses.

        This function constructs a CSV file containing various details about Rigvedic verses,
        including the author (poet), unique verse identifier, metronomic rhythm representation,
        and meter information. The function loads verse text from an existing JSON file or
        generates it using the specified method, reads additional information from a CSV file,
        and performs necessary transformations to create the final CSV.

        Note:
            - If a JSON file with verse text exists, it is loaded. Otherwise, text is generated
            using the _create_text_dict method.
            - The CSV file "rv_info.csv" is expected to be present in the source directory
            (src_dir), containing columns: "bookNum", "hymnNum", "verseNum", "poet", and "meter".
            - The "work" column in the CSV is created by combining "bookNum", "hymnNum", and "verseNum"
            using the _create_id method.
            - The "text" column in the CSV is populated with verse text from the loaded text_dict.
            - The "metronome" column is generated by applying the _transform_text_to_metronome method
            to the "text" column.
            - The "poet" column is renamed to "author" for consistency.
            - The CSV is saved as "rigveda.csv" in the base directory (base_dir).

        Example:
            Given existing JSON file with verse text and "rv_info.csv" containing verse information,
            the resulting "rigveda.csv" will have columns: "author", "work", "metronome", "meter",
            "text", and relevant verse details.
        '''

        # load text from json or make text
        if os.path.exists(self.json_path):
            with open(self.json_path, mode='r') as f:
                text_dict = json.load(f)
        else:
            text_dict = self._create_text_dict()

        # read poet, meter, deity, verse number CSV
        df = pd.read_csv(os.path.join(self.src_dir, "rv_info.csv"))

        # add "work" columns which is identical to verse ID
        df["work"] = df.apply(lambda row: self._create_id(row["bookNum"], row["hymnNum"], row["verseNum"]), axis=1)

        # add "text" columns
        df["text"] = df["work"].map(text_dict.get)

        # add "metronome" columns converted from text columns
        df["metronome"] = df["text"].apply(self._transform_text_to_metronome)

        # rename a column "poet" with "author"
        df.rename(columns={"poet": "author"}, inplace=True)
        new_header = ["author", "work", "metronome", "meter"]

        # save dataframe as CSV
        csv_path = os.path.join(self.base_dir, "rigveda.csv")
        df[new_header].to_csv(csv_path, index=False)


    def preprocess(self):
        '''Perform preprocessing steps to generate a CSV file with Rigvedic verse information.

        This function orchestrates the preprocessing of Rigvedic verse data, including the creation
        of a CSV file containing detailed information about each verse. The function invokes the
        internal method _create_csv, which performs the necessary data extraction, transformation,
        and saving processes.

        Note:
            - The function relies on the _create_csv method to generate the final CSV file.
            - The generated CSV file will contain columns such as "author", "work", "metronome",
            "meter", and "text", capturing important details about Rigvedic verses.
            - Data required for preprocessing, such as verse text and additional verse information,
            are sourced from existing JSON and CSV files.

        Example:
            Calling the preprocess method initiates the generation of a "rigveda.csv" file, which
            serves as a processed and structured dataset for further analysis and usage.
        '''
        
        print("Start preprocessing.")

        # Call the internal method to create the CSV
        self._create_csv()

        print("Complete.")


    def _basic_scoring(self, df: pd.DataFrame):
        '''Perform basic scoring on metronomic rhythm representations.

        This function calculates a basic scoring metric for metronomic rhythm representations
        of Rigvedic verses. The scoring is performed using the provided Scorer class from
        the met library. The resulting distance matrix is saved as a CSV file for further
        analysis and exploration.

        Args:
            df (DataFrame): A DataFrame containing verse information, including metronomic
            rhythm representations.

        Note:
            - The Scorer class is used from the met library to calculate a distance matrix
            based on metronomic rhythm representations.
            - The metronomic rhythm representations are expected to be present in the "metronome"
            column of the DataFrame.
            - The calculated distance matrix is saved as a CSV file named "basic_{branch_number}.csv"
            in the "data" directory within the base directory.
        '''

        print("Start Basic Scoring.")
        scorer = met.scoring.Scorer()
        # metronome is also the default column name
        df1 = scorer.dist_matrix(df, col='metronome')
        basic_scoring_df_path = os.path.join(self.base_dir, "data", f"basic_{self.branch_number}.csv")
        df1.to_csv(basic_scoring_df_path, index=False)
        print(f"Saved {os.path.basename(basic_scoring_df_path)}")


    def _fast_scoring(self, df: pd.DataFrame):
        '''Perform fast parallelized scoring on metronomic rhythm representations.

        This function calculates a fast parallelized scoring metric for metronomic rhythm
        representations of Rigvedic verses. The scoring is performed using the provided Scorer
        class from the met library. The resulting distance matrix is saved as a CSV file for
        further analysis and exploration.

        Args:
            df (DataFrame): A DataFrame containing verse information, including metronomic
            rhythm representations.

        Note:
            - The Scorer class is used from the met library to calculate a distance matrix
            based on metronomic rhythm representations.
            - The metronomic rhythm representations are expected to be present in the "metronome"
            column of the DataFrame.
            - The calculated distance matrix is saved as a CSV file named "fast_{branch_number}.csv"
            in the "data" directory within the base directory.
        '''

        print("Start Fast Scoring.")
        
        scorer = met.scoring.Scorer()
        # metronome is also the default column name
        df2 = scorer.dist_matrix_parallel(df, col='metronome')
        fast_scoring_df_path = os.path.join(self.base_dir, "data", f"fast_{self.branch_number}.csv")
        df2.to_csv(fast_scoring_df_path, index=False)

        print(f"Saved {os.path.basename(fast_scoring_df_path)}")


    def scoring(self, df: pd.DataFrame, books: list):
        '''Perform metronomic rhythm scoring on Rigvedic verses.

        This function orchestrates the process of scoring metronomic rhythm representations
        for specified Rigvedic verses. It calculates both basic and fast parallelized scoring
        metrics using provided Scorer classes from the met library. The resulting distance
        matrices are saved as CSV files for further analysis and exploration.

        Args:
            df (DataFrame): A DataFrame containing verse information, including metronomic
            rhythm representations.
            books (list): A list of book numbers to be included in the scoring process.

        Note:
            - The DataFrame contains verse information with metronomic rhythm representations,
            and the book numbers specified in the "books" list.
            - The scoring process involves invoking internal methods for basic and fast scoring.
            - The calculated distance matrices are saved as CSV files named "basic_{branch_number}.csv"
            and "fast_{branch_number}.csv" in the "data" directory within the base directory.

        Example:
            Given a DataFrame with verse information and a list of book numbers, calling this method
            performs both basic and fast scoring on the specified verses and saves the resulting
            distance matrices for further analysis.
        '''

        self.df = df
        self.books = books
        self.branch_number = '-'.join([str(book) for book in self.books])

        # follow example code
        # Basic Scoring
        self._basic_scoring(self.df)

        # Fast Scoring (runs ray locally, may be more fragile)
        self._fast_scoring(self.df)


    def _save_label_color_map(self, dendro):
        '''Save a label-color mapping for dendrogram visualization.

        This function generates and saves a mapping between labels and colors for the purpose
        of dendrogram visualization. The mapping is saved as a tab-separated values (TSV) file,
        where each row consists of a color and its corresponding label.

        Args:
            dendro (dict): A dictionary containing dendrogram information, including leaves color
            list and intercalation vector list (ivl).

        Note:
            - The dendrogram information is used to extract leaf color information and labels.
            - The color palette used for mapping is derived from the "tab10" seaborn color palette.
            - The generated mapping TSV file is named "text_{books}.tsv" and is saved in the "data"
            directory within the base directory.
        '''

        tsv_path = os.path.join(self.base_dir, "data", f"text_{'-'.join(self.books)}.tsv")
        with open(tsv_path, 'w') as tsvfile:
            lv_color_list = dendro['leaves_color_list']

            writer = csv.writer(tsvfile, delimiter="\t")
            color_set = sns.color_palette("tab10", n_colors=len(set(lv_color_list))+1)
            for color, label in zip(lv_color_list, dendro['ivl']):
                writer.writerow([color_set[int(color[-1])], f"{label[1]}_{label[0]}"])


    def create_text_plot(self, tsv_path: str):
        '''Create a colored text plot for dendrogram labels.

        This function generates a colored text plot for dendrogram labels based on the provided
        TSV (tab-separated values) file. Each row in the TSV file consists of a color and a label,
        which are used to create the text plot. The resulting plot is saved as a PNG image file.

        Args:
            tsv_path (str): The path to the TSV file containing color and label information.

        Note:
            - The TSV file is expected to have two columns: "color" and "label".
            - The "color" column contains color information in a format that can be evaluated using
            the ast.literal_eval function to retrieve a list of color values.
            - The "label" column contains the text labels associated with the colors.
            - The text plot is created with each label displayed in a specific color and arranged
            in ascending order.
            - The generated PNG image file is saved in the "fig" directory within the base directory.
        '''

        # TSVファイルをpandasのDataFrameとして読み込みます
        df = pd.read_csv(tsv_path, sep='\t')

        df.columns = ['color', 'label']

        # 文字列を昇順に並び替えます
        df.sort_values(by='label', inplace=True)

        # 図のサイズを設定します
        plt.figure(figsize=(5, 20))

        num_rows = df.shape[0]
        # 各行の色と文字列を取得して図を作成します
        i=0
        for index, row in df.iterrows():
            color = [x for x in ast.literal_eval(row['color'])]
            text = row['label']
            # print(text, color)
            plt.text(0.5, (i / num_rows), text, fontsize=5, color=color, ha='center', va='center')
            i+=1

        plt.axis('off')  # 軸を非表示にする
        plt.savefig(os.path.join(self.base_dir, "fig", f"color_{os.path.splitext(os.path.basename(tsv_path))[0]}.png"))


    def dendrogram(self, csv_path: str, id_to_label: dict):
        '''Generate a dendrogram visualization for clustering of Rigvedic verses.

        This function performs hierarchical clustering on Rigvedic verses based on a distance matrix
        calculated from provided data. It then generates a dendrogram visualization to depict the
        clustering structure. The resulting dendrogram plot is saved as a PNG image file.

        Args:
            csv_path (str): The path to the CSV file containing data for clustering.
            id_to_label (dict): A dictionary mapping unique verse IDs to corresponding labels.

        Note:
            - The CSV file should contain data suitable for hierarchical clustering, such as a distance
            matrix calculated based on metronomic rhythm representations.
            - The id_to_label dictionary provides a mapping between unique verse IDs and their respective
            labels, which will be used for labeling the dendrogram leaves.
            - The generated dendrogram plot is saved in the "fig" directory within the base directory.

        Example:
            Given a CSV file with distance matrix data and a mapping between verse IDs and labels, calling
            this method performs hierarchical clustering and generates a dendrogram visualization.
        '''

        self.id_to_label = id_to_label
        self.labels = list(id_to_label.values())
        self.books = csv_path.split('.')[0].split('_')[1].split('-')

        df = pd.read_csv(os.path.join(self.base_dir, csv_path))

        # 距離行列の計算
        distance_matrix = pdist(df)

        # linkage関数に距離行列を渡して階層的クラスタリングを行う
        self.Z = linkage(distance_matrix, method='ward')


        # デンドログラムの描画
        plt.figure(figsize=(160, 90), dpi=200)
        color_threshold = 15
        dendro = dendrogram(self.Z,
                labels=self.labels,
                orientation='right',
                # link_color_func=self._link_color_func
                color_threshold=color_threshold
                )
        self._save_label_color_map(dendro)
        
        # ラベルに色をつける
        meter_set = set([label[0] for label in self.labels])
        palette = sns.color_palette("husl", len(meter_set))
        # palette = sns.husl_palette(len(meter_set))
        # palette = sns.hls_palette(len(meter_set))
        self.label_to_color = {label: color for label, color in zip(meter_set, palette)}
        ax = plt.gca()
        ylbls = ax.get_ymajorticklabels()
        for lbl in ylbls:
            meter = ast.literal_eval(lbl.get_text())[0]
            lbl.set_color(self.label_to_color[meter])

        plt.title(f"Rigveda {' '.join([str(book) for book in self.books])}", fontsize=90)
        plt.xlabel('Distance', fontsize=60)
        plt.ylabel('Labels', fontsize=60)
        plt.xticks(fontsize=50)

        if self.books == [1] or self.books == [8] or self.books == [10]:
            plt.yticks(fontsize=4)

        bn = os.path.splitext(os.path.basename(csv_path))[0]
        img_path = os.path.join(self.base_dir, "fig", f"clustering_{bn}.png")
        plt.savefig(img_path)


def get_args():
    '''Parse command-line arguments for the VedaMetronome script.

    This function sets up and configures an argument parser to handle command-line options and
    arguments for the VedaMetronome script. It defines the available command-line flags and their
    associated help messages, and then parses the command-line arguments.

    Returns:
        argparse.Namespace: A namespace containing the parsed command-line arguments.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", action="store_true", help="Perform preprocessing")
    parser.add_argument("-m", "--metronome", action="store_true", help="Scoring")
    parser.add_argument("-d", "--dendrogram", action="store_true", help="Visualization (dendrogram)")
    parser.add_argument("-t", "--text", action="store_true", help="Visualization (text, sorted labels)")
    parser.add_argument("-b", "--books", nargs='+', type=int, help="Specify up to 10 values")
    args = parser.parse_args()

    return args


def main():
    '''VedaMetronome Script

    This is the main entry point for the VedaMetronome script. It processes command-line options
    and arguments to control various tasks related to preprocessing, scoring, dendrogram
    visualization, and text visualization of Rigvedic verses. The script performs the specified
    tasks based on the provided command-line options and arguments.

    Usage:
        python vedametronome.py -p -m -d -t -b 1 2 3 ...

    Options:
        -p, --preprocess   Perform preprocessing on Rigvedic text files.
        -m, --metronome    Perform scoring of metronomic rhythm and generate scores.
        -d, --dendrogram   Generate dendrogram visualization for clustering.
        -t, --text         Generate text visualization with sorted labels.
        -b, --books        Specify up to 10 book values for analysis.

    Example:
        Calling this script with appropriate command-line options and arguments allows you to
        perform various analyses on Rigvedic verses, including preprocessing, scoring, and
        visualization.
    '''


    args = get_args()

    rv_metronome = VedaMetronome()

    # create a text
    if args.preprocess:
        rv_metronome.preprocess()

    # create a data
    df = pd.read_csv("rigveda.csv")
    books = sorted(args.books[:10])
    pattern = "|".join(["b{:02d}".format(book) for book in books])
    df_data = df[df["work"].str.contains(pattern)]
    labels = df_data.apply(lambda row: (row["meter"], row["work"]), axis=1).tolist()
    id_to_label = {idx: label for idx, label in zip(df_data.index, labels)}

    bn = '-'.join([str(book) for book in books])


    # score
    if args.metronome:
        rv_metronome.scoring(df_data, books)
    
    # clustering
    if args.dendrogram:
        csv_path = os.path.join("data", f"basic_{bn}.csv")
        rv_metronome.dendrogram(csv_path, id_to_label)
        # rv_metronome.color_bar()

    # coloring text
    if args.text:
        tsv_path = os.path.join("data", f"text_{bn}.tsv")
        rv_metronome.create_text_plot(tsv_path)


if __name__ == '__main__':
    main()