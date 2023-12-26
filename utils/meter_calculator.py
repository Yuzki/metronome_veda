import csv
import os
import re

from transliteration import transliterate


class VedicMeter:
    def __init__(self, text: str, transliteration: str = "iast") -> None:
        transliteration_list = self._load_transliteration_methods()

        if transliteration.lower() in transliteration_list:
            _text = transliterate(text, transliteration, "slp1")
            text_transliterated = self._clean_text(_text)
        else:
            print(f"Incompatible transliteration method: {transliteration}")

        self._set_phoneme()

        self.skeleton = self._convert_to_skeleton(text_transliterated)
        self.meter = self._calculate_meter(self.skeleton)

    def _load_transliteration_methods(self) -> list:
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "transliteration.csv"
        )
        with open(csv_path, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            transliteration_list = next(csv_reader)

        return transliteration_list

    def _set_phoneme(self):
        self.short_vowel_list = [
            # short vowel a
            "a",
            # short vowel i
            "i",
            # short vowel u
            "u",
            # short vowel r̥
            "f",
            # short vowel l̥
            "x",
        ]
        self.long_vowel_list = [
            # long vowel ā
            "A",
            # long vowel ī
            "I",
            # long vowel ū
            "U",
            # long vowel r̥̄
            "F",
            # diphthong e
            "e",
            # diphthong o
            "o",
            # diphthong ai
            "E",
            # diphthong au
            "O",
        ]
        self.consonant_list = [
            # velar; voiceless unasprirated k
            "k",
            # velar; voiceless asprirated kh
            "K",
            # velar; voiced unasprirated g
            "g",
            # velar; voiced asprirated gh
            "G",
            # velar; nasal ṅ
            "N",
            # palatal; voiceless unasprirated c
            "c",
            # palatal; voiceed unasprirated j
            "j",
            # palatal; voiced asprirated jh
            "J",
            # palatal; nasal ñ
            "Y",
            # retroflex; voiceless unasprirated ṭ
            "w",
            # retroflex; voiceless asprirated ṭh
            "W",
            # retroflex; voiced unasprirated ḍ
            "q",
            # retroflex; nasal ṇ
            "R",
            # dental; voiceless unasprirated t
            "t",
            # dental; voiceless asprirated th
            "T",
            # dental; voiced unasprirated d
            "d",
            # dental; voiced asprirated dh
            "D",
            # dental; nasal n
            "n",
            # labial; voiceless unasprirated p
            "p",
            # labial; voiceless asprirated ph
            "P",
            # labial; voiced unasprirated b
            "b",
            # labial; voiced asprirated bh
            "B",
            # labial; nasal m
            "m",
            # semivowel y
            "y",
            # semivowel r
            "r",
            # smivowel l
            "l",
            # semivowel v
            "v",
            # sibilant ś
            "S",
            # sibilant ṣ
            "z",
            # sibilant s
            "s",
            # h
            "h",
            # visarga ḥ
            "H",
            # anusvara ṃ
            "M",
        ]
        self.double_consonant_list = [
            # ch
            "C",
            # ḍh
            "Q",
        ]

    def _clean_text(self, text: str):
        text = re.sub(r"[・3@\\\-\+\'&~\*/\`\u0300]", "", text)

        return text

    def _convert_to_skeleton(self, text: str):
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

    def _calculate_meter(self, skeleton: str) -> str:
        return ""


if __name__ == "__main__":
    vm = VedicMeter("agn;im ii.le purohit;a.m", "tf")
    print(vm.skeleton)
