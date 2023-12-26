from utils.meter_calculator import VedicMeter


def convert_text_to_metronome(text: str, transliteration: str = "iast"):
    # define the weight of vowels
    vowel_weight = {"V": 0, "W": 1}

    # get a skeleton (V, W, C, H, ., |)
    vm = VedicMeter(text, transliteration)
    skeleton = vm.skeleton
    # to calculate the final syllable
    skeleton += "C"

    # index of vowels in the skeleton
    vowel_idx_list = [
        idx for idx, phoneme in enumerate(skeleton) if phoneme in vowel_weight
    ]

    # key: index of vowel, value: mora
    vowel_mora_dict = {}
    for i, vowel_idx in enumerate(vowel_idx_list):
        consonant_counter = 0

        # set the next index
        if i == len(vowel_idx_list) - 1:
            fin = len(skeleton)
        else:
            fin = vowel_idx_list[i + 1]

        # sum up consonants
        for phoneme_idx in range(vowel_idx + 1, fin):
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
            if idx in vowel_mora_dict:
                mora = vowel_mora_dict[idx]
                metronome += "S" if mora > 0 else "w"

    return metronome


if __name__ == "__main__":
    m = convert_text_to_metronome("agn;im ii;le purohit;a.m|", "tf")
    print(m)
