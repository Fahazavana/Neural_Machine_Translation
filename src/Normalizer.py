import re


def process_line(line):
    line = re.sub(r"(\.$|!$|\?$)", r" \1", line)
    line = re.sub(r"(\.$|!$|\?$)", r"<eos>", line)
    line = re.sub(r"('[a-zA-Z]{1})\b", r" \1", line)
    line = re.sub(r"(:|,|``|'')", r" \1 ", line)

    line = re.sub(r"\s{2,}", " ", line)

    pattern = r"\$([^$]+)\$"
    new_text = ""
    last_end = 0

    for match in re.finditer(pattern, line):
        start, end = match.span()
        new_text += line[last_end:start] + " $" + match.group(1).replace(" ", "") + "$ "
        last_end = end

    new_text += line[last_end:]
    text = new_text
    text = re.sub(r"\s{2,}", " ", text)
    text = text.lower()
    line = re.sub(r"\s{2,}", " ", line)  # ???? But it works
    text = text.strip().split()
    if text[-1] != "<eos>":
        text.append("<eos>")
    return ["<sos>"] + text


def preprocess_data(train_raw_dir, train_data_dir, language_files, language_name):
    with open(f"{train_data_dir}/{language_name}.txt", "w+") as output_file:
        for file_name in language_files:
            with open(f"{train_raw_dir}/{file_name}") as input_file:
                for line in input_file:
                    line = line.strip()
                    if len(line) > 0 and line[0] != "%":
                        processed_line = process_line(line)
                        output_file.write(" ".join(processed_line) + "\n")
    print(f"Done for {language_name}!")
