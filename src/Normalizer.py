import re


def process_line(line):
	line = re.sub(r'([.!?])', ' <eos> ', line)
	line = re.sub(r'([{[\]()},$])', r' \1 ', line)
	line = re.sub(r'\s{2,}', ' ', line)
	line = line.lower().strip()

	tokens = line.split()

	if len(tokens) >= 2 and tokens[-2] == "<eos>":
		tokens[-2], tokens[-1] = tokens[-1], tokens[-2]

	latex_mode = False
	latex_stack = ""
	tokenized_line = ['<sos>']

	for token in tokens:
		if latex_mode:
			latex_stack += token
			if token == "$":
				latex_mode = False
				# latex_stack = latex_stack.replace(" ", "")
				latex_stack = '<ltx>'
				tokenized_line.append(latex_stack)
				latex_stack = ""
		elif token == "$":
			latex_mode = True
			latex_stack += token
		else:
			if latex_stack:
				latex_stack = '<ltx>'
				tokenized_line.append(latex_stack)
				latex_stack = ""
			tokenized_line.append(token)

	if latex_stack:
		latex_stack = '<ltx>'
		tokenized_line.append(latex_stack)
	text = " ".join(tokenized_line)
	text = re.sub(r"\\", "  ", text)
	text = re.sub(r"(<eos>)(\w)+", "\1", text)
	text = re.sub(r"(\d)+", " <num> ", text)
	text = re.sub(r"[\(\[\{]", " <opn> ", text)
	text = re.sub(r"[\)\]\}]", " <cld> ", text)
	text = re.sub(r"\\\w+\b", " <ltx> ", text)
	text = re.sub(r"(\\%|%)", "<prc>", text)
	text = re.sub(r",", "<com>", text)
	text = re.sub(r"'", "<apo>", text)
	text = re.sub(r"<apo><apo>", " ", text)
	text = re.sub(r"/", " ", text)
	text = re.sub(r"-", " ", text)
	text = re.sub(r"[^\w<>]", " ", text)
	text = re.sub(r"\s{2,}", " ", text)
	text = text.split()
	if text[-1] != "<eos>":
		text.append("<eos>")
	return text


def preprocess_data(train_raw_dir, train_data_dir, language_files, language_name):
	with open(f"{train_data_dir}/{language_name}.txt", 'w+') as output_file:
		for file_name in language_files:
			with open(f"{train_raw_dir}/{file_name}") as input_file:
				for line in input_file:
					line = line.strip()
					if len(line) > 0 and line[0] != "%":
						processed_line = process_line(line)
						output_file.write(" ".join(processed_line) + "\n")
	print(f"Done for {language_name}!")
