import torch 

class BeamSearchNode:
	"""
	Data structure to contains all the nodes of the beam search
	"""
	def __init__(self, hidden, previous_node, token, log_prob, length):
		self.hidden = hidden
		self.previous_node = previous_node
		self.token = token
		self.log_prob = log_prob
		self.length = length

	def eval(self):
		return self.log_prob / float(self.length)


def greedy_search(model, source, max_len=20):
	end_token = 2
	inputs = source[0]
	sequence = [1]
	max_len = max(source.shape[0]+5, max_len)
	model.eval()
	with torch.no_grad():
		_, hidden = model.encoder(source)
		for _ in range(max_len):
			output, hidden = model.decoder(inputs, hidden)
			top1 = output.argmax(1)
			next_token = top1.item()
			sequence.append(next_token)

			if next_token == end_token:
				break

			inputs = top1

	return sequence


def beam_search(model, source, beam_width=3, max_len=20):

	start_token, end_token = source[0], 2

	# Get the encoder hidden state
	model.eval()
	with torch.no_grad():
		_, hidden = model.encoder(source)

		# Initialize
		nodes = [BeamSearchNode(hidden, None, start_token, 0, 1)]
		end_nodes = []

		# Search until reaching the max_len
		for _ in range(max_len):
			all_candidates = []
			for node in nodes:
				# Stop if end of sentece
				if node.token.item() == end_token:
					end_nodes.append(node)
					continue

				# pedict next word for the current node
				output, hidden = model.decoder(node.token, node.hidden)
				log_probs, indices = output.log_softmax(dim=1).topk(k=beam_width)

				# create new node for each top-k
				for i in range(beam_width):
					token = indices[0, i].unsqueeze(0)
					log_prob = log_probs[0, i].item()
					candidate = BeamSearchNode(
						hidden, node, token, node.log_prob + log_prob, node.length + 1
					)
					all_candidates.append(candidate)

			# Sort all curent candidates, and consider only the top-k
			nodes = sorted(all_candidates, key=lambda x: x.eval(), reverse=True)
			nodes = nodes[:beam_width]

	# If never reaches any end of sentence
	if len(end_nodes) == 0:
		end_nodes = nodes

	# Back-tracking
	best_node = sorted(end_nodes, key=lambda x: x.eval(), reverse=True)
	sequence = []
	node = best_node[0]
	while node is not None:
		sequence.append(node.token.item())
		node = node.previous_node

	sequence = sequence[::-1]
	return sequence


class Translator:
	def __init__(self, model, source_lang, target_lang, device):
		self.model = model
		self.source_lang = source_lang
		self.target_lang = target_lang
		self.device = device

	def translate_sentence(self, sentence, method="greedy", beam_width=3, max_len=20):
		text = [
			(
				self.source_lang.stoi[word]
				if word in self.source_lang.stoi
				else self.source_lang.stoi["<unk>"]
			)
			for word in sentence.strip().split()
		]
		text = torch.tensor(text, dtype=torch.long).unsqueeze(1).to(self.device)

		if method == "greedy":
			translated = greedy_search(self.model, text, max_len)
		elif method == "beam":
			translated = beam_search(self.model, text, beam_width, max_len)
		else:
			raise ValueError("Unknown method: choose between 'greedy' or 'beam'")

		return " ".join([self.target_lang.itos[idx] for idx in translated])
