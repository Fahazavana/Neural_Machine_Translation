
from .Translator import Translator,  BeamSearchNode
import torch 


class BeamSearchNodeAtt(BeamSearchNode):
	"""
	Data structure to contains all the nodes of the beam search
	"""
	def __init__(self, hidden, encoder_out, previous_node, token, log_prob, length):
		super(BeamSearchNodeAtt, self).__init__(
			hidden, previous_node, token, log_prob, length
		)
		self.encoder_out = encoder_out


def beam_search_att(model, source, beam_width=3, max_len=20, lstm=False):
	
	start_token, end_token = source[0], 2
	model.eval()
	with torch.no_grad():
		# Get the compresed version
		encoder_out, hidden = model.encoder(source)
		hidden = (torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1])) if lstm else torch.zeros_like(hidden)
		# Initialize
		nodes = [BeamSearchNodeAtt(hidden, encoder_out, None, start_token, 0, 1)]
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
				output, hidden = model.decoder(node.token, hidden, node.encoder_out)
				log_probs, indices = output.log_softmax(dim=1).topk(k=beam_width)

				# create new node for each top-k
				for i in range(beam_width):
					token = indices[0, i].unsqueeze(0)
					log_prob = log_probs[0, i].item()
					candidate = BeamSearchNodeAtt(
						hidden,
						encoder_out,
						node,
						token,
						node.log_prob + log_prob,
						node.length + 1,
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


def greedy_search_att(model, source, max_len=20, lstm=False):
	end_token = 2
	inputs = source[0]
	sequence = [1]
	model.eval()
	with torch.no_grad():
		encoder_out, hidden = model.encoder(source)
		hidden = (torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1])) if lstm else torch.zeros_like(hidden)
		for _ in range(max_len):
			output, hidden = model.decoder(inputs, hidden, encoder_out)
			top1 = output.argmax(1)
			next_token = top1.item()
			sequence.append(next_token)

			if next_token == end_token:
				break

			inputs = top1

	return sequence


class TranslatorAtt(Translator):
	def __init__(self, model, source_lang, target_lang, device, lstm=False):
		super(TranslatorAtt, self).__init__(model, source_lang, target_lang, device)
		self.lstm = lstm
		
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
		max_len = max(text.shape[0], 20)
		if method == "greedy":
			translated = greedy_search_att(self.model, text, max_len, self.lstm)
		elif method == "beam":
			translated = beam_search_att(self.model, text, beam_width, max_len, self.lstm)
		else:
			raise ValueError("Unknown method: choose between 'greedy' or 'beam'")

		return " ".join([self.target_lang.itos[idx] for idx in translated])
