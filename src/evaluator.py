# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
from collections import OrderedDict
from logging import getLogger
import numpy as np
import torch
from torch import nn

from .utils import restore_segmentation


logger = getLogger()


TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
BLEU_SCRIPT_PATH = os.path.join(TOOLS_PATH, 'mosesdecoder/scripts/generic/multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH), "Moses not found. Please be sure you downloaded Moses in %s" % TOOLS_PATH


class EvaluatorMT(object):

	def __init__(self, trainer, data, params):
		"""
		Initialize evaluator.
		"""
		self.encoder = trainer.encoder
		self.decoder = trainer.decoder
		self.discriminator = trainer.discriminator
		self.data = data
		self.dico = data['dico']
		self.params = params

		# create reference files for BLEU evaluation
		self.create_reference_files()
		print (self.data)

	def get_pair_for_mono(self, lang):
		"""
		Find a language pair for monolingual data.
		"""
		candidates = [(l1, l2) for (l1, l2) in self.data['para'].keys() if l1 == lang or l2 == lang]
		assert len(candidates) > 0
		return sorted(candidates)[0]

	def mono_iterator(self, data_type, lang):
		"""
		If we do not have monolingual validation / test sets, we take one from parallel data.
		"""
		dataset = self.data['mono'][lang][data_type]
		if dataset is None:
			pair = self.get_pair_for_mono(lang)
			dataset = self.data['para'][pair][data_type]
			i = 0 if pair[0] == lang else 1
		else:
			i = None
		dataset.batch_size = self.params.batch_size
		for batch in dataset.get_iterator(shuffle=False, group_by_size=False)():
			yield batch if i is None else batch[i]

	def get_iterator(self, data_type, lang1, lang2):
		"""
		Create a new iterator for a dataset.
		"""
# 		assert data_type in ['valid', 'test'] #Ishan
		if lang2 is None or lang1 == lang2:
			for batch in self.mono_iterator(data_type, lang1):
				yield batch if lang2 is None else (batch, batch)
		else:
			k = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
			dataset = self.data['para'][k][data_type]
			dataset.batch_size = self.params.batch_size
			for batch in dataset.get_iterator(shuffle=False, group_by_size=False)():
				yield batch if lang1 < lang2 else batch[::-1]

	def create_reference_files(self):
		"""
		Create reference files for BLEU evaluation.
		"""
		params = self.params
		params.ref_paths = {}

		for (lang1, lang2), v in self.data['para'].items():

			assert lang1 < lang2
			lang1_id = params.lang2id[lang1]
			lang2_id = params.lang2id[lang2]

			for data_type in ['valid', 'test']:

				lang1_path = os.path.join(params.dump_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_type))
				lang2_path = os.path.join(params.dump_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type))

				lang1_txt = []
				lang2_txt = []

				# convert to text
				for (sent1, len1), (sent2, len2) in self.get_iterator(data_type, lang1, lang2):
					lang1_txt.extend(convert_to_text(sent1, len1, self.dico[lang1], lang1_id, params))
					lang2_txt.extend(convert_to_text(sent2, len2, self.dico[lang2], lang2_id, params))

				# replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
				lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
				lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

				# export hypothesis
				with open(lang1_path, 'w', encoding='utf-8') as f:
					f.write('\n'.join(lang1_txt) + '\n')
				with open(lang2_path, 'w', encoding='utf-8') as f:
					f.write('\n'.join(lang2_txt) + '\n')

				# restore original segmentation
				restore_segmentation(lang1_path)
				restore_segmentation(lang2_path)

				# store data paths
				params.ref_paths[(lang2, lang1, data_type)] = lang1_path
				params.ref_paths[(lang1, lang2, data_type)] = lang2_path

	def custom_discriminator_step(self, data_type):
		"""
		Train the discriminator on the latent space.
		"""
		self.encoder.eval()
		self.decoder.eval()
		self.discriminator.eval()

		
		real_correct = 0
		real_total = 0
		fake_correct = 0
		fake_total = 0

		lang1 = 'en'
		for batch in self.get_iterator(data_type, lang1, None):
			encoded = []
			(sent1, len1) = batch
			sent1 = sent1.cuda()
			encoded.append(self.encoder(sent1.cuda(), len1, 0))

		# discriminator
			dis_inputs = [x.dis_input.view(-1, x.dis_input.size(-1)) for x in encoded]
			ntokens = [dis_input.size(0) for dis_input in dis_inputs]
			encoded = torch.cat(dis_inputs, 0)
			predictions = self.discriminator(encoded.data)
			fake_correct += torch.sum(predictions[:,0] > predictions[:,1])
			fake_total += predictions.shape[0]
			
		print ('fake',fake_correct,'/',fake_total)
		
		lang1 = 'hi'
		for batch in self.get_iterator(data_type, lang1, None):
			encoded = []
			(sent1, len1) = batch
			sent1 = sent1.cuda()
			encoded.append(self.encoder(sent1.cuda(), len1, 0))

		# discriminator
			dis_inputs = [x.dis_input.view(-1, x.dis_input.size(-1)) for x in encoded]
			ntokens = [dis_input.size(0) for dis_input in dis_inputs]
			encoded = torch.cat(dis_inputs, 0)
			predictions = self.discriminator(encoded.data)
			real_correct += torch.sum(predictions[:,0] < predictions[:,1])
			real_total += predictions.shape[0]
			
		print ('real',real_correct,'/',real_total)
		# loss
# 		self.dis_target = torch.cat([torch.zeros(sz).fill_(i) for i, sz in enumerate(ntokens)]) # hi should be given to real
# 		self.dis_target = self.dis_target.contiguous().long().cuda()
# 		y = self.dis_target

# 		loss = F.cross_entropy(predictions, y)
# 		self.stats['dis_costs'].append(loss.item())

# 		# optimizer
# 		self.zero_grad('dis')
# 		loss.backward()
# 		self.update_params('dis')
# 		clip_parameters(self.discriminator, self.params.dis_clip)
# 		return loss.item()
	
	def eval_para(self, lang1, lang2, data_type, scores):
		"""
		Evaluate lang1 -> lang2 perplexity and BLEU scores.
		"""
		logger.info("Evaluating %s -> %s (%s) ..." % (lang1, lang2, data_type))
		assert data_type in ['valid', 'test']
		self.encoder.eval()
		self.decoder.eval()
		params = self.params
		lang1_id = params.lang2id[lang1]
		lang2_id = params.lang2id[lang2]

		# hypothesis
		txt = []

		# for perplexity
		loss_fn2 = nn.CrossEntropyLoss(weight=self.decoder.loss_fn[lang2_id].weight, size_average=False)
		n_words2 = self.params.n_words[lang2_id]
		count = 0
		xe_loss = 0

		for batch in self.get_iterator(data_type, lang1, lang2):

			# batch
			(sent1, len1), (sent2, len2) = batch
			sent1, sent2 = sent1.cuda(), sent2.cuda()

			# encode / decode / generate
			encoded = self.encoder(sent1, len1, lang1_id)
			decoded = self.decoder(encoded, sent2[:-1], lang2_id)
			sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

			# cross-entropy loss
			xe_loss += loss_fn2(decoded.view(-1, n_words2), sent2[1:].view(-1)).item()
			count += (len2 - 1).sum().item()  # skip BOS word

			# convert to text
			txt.extend(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params))

		# hypothesis / reference paths
		hyp_name = 'hyp.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type)
		hyp_path = os.path.join(params.dump_path, hyp_name)
		ref_path = params.ref_paths[(lang1, lang2, data_type)]

		# export sentences to hypothesis file / restore BPE segmentation
		with open(hyp_path, 'w', encoding='utf-8') as f:
			f.write('\n'.join(txt) + '\n')
		restore_segmentation(hyp_path)

		# evaluate BLEU score
		bleu = eval_moses_bleu(ref_path, hyp_path)
		logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))

		# update scores
		scores['ppl_%s_%s_%s' % (lang1, lang2, data_type)] = np.exp(xe_loss / count)
		scores['bleu_%s_%s_%s' % (lang1, lang2, data_type)] = bleu
	
	def eval_gen(self, lang1, lang2, data_type, scores):
		"""
		Evaluate lang1 -> lang2 perplexity and BLEU scores.
		"""
		logger.info("Evaluating %s -> %s (%s) ..." % (lang1, lang2, data_type))
		self.encoder.eval()
		self.decoder.eval()
		params = self.params
		lang1_id = params.lang2id[lang1]
		lang2_id = params.lang2id[lang2]

		# hypothesis
		txt0 = []
		txt1 = []
		txt2 = []
		txt3 = []
		txt4 = []

		for batch in self.get_iterator(data_type, lang1, None):

			# batch
			(sent1, len1) = batch
			sent1 = sent1.cuda()

			# encode / decode / generate
			encoded = self.encoder(sent1, len1, lang1_id)

			#sent2_0, len2_0, sent2_1, len2_1, sent2_2, len2_2, sent2_3, len2_3, sent2_4, len2_4, _ = self.decoder.generate(encoded, lang2_id)
			sent2_0, len2_0 = self.decoder.generate(encoded, lang2_id)
			# convert to text
			txt0.extend(convert_to_text(sent2_0, len2_0, self.dico[lang2], lang2_id, self.params))
			#txt1.extend(convert_to_text(sent2_1, len2_1, self.dico[lang2], lang2_id, self.params))
			#txt2.extend(convert_to_text(sent2_2, len2_2, self.dico[lang2], lang2_id, self.params))
			#txt3.extend(convert_to_text(sent2_3, len2_3, self.dico[lang2], lang2_id, self.params))
			#txt4.extend(convert_to_text(sent2_4, len2_4, self.dico[lang2], lang2_id, self.params))

		hyp_name0 = 'topk0.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type)
		hyp_path0 = os.path.join(params.dump_path, hyp_name0)
		'''
		hyp_name1 = 'topk1.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type)
		hyp_path1 = os.path.join(params.dump_path, hyp_name1)
		
		hyp_name2 = 'topk2.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type)
		hyp_path2 = os.path.join(params.dump_path, hyp_name2)
		
		hyp_name3 = 'topk3.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type)
		hyp_path3 = os.path.join(params.dump_path, hyp_name3)
		
		hyp_name4 = 'topk4.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type)
		hyp_path4 = os.path.join(params.dump_path, hyp_name4)
# 		ref_path = params.ref_paths[(lang1, lang2, data_type)]
		'''
		# export sentences to hypothesis file / restore BPE segmentation
		with open(hyp_path0, 'w', encoding='utf-8') as f:
			f.write('\n'.join(txt0) + '\n')
		restore_segmentation(hyp_path0)
		'''
		with open(hyp_path1, 'w', encoding='utf-8') as f:
			f.write('\n'.join(txt1) + '\n')
		restore_segmentation(hyp_path1)
		
		with open(hyp_path2, 'w', encoding='utf-8') as f:
			f.write('\n'.join(txt2) + '\n')
		restore_segmentation(hyp_path2)
		
		with open(hyp_path3, 'w', encoding='utf-8') as f:
			f.write('\n'.join(txt3) + '\n')
		restore_segmentation(hyp_path3)
		
		with open(hyp_path4, 'w', encoding='utf-8') as f:
			f.write('\n'.join(txt4) + '\n')
		restore_segmentation(hyp_path4)
		'''
		# evaluate BLEU score
# 		bleu = eval_moses_bleu(ref_path, hyp_path)
# 		logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))

		# update scores
# 		scores['ppl_%s_%s_%s' % (lang1, lang2, data_type)] = np.exp(xe_loss / count)
# 		scores['bleu_%s_%s_%s' % (lang1, lang2, data_type)] = bleu

	def eval_back(self, lang1, lang2, lang3, data_type, scores):
		"""
		Compute lang1 -> lang2 -> lang3 perplexity and BLEU scores.
		"""
		logger.info("Evaluating %s -> %s -> %s (%s) ..." % (lang1, lang2, lang3, data_type))
		assert data_type in ['valid', 'test']
		self.encoder.eval()
		self.decoder.eval()
		params = self.params
		lang1_id = params.lang2id[lang1]
		lang2_id = params.lang2id[lang2]
		lang3_id = params.lang2id[lang3]

		# hypothesis
		txt = []

		# for perplexity
		loss_fn3 = nn.CrossEntropyLoss(weight=self.decoder.loss_fn[lang3_id].weight, size_average=False)
		n_words3 = self.params.n_words[lang3_id]
		count = 0
		xe_loss = 0

		for batch in self.get_iterator(data_type, lang1, lang3):

			# batch
			(sent1, len1), (sent3, len3) = batch
			sent1, sent3 = sent1.cuda(), sent3.cuda()

			# encode / generate lang1 -> lang2
			encoded = self.encoder(sent1, len1, lang1_id)
			sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

			# encode / decode / generate lang2 -> lang3
			encoded = self.encoder(sent2_.cuda(), len2_, lang2_id)
			decoded = self.decoder(encoded, sent3[:-1], lang3_id)
			sent3_, len3_, _ = self.decoder.generate(encoded, lang3_id)

			# cross-entropy loss
			xe_loss += loss_fn3(decoded.view(-1, n_words3), sent3[1:].view(-1)).item()
			count += (len3 - 1).sum().item()  # skip BOS word

			# convert to text
			txt.extend(convert_to_text(sent3_, len3_, self.dico[lang3], lang3_id, self.params))

		# hypothesis / reference paths
		hyp_name = 'hyp.{0}-{1}-{2}.{3}.txt'.format(lang1, lang2, lang3, data_type)
		hyp_path = os.path.join(params.dump_path, hyp_name)
		if lang1 == lang3:
			_lang1, _lang3 = self.get_pair_for_mono(lang1)
			if lang3 != _lang3:
				_lang1, _lang3 = _lang3, _lang1
			ref_path = params.ref_paths[(_lang1, _lang3, data_type)]
		else:
			ref_path = params.ref_paths[(lang1, lang3, data_type)]

		# export sentences to hypothesis file / restore BPE segmentation
		with open(hyp_path, 'w', encoding='utf-8') as f:
			f.write('\n'.join(txt) + '\n')
		restore_segmentation(hyp_path)

		# evaluate BLEU score
		bleu = eval_moses_bleu(ref_path, hyp_path)
		logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))

		# update scores
		scores['ppl_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type)] = np.exp(xe_loss / count)
		scores['bleu_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type)] = bleu

	def run_all_evals(self, epoch):
		"""
		Run all evaluations.
		"""
		scores = OrderedDict({'epoch': epoch})

		with torch.no_grad():

			for lang1, lang2 in self.data['para'].keys():
				for data_type in ['valid', 'test']:
					self.eval_para(lang1, lang2, data_type, scores)
					self.eval_para(lang2, lang1, data_type, scores)
# 			for lang1, lang2 in self.data['para'].keys():
# 				for data_type in ['valid','test','train']:
# 					self.eval_gen(lang1, lang2, data_type, scores)
# 					self.eval_gen(lang2, lang1, data_type, scores)

			for lang1, lang2, lang3 in self.params.pivo_directions:
				for data_type in ['valid', 'test']:
					self.eval_back(lang1, lang2, lang3, data_type, scores)

		return scores


def eval_moses_bleu(ref, hyp):
	"""
	Given a file of hypothesis and reference files,
	evaluate the BLEU score using Moses scripts.
	"""
	assert os.path.isfile(ref) and os.path.isfile(hyp)
	command = BLEU_SCRIPT_PATH + ' %s < %s'
	p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
	result = p.communicate()[0].decode("utf-8")
	if result.startswith('BLEU'):
		return float(result[7:result.index(',')])
	else:
		logger.warning('Impossible to parse BLEU score! "%s"' % result)
		return -1


def convert_to_text(batch, lengths, dico, lang_id, params):
	"""
	Convert a batch of sentences to a list of text sentences.
	"""
	batch = batch.cpu().numpy()
	lengths = lengths.cpu().numpy()
	bos_index = params.bos_index[lang_id]

	slen, bs = batch.shape
	assert lengths.max() == slen and lengths.shape[0] == bs
	assert (batch[0] == bos_index).sum() == bs
	assert (batch == params.eos_index).sum() == bs
	sentences = []

	for j in range(bs):
		words = []
		for k in range(1, lengths[j]):
			if batch[k, j] == params.eos_index:
				break
			words.append(dico[batch[k, j]])
		sentences.append(" ".join(words))
	return sentences

