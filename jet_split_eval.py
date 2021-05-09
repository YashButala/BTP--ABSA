f1 = open('jet_m4_15res_result.txt')
f2 = open('test.txt','r')
def fun(tups):
	a = []
	b = ''
	fl = 0
	for i in range(len(tups)):
		if tups[i]=='(':
			fl = 1
		if fl:
			b += tups[i]
		if tups[i]==')':
			a.append(b)
			b = ''
			fl = 0

	return a	
total_entity = 0	
total_entity_s = 0
total_entity_m = 0
total_entity_c = 0
total_entity_o = 0

total_predict = 0
total_predict_s = 0
total_predict_m = 0
total_predict_c = 0
total_predict_o = 0

pp = 0
pp_s = 0
pp_m = 0
pp_c = 0
pp_o = 0

for line in f1:
	sent = f2.readline()
	gold = f1.readline()
	op = f1.readline()
	tmp1 = gold.find(']')
	tmp2 = op.find(']')
	gold = gold[tmp1+4:-3]
	op = op[tmp2+4:-3]
	f1.readline()
	# print(gold)
	# print(op)
	tups = sent.split('####')[3][2:-3]
	ptrs = tups.split('), (')
	rel = set()
	overlap = set()
	oflag = 0
	for ptr in ptrs:
		asp = ptr.split('],')[0][1:]
		opi = ptr.split('],')[1][2:]
		emo = ptr.split('],')[2][2:-1]
		
		s1 = int(asp.split(',')[0])
		e1 = int(asp.split(',')[-1])
		s2 = int(opi.split(',')[0])
		e2 = int(opi.split(',')[-1])
		# print(ptr)
		# print(str(s1),str(e1),str(s2),str(e2),emo)

		rel.add(emo)
		if (s1,e1) in overlap:
			oflag = 1
		if (s2,e2) in overlap:
			oflag = 1
		overlap.add((s1,e1))
		overlap.add((s2,e2))
	

	gt = fun(gold)
	ot = fun(op)
	print(gt)
	print(ot)
	total_entity += len(gt)
	total_predict += len(ot)
	for oti in ot:
		for gti in gt:
			if oti==gti:
				pp += 1
	if len(ptrs)>1:
		total_entity_m += len(gt)
		total_predict_m += len(ot)
		for oti in ot:
			for gti in gt:
				if oti==gti:
					pp_m += 1	

	else:
	 	total_entity_s += len(gt)
	 	total_predict_s += len(ot)
	 	for oti in ot:
	 		for gti in gt:
	 			if oti==gti:
	 				pp_s += 1

	if(len(rel))>1:
		total_entity_c += len(gt)
		total_predict_c += len(ot)
		for oti in ot:
			for gti in gt:
				if oti==gti:
					pp_c += 1

	if oflag:
		total_entity_o += len(gt)
		total_predict_o += len(ot)
		for oti in ot:
			for gti in gt:
				if oti==gti:
					pp_o += 1
	
precision = pp_s * 1.0 / total_predict_s  if total_predict_s != 0 else 0
recall = pp_s * 1.0 / total_entity_s  if total_entity_s != 0 else 0
fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

print('Single: ')
print(precision, recall, fscore)

precision = pp_m * 1.0 / total_predict_m  if total_predict_m != 0 else 0
recall = pp_m * 1.0 / total_entity_m  if total_entity_m != 0 else 0
fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

print('Mult: ')
print(precision, recall, fscore)

precision = pp_c * 1.0 / total_predict_c  if total_predict_c != 0 else 0
recall = pp_c * 1.0 / total_entity_c  if total_entity_c != 0 else 0
fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

print('Contrasts: ')
print(precision, recall, fscore)

precision = pp_o * 1.0 / total_predict_o  if total_predict_o != 0 else 0
recall = pp_o * 1.0 / total_entity_o  if total_entity_o != 0 else 0
fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

print('Overlap: ')
print(precision, recall, fscore)


precision = pp * 1.0 / total_predict  if total_predict != 0 else 0
recall = pp * 1.0 / total_entity  if total_entity != 0 else 0
fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

print('Combined: ')
print(precision, recall, fscore)

