def is_full_match(triplet, triplets):
	for t in triplets:
		if t[0] == triplet[0] and t[1] == triplet[1] and t[2] == triplet[2]:
			return True
	return False

gt_ap = 0
predicted_ap = 0
correct_ap = 0
gt_ap_list = []
predicted_ap_list = []
correct_ap_list = []

gt_op = 0
predicted_op = 0
correct_op = 0
gt_op_list = []
predicted_op_list = []
correct_op_list = []

with open('test.out', 'r') as f_in:
    lines = f_in.readlines()
lineCount = len(lines)
start = 1
while start < lineCount:
    desired_lines = lines[start:start+2]
    exp = desired_lines[0].strip()[9:].strip()
    gt_triplets = exp.split('|')
    exp_ap = []
    exp_op = []
    for triplet in gt_triplets:
        exp_ap.append(triplet.split(';')[0].strip())
        exp_op.append(triplet.split(';')[1].strip())
    pred = desired_lines[1].strip()[10:].strip()
    pred_triplets = pred.split('|')
    pred_ap = []
    pred_op = []    
    for triplet in pred_triplets:
        pred_ap.append(triplet.split(';')[0].strip())
        pred_op.append(triplet.split(';')[1].strip())
    
    exp_ap = set(exp_ap)
    pred_ap = set(pred_ap)
    gt_ap += len(exp_ap)
    predicted_ap += len(pred_ap)
    correct_ap += len(exp_ap.intersection(pred_ap))

    gt_ap_list.append(len(exp_ap))
    predicted_ap_list.append(len(pred_ap))
    correct_ap_list.append(len(exp_ap.intersection(pred_ap)))

    exp_op = set(exp_op)
    pred_op = set(pred_op)
    gt_op += len(exp_op)
    predicted_op += len(pred_op)
    correct_op += len(exp_op.intersection(pred_op))

    gt_op_list.append(len(exp_op))
    predicted_op_list.append(len(pred_op))
    correct_op_list.append(len(exp_op.intersection(pred_op)))

    start += 4

p_ap = float(correct_ap) / (predicted_ap + 1e-8)
r_ap = float(correct_ap) / (gt_ap + 1e-8)
f1_ap = (2 * p_ap * r_ap) / (p_ap + r_ap + 1e-8)
print('Aspect Prediction:')
print(f'Precision: {round(p_ap,3)}')
print(f'Recall: {round(r_ap,3)}')
print(f'F1: {round(f1_ap,3)}')
p_ap = float(sum(correct_ap_list)) / (sum(predicted_ap_list) + 1e-8)
r_ap = float(sum(correct_ap_list)) / (sum(gt_ap_list) + 1e-8)
f1_ap = (2 * p_ap * r_ap) / (p_ap + r_ap + 1e-8)
print('After rechecking:')
print(f'Precision: {round(p_ap,3)}')
print(f'Recall: {round(r_ap,3)}')
print(f'F1: {round(f1_ap,3)}')
print("\n")

p_op = float(correct_op) / (predicted_op + 1e-8)
r_op = float(correct_op) / (gt_op + 1e-8)
f1_op = (2 * p_op * r_op) / (p_op + r_op + 1e-8)
print('Opinion Prediction:')
print(f'Precision: {round(p_op,3)}')
print(f'Recall: {round(r_op,3)}')
print(f'F1: {round(f1_op,3)}')
p_op = float(sum(correct_op_list)) / (sum(predicted_op_list) + 1e-8)
r_op = float(sum(correct_op_list)) / (sum(gt_op_list) + 1e-8)
f1_op = (2 * p_ap * r_ap) / (p_ap + r_ap + 1e-8)
print('After rechecking:')
print(f'Precision: {round(p_op,3)}')
print(f'Recall: {round(r_op,3)}')
print(f'F1: {round(f1_op,3)}')
print("\n")

print(exp_ap)
print(pred_ap)
print(gt_ap_list[-1], predicted_ap_list[-1], correct_ap_list[-1])
print(exp_op)
print(pred_op)
print(gt_op_list[-1], predicted_op_list[-1], correct_op_list[-1])