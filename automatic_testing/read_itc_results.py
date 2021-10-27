results_path = "res.save"
packet_size = 8

cnn_idx = 1
pers_idx = 2
id_idx = 3
kshot_idx = 6

total_kshot_vals = 3
total_ids = 10
train_cnn_options = 2

res = []
with open(results_path, 'r') as in_stream:
    for line in in_stream:
        res.append(line)

# res = "".join(res)

train_new_cnn = True

# Index target player
## index which value of k we used
### index whether or not we trained a new cnn (0 for false, 1 for true)
#### index which result we want (0 for cnn_acc's, 1 for pers_acc's)
##### index individual accuracy results
full_result_table = [[[[[], []] for k in range(train_cnn_options)] for j in range(total_kshot_vals)] for i in
                     range(total_ids)]

kshot_table = {1: 0, 5: 1, 10: 2}

for i in range(len(res) // packet_size):
    idx = i * packet_size

    # alternates every 5 packets
    if (i + 1) % 5 == 0:
        train_new_cnn = not train_new_cnn

    cnn_acc = float(res[idx + cnn_idx].split(" = ")[1])
    pers_acc = float(res[idx + pers_idx].split(" = ")[1])
    p_id = int(res[idx + id_idx].split(" = ")[1]) - 1
    k = kshot_table[int(res[idx + kshot_idx].split(" = ")[1])]

    if cnn_acc > 1.0:
        cnn_acc /= 100.0

    if pers_acc > 1.0:
        pers_acc /= 100.0

    full_result_table[p_id][k][int(train_new_cnn)][0].append(cnn_acc)
    full_result_table[p_id][k][int(train_new_cnn)][1].append(pers_acc)

stat_results = []

kshot_table_rev = {0: 1, 1: 5, 2: 10}


import numpy as np


total_cnn_var = []
total_pers_var = []

for p_id in range(total_ids):
    stat_results.append("target id = {a}\n\n".format(a=p_id + 1))
    for k in range(total_kshot_vals):
        stat_results.append("  k-shot = {a}\n\n".format(a=kshot_table_rev[k]))
        for train_new_cnn_idx in range(train_cnn_options):
            stat_results.append("    trained a new cnn = {a}\n\n".format(a=bool(train_new_cnn_idx)))

            cnn_result_vec = full_result_table[p_id][k][train_new_cnn_idx][0]
            pers_result_vec = full_result_table[p_id][k][train_new_cnn_idx][1]

            stat_results.append("      CNN results:\n")
            stat_results.append("      min = {a}, max = {b}\n".format(a=np.min(cnn_result_vec),
                                                                      b=np.max(cnn_result_vec)))
            stat_results.append("      mean = {a}, var = {b}\n\n".format(a=np.mean(cnn_result_vec),
                                                                         b=np.var(cnn_result_vec)))

            stat_results.append("      Personalization results:\n")
            stat_results.append("      min = {a}, max = {b}\n".format(a=np.min(pers_result_vec),
                                                                      b=np.max(pers_result_vec)))
            stat_results.append("      mean = {a}, var = {b}\n\n".format(a=np.mean(pers_result_vec),
                                                                         b=np.var(pers_result_vec)))

            total_cnn_var.append(np.var(cnn_result_vec))
            total_pers_var.append(np.var(pers_result_vec))

stat_results = "".join(stat_results)

mean_cnn_var = np.mean(total_cnn_var)
mean_pers_var = np.mean(total_pers_var)

l_or_h = "lower"
if mean_cnn_var - mean_pers_var < 0:
    l_or_h = "higher"

var_result_str = "Overall, the variance of the personalization method is ..."


print(stat_results)

with open("stats.results", 'w') as stream:
    stream.write(stat_results)
