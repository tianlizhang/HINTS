import matplotlib.pyplot as plt
import numpy as np

result = [0.86935334, 1.14519809, 0.70537772, 0.89355895, 0.85935305, 1.10467529, \
     0.91182327, 1.19243881, 0.9332086,  1.23694128, 0.93700404, 1.25805918]
result_gcn = [0.81997577, 1.07546867, 0.68153265, 0.86136241, 0.83108572, 1.06564531,\
 0.86677978, 1.13007487, 0.86807977, 1.14746493, 0.85240092, 1.14330494]

result_s = [0.74475747, 1.02531743, 0.58652932, 0.77527883, 0.71336954, 0.97419123,\
 0.78575819, 1.06030098, 0.76652738, 1.07684674, 0.87160291, 1.18108764]



xx = [ [] for _ in range(5)]
# xx[0] = [result[0], result[2], result[4], result[6], result[8], result[10]]
# xx[1] = [result_gcn[0], result_gcn[2], result_gcn[4], result_gcn[6], result_gcn[8], result_gcn[10]]
# xx[2] = [result_s[0], result_s[2], result_s[4], result_s[6], result_s[8], result_s[10]]

xx[0] = [result[1], result[3], result[5], result[7], result[9], result[11]]
xx[1] = [result_gcn[1], result_gcn[3], result_gcn[5], result_gcn[7], result_gcn[9], result_gcn[11]]
xx[2] = [result_s[1], result_s[3], result_s[5], result_s[7], result_s[9], result_s[11]]

plt.plot(np.arange(6), xx[0])
plt.plot(np.arange(6), xx[1])
plt.plot(np.arange(6), xx[2])
plt.legend(['HINTS', 'HINTS-GCN', 'HINTS-Seq2Seq'])
plt.xlabel('year')
plt.ylabel('Err')
plt.title('DBLP-V11 RMSLE')
plt.savefig('DBLP-V11 RMSLE.jpg')



# result = [0.50467382, 0.68983492, 0.48112889, 0.61592657, 0.53236706, 0.69329664,\
#  0.52784028, 0.71486322, 0.50091388, 0.70958402, 0.48111897, 0.71045112]


# result_gcn = [0.49996643, 0.70286253, 0.50448571, 0.6283476,  0.5177614,  0.70847293,\
#  0.51396504, 0.72890453, 0.49068978, 0.7222613,  0.4729302,  0.72128525]

# result_s =  [0.47950536, 0.72460657, 0.44748667, 0.64386666, 0.50847387, 0.72579159,\
#  0.50180064, 0.75636491, 0.47886538, 0.74677122, 0.46090024, 0.74417419]

# xx = [ [] for _ in range(6)]
# xx[0] = [result[0], result[2], result[4], result[6], result[8], result[10]]
# xx[1] = [result_gcn[0], result_gcn[2], result_gcn[4], result_gcn[6], result_gcn[8], result_gcn[10]]
# xx[2] = [result_s[0], result_s[2], result_s[4], result_s[6], result_s[8], result_s[10]]

# # xx[0] = [result[1], result[3], result[5], result[7], result[9], result[11]]
# # xx[1] = [result_gcn[1], result_gcn[3], result_gcn[5], result_gcn[7], result_gcn[9], result_gcn[11]]
# # xx[2] = [result_s[1], result_s[3], result_s[5], result_s[7], result_s[9], result_s[11]]

# plt.plot(np.arange(6), xx[0])
# plt.plot(np.arange(6), xx[1])
# plt.plot(np.arange(6), xx[2])
# plt.legend(['HINTS', 'HINTS-GCN', 'HINTS-Seq2Seq'])
# plt.xlabel('year')
# plt.ylabel('Err')
# plt.title('DBLP-V13 MALE')
# plt.savefig('DBLP-V13_MALE.jpg')