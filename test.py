obs_seq = 'AGACTGCATATATAAGGGGCAGGCTG'
st_seq =  '00000000111111100000000000'

import mchmm as mc
a = mc.HiddenMarkovModel()
a = a.from_seq(obs_seq, st_seq)
print(a.states)
print(a.observations)
print(a.ep)
import pandas as pd
print(pd.DataFrame(a.ep, index=a.states, columns=a.observations))
x, y = a.viterbi('GGCATTGGGCTATAAGAGGAGCTTG')
x2, y2 = a.baum_welch('GGCATTGGGCTATAAGAGGAGCTTG', iters=5)

print('BW', "".join(x2))
print('RE', 'GGCATTGGGCTATAAGAGGAGCTTG')
print('VI', "".join(x))

#AGACTGCATATATAAGGGGCAGGCTG
#00000000111111100000000000

#TGGCATTGGGCTATAAGAGGAGCTTG
