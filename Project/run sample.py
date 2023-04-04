import tradition
import deep
import os
import pandas as pd
import util
input = 'sample.mp4'

[dn1, dn_t] = tradition.tradition_stabilization(input, "Sample/T")
# [dn2, dn_p] = deep.deep_stabilization(
#     input, "Sample/P1", iterate_num=10000, lrate=0.02)

[dn2, dn_p] = deep.deep_stabilization(
    input, "Sample/P1", iterate_num=10000, lrate=0.025)
# [dn2, dn_p] = deep.deep_stabilization(input, "Sample/P2")

print(dn1, dn2)
print(dn_t, dn_p)
