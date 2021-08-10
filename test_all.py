
import scripts.test as t2

import os

import warnings
warnings.filterwarnings("ignore")



imgfile_i = './data_2/test'


for root, dirs, files in os.walk('data_2/test'):

    for dir in dirs:

        one_case_root = os.path.join(root, dir)

        t2.test_s2(one_case_root, dir)

