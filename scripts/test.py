from scripts.data import *
from scripts.model import *

def test_s2(case_path,case_name):
    testGene = testGenerator(case_path)
    lens = getFileNum(case_path)
    model = unet(num_class=20)
    model.load_weights("saved_models/D2_DEL.hdf5")
    results = model.predict_generator(testGene,lens,max_queue_size=1,verbose=1)
    if not(os.path.exists('results_1/D2_DEL/'+case_name)):
        os.makedirs('results_1/D2_DEL/'+case_name)
    saveResult('results_1/D2_DEL/'+case_name,results)
