from scripts.data import *
from scripts.model import *

aug_args = dict()

train_gene = trainGenerator(batch_size=2,aug_dict=aug_args,train_path='data_2/',
                        image_folder='MR2D',label_folder='Mask2D',
                        image_color_mode='rgb',label_color_mode='rgb',
                        image_save_prefix='image',label_save_prefix='label',
                        flag_multi_class=True,save_to_dir=None)

model = unet(num_class=20)

model_checkpoint = ModelCheckpoint('saved_models/D2_DEL.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(train_gene,
                              steps_per_epoch=2000,
                              epochs=10,
                              verbose=1,
                              callbacks=[model_checkpoint])

