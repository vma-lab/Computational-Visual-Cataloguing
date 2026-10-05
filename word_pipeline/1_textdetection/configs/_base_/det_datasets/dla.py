dataset_type = 'OCRCLIPDataset'
data_root = 'data/DLA'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/img',
    data_name='dla',
    pipeline=None)

val = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_val.json',
    img_prefix=f'{data_root}/img',
    data_name='dla',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/img',  
    #img_prefix=f'{data_root}',
    data_name='dla',
    pipeline=None)

train_list_author = [train]

val_list = [val]

test_list = [test]

train_list = train_list_author 
