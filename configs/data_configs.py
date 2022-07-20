from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test']
	},
	"cars_encode_prev": {
		'transforms': transforms_config.CarsEncodeTransforms,
		'train_source_root': dataset_paths['cars_train'],
		'train_target_root': dataset_paths['cars_train'],
		'test_source_root': dataset_paths['cars_test'],
		'test_target_root': dataset_paths['cars_test']
	},
	"church_encode": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['church_train'],
		'train_target_root': dataset_paths['church_train'],
		'test_source_root': dataset_paths['church_test'],
		'test_target_root': dataset_paths['church_test']
	},
	"horse_encode": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['horse_train'],
		'train_target_root': dataset_paths['horse_train'],
		'test_source_root': dataset_paths['horse_test'],
		'test_target_root': dataset_paths['horse_test']
	},
	"afhq_wild_encode": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['afhq_wild_train'],
		'train_target_root': dataset_paths['afhq_wild_train'],
		'test_source_root': dataset_paths['afhq_wild_test'],
		'test_target_root': dataset_paths['afhq_wild_test']
	},
	"toonify": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test']
	},
    'paint_ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['paint_train_source_ffhq'],
		'train_target_root': dataset_paths['paint_train_target_ffhq'],
		'test_source_root': dataset_paths['paint_test_source_ffhq'],
		'test_target_root': dataset_paths['paint_test_target_ffhq'],
	},
    'paint_ffhq_encode2': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['paint_train_source_ffhq2'],
		'train_target_root': dataset_paths['paint_train_target_ffhq2'],
		'test_source_root': dataset_paths['paint_test_source_ffhq2'],
		'test_target_root': dataset_paths['paint_test_target_ffhq2'],
	},
	'paint_ffhq_encode_id': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['paint_train_target_ffhq2'],
		'train_target_root': dataset_paths['paint_train_source_ffhq2'],
		'test_source_root': dataset_paths['paint_test_target_ffhq2'],
		'test_target_root': dataset_paths['paint_test_source_ffhq2'],
	},
    'cars_encode': {
		'transforms': transforms_config.CarsEncodeTransforms,
		'train_source_root': dataset_paths['paint_train_source'],
		'train_target_root': dataset_paths['paint_train_target'],
		'test_source_root': dataset_paths['paint_test_source'],
		'test_target_root': dataset_paths['paint_test_target'],
	},
}