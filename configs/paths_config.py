dataset_paths = {
	'celeba': '',
	'celeba_test': '',

	'cars_train': '',
	'cars_test': '',

	'church_train': '',
	'church_test': '',

	'horse_train': '',
	'horse_test': '',

	'afhq_wild_train': '',
	'afhq_wild_test': '',
    
    # Painting completion ffhq
    'paint_train_source_ffhq': 'data/celeba/intelli-paint/train/painting',
    'paint_train_target_ffhq': 'data/celeba/intelli-paint/train/target',
    'paint_test_source_ffhq': 'data/celeba/intelli-paint/test/painting',
    'paint_test_target_ffhq': 'data/celeba/intelli-paint/test/target',
     
    # Painting completion cars
    'paint_train_source': 'data/intelli-paint/train/painting',
    'paint_train_target': 'data/intelli-paint/train/target',
    'paint_test_source': 'data/intelli-paint/test/painting',
    'paint_test_target': 'data/intelli-paint/test/target',
}

model_paths = {
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'resnet34': 'pretrained_models/resnet34-333f7ec4.pth',
	'stylegan_celeba': 'pretrained_models/stylegan2-celeba-config-f.pt',
	'stylegan_cars': 'pretrained_models/stylegan2-car-config-f.pt',
	'stylegan_church': 'pretrained_models/stylegan2-church-config-f.pt',
	'stylegan_horse': 'pretrained_models/stylegan2-horse-config-f.pt',
	'stylegan_ada_wild': 'pretrained_models/afhqwild.pt',
	'stylegan_toonify': 'pretrained_models/celeba_cartoon_blended.pt',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth',
	'id_encoder': 'pretrained_models/id-encoder.pt',
	'canvas_encoder': 'pretrained_models/canvas-encoder.pt',
	'stylegan_watercolor': 'pretrained_models/stylegan2-watercolor.pt',
}
