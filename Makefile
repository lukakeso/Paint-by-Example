Quantitative_test:
	python scripts/inference_test_bench.py \
	--plms \
	--outdir results/test_bench \
	--config configs/v1.yaml \
	--ckpt checkpoints/model.ckpt \
	--scale 5

FID_test:
	python eval_tool/fid/fid_score.py --device cuda \
	test_bench/test_set_GT \
	results/test_bench/results

QS_test:
	python eval_tool/gmm/gmm_score_coco.py results/test_bench/results \
	--gmm_path eval_tool/gmm/coco2017_gmm_k20 \
	--gpu 1

CLIP_test:
	python eval_tool/clip_score/region_clip_score.py \
	--result_dir results/test_bench/results