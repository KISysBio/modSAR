nohup docker-compose run --rm modsar ipython3 scripts/run_cross_validation.py -- --dataset NYPR2 --datasplit 1 > .test.log 2>&1 &
