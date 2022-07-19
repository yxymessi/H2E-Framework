
# stage_1
python stage_1.py  --num_epochs 180  --cfg './noise_longtail/config/animal_10N/animal.yaml'  --lr 0.2 --weight_decay 1e-4   --save_model 1  --randaug  1
# stage_2
python stage_2.py   --cfg './noise_longtail/config/animal_10N/animal.yaml'  --num_epochs 20  --lr 0.01 --weight_decay 1e-5  --freeze 1 --save_model 1