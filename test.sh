CUDA_VISIBLE_DEVICES=0 python3 test.py \
--model_name bert-base-chinese \
--model_path workspace/wb/best.pt \
--num_labels 6 \
--test_data_path data/clean/usual_test_labeled.txt \
--batch_size 64 \
--dataloader_num_workors 8 \

