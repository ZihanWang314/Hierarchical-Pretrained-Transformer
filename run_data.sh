data_root=../condqa_files/data
model_root=../condqa_files/model

echo "processing data to inputs..."
python process_data.py --data_root=${data_root} --model_root=${model_root} # --easy_passage # --yesno_only # --conditional_only