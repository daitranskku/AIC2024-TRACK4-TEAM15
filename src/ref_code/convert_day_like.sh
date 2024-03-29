echo "Converting train images"
python test_unpaired.py --input /home/daitranskku/code/cvpr2024/aicity/github_submission/sample_dataset/NAFNet_Output/train/ --save_dir /home/daitranskku/code/cvpr2024/aicity/github_submission/sample_dataset/GSAD_Output/train
echo "Converting val images"
python test_unpaired.py --input /home/daitranskku/code/cvpr2024/aicity/github_submission/sample_dataset/NAFNet_Output/val/ --save_dir /home/daitranskku/code/cvpr2024/aicity/github_submission/sample_dataset/GSAD_Output/val
echo "Converting test images"
python test_unpaired.py --input /home/daitranskku/code/cvpr2024/aicity/github_submission/sample_dataset/NAFNet_Output/cvpr_test/ --save_dir /home/daitranskku/code/cvpr2024/aicity/github_submission/sample_dataset/GSAD_Output/cvpr_test
