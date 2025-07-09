python vai_q_yolo.py  --model_name "yolov8n.pt" --batch_size 16 --target DPUCZDX8G_ISA1_B4096 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "yolov8n.pt" --batch_size 1 --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v8n
