# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM


- **MSSV:*2A202603582*
- **Họ và tên:*Nguyễn Minh Đạt*
- **SEED** (in ở ô đầu notebook): 6021

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.974   0.781      0.867        0.889   
2. BoW + MLP (deep learning)         0.978   0.917      0.946        0.974   
3a. Encoder (MiniLM) + kNN           0.893   0.958      0.925        1.000   
3b. Encoder (MiniLM) + LogReg        0.978   0.948      0.963        1.000   
3c. Encoder (MiniLM) + MLP head      0.989   0.948      0.968        1.000   
4a. LLM nhỏ zero-shot                0.833   1.000      0.909        0.909   
4b. LLM lớn zero-shot                0.952   1.000      0.976        0.976   
4c. LLM nhỏ few-shot                 0.950   0.950      0.950        0.950   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            5.798      15658.408       0.046   
2. BoW + MLP (deep learning)        13.504      17998.438       0.314   
3a. Encoder (MiniLM) + kNN          66.499         18.108      35.292   
3b. Encoder (MiniLM) + LogReg       66.956         32.495      23.200   
3c. Encoder (MiniLM) + MLP head     67.475         29.035      22.722   
4a. LLM nhỏ zero-shot                0.000          1.149     379.925   
4b. LLM lớn zero-shot                0.000          1.220     322.955   
4c. LLM nhỏ few-shot                 0.000          1.109     377.467   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.079           0.0  
2. BoW + MLP (deep learning)          0.416           0.0  
3a. Encoder (MiniLM) + kNN           56.758           0.0  
3b. Encoder (MiniLM) + LogReg        34.082           0.0  
3c. Encoder (MiniLM) + MLP head      33.232           0.0  
4a. LLM nhỏ zero-shot              2378.956           0.0  
4b. LLM lớn zero-shot              2430.145           0.0  
4c. LLM nhỏ few-shot               2494.902           0.0  

## 3. Dự đoán và thực tế


1. Learning rate: Dự đoán lr=50 có thể khiến loss dao động hoặc tăng. Thực tế loss tăng vọt ban đầu rồi giảm còn khoảng 0.005. Bước cập nhật lớn gây mất ổn định lúc đầu nhưng chưa khiến quá trình học mất khả năng hội tụ.

2. Tin ngụy trang: Dự đoán Encoder + LogReg đúng nhiều hơn BoW. Thực tế Encoder đúng 4/6, BoW đúng 5/6. Embedding biểu diễn ngữ nghĩa tốt nhưng đầu phân loại vẫn có thể bỏ sót dấu hiệu spam.

3. Model lớn và nhỏ: Thực tế model lớn đạt 0.976, model nhỏ đạt 0.909, chênh 0.067. Một số tin ham khó khiến model nhỏ chặn nhầm; SUB nhỏ nên vài lỗi đã tạo chênh lệch đáng kể.

## 4. Phân tích 50 case khó

BoW + MLP tụt F1 nhiều nhất, từ 0.974 xuống 0.378, giảm 0.596. BoW + LogReg có F1 trên bộ khó thấp nhất (0.333).

## 5. Thử thách cá nhân

Đề #4: So sánh kNN với k = 1, 5, 25 trên cùng embedding và tập test.
tập train và tập test; lần lượt chạy knn_predict với k = 1, 5, 25. Với mỗi k, mô hình lấy k tin train gần nhất để bỏ phiếu, sau đó so nhãn dự đoán với nhãn thật bằng hàm metrics để tính F1. 

k quá nhỏ dễ bị ảnh hưởng bởi một hàng xóm nhiễu. k quá lớn đưa thêm các tin ít tương đồng vào bỏ phiếu và có thể thiên về ham do dữ liệu mất cân bằng. Trong ba mức đã thử, k = 5 cho F1 cao nhất; khi chọn cấu hình thực tế nên dùng tập validation.

## 6. Phương án cho sản phẩm thật

Em chọn BoW + MLP: F1 test 0.946, p95 0.416 ms và throughput khoảng 17.998 tin/giây trong phép đo, đáp ứng tải trung bình khoảng 116 tin/giây. Đánh đổi là F1 thấp hơn Encoder + LogReg/MLP head.

## 7. AI log và điều đã học


Mỗi lần dùng AI em sẽ hỏi những thứ trước mắt cần xử lý mà em chưa hiểu hoặc chưa biết làm, nhận được cấu trả lời sẽ kiểm tra và xác thực lại xem là thông tin đã chính xác hay chưa có đáng tin cậy hay không, em sẽ sửa lại theo văn phong và theo cách hiểu của mình hơn là chép của AI.
Qua hôm nay em hiểu rằng accuracy cao chưa chắc bắt spam tốt và mô hình phức tạp hơn không luôn thắng. Khi chọn mô hình, cần cân nhắc cả chất lượng, tốc độ và chi phí. 
