# TÓM TẮT VĂN BẢN BẰNG TF-IDF VÀ TEXTRANK

## Tổng quan

Thực hiện tóm tắt văn bản tự động sử dụng thuật toán TextRank dựa trên TF-IDF:

- Đọc và xử lý các file XML từ dataset DUC (Document Understanding Conference)
- Biểu diễn văn bản dưới dạng vector TF-IDF
- Tính toán độ tương đồng cosine giữa các câu
- Mô hình hóa văn bản thành đồ thị
- Áp dụng thuật toán TextRank để xếp hạng câu
- Tạo bản tóm tắt và đánh giá chất lượng bằng ROUGE metrics

## Cài đặt và Chạy

### Yêu cầu hệ thống

- Python 3.8 trở lên
- Jupyter Notebook
- Các thư viện: numpy, python-docx

### Cài đặt

1. **Cài đặt thư viện:**

```bash
pip install nltk spacy python-docx numpy
```

## Cách sử dụng

### 1. Chọn file để xử lý

Khi chạy notebook, hệ thống sẽ hiển thị danh sách các file có sẵn trong thư mục `DUC_TEXT/train/`. Bạn có thể:

- Nhập tên file cụ thể (ví dụ: `d061j`)
- Nhấn Enter để sử dụng file đầu tiên

### 2. Theo dõi quá trình xử lý

- **Bước 1:** Đọc và tiền xử lý văn bản
- **Bước 2:** Tính toán ma trận TF-IDF
- **Bước 3:** Tính độ tương đồng cosine
- **Bước 4:** Tạo đồ thị từ văn bản
- **Bước 5:** Áp dụng thuật toán TextRank
- **Bước 6:** Tạo bản tóm tắt
- **Bước 7:** Lưu kết quả vào Word
- **Bước 8:** Đọc reference summary
- **Bước 9:** Đánh giá bằng ROUGE

### 3. Xem kết quả

Sau khi chạy xong, bạn sẽ có:

- `input.docx`: Văn bản gốc đã được xử lý
- `output_summary.docx`: Bản tóm tắt tự động
- `Test_DUC_SUM.docx`: Reference summary từ DUC
- Điểm ROUGE trên notebook

## Kết quả và Đánh giá

### Metrics sử dụng

- **ROUGE-1**: Đo overlap của unigrams (từ đơn)
- **ROUGE-2**: Đo overlap của bigrams (cặp từ)
- **ROUGE-L**: Đo Longest Common Subsequence

### Thang điểm chất lượng

- **≥ 0.5**: Xuất sắc
- **0.3-0.5**: Tốt
- **0.2-0.3**: Khá
- **0.1-0.2**: Trung bình
- **< 0.1**: Cần cải thiện

### Thông số có thể điều chỉnh

```python
# Trong class TextSummarizerTFIDFTextRank
damping_factor = 0.85      # Tham số damping cho TextRank (0.5-0.9)
max_iterations = 100       # Số vòng lặp tối đa
tolerance = 1e-6           # Ngưỡng hội tụ
summary_ratio = 0.1        # Tỷ lệ tóm tắt (5%-20%)
threshold = 0.1            # Ngưỡng similarity để tạo cạnh (0.05-0.3)
```

## 📚 Tài liệu tham khảo

1. **TextRank Algorithm**: Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into texts.
2. **TF-IDF**: Salton, G., & McGill, M. J. (1983). Introduction to modern information retrieval.
3. **ROUGE Metrics**: Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries.
4. **DUC Dataset**: Document Understanding Conference dataset.
