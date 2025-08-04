"""
Chương trình tính TF-IDF
Áp dụng trên dữ liệu DUC (Document Understanding Conference)

Công thức TF-IDF:
- TF (Term Frequency) = số lần xuất hiện của từ trong tài liệu / tổng số từ trong tài liệu
- IDF (Inverse Document Frequency) = log(tổng số tài liệu / số tài liệu chứa từ đó)
- TF-IDF = TF × IDF
"""

import os
import re
import math
from collections import defaultdict, Counter
import numpy as np

class TFIDFCalculator:
    def __init__(self):
        self.documents = []  # Danh sách các tài liệu
        self.vocabulary = set()  # Từ vựng của toàn bộ corpus
        self.document_count = 0  # Tổng số tài liệu
        self.word_doc_count = defaultdict(int)  # Số tài liệu chứa mỗi từ
        
    def preprocess_text(self, text):
        """
        Tiền xử lý văn bản:
        - Loại bỏ thẻ XML
        - Chuyển về chữ thường
        - Loại bỏ dấu câu
        - Tách từ
        """
        # Loại bỏ thẻ XML
        text = re.sub(r'<[^>]+>', '', text)
        # Chuyển về chữ thường
        text = text.lower()
        # Loại bỏ dấu câu và ký tự đặc biệt, chỉ giữ lại chữ cái và số
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Tách từ
        words = text.split()
        # Loại bỏ từ rỗng
        words = [word for word in words if len(word) > 0]
        return words
    
    def load_document(self, file_path):
        """Đọc và xử lý một tài liệu"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            words = self.preprocess_text(content)
            return words
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
            return []
    
    def load_corpus(self, folder_path):
        """Đọc toàn bộ corpus từ một thư mục"""
        print(f"Đang tải corpus từ: {folder_path}")
        
        for filename in sorted(os.listdir(folder_path)):
            if filename.startswith('.'):  # Bỏ qua file ẩn
                continue
                
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                words = self.load_document(file_path)
                if words:
                    self.documents.append({
                        'filename': filename,
                        'words': words,
                        'word_count': len(words)
                    })
                    # Cập nhật từ vựng
                    unique_words = set(words)
                    self.vocabulary.update(unique_words)
                    
                    # Đếm số tài liệu chứa mỗi từ
                    for word in unique_words:
                        self.word_doc_count[word] += 1
        
        self.document_count = len(self.documents)
        print(f"Đã tải {self.document_count} tài liệu")
        print(f"Tổng từ vựng: {len(self.vocabulary)} từ")
    
    def calculate_tf(self, word, document):
        """
        Tính Term Frequency (TF)
        TF = số lần xuất hiện của từ / tổng số từ trong tài liệu
        """
        word_count = document['words'].count(word)
        total_words = document['word_count']
        tf = word_count / total_words if total_words > 0 else 0
        return tf
    
    def calculate_idf(self, word):
        """
        Tính Inverse Document Frequency (IDF)
        IDF = log(tổng số tài liệu / số tài liệu chứa từ)
        """
        docs_containing_word = self.word_doc_count[word]
        if docs_containing_word == 0:
            return 0
        idf = math.log(self.document_count / docs_containing_word)
        return idf
    
    def calculate_tfidf(self, word, document):
        """
        Tính TF-IDF
        TF-IDF = TF × IDF
        """
        tf = self.calculate_tf(word, document)
        idf = self.calculate_idf(word)
        tfidf = tf * idf
        return tfidf
    
    def get_tfidf_vector(self, document_index):
        """Tính vector TF-IDF cho một tài liệu"""
        if document_index >= len(self.documents):
            print("Chỉ số tài liệu không hợp lệ")
            return {}
        
        document = self.documents[document_index]
        tfidf_vector = {}
        
        # Chỉ tính TF-IDF cho các từ có trong tài liệu này
        unique_words = set(document['words'])
        
        for word in unique_words:
            tfidf_vector[word] = self.calculate_tfidf(word, document)
        
        return tfidf_vector
    
    def get_top_words(self, document_index, top_k=10):
        """Lấy top k từ có TF-IDF cao nhất trong tài liệu"""
        tfidf_vector = self.get_tfidf_vector(document_index)
        # Sắp xếp theo giá trị TF-IDF giảm dần
        sorted_words = sorted(tfidf_vector.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_k]
    
    def analyze_document(self, document_index):
        """Phân tích chi tiết một tài liệu"""
        if document_index >= len(self.documents):
            print("Chỉ số tài liệu không hợp lệ")
            return
        
        document = self.documents[document_index]
        print(f"\n=== PHÂN TÍCH TÀI LIỆU: {document['filename']} ===")
        print(f"Tổng số từ: {document['word_count']}")
        print(f"Số từ duy nhất: {len(set(document['words']))}")
        
        # Hiển thị top 10 từ có TF-IDF cao nhất
        print(f"\nTop 10 từ có TF-IDF cao nhất:")
        print(f"{'Từ':<15} {'TF':<10} {'IDF':<10} {'TF-IDF':<10}")
        print("-" * 50)
        
        top_words = self.get_top_words(document_index, 10)
        for word, tfidf in top_words:
            tf = self.calculate_tf(word, document)
            idf = self.calculate_idf(word)
            print(f"{word:<15} {tf:<10.4f} {idf:<10.4f} {tfidf:<10.4f}")
    
    def demonstrate_calculation(self, word, document_index):
        """Minh họa chi tiết cách tính TF-IDF cho một từ cụ thể"""
        if document_index >= len(self.documents):
            print("Chỉ số tài liệu không hợp lệ")
            return
        
        document = self.documents[document_index]
        
        print(f"\n=== MINH HỌA TÍNH TF-IDF CHO TỪ '{word}' ===")
        print(f"Tài liệu: {document['filename']}")
        
        # Đếm số lần xuất hiện của từ
        word_count = document['words'].count(word)
        total_words = document['word_count']
        
        print(f"\n1. TÍNH TF (Term Frequency):")
        print(f"   Số lần xuất hiện của '{word}': {word_count}")
        print(f"   Tổng số từ trong tài liệu: {total_words}")
        print(f"   TF = {word_count} / {total_words} = {word_count/total_words:.6f}")
        
        # Tính IDF
        docs_with_word = self.word_doc_count[word]
        total_docs = self.document_count
        
        print(f"\n2. TÍNH IDF (Inverse Document Frequency):")
        print(f"   Tổng số tài liệu: {total_docs}")
        print(f"   Số tài liệu chứa '{word}': {docs_with_word}")
        print(f"   IDF = log({total_docs} / {docs_with_word}) = {math.log(total_docs/docs_with_word):.6f}")
        
        # Tính TF-IDF
        tf = word_count / total_words
        idf = math.log(total_docs / docs_with_word) if docs_with_word > 0 else 0
        tfidf = tf * idf
        
        print(f"\n3. TÍNH TF-IDF:")
        print(f"   TF-IDF = TF × IDF = {tf:.6f} × {idf:.6f} = {tfidf:.6f}")
        
        return tfidf
    
    def build_tfidf_matrix(self):
        """
        Xây dựng ma trận TF-IDF cho toàn bộ corpus
        Mỗi hàng là một tài liệu, mỗi cột là một từ trong từ vựng
        """
        print("\nĐang xây dựng ma trận TF-IDF...")
        
        # Tạo danh sách từ vựng theo thứ tự cố định
        vocab_list = sorted(list(self.vocabulary))
        vocab_size = len(vocab_list)
        word_to_index = {word: i for i, word in enumerate(vocab_list)}
        
        # Khởi tạo ma trận TF-IDF
        tfidf_matrix = np.zeros((self.document_count, vocab_size))
        
        # Tính TF-IDF cho từng tài liệu
        for doc_idx, document in enumerate(self.documents):
            for word in set(document['words']):  # Chỉ tính cho từ có trong tài liệu
                word_idx = word_to_index[word]
                tfidf_matrix[doc_idx, word_idx] = self.calculate_tfidf(word, document)
        
        print(f"Ma trận TF-IDF: {tfidf_matrix.shape} (tài liệu × từ vựng)")
        return tfidf_matrix, vocab_list, word_to_index
    
    def calculate_cosine_similarity(self, vector1, vector2):
        """
        Tính cosine similarity giữa hai vector
        cosine_similarity = (A · B) / (|A| × |B|)
        """
        # Tính tích vô hướng
        dot_product = np.dot(vector1, vector2)
        
        # Tính độ dài của các vector
        norm_a = np.linalg.norm(vector1)
        norm_b = np.linalg.norm(vector2)
        
        # Tránh chia cho 0
        if norm_a == 0 or norm_b == 0:
            return 0
        
        cosine_sim = dot_product / (norm_a * norm_b)
        return cosine_sim
    
    def build_cosine_matrix(self, tfidf_matrix):
        """
        Xây dựng ma trận cosine similarity giữa các tài liệu
        """
        print("\nĐang xây dựng ma trận cosine similarity...")
        
        num_docs = tfidf_matrix.shape[0]
        cosine_matrix = np.zeros((num_docs, num_docs))
        
        for i in range(num_docs):
            for j in range(num_docs):
                cosine_matrix[i, j] = self.calculate_cosine_similarity(
                    tfidf_matrix[i], tfidf_matrix[j]
                )
        
        print(f"Ma trận cosine similarity: {cosine_matrix.shape}")
        return cosine_matrix
    
    def extractive_summarization(self, tfidf_matrix, top_sentences=3):
        """
        Tóm tắt văn bản bằng phương pháp extractive dựa trên TF-IDF
        Chọn các câu có điểm TF-IDF cao nhất
        """
        print(f"\nĐang tóm tắt văn bản (chọn {top_sentences} câu quan trọng nhất)...")
        
        # Tính điểm quan trọng của mỗi tài liệu (tổng TF-IDF)
        document_scores = []
        for doc_idx in range(len(self.documents)):
            # Tổng TF-IDF của tất cả từ trong tài liệu
            total_score = np.sum(tfidf_matrix[doc_idx])
            document_scores.append({
                'index': doc_idx,
                'filename': self.documents[doc_idx]['filename'],
                'score': total_score,
                'word_count': self.documents[doc_idx]['word_count']
            })
        
        # Sắp xếp theo điểm giảm dần
        document_scores.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\nTop {top_sentences} tài liệu quan trọng nhất (dựa trên TF-IDF):")
        print(f"{'Thứ hạng':<10} {'Tài liệu':<15} {'Điểm TF-IDF':<15} {'Số từ':<10}")
        print("-" * 55)
        
        summary_docs = []
        for i in range(min(top_sentences, len(document_scores))):
            doc_info = document_scores[i]
            print(f"{i+1:<10} {doc_info['filename']:<15} {doc_info['score']:<15.4f} {doc_info['word_count']:<10}")
            summary_docs.append(doc_info)
        
        return summary_docs
    
    def find_similar_documents(self, cosine_matrix, doc_index, top_k=5):
        """
        Tìm các tài liệu tương tự nhất với tài liệu cho trước
        """
        if doc_index >= len(self.documents):
            print("Chỉ số tài liệu không hợp lệ")
            return []
        
        # Lấy độ tương tự của tài liệu với tất cả tài liệu khác
        similarities = cosine_matrix[doc_index]
        
        # Tạo danh sách (index, similarity) và sắp xếp
        similarity_pairs = [(i, sim) for i, sim in enumerate(similarities) if i != doc_index]
        similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTài liệu tương tự nhất với '{self.documents[doc_index]['filename']}':")
        print(f"{'Thứ hạng':<10} {'Tài liệu':<15} {'Độ tương tự':<15}")
        print("-" * 45)
        
        similar_docs = []
        for i, (doc_idx, similarity) in enumerate(similarity_pairs[:top_k]):
            print(f"{i+1:<10} {self.documents[doc_idx]['filename']:<15} {similarity:<15.4f}")
            similar_docs.append({
                'index': doc_idx,
                'filename': self.documents[doc_idx]['filename'],
                'similarity': similarity
            })
        
        return similar_docs
    
    def demonstrate_cosine_calculation(self, doc1_index, doc2_index, tfidf_matrix):
        """
        Minh họa chi tiết cách tính cosine similarity giữa hai tài liệu
        """
        if doc1_index >= len(self.documents) or doc2_index >= len(self.documents):
            print("Chỉ số tài liệu không hợp lệ")
            return
        
        doc1 = self.documents[doc1_index]
        doc2 = self.documents[doc2_index]
        
        print(f"\n=== MINH HỌA TÍNH COSINE SIMILARITY ===")
        print(f"Tài liệu 1: {doc1['filename']}")
        print(f"Tài liệu 2: {doc2['filename']}")
        
        # Lấy vector TF-IDF của hai tài liệu
        vector1 = tfidf_matrix[doc1_index]
        vector2 = tfidf_matrix[doc2_index]
        
        # Tính tích vô hướng
        dot_product = np.dot(vector1, vector2)
        print(f"\n1. TÍCH VÔ HƯỚNG (A · B):")
        print(f"   Dot product = {dot_product:.6f}")
        
        # Tính độ dài vector
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        print(f"\n2. ĐỘ DÀI VECTOR:")
        print(f"   |A| = √(Σ(ai²)) = {norm1:.6f}")
        print(f"   |B| = √(Σ(bi²)) = {norm2:.6f}")
        
        # Tính cosine similarity
        cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        print(f"\n3. COSINE SIMILARITY:")
        print(f"   cosine_similarity = (A · B) / (|A| × |B|)")
        print(f"   cosine_similarity = {dot_product:.6f} / ({norm1:.6f} × {norm2:.6f})")
        print(f"   cosine_similarity = {cosine_sim:.6f}")
        
        # Giải thích ý nghĩa
        if cosine_sim > 0.8:
            interpretation = "rất tương tự"
        elif cosine_sim > 0.6:
            interpretation = "tương tự"
        elif cosine_sim > 0.4:
            interpretation = "hơi tương tự"
        elif cosine_sim > 0.2:
            interpretation = "ít tương tự"
        else:
            interpretation = "không tương tự"
        
        print(f"\n4. GIẢI THÍCH:")
        print(f"   Giá trị {cosine_sim:.6f} cho thấy hai tài liệu {interpretation}")
        
        return cosine_sim


def main():
    # Khởi tạo calculator
    tfidf_calc = TFIDFCalculator()
    
    # Đường dẫn đến thư mục chứa dữ liệu
    base_path = "/Users/yoliephan/Library/CloudStorage/OneDrive-Personal/Tài liệu/MASTER 2025/Đợt 1/Xử lý Ngôn Ngữ Tự Nhiên/MY"
    
    # Chọn thư mục để phân tích (có thể thay đổi)
    data_folder = os.path.join(base_path, "DUC_TEXT", "train")
    
    # Tải corpus
    tfidf_calc.load_corpus(data_folder)
    
    if tfidf_calc.document_count == 0:
        print("Không tìm thấy tài liệu nào!")
        return
    
    # Phân tích một vài tài liệu đầu
    print("\n" + "="*60)
    print("PHÂN TÍCH TF-IDF CHO CÁC TÀI LIỆU")
    print("="*60)
    
    # Phân tích 3 tài liệu đầu tiên
    for i in range(min(3, tfidf_calc.document_count)):
        tfidf_calc.analyze_document(i)
    
    # Minh họa chi tiết cách tính cho một từ cụ thể
    print("\n" + "="*60)
    print("MINH HỌA CHI TIẾT CÁCH TÍNH")
    print("="*60)
    
    # Chọn tài liệu đầu tiên và từ "hurricane" để minh họa
    if tfidf_calc.document_count > 0:
        document = tfidf_calc.documents[0]
        # Tìm một từ phổ biến để minh họa
        word_counts = Counter(document['words'])
        common_words = [word for word, count in word_counts.most_common(10) 
                       if len(word) > 3]  # Bỏ qua từ quá ngắn
        
        if common_words:
            demo_word = common_words[0]
            tfidf_calc.demonstrate_calculation(demo_word, 0)
    
    # Thống kê tổng quan
    print(f"\n" + "="*60)
    print("THỐNG KÊ TỔNG QUAN")
    print("="*60)
    print(f"Tổng số tài liệu: {tfidf_calc.document_count}")
    print(f"Tổng từ vựng: {len(tfidf_calc.vocabulary)}")
    
    # Hiển thị một số từ xuất hiện trong nhiều tài liệu nhất
    print(f"\nTop 10 từ xuất hiện trong nhiều tài liệu nhất:")
    sorted_words = sorted(tfidf_calc.word_doc_count.items(), 
                         key=lambda x: x[1], reverse=True)
    for word, doc_count in sorted_words[:10]:
        if len(word) > 2:  # Bỏ qua từ quá ngắn
            percentage = (doc_count / tfidf_calc.document_count) * 100
            print(f"  {word}: {doc_count} tài liệu ({percentage:.1f}%)")
    
    # ===== PHẦN MỚI: TÓM TẮT VÀ MA TRẬN COSINE =====
    print(f"\n" + "="*60)
    print("XÂY DỰNG MA TRẬN TF-IDF VÀ COSINE SIMILARITY")
    print("="*60)
    
    # Xây dựng ma trận TF-IDF
    tfidf_matrix, vocab_list, word_to_index = tfidf_calc.build_tfidf_matrix()
    
    # Xây dựng ma trận cosine similarity
    cosine_matrix = tfidf_calc.build_cosine_matrix(tfidf_matrix)
    
    # Tóm tắt văn bản
    print(f"\n" + "="*60)
    print("TÓM TẮT VÀN BẢN BẰNG TF-IDF")
    print("="*60)
    
    summary_docs = tfidf_calc.extractive_summarization(tfidf_matrix, top_sentences=5)
    
    # Tìm tài liệu tương tự
    print(f"\n" + "="*60)
    print("TÌM TÀI LIỆU TƯƠNG TỰ")
    print("="*60)
    
    if tfidf_calc.document_count > 1:
        # Tìm tài liệu tương tự với tài liệu đầu tiên
        similar_docs = tfidf_calc.find_similar_documents(cosine_matrix, 0, top_k=5)
    
    # Minh họa tính cosine similarity
    print(f"\n" + "="*60)
    print("MINH HỌA TÍNH COSINE SIMILARITY")
    print("="*60)
    
    if tfidf_calc.document_count >= 2:
        tfidf_calc.demonstrate_cosine_calculation(0, 1, tfidf_matrix)
    
    # Hiển thị một phần ma trận cosine
    print(f"\n" + "="*60)
    print("MA TRẬN COSINE SIMILARITY (5x5 đầu tiên)")
    print("="*60)
    
    display_size = min(5, tfidf_calc.document_count)
    print(f"\n{'Tài liệu':<12}", end="")
    for j in range(display_size):
        print(f"{tfidf_calc.documents[j]['filename'][:8]:<10}", end="")
    print()
    
    for i in range(display_size):
        print(f"{tfidf_calc.documents[i]['filename'][:10]:<12}", end="")
        for j in range(display_size):
            print(f"{cosine_matrix[i,j]:<10.4f}", end="")
        print()
    
    # Thống kê ma trận
    print(f"\nThống kê ma trận cosine similarity:")
    print(f"  - Giá trị trung bình: {np.mean(cosine_matrix):.4f}")
    print(f"  - Giá trị cao nhất (không tính đường chéo): {np.max(cosine_matrix - np.eye(len(cosine_matrix))):.4f}")
    print(f"  - Giá trị thấp nhất: {np.min(cosine_matrix):.4f}")


if __name__ == "__main__":
    main()
