"""
Chương trình tính PageRank
Áp dụng trên dữ liệu DUC để đánh giá tầm quan trọng của tài liệu

Công thức PageRank:
PR(A) = (1-d)/N + d * Σ(PR(T_i)/C(T_i))

Trong đó:
- PR(A): PageRank của trang A
- d: damping factor (thường là 0.85)
- N: tổng số trang
- T_i: các trang liên kết đến trang A
- C(T_i): số liên kết ra từ trang T_i
"""

import os
import numpy as np
from tf_idf_calculator import TFIDFCalculator

class PageRankCalculator:
    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        self.damping_factor = damping_factor  # Hệ số damping (d)
        self.max_iterations = max_iterations  # Số vòng lặp tối đa
        self.tolerance = tolerance  # Ngưỡng hội tụ
        self.pagerank_scores = None
        self.adjacency_matrix = None
        self.transition_matrix = None
        
    def create_adjacency_matrix(self, cosine_matrix, threshold=0.3):
        """
        Tạo ma trận kề từ ma trận cosine similarity
        Nếu similarity > threshold thì có liên kết
        """
        print(f"\nTạo ma trận kề từ ma trận cosine similarity (ngưỡng: {threshold})")
        
        n = cosine_matrix.shape[0]
        self.adjacency_matrix = np.zeros((n, n))
        
        # Tạo liên kết dựa trên ngưỡng similarity
        for i in range(n):
            for j in range(n):
                if i != j and cosine_matrix[i, j] > threshold:
                    self.adjacency_matrix[i, j] = 1
        
        # Đếm số liên kết
        total_links = np.sum(self.adjacency_matrix)
        print(f"Tổng số liên kết: {int(total_links)}")
        
        # Hiển thị ma trận kề (5x5 đầu tiên)
        display_size = min(5, n)
        print(f"\nMa trận kề ({display_size}x{display_size} đầu tiên):")
        print("   ", end="")
        for j in range(display_size):
            print(f"D{j:<3}", end="")
        print()
        
        for i in range(display_size):
            print(f"D{i:<2}", end="")
            for j in range(display_size):
                print(f"{int(self.adjacency_matrix[i,j]):<4}", end="")
            print()
        
        return self.adjacency_matrix
    
    def create_transition_matrix(self):
        """
        Tạo ma trận chuyển tiếp từ ma trận kề
        Mỗi hàng được chuẩn hóa để tổng = 1
        """
        print(f"\nTạo ma trận chuyển tiếp...")
        
        n = self.adjacency_matrix.shape[0]
        self.transition_matrix = np.copy(self.adjacency_matrix).astype(float)
        
        # Chuẩn hóa từng hàng
        for i in range(n):
            row_sum = np.sum(self.transition_matrix[i])
            if row_sum > 0:
                self.transition_matrix[i] = self.transition_matrix[i] / row_sum
            else:
                # Nếu không có liên kết ra, phân bố đều cho tất cả các trang
                self.transition_matrix[i] = np.ones(n) / n
        
        print(f"Ma trận chuyển tiếp: {self.transition_matrix.shape}")
        
        # Hiển thị ma trận chuyển tiếp (5x5 đầu tiên)
        display_size = min(5, n)
        print(f"\nMa trận chuyển tiếp ({display_size}x{display_size} đầu tiên):")
        print("     ", end="")
        for j in range(display_size):
            print(f"D{j:<8}", end="")
        print()
        
        for i in range(display_size):
            print(f"D{i:<4}", end="")
            for j in range(display_size):
                print(f"{self.transition_matrix[i,j]:<8.4f}", end="")
            print()
        
        return self.transition_matrix
    
    def calculate_pagerank_manual(self, documents_info):
        """
        Tính PageRank bằng phương pháp power iteration (tính tay)
        """
        print(f"\n" + "="*60)
        print("TÍNH PAGERANK BẰNG POWER ITERATION")
        print("="*60)
        
        n = self.transition_matrix.shape[0]
        
        # Khởi tạo vector PageRank (phân bố đều)
        pagerank_vector = np.ones(n) / n
        print(f"Khởi tạo PageRank: phân bố đều = 1/{n} = {1/n:.6f}")
        
        print(f"\nCông thức PageRank:")
        print(f"PR_new = (1-d)/N * 1 + d * M^T * PR_old")
        print(f"Trong đó:")
        print(f"  - d (damping factor) = {self.damping_factor}")
        print(f"  - N (số tài liệu) = {n}")
        print(f"  - M^T = ma trận chuyển tiếp chuyển vị")
        print(f"  - (1-d)/N = {(1-self.damping_factor)/n:.6f}")
        
        # Lưu lịch sử để theo dõi hội tụ
        history = []
        
        print(f"\nQuá trình lặp:")
        print(f"{'Vòng':<6} {'Thay đổi':<12} {'Top 3 PageRank':<30}")
        print("-" * 55)
        
        for iteration in range(self.max_iterations):
            # Lưu vector cũ
            old_pagerank = np.copy(pagerank_vector)
            
            # Tính PageRank mới theo công thức
            # PR = (1-d)/N + d * M^T * PR_old
            teleport_term = (1 - self.damping_factor) / n
            random_walk_term = self.damping_factor * np.dot(self.transition_matrix.T, pagerank_vector)
            pagerank_vector = teleport_term + random_walk_term
            
            # Tính sự thay đổi
            change = np.linalg.norm(pagerank_vector - old_pagerank)
            history.append((iteration + 1, change, np.copy(pagerank_vector)))
            
            # Hiển thị top 3 tài liệu có PageRank cao nhất
            top_indices = np.argsort(pagerank_vector)[-3:][::-1]
            top_scores = [f"{pagerank_vector[i]:.4f}" for i in top_indices]
            top_info = ", ".join(top_scores)
            
            print(f"{iteration+1:<6} {change:<12.8f} {top_info:<30}")
            
            # Kiểm tra hội tụ
            if change < self.tolerance:
                print(f"\nHội tụ sau {iteration + 1} vòng lặp!")
                break
        
        self.pagerank_scores = pagerank_vector
        
        # Hiển thị kết quả chi tiết
        self.display_pagerank_results(documents_info)
        
        return pagerank_vector, history
    
    def display_pagerank_results(self, documents_info):
        """
        Hiển thị kết quả PageRank chi tiết
        """
        print(f"\n" + "="*60)
        print("KẾT QUẢ PAGERANK")
        print("="*60)
        
        n = len(self.pagerank_scores)
        
        # Tạo danh sách kết quả
        results = []
        for i in range(n):
            results.append({
                'index': i,
                'filename': documents_info[i]['filename'] if i < len(documents_info) else f"Doc_{i}",
                'pagerank': self.pagerank_scores[i],
                'word_count': documents_info[i]['word_count'] if i < len(documents_info) else 0
            })
        
        # Sắp xếp theo PageRank giảm dần
        results.sort(key=lambda x: x['pagerank'], reverse=True)
        
        print(f"{'Hạng':<6} {'Tài liệu':<15} {'PageRank':<12} {'Số từ':<8} {'%':<8}")
        print("-" * 55)
        
        total_pagerank = sum(self.pagerank_scores)
        for rank, result in enumerate(results[:10], 1):
            percentage = (result['pagerank'] / total_pagerank) * 100
            print(f"{rank:<6} {result['filename'][:13]:<15} {result['pagerank']:<12.6f} "
                  f"{result['word_count']:<8} {percentage:<8.2f}")
        
        # Thống kê
        print(f"\nThống kê PageRank:")
        print(f"  - Tổng PageRank: {total_pagerank:.6f}")
        print(f"  - Trung bình: {np.mean(self.pagerank_scores):.6f}")
        print(f"  - Độ lệch chuẩn: {np.std(self.pagerank_scores):.6f}")
        print(f"  - Max: {np.max(self.pagerank_scores):.6f}")
        print(f"  - Min: {np.min(self.pagerank_scores):.6f}")
    
    def demonstrate_pagerank_calculation(self, documents_info, doc_index=0):
        """
        Minh họa chi tiết cách tính PageRank cho một tài liệu cụ thể
        """
        if doc_index >= len(self.pagerank_scores):
            print("Chỉ số tài liệu không hợp lệ")
            return
        
        print(f"\n" + "="*60)
        print(f"MINH HỌA TÍNH PAGERANK CHO TÀI LIỆU {doc_index}")
        print("="*60)
        
        filename = documents_info[doc_index]['filename'] if doc_index < len(documents_info) else f"Doc_{doc_index}"
        print(f"Tài liệu: {filename}")
        
        n = len(self.pagerank_scores)
        
        # Tìm các tài liệu liên kết đến tài liệu này
        incoming_links = []
        for i in range(n):
            if self.adjacency_matrix[i, doc_index] == 1:
                incoming_links.append(i)
        
        print(f"\n1. CÁC TÀI LIỆU LIÊN KẾT ĐẾN:")
        if incoming_links:
            for link_idx in incoming_links:
                link_filename = documents_info[link_idx]['filename'] if link_idx < len(documents_info) else f"Doc_{link_idx}"
                outgoing_count = np.sum(self.adjacency_matrix[link_idx])
                print(f"   - {link_filename} (có {int(outgoing_count)} liên kết ra)")
        else:
            print("   - Không có liên kết đến")
        
        print(f"\n2. CÔNG THỨC PAGERANK:")
        print(f"   PR({filename}) = (1-d)/N + d * Σ(PR(T_i)/C(T_i))")
        print(f"   Trong đó:")
        print(f"     - d = {self.damping_factor} (damping factor)")
        print(f"     - N = {n} (tổng số tài liệu)")
        print(f"     - T_i: các tài liệu liên kết đến {filename}")
        print(f"     - C(T_i): số liên kết ra từ T_i")
        
        # Tính từng thành phần
        teleport_term = (1 - self.damping_factor) / n
        print(f"\n3. TÍNH TOÁN CHI TIẾT:")
        print(f"   a) Teleport term = (1-d)/N = (1-{self.damping_factor})/{n} = {teleport_term:.6f}")
        
        link_contribution = 0
        print(f"   b) Link contribution:")
        if incoming_links:
            for link_idx in incoming_links:
                link_filename = documents_info[link_idx]['filename'] if link_idx < len(documents_info) else f"Doc_{link_idx}"
                pr_link = self.pagerank_scores[link_idx]
                outgoing_count = np.sum(self.adjacency_matrix[link_idx])
                contribution = pr_link / outgoing_count if outgoing_count > 0 else 0
                link_contribution += contribution
                print(f"      - Từ {link_filename}: PR = {pr_link:.6f}, "
                      f"Outgoing = {int(outgoing_count)}, "
                      f"Contribution = {pr_link:.6f}/{int(outgoing_count)} = {contribution:.6f}")
            
            print(f"      Tổng link contribution = {link_contribution:.6f}")
        else:
            print(f"      Không có liên kết → contribution = 0")
        
        final_pagerank = teleport_term + self.damping_factor * link_contribution
        print(f"\n4. KẾT QUẢ CUỐI CÙNG:")
        print(f"   PR({filename}) = {teleport_term:.6f} + {self.damping_factor} * {link_contribution:.6f}")
        print(f"   PR({filename}) = {teleport_term:.6f} + {self.damping_factor * link_contribution:.6f}")
        print(f"   PR({filename}) = {final_pagerank:.6f}")
        
        actual_pagerank = self.pagerank_scores[doc_index]
        print(f"\n   PageRank thực tế: {actual_pagerank:.6f}")
        print(f"   Sai số: {abs(final_pagerank - actual_pagerank):.8f}")
    
    def compare_with_tfidf(self, tfidf_scores, documents_info):
        """
        So sánh kết quả PageRank với TF-IDF
        """
        print(f"\n" + "="*60)
        print("SO SÁNH PAGERANK VỚI TF-IDF")
        print("="*60)
        
        n = len(self.pagerank_scores)
        
        # Tạo bảng so sánh
        comparison = []
        for i in range(n):
            filename = documents_info[i]['filename'] if i < len(documents_info) else f"Doc_{i}"
            comparison.append({
                'filename': filename,
                'pagerank': self.pagerank_scores[i],
                'tfidf': tfidf_scores[i] if i < len(tfidf_scores) else 0,
                'pagerank_rank': 0,
                'tfidf_rank': 0
            })
        
        # Tính thứ hạng
        sorted_by_pagerank = sorted(comparison, key=lambda x: x['pagerank'], reverse=True)
        sorted_by_tfidf = sorted(comparison, key=lambda x: x['tfidf'], reverse=True)
        
        for rank, item in enumerate(sorted_by_pagerank):
            item['pagerank_rank'] = rank + 1
        
        for rank, item in enumerate(sorted_by_tfidf):
            item['tfidf_rank'] = rank + 1
        
        # Hiển thị so sánh
        print(f"{'Tài liệu':<15} {'PageRank':<12} {'Hạng PR':<8} {'TF-IDF':<12} {'Hạng TF':<8} {'Chênh lệch':<10}")
        print("-" * 75)
        
        for item in sorted(comparison, key=lambda x: x['pagerank'], reverse=True)[:10]:
            rank_diff = abs(item['pagerank_rank'] - item['tfidf_rank'])
            print(f"{item['filename'][:13]:<15} {item['pagerank']:<12.6f} "
                  f"{item['pagerank_rank']:<8} {item['tfidf']:<12.4f} "
                  f"{item['tfidf_rank']:<8} {rank_diff:<10}")
        
        # Tính correlation
        pagerank_values = [item['pagerank'] for item in comparison]
        tfidf_values = [item['tfidf'] for item in comparison]
        correlation = np.corrcoef(pagerank_values, tfidf_values)[0, 1]
        
        print(f"\nHệ số tương quan giữa PageRank và TF-IDF: {correlation:.4f}")
        
        if correlation > 0.7:
            interpretation = "tương quan mạnh"
        elif correlation > 0.4:
            interpretation = "tương quan vừa"
        elif correlation > 0.2:
            interpretation = "tương quan yếu"
        else:
            interpretation = "không tương quan"
        
        print(f"Giải thích: {interpretation}")


def main():
    print("="*60)
    print("CHƯƠNG TRÌNH TÍNH PAGERANK CHO TÀI LIỆU")
    print("="*60)
    
    # Khởi tạo TF-IDF calculator để có dữ liệu
    tfidf_calc = TFIDFCalculator()
    base_path = "/Users/yoliephan/Library/CloudStorage/OneDrive-Personal/Tài liệu/MASTER 2025/Đợt 1/Xử lý Ngôn Ngữ Tự Nhiên/MY"
    data_folder = os.path.join(base_path, "DUC_TEXT", "train")
    
    # Tải dữ liệu
    tfidf_calc.load_corpus(data_folder)
    
    if tfidf_calc.document_count == 0:
        print("Không tìm thấy tài liệu nào!")
        return
    
    # Xây dựng ma trận TF-IDF và cosine similarity
    print(f"\nXây dựng ma trận để tính PageRank...")
    tfidf_matrix, vocab_list, word_to_index = tfidf_calc.build_tfidf_matrix()
    cosine_matrix = tfidf_calc.build_cosine_matrix(tfidf_matrix)
    
    # Khởi tạo PageRank calculator
    pagerank_calc = PageRankCalculator(damping_factor=0.85, max_iterations=50, tolerance=1e-6)
    
    # Tạo ma trận kề từ cosine similarity
    adjacency_matrix = pagerank_calc.create_adjacency_matrix(cosine_matrix, threshold=0.4)
    
    # Tạo ma trận chuyển tiếp
    transition_matrix = pagerank_calc.create_transition_matrix()
    
    # Tính PageRank
    pagerank_scores, history = pagerank_calc.calculate_pagerank_manual(tfidf_calc.documents)
    
    # Minh họa chi tiết cho một tài liệu
    if tfidf_calc.document_count > 0:
        pagerank_calc.demonstrate_pagerank_calculation(tfidf_calc.documents, doc_index=0)
    
    # So sánh với TF-IDF
    if tfidf_calc.document_count > 0:
        # Tính tổng TF-IDF cho mỗi tài liệu
        tfidf_scores = [np.sum(tfidf_matrix[i]) for i in range(tfidf_calc.document_count)]
        pagerank_calc.compare_with_tfidf(tfidf_scores, tfidf_calc.documents)
    
    # Hiển thị biểu đồ hội tụ
    print(f"\n" + "="*60)
    print("QUÁ TRÌNH HỘI TỤ PAGERANK")
    print("="*60)
    
    print(f"{'Vòng':<6} {'Thay đổi':<15} {'Trạng thái':<15}")
    print("-" * 40)
    
    for iteration, change, scores in history[-10:]:  # Hiển thị 10 vòng cuối
        if change < pagerank_calc.tolerance:
            status = "Đã hội tụ"
        elif change < 0.001:
            status = "Gần hội tụ"
        else:
            status = "Đang hội tụ"
        print(f"{iteration:<6} {change:<15.8f} {status:<15}")


if __name__ == "__main__":
    main()
