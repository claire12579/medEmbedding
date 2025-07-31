import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

class MedicalDocumentSearch:
    def __init__(self, model_name="./bge-m3"):
        """
        初始化医学文档检索系统
        """
        print("初始化医学文档检索系统...")
        
        try:
            print(f"正在加载本地BGE-M3模型: {model_name}")
            self.model = SentenceTransformer(model_name)
            print(f"✅ 已成功加载BGE-M3模型: {model_name}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("请检查模型文件是否完整")
            raise e
        
        self.index = None
        self.documents = []
        self.chunks = []
        self.chunk_embeddings = None
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF文件中提取文本
        """
        print(f"正在从PDF提取文本: {pdf_path}")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        print(f"已处理第 {page_num + 1} 页")
        except Exception as e:
            print(f"PDF读取错误: {e}")
            return ""
        
        print(f"PDF文本提取完成，总长度: {len(text)} 字符")
        return text
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        将文本分割成块，保持语义完整性
        """
        print(f"正在分割文本，块大小: {chunk_size}, 重叠: {overlap}")
        
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 如果当前段落加上当前块超过限制，则保存当前块
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': len(current_chunk)
                })
                chunk_id += 1
                
                # 保留重叠部分
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk += "\n" + para if current_chunk else para
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk)
            })
        
        print(f"文本分割完成，生成 {len(chunks)} 个文本块")
        return chunks
    
    def build_index(self, pdf_path: str):
        """
        构建文档索引
        """
        print("开始构建文档索引...")
        
        # 提取PDF文本
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("无法从PDF中提取文本")
        
        # 分割文本
        self.chunks = self.split_text_into_chunks(text)
        if not self.chunks:
            raise ValueError("文本分割失败")
        
        # 生成embeddings
        print("正在生成文本embeddings...")
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        self.chunk_embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        
        # 构建FAISS索引
        print("正在构建FAISS索引...")
        dimension = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        
        # 归一化embeddings（用于余弦相似度）
        faiss.normalize_L2(self.chunk_embeddings)
        self.index.add(self.chunk_embeddings.astype('float32'))
        
        print(f"索引构建完成! 索引维度: {dimension}, 文档块数量: {len(self.chunks)}")
        
        # 保存索引和数据
        self.save_index()
        
    def save_index(self, index_dir: str = "index"):
        """
        保存索引和相关数据
        """
        os.makedirs(index_dir, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(index_dir, "document.index"))
        
        # 保存文档块和embeddings
        with open(os.path.join(index_dir, "chunks.pkl"), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        np.save(os.path.join(index_dir, "embeddings.npy"), self.chunk_embeddings)
        
        print(f"索引已保存到 {index_dir} 目录")
        
    def load_index(self, index_dir: str = "index"):
        """
        加载已保存的索引
        """
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(os.path.join(index_dir, "document.index"))
            
            # 加载文档块
            with open(os.path.join(index_dir, "chunks.pkl"), 'rb') as f:
                self.chunks = pickle.load(f)
            
            # 加载embeddings
            self.chunk_embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
            
            print(f"索引已从 {index_dir} 目录加载")
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索相关文档片段
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用build_index或load_index")
        
        print(f"搜索查询: '{query}'")
        
        # 生成查询embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # 格式化结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'chunk_id': chunk['id'],
                    'text': chunk['text'],
                    'length': chunk['length']
                }
                results.append(result)
        
        print(f"搜索完成，返回 {len(results)} 个结果")
        return results
    
    def highlight_text(self, text: str, query: str) -> str:
        """
        高亮文本中的关键词
        """
        # 简单的关键词高亮
        words = query.split()
        highlighted_text = text
        
        for word in words:
            if len(word) > 1:  # 忽略单个字符
                # 使用HTML标记高亮
                highlighted_text = re.sub(
                    f'({re.escape(word)})', 
                    r'<mark>\1</mark>', 
                    highlighted_text, 
                    flags=re.IGNORECASE
                )
        
        return highlighted_text

# Flask Web应用
app = Flask(__name__)
CORS(app)

# 全局搜索引擎实例
search_engine = None

@app.route('/')
def index():
    """
    主页
    """
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def api_search():
    """
    搜索API接口
    """
    global search_engine
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': '查询不能为空'}), 400
        
        if search_engine is None:
            return jsonify({'error': '搜索引擎未初始化'}), 500
        
        # 执行搜索
        results = search_engine.search(query, top_k)
        
        # 添加高亮
        for result in results:
            result['highlighted_text'] = search_engine.highlight_text(result['text'], query)
        
        return jsonify({
            'query': query,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    """
    系统状态API
    """
    global search_engine
    
    status = {
        'initialized': search_engine is not None,
        'index_built': search_engine is not None and search_engine.index is not None,
        'chunk_count': len(search_engine.chunks) if search_engine and search_engine.chunks else 0
    }
    
    return jsonify(status)

def initialize_search_engine():
    """
    初始化搜索引擎
    """
    global search_engine
    
    print("初始化医学文档搜索引擎...")
    search_engine = MedicalDocumentSearch()
    
    # 尝试加载已有索引
    if search_engine.load_index():
        print("已加载现有索引")
        return
    
    # 寻找PDF文件
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf') or f.endswith('.crdownload')]
    
    if pdf_files:
        pdf_file = pdf_files[0]
        print(f"找到PDF文件: {pdf_file}")
        
        try:
            search_engine.build_index(pdf_file)
            print("索引构建完成!")
        except Exception as e:
            print(f"索引构建失败: {e}")
    else:
        print("未找到PDF文件")

if __name__ == '__main__':
    # 初始化搜索引擎
    initialize_search_engine()
    
    # 启动Flask应用
    print("启动Web服务器...")
    app.run(debug=True, host='0.0.0.0', port=5001) 