#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

def check_environment():
    """检查Python环境和依赖"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    # 检查必要的包
    required_packages = [
        'sentence_transformers',
        'faiss',
        'PyPDF2', 
        'flask',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def find_pdf_files():
    """查找PDF文件"""
    pdf_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.pdf', '.crdownload')):
            pdf_files.append(file)
    return pdf_files

def main():
    print("🏥 医学文档智能检索系统")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请安装必要的依赖包")
        return
    
    # 检查PDF文件
    pdf_files = find_pdf_files()
    if not pdf_files:
        print("\n⚠️  未找到PDF文件")
        print("请将PDF医学文档放在当前目录下")
        return
    
    print(f"\n📄 找到PDF文件: {pdf_files[0]}")
    
    # 启动系统
    print("\n🚀 启动医学检索系统...")
    print("📝 系统将自动:")
    print("   1. 解析PDF文档")
    print("   2. 生成文本embedding")
    print("   3. 构建检索索引")
    print("   4. 启动Web服务")
    print("\n⏳ 首次运行需要下载模型，请耐心等待...")
    print("🌐 启动后请访问: http://localhost:5001")
    print("\n" + "=" * 50)
    
    try:
        # 导入并运行主程序
        from medical_search import initialize_search_engine, app
        
        # 初始化搜索引擎
        initialize_search_engine()
        
        # 启动Flask应用
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except KeyboardInterrupt:
        print("\n\n👋 系统已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")

if __name__ == "__main__":
    main() 