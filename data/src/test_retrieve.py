# src/test_retrieve.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 初始化和构建时一致的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh"
)

# 加载本地持久化的向量库
vector_db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

def search_knowledge(query):
    # 相似度检索，返回top3最相似结果
    results = vector_db.similarity_search(query, k=3)
    for idx, r in enumerate(results):
        print(f"===== 第{idx+1}条检索结果 =====")
        print(f"元数据信息：{r.metadata}")
        print(f"匹配内容：{r.page_content}\n")
    return results

# 测试用例1：发烧请假（预期返回请假流程内容）
if __name__ == "__main__":
    print("【测试1：查询发烧请假】")
    search_knowledge("我发烧了怎么办？")

    print("="*50)
    # 测试用例2：查询奖学金
    print("【测试2：查询奖学金申请条件】")
    search_knowledge("奖学金怎么申请？")