import os
import fitz
import warnings
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
from langchain.tools import Tool
from langgraph.graph import StateGraph
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("❌ HUGGINGFACE_TOKEN bulunamadı! Lütfen .env dosyanızı kontrol edin.")

login(hf_token)

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=hf_token,
    temperature=0.5,
    model_kwargs={"max_length": 1024}
)

project_dir = os.path.dirname(__file__)
pdf_file_path = os.path.join(project_dir, "files")
os.makedirs(pdf_file_path, exist_ok=True)

def get_pdf():
    """'files' klasöründeki tüm PDF dosyalarını liste olarak döndürür."""
    return [f for f in os.listdir(pdf_file_path) if f.endswith(".pdf")]


def extract_text_from_pdf(pdf_filename):
    """Belirtilen PDF dosyasından metni çıkarır."""

    pdf_path = pdf_filename if os.path.isabs(pdf_filename) else os.path.join(pdf_file_path, pdf_filename)

    if not os.path.exists(pdf_path):
        return f"⚠️ Dosya bulunamadı: {pdf_path}"

    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join([page.get_text("text") for page in doc])
            return text.strip() if text.strip() else "⚠️ PDF içeriği boş veya okunamadı."
    except Exception as e:
        return f"Hata: {e}"


def summarize_text(text):
    """Qwen modeli ile metni özetler."""
    try:
        truncated_text = text[:2000]  # API sınırlarına göre kesildi
        prompt = f"Bu metni özetle:\n\n{truncated_text}"
        response = llm.run(prompt)
        return response
    except Exception as e:
        return f"Hata: {e}"


def translate_text(text, target_language="English"):
    """Metni hedef dile çevirir (varsayılan: İngilizce)."""
    try:
        truncated_text = text[:2000]
        prompt = f"Bu metni {target_language} diline çevir:\n\n{truncated_text}"
        response = llm.run(prompt)
        return response
    except Exception as e:
        return f"Hata: {e}"


def answer_questions(text, question):
    """Kullanıcının metinle ilgili sorduğu soruya cevap verir."""
    try:
        truncated_text = text[:2000]
        prompt = f"Aşağıdaki metne dayanarak şu soruya cevap ver: {question}\n\n{truncated_text}"
        response = llm.run(prompt)
        return response
    except Exception as e:
        return f"Hata: {e}"

tools = [
    Tool(name="PDF Listeleme", func=get_pdf, description="Klasördeki PDF dosyalarını listeler."),
    Tool(name="PDF Okuma", func=extract_text_from_pdf, description="Belirtilen bir PDF dosyasındaki metni okur."),
    Tool(name="Metin Özetleme", func=summarize_text, description="Metni özetler."),
    Tool(name="Çeviri", func=translate_text, description="Metni başka bir dile çevirir."),
    Tool(name="Soru-Cevap", func=answer_questions, description="Metne dayalı sorulara cevap verir."),
]

class AgentState(BaseModel):
    input: str

def my_agent(state: AgentState):
    """LangGraph için ana Agent fonksiyonu."""
    user_input = state.input
    response = llm.run(user_input)
    return {"output": response}

graph = StateGraph(AgentState)
graph.add_node("agent", my_agent)
graph.set_entry_point("agent")

agent_executor = graph.compile()

if __name__ == "__main__":
    print("🚀 LangGraph tabanlı Agent başlatıldı!")

    while True:
        command = input("\n🔍 Ne yapmak istiyorsunuz? (çıkmak için 'exit' yazın) ➡️ ")

        if command.lower() == "exit":
            print("🚀 Agent kapatılıyor...")
            break

        elif "listele" in command.lower():
            pdf_files = get_pdf()
            if pdf_files:
                print(f"📂 Klasörde bulunan PDF dosyaları: {', '.join(pdf_files)}")
            else:
                print("⚠️ Klasörde hiç PDF dosyası bulunamadı!")

        elif "oku" in command.lower():
            pdf_files = get_pdf()
            if not pdf_files:
                print("⚠️ Klasörde hiç PDF dosyası bulunamadı!")
                continue

            pdf_name = input("📄 Hangi PDF dosyasını okumak istiyorsunuz? (Tam dosya yolu veya sadece dosya adı) ➡️ ")

            pdf_text = extract_text_from_pdf(pdf_name)
            print(f"\n📖 PDF İçeriği:\n{pdf_text}")

        else:
            response = agent_executor.invoke(AgentState(input=command))
            print(f"\n✅ Agent Cevabı: {response['output']}")
