import pymupdf
import os
from huggingface_hub import login
from dotenv import load_dotenv
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(hf_token)

# PDF dosyalarının saklandığı klasör
project_dir = os.path.dirname(__file__)
pdf_file_path = os.path.join(project_dir, "files")
os.makedirs(pdf_file_path, exist_ok=True)

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

def get_pdf():
    """'files' klasöründeki tüm PDF dosyalarını liste halinde döndürür."""
    return [os.path.join(pdf_file_path, f) for f in os.listdir(pdf_file_path) if f.endswith(".pdf")]

def extract_text_from_pdf(pdf_path):
    """PDF'ten metin çıkarma fonksiyonu."""
    try:
        doc = pymupdf.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip() if text.strip() else "PDF içeriği okunamadı."
    except Exception as e:
        return f"Hata: {e}"

def process_text_with_qwen(text, message):
    """Qwen modeli ile kullanıcının verdiği isteğe göre işlem yapar."""
    try:
        prompt = f"{message}:\n\n{text[:2000]}"
        response = llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        return f"İşlem hatası: {e}"

def process_pdf(pdf_path, message):
    """PDF'i okur, metni çıkarır ve kullanıcının isteğine göre işler."""
    text = extract_text_from_pdf(pdf_path)
    if text.startswith("Hata:") or not text:
        return "Bu PDF boş veya okunamıyor."
    return process_text_with_qwen(text, message)

if __name__ == "__main__":
    pdf_files = get_pdf()
    if not pdf_files:
        print("⚠️ 'files' klasöründe hiç PDF dosyası bulunamadı!")
    else:
        answer = input("Ne yapmak istiyorsunuz? ")
        for pdf_path in pdf_files:
            result = process_pdf(pdf_path, answer)
            print(f"\nİstek gerçekleştirildi: {result}")
