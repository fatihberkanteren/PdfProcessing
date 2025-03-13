import pymupdf
from transformers import pipeline
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(hf_token)

project_dir = os.path.dirname(__file__)
pdf_file_path = os.path.join(project_dir, "files")
os.makedirs(pdf_file_path, exist_ok=True)

def get_pdf():
    """'files' klasöründeki tüm PDF dosyalarını liste halinde döndürür."""
    pdf_files = [os.path.join(pdf_file_path, f) for f in os.listdir(pdf_file_path) if f.endswith(".pdf")]
    return pdf_files

def extract_text_from_pdf(pdf_path):
    """PDF'ten metin çıkarma fonksiyonu."""
    try:
        doc = pymupdf.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text if text.strip() else "PDF içeriği okunamadı."
    except Exception as e:
        return f"Hata: {e}"

# Hugging Face modelini TEK SEFERDE yükleyerek hız artırıldı
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    """Metni özetleyen Hugging Face modeli."""
    try:
        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Özetleme hatası: {e}"

def process_pdf(pdf_path):
    """PDF'i okur, metni çıkarır ve özetler."""
    text = extract_text_from_pdf(pdf_path)
    if "Hata:" in text or text.strip() == "":
        return "Bu PDF boş veya okunamıyor."
    return summarize_text(text)

if __name__ == "__main__":
    pdf_files = get_pdf()
    if not pdf_files:
        print("⚠️ 'files' klasöründe hiç PDF dosyası bulunamadı!")
    else:
        for pdf_path in pdf_files:
            summary = process_pdf(pdf_path)
            print(f"\n{pdf_path} Özet:", summary)
